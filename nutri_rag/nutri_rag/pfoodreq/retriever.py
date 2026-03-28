"""Recipe retrieval for PFoodReq benchmark.

Config C pipeline:
  all tag recipes → deterministic constraint filter → GAT+text re-rank → top-k → LLM
"""

from __future__ import annotations

import duckdb
import numpy as np

from nutri_rag.config import DB_PATH, PFOODREQ_LAMBDA, PFOODREQ_TOP_K
from nutri_rag.embedding import RecipeVectorIndex, TextEmbedder, FOOD_SEARCH_INSTRUCTION

# Map PFoodReq nutrient names to DB column(s)
_NUTRIENT_COL_MAP = {
    "carbohydrates": ["carbohydrates"],
    "protein": ["protein"],
    "fat": ["saturated_fat", "monounsaturated_fat", "polyunsaturated_fat"],
    "calories": ["calories"],
}


class RecipeRetriever:
    """Retrieve and score recipes for PFoodReq queries."""

    def __init__(
        self,
        db_path: str = DB_PATH,
        embedder: TextEmbedder | None = None,
        recipe_index: RecipeVectorIndex | None = None,
    ):
        self.db_path = db_path
        self.embedder = embedder
        self.recipe_index = recipe_index or RecipeVectorIndex()

    def _ensure_embedder(self):
        if self.embedder is None:
            self.embedder = TextEmbedder()

    def get_candidates_by_tag(self, tag_name: str) -> list[dict]:
        """Get all recipes under a cuisine tag with full context.

        Returns list of recipe dicts with ingredients and nutrients.
        """
        con = duckdb.connect(self.db_path, read_only=True)

        # Get recipes + nutrients
        recipes_rows = con.execute("""
            SELECT r.recipe_id, r.recipe_name, r.calories, r.protein,
                   r.carbohydrates, r.saturated_fat, r.monounsaturated_fat,
                   r.polyunsaturated_fat
            FROM nodes_recipe r
            JOIN edges_recipe_has_tag e ON r.recipe_id = e.recipe_id
            JOIN nodes_tag t ON e.tag_id = t.tag_id
            WHERE LOWER(t.tag_name) = LOWER(?)
        """, [tag_name]).fetchall()

        if not recipes_rows:
            con.close()
            return []

        recipe_map = {}
        for r in recipes_rows:
            recipe_map[r[0]] = {
                "recipe_id": r[0],
                "recipe_name": r[1],
                "calories": r[2],
                "protein": r[3],
                "carbohydrates": r[4],
                "saturated_fat": r[5],
                "monounsaturated_fat": r[6],
                "polyunsaturated_fat": r[7],
                "ingredients": [],
            }

        # Get ingredients
        placeholders = ",".join(["?"] * len(recipe_map))
        rids = list(recipe_map.keys())
        ingredients = con.execute(f"""
            SELECT e.recipe_id, e.ingredient_name
            FROM edges_recipe_uses_food e
            WHERE e.recipe_id IN ({placeholders})
            ORDER BY e.recipe_id, e.ingredient_name
        """, rids).fetchall()

        for recipe_id, ing_name in ingredients:
            if recipe_id in recipe_map:
                recipe_map[recipe_id]["ingredients"].append(ing_name)

        con.close()
        return list(recipe_map.values())

    def filter_by_constraints(
        self,
        candidates: list[dict],
        positive_ingredients: list[str],
        negative_ingredients: list[str],
        nutrient_constraints: list[dict],
    ) -> list[dict]:
        """Deterministic constraint filter.

        Removes recipes that:
        - contain any negative ingredient
        - are missing any positive ingredient
        - fall outside nutrient ranges
        """
        filtered = []
        for recipe in candidates:
            ing_set = {name.lower() for name in recipe.get("ingredients", [])}

            # Check negative ingredients (must NOT contain)
            skip = False
            for neg in negative_ingredients:
                if neg.lower() in ing_set:
                    skip = True
                    break
            if skip:
                continue

            # Check positive ingredients (must contain)
            for pos in positive_ingredients:
                if pos.lower() not in ing_set:
                    skip = True
                    break
            if skip:
                continue

            # Check nutrient constraints
            for nc in nutrient_constraints:
                nutrient = nc.get("nutrient", "")
                rng = nc.get("range", [])
                if len(rng) != 2:
                    continue
                cols = _NUTRIENT_COL_MAP.get(nutrient)
                if not cols:
                    continue
                val = sum(recipe.get(c, 0.0) or 0.0 for c in cols)
                if val < rng[0] or val > rng[1]:
                    skip = True
                    break
            if skip:
                continue

            filtered.append(recipe)

        return filtered

    def score_candidates(
        self,
        query_text: str,
        candidate_recipe_ids: list[int],
        top_k: int = PFOODREQ_TOP_K,
        lam: float = PFOODREQ_LAMBDA,
    ) -> list[dict]:
        """Score candidate recipes using text + GAT embeddings.

        Returns list of {recipe_id, combined_score, text_score, gat_score},
        sorted by combined_score descending.
        """
        self._ensure_embedder()

        query_vec = self.embedder.encode(
            [query_text],
            task_instruction=FOOD_SEARCH_INSTRUCTION,
        )[0]  # (dim,)

        results = self.recipe_index.search_by_ids(
            query_vector=query_vec,
            candidate_recipe_ids=candidate_recipe_ids,
            k=top_k,
            lam=lam,
        )

        return [
            {
                "recipe_id": rid,
                "combined_score": cs,
                "text_score": ts,
                "gat_score": gs,
            }
            for rid, cs, ts, gs in results
        ]

    def retrieve(
        self,
        query_text: str,
        tag_name: str,
        top_k: int = PFOODREQ_TOP_K,
        lam: float = PFOODREQ_LAMBDA,
        positive_ingredients: list[str] | None = None,
        negative_ingredients: list[str] | None = None,
        nutrient_constraints: list[dict] | None = None,
    ) -> list[dict]:
        """Config C pipeline: tag → deterministic filter → GAT+text re-rank → top-k.

        Returns list of recipe dicts with scores and full context,
        sorted by combined score descending.
        """
        # Step 1: Get ALL candidates by tag (with context)
        candidates = self.get_candidates_by_tag(tag_name)
        if not candidates:
            return []

        n_before = len(candidates)

        # Step 2: Deterministic constraint filter
        candidates = self.filter_by_constraints(
            candidates,
            positive_ingredients=positive_ingredients or [],
            negative_ingredients=negative_ingredients or [],
            nutrient_constraints=nutrient_constraints or [],
        )

        if not candidates:
            return []

        n_after = len(candidates)

        # Step 3: GAT+text re-ranking on filtered set, take top-k
        candidate_ids = [c["recipe_id"] for c in candidates]
        scored = self.score_candidates(query_text, candidate_ids, top_k=top_k, lam=lam)

        if not scored:
            return candidates[:top_k]  # fallback: return unscored

        # Merge scores into candidate context
        score_map = {s["recipe_id"]: s for s in scored}
        result = []
        for ctx in candidates:
            rid = ctx["recipe_id"]
            if rid in score_map:
                ctx["combined_score"] = score_map[rid]["combined_score"]
                ctx["text_score"] = score_map[rid]["text_score"]
                ctx["gat_score"] = score_map[rid]["gat_score"]
                result.append(ctx)

        result.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        return result[:top_k]
