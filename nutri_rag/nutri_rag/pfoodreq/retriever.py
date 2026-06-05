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
        mode: str | None = None,
    ) -> list[dict]:
        """Score candidate recipes using text + GAT embeddings.

        Args:
            mode: GAT scoring mode passed through to
                `RecipeVectorIndex.search_by_ids`. None falls back to the
                `RECIPE_SCORE_MODE` env var, which defaults to "hybrid"
                (query-conditioned pseudo-anchor) — the new default after
                Phase D Gap 1. Set to "pool_centroid" to reproduce the
                historical PFoodReq baseline scoring.

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
            mode=mode,
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
        mode: str | None = None,
    ) -> list[dict]:
        """Config C pipeline: tag → deterministic filter → GAT+text re-rank → top-k.

        Args:
            mode: GAT scoring mode (hybrid/pool_centroid/external_gat) —
                see `score_candidates`. Defaults to RECIPE_SCORE_MODE env var,
                which defaults to "hybrid" after Phase D Gap 1.

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
        scored = self.score_candidates(query_text, candidate_ids, top_k=top_k,
                                        lam=lam, mode=mode)

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

    # ────────────────────────────────────────────────────────────────────
    # Phase D-test: embedding-first retrieval that shares the exact same
    # code path the robot's MealRecommender uses (hybrid_rank_recipes).
    # ────────────────────────────────────────────────────────────────────

    def retrieve_embedding_first(
        self,
        query_text: str,
        positive_ingredients: list[str] | None = None,
        negative_ingredients: list[str] | None = None,
        nutrient_constraints: list[dict] | None = None,
        tag_name: str | None = None,
        top_k: int = PFOODREQ_TOP_K,
        lam: float = PFOODREQ_LAMBDA,
        constraints: str = "hard",
    ) -> list[dict]:
        """Embedding-first PFoodReq retrieval.

        No tag pre-filter, no constraint pre-filter. Hybrid-ranks ALL recipes
        via the same hybrid_rank_recipes primitive the robot meal layer uses.
        Then applies PFoodReq's constraints either as a hard filter (preserves
        task semantics) or as soft score penalties (research mode).

        constraints:
            "hard" — recipes violating any constraint are removed before top-K
                     (≈ legacy task definition; uses Phase D's retrieval path).
            "soft" — violations become score penalties; PFoodReq can measure
                     "how well does retrieval alone do?".
        """
        from nutri_rag.search import hybrid_rank_recipes
        self._ensure_embedder()

        query_vec = self.embedder.encode(
            [query_text],
            task_instruction=FOOD_SEARCH_INSTRUCTION,
        )[0]

        # When a tag is provided (PFoodReq case), restrict the candidate pool
        # to tag-matched recipes — PFoodReq queries are tag-anchored, and the
        # ground-truth recipes are specifically those tagged with `tag_name`.
        # "Embedding-first" here means embeddings are the primary RANKING
        # signal (vs pool centrality); it does NOT mean discarding obvious
        # hard signals like the tag itself. For the robot pipeline, no tag is
        # passed → score all 82k recipes (Phase D behavior preserved).
        tag_candidates_ctx = None
        if tag_name:
            tag_candidates_ctx = self.get_candidates_by_tag(tag_name)
            tag_candidate_ids = [int(c["recipe_id"]) for c in tag_candidates_ctx]
            if not tag_candidate_ids:
                return []
        else:
            tag_candidate_ids = None  # all 82k

        # We need recipe ingredient + nutrient context to apply constraints.
        # Pull all tag-matched recipes (small pool, ~hundreds-thousands) or a
        # generous top-N from the full corpus.
        if tag_candidate_ids is not None:
            pull = len(tag_candidate_ids)
        else:
            pull = max(top_k * 50, 1000)
        ranked = hybrid_rank_recipes(
            q_text=query_vec,
            candidate_recipe_ids=tag_candidate_ids,
            alpha=lam,
            k=pull,
            db_path=self.db_path,
        )
        if ranked.empty:
            return []

        survivor_ids = ranked["recipe_id"].astype(int).tolist()

        # Reuse the tag-fetched context if available (it already has ingredients
        # + nutrients). Otherwise fetch context for the embedding survivors.
        recipe_map = {}
        if tag_candidates_ctx is not None:
            for c in tag_candidates_ctx:
                # Provide the same keys as the legacy path for constraint filter
                recipe_map[int(c["recipe_id"])] = {
                    "recipe_id": int(c["recipe_id"]),
                    "recipe_name": c["recipe_name"],
                    "calories": c.get("calories"),
                    "protein": c.get("protein"),
                    "carbohydrates": c.get("carbohydrates"),
                    "saturated_fat": c.get("saturated_fat"),
                    "monounsaturated_fat": c.get("monounsaturated_fat"),
                    "polyunsaturated_fat": c.get("polyunsaturated_fat"),
                    "ingredients": c.get("ingredients", []),
                }
        else:
            ids_str = ",".join(str(r) for r in survivor_ids)
            con = duckdb.connect(self.db_path, read_only=True)
            try:
                recipes_df = con.execute(f"""
                    SELECT recipe_id, recipe_name, calories, protein, carbohydrates,
                           saturated_fat, monounsaturated_fat, polyunsaturated_fat
                    FROM nodes_recipe
                    WHERE recipe_id IN ({ids_str})
                """).df()
                ing_df = con.execute(f"""
                    SELECT recipe_id, ingredient_name
                    FROM edges_recipe_uses_food
                    WHERE recipe_id IN ({ids_str})
                    ORDER BY recipe_id, ingredient_name
                """).df()
            finally:
                con.close()

            for r in recipes_df.itertuples(index=False):
                recipe_map[int(r.recipe_id)] = {
                    "recipe_id": int(r.recipe_id),
                    "recipe_name": r.recipe_name,
                    "calories": r.calories,
                    "protein": r.protein,
                    "carbohydrates": r.carbohydrates,
                    "saturated_fat": r.saturated_fat,
                    "monounsaturated_fat": r.monounsaturated_fat,
                    "polyunsaturated_fat": r.polyunsaturated_fat,
                    "ingredients": [],
                }
            for r in ing_df.itertuples(index=False):
                if int(r.recipe_id) in recipe_map:
                    recipe_map[int(r.recipe_id)]["ingredients"].append(r.ingredient_name)

        positive_ingredients = positive_ingredients or []
        negative_ingredients = negative_ingredients or []
        nutrient_constraints = nutrient_constraints or []

        # Apply constraints
        candidates = [recipe_map[rid] for rid in survivor_ids if rid in recipe_map]
        if constraints == "hard":
            candidates = self.filter_by_constraints(
                candidates,
                positive_ingredients=positive_ingredients,
                negative_ingredients=negative_ingredients,
                nutrient_constraints=nutrient_constraints,
            )
        elif constraints == "soft":
            # Add a per-recipe "violation penalty" attribute used for re-sort
            for c in candidates:
                c["_violation_penalty"] = self._soft_constraint_penalty(
                    c, positive_ingredients, negative_ingredients, nutrient_constraints
                )
        else:
            raise ValueError(f"unknown constraints mode: {constraints!r}")

        # Merge embedding scores back, then optionally re-sort by combined-minus-penalty
        score_map = {int(row.recipe_id): (float(row.total), float(row.text_sim), float(row.gat_sim))
                     for row in ranked.itertuples(index=False)}
        for c in candidates:
            sc = score_map.get(int(c["recipe_id"]), (0.0, 0.0, 0.0))
            c["combined_score"] = sc[0]
            c["text_score"] = sc[1]
            c["gat_score"] = sc[2]
            if constraints == "soft":
                c["combined_score"] -= c.get("_violation_penalty", 0.0)

        candidates.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _soft_constraint_penalty(recipe, positive_ingredients, negative_ingredients,
                                   nutrient_constraints) -> float:
        """Per-recipe penalty in [0, ~3] for constraint violations (soft mode)."""
        penalty = 0.0
        ing_set = {n.lower() for n in recipe.get("ingredients", [])}
        # Each missing positive ingredient: -0.5
        for pos in positive_ingredients:
            if pos.lower() not in ing_set:
                penalty += 0.5
        # Each present negative ingredient: -0.5
        for neg in negative_ingredients:
            if neg.lower() in ing_set:
                penalty += 0.5
        # Each nutrient out of range: -0.5
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
                penalty += 0.5
        return penalty
