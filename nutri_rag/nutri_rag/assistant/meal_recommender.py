"""Recipe-layer meal composition for the robot assistant (Phase D).

After Phases A/B/C produce a list of foods that fill the user's nutritional
gap, this stage uses those foods plus the user's pantry to retrieve concrete
meal candidates from the recipe corpus.

Pipeline:
    Step 1 — Build query vectors from the recommended foods.
             q_text = prose description of the target meal.
             q_gat  = normalized mean of GAT_food[recommended_fdc_ids].
                      Food and recipe GAT vectors share the same node-embedding
                      space (trained jointly), so this is a valid graph-space
                      query for recipes — no SQL pre-filter needed.

    Step 2 — Hybrid rank ALL recipes via hybrid_rank_recipes (recipe-store
             analog of hybrid_rank). Top-K candidates by embedding score.

    Step 3 — SQL post-fetch for survivors: ingredient sets, nutrient totals.
             Compute structured signals:
               overlap = |ingredients ∩ recommended_fdc_ids| / |recommended|
               missing = |ingredients \ available_fdc_ids| / |ingredients|
             (overlap rewards using the gap-filling foods; missing penalizes
              pantry gaps. User's spec: "contain required food = enough; less
              unavailable the better".)

    Step 4 — Re-rank with structured terms:
             final = embed_score + gamma·overlap - beta·missing

    Step 5 — Top-K recipes returned to caller (pipeline → LLM).

Design notes:
    - No hard pre-filter; all 82k recipes are scored. The recipe→food joint
      training ensures recipes that USE recommended foods cluster near them
      in graph space.
    - No availability hard exclusion; missing-ingredient count is a soft
      penalty only.
    - Tag/meal-type filter (lunch/breakfast) is opt-in via meal_type_tags
      argument.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import duckdb
import numpy as np

from nutri_rag.config import DB_PATH, FOOD_EMBEDDINGS_PATH
from nutri_rag.embedding import FOOD_SEARCH_INSTRUCTION
from nutri_rag.search import _get_embedder, hybrid_rank_recipes

# Defaults — overridable per call or via env vars
DEFAULT_MEAL_ALPHA = float(os.environ.get("MEAL_ALPHA", "0.5"))
DEFAULT_MEAL_GAMMA = float(os.environ.get("MEAL_GAMMA", "0.5"))
DEFAULT_MEAL_BETA = float(os.environ.get("MEAL_BETA", "0.3"))
DEFAULT_MEAL_TOP_K_CANDIDATES = int(os.environ.get("MEAL_TOP_K_CANDIDATES", "30"))
DEFAULT_MEAL_TOP_K_FINAL = int(os.environ.get("MEAL_TOP_K_FINAL", "5"))
DEFAULT_MEAL_MIN_OVERLAP = int(os.environ.get("MEAL_MIN_OVERLAP", "1"))
DEFAULT_MEAL_MACRO_TOLERANCE = float(os.environ.get("MEAL_MACRO_TOLERANCE", "0.5"))

# next_meal → DB tag name. Verified against nodes_tag:
#   breakfast=200 recipes, lunch=200, dinner-party=200, snacks=200.
# (no plain "dinner" or "snack" tag exists; "main-dish" also absent.)
_NEXT_MEAL_TO_TAG = {
    "breakfast": "breakfast",
    "lunch": "lunch",
    "dinner": "dinner-party",
    "snack": "snacks",
    "snacks": "snacks",
}


def _next_meal_to_tag(next_meal: str) -> str:
    """Map user-facing meal name to the corresponding DB tag (lowercased)."""
    return _NEXT_MEAL_TO_TAG.get((next_meal or "").lower().strip(), (next_meal or "lunch").lower())


def _ingredients_contain_any(ingredients: list[str], terms: list[str]) -> bool:
    """Case-insensitive substring match: any term ⊂ any ingredient (or vice versa).

    Handles the mismatch between USDA preference descriptions ("Almonds, raw")
    and recipe ingredient strings ("almonds", "blanched almonds"). Splits the
    head before any comma so "Almonds, raw" → "almonds".
    """
    if not terms:
        return False
    norm_terms = [t.split(",")[0].strip().lower() for t in terms if t]
    norm_terms = [t for t in norm_terms if len(t) >= 3]  # avoid spurious 2-char matches
    if not norm_terms:
        return False
    for ing in ingredients:
        ing_lc = (ing or "").lower()
        if not ing_lc:
            continue
        for term in norm_terms:
            if term in ing_lc:
                return True
    return False


def _macros_in_range(
    context: dict,
    targets: dict[str, float] | None,
    tolerance: float,
) -> bool:
    """Per-recipe macros within ±tolerance of each target. tolerance=0.5 → ±50%."""
    if not targets or tolerance <= 0:
        return True
    pairs = [
        ("protein", "protein_g"),
        ("carbohydrates", "carb_g"),
        ("calories", "energy_kcal"),
        # fat sum already computed as fat_total
    ]
    for ctx_key, target_key in pairs:
        t = float(targets.get(target_key, 0) or 0)
        if t <= 0:
            continue
        v = float(context.get(ctx_key) or 0)
        lo = t * (1.0 - tolerance)
        hi = t * (1.0 + tolerance)
        if v < lo or v > hi:
            return False
    return True


@dataclass
class MealOption:
    """A meal candidate returned by MealRecommender."""
    recipe_id: int
    recipe_name: str
    ingredients: list[str] = field(default_factory=list)  # ingredient_name strings
    ingredient_fdc_ids: set[int] = field(default_factory=set)
    nutrients: dict[str, float] = field(default_factory=dict)
    text_sim: float = 0.0
    gat_sim: float = 0.0
    embed_score: float = 0.0           # alpha-blended embedding score
    overlap_count: int = 0             # count of recommended_fdc_ids present
    overlap_ratio: float = 0.0         # overlap_count / |recommended_fdc_ids|
    missing_count: int = 0             # count of ingredients not in pantry
    missing_ratio: float = 0.0         # missing_count / |ingredients|
    final_score: float = 0.0


class MealRecommender:
    """Retrieve recipes that fit the gap-filling foods + pantry constraints.

    Reuses Phase A's hybrid_rank_recipes primitive — same algorithm that
    PFoodReq's --retrieval-style embedding_first mode uses for evaluation.
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        food_gat_embeddings_path: str = FOOD_EMBEDDINGS_PATH,
    ):
        self._db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None
        # Load and normalize food GAT embeddings for q_gat construction.
        # We need food→graph-space mapping to build q_gat from recommended_fdc_ids.
        raw = np.load(food_gat_embeddings_path)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._food_gat = (raw / norms).astype(np.float32)
        self._fdc_to_food_idx: dict[int, int] | None = None

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(self._db_path, read_only=True)
        return self._con

    def _ensure_food_idx(self):
        if self._fdc_to_food_idx is not None:
            return
        df = self.con.execute("SELECT fdc_id FROM nodes_food").df()
        self._fdc_to_food_idx = {int(fid): i for i, fid in enumerate(df["fdc_id"].tolist())}

    def _build_q_gat(self, recommended_fdc_ids: list[int]) -> np.ndarray | None:
        """Mean of GAT_food[recommended_fdc_ids], normalized.

        Recipe-GAT lives in the same node-embedding space, so this is a
        valid graph-space query for recipes.
        """
        self._ensure_food_idx()
        vecs = []
        for fid in recommended_fdc_ids:
            idx = self._fdc_to_food_idx.get(int(fid))
            if idx is not None:
                vecs.append(self._food_gat[idx])
        if not vecs:
            return None
        q_gat = np.stack(vecs).mean(axis=0)
        n = float(np.linalg.norm(q_gat))
        if n == 0:
            return None
        return (q_gat / n).astype(np.float32)

    def _build_q_text(
        self,
        recommended_food_names: list[str],
        targets: dict[str, float] | None,
        next_meal: str,
    ) -> np.ndarray:
        """Prose descriptor of the target meal → text embedding."""
        # Limit food name list to keep prompts short
        head = ", ".join(recommended_food_names[:6])
        parts = [f"A balanced {next_meal} featuring {head}." if head
                 else f"A balanced {next_meal}."]
        if targets:
            parts.append(
                f"Approximately {targets.get('protein_g', 0):.0f}g protein, "
                f"{targets.get('fat_g', 0):.0f}g fat, "
                f"{targets.get('carb_g', 0):.0f}g carbohydrate, "
                f"and {targets.get('energy_kcal', 0):.0f} kcal per serving."
            )
        prose = " ".join(parts)
        q = _get_embedder().encode([prose], task_instruction=FOOD_SEARCH_INSTRUCTION)[0]
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        return q.astype(np.float32)

    def _fetch_recipe_context(self, recipe_ids: list[int]) -> dict[int, dict]:
        """Single batched lookup: ingredient list + nutrient totals for many recipes."""
        if not recipe_ids:
            return {}
        ids_str = ",".join(str(int(r)) for r in recipe_ids)
        # Recipe metadata + nutrient totals
        recipes_df = self.con.execute(f"""
            SELECT recipe_id, recipe_name, calories, protein, carbohydrates,
                   saturated_fat, monounsaturated_fat, polyunsaturated_fat
            FROM nodes_recipe
            WHERE recipe_id IN ({ids_str})
        """).df()
        # Ingredient list with fdc_ids
        ing_df = self.con.execute(f"""
            SELECT recipe_id, fdc_id, ingredient_name
            FROM edges_recipe_uses_food
            WHERE recipe_id IN ({ids_str})
        """).df()

        ctx: dict[int, dict] = {}
        for _, row in recipes_df.iterrows():
            rid = int(row["recipe_id"])
            ctx[rid] = {
                "recipe_name": row["recipe_name"],
                "calories": float(row["calories"] or 0),
                "protein": float(row["protein"] or 0),
                "carbohydrates": float(row["carbohydrates"] or 0),
                "fat_total": float(
                    (row.get("saturated_fat") or 0)
                    + (row.get("monounsaturated_fat") or 0)
                    + (row.get("polyunsaturated_fat") or 0)
                ),
                "ingredients": [],
                "ingredient_fdc_ids": set(),
            }
        for _, row in ing_df.iterrows():
            rid = int(row["recipe_id"])
            if rid in ctx:
                ctx[rid]["ingredients"].append(str(row["ingredient_name"]))
                fid = row.get("fdc_id")
                if fid is not None:
                    ctx[rid]["ingredient_fdc_ids"].add(int(fid))
        return ctx

    def _meal_type_candidate_ids(self, meal_type_tags: list[str] | None) -> list[int] | None:
        """Optional soft pre-filter by meal-type tag. None = no restriction."""
        if not meal_type_tags:
            return None
        tag_csv = ",".join(f"'{t.lower()}'" for t in meal_type_tags)
        df = self.con.execute(f"""
            SELECT DISTINCT e.recipe_id
            FROM edges_recipe_has_tag e
            JOIN nodes_tag t ON e.tag_id = t.tag_id
            WHERE LOWER(t.tag_name) IN ({tag_csv})
        """).df()
        ids = df["recipe_id"].astype(int).tolist()
        return ids if ids else None

    def _filter_with_relaxation(self, all_rows: list[dict], min_overlap: int) -> list[dict]:
        """Apply hard filters with progressive relaxation when zero survive.

        Cascade tiers (per user spec: "Relax constraints in order"):
          Tier 1: tag + no_disliked + overlap>=min + macros_ok  (strictest)
          Tier 2: tag + no_disliked + overlap>=min               (drop macros)
          Tier 3: tag + no_disliked + overlap>=1                 (drop macros, looser overlap)
          Tier 4: tag + no_disliked                               (drop overlap entirely)
          Tier 5: tag                                              (drop disliked — last resort)

        Returns the survivors at the first non-empty tier. Tier reached is
        attached to each row as `_tier_used` for debugging / transparency.
        """
        # Tier 1: strict
        survivors = [r for r in all_rows
                     if not r["_has_disliked"]
                     and r["overlap_count"] >= min_overlap
                     and r["_macros_ok"]]
        if survivors:
            for r in survivors:
                r["_tier_used"] = "strict"
            return survivors

        # Tier 2: drop macro range filter
        survivors = [r for r in all_rows
                     if not r["_has_disliked"]
                     and r["overlap_count"] >= min_overlap]
        if survivors:
            for r in survivors:
                r["_tier_used"] = "drop_macros"
            return survivors

        # Tier 3: loosen overlap to >=1
        if min_overlap > 1:
            survivors = [r for r in all_rows
                         if not r["_has_disliked"]
                         and r["overlap_count"] >= 1]
            if survivors:
                for r in survivors:
                    r["_tier_used"] = "drop_macros_loose_overlap"
                return survivors

        # Tier 4: drop overlap entirely
        survivors = [r for r in all_rows if not r["_has_disliked"]]
        if survivors:
            for r in survivors:
                r["_tier_used"] = "drop_overlap"
            return survivors

        # Tier 5: last resort — drop disliked filter too
        # (only happens when every tag-matched recipe contains a disliked ingredient)
        for r in all_rows:
            r["_tier_used"] = "all_relaxed"
        return list(all_rows)

    def recommend_meal(
        self,
        recommended_foods: list,                # list[FoodOption] from FoodRecommender
        targets: dict[str, float] | None = None,
        available_fdc_ids: set[int] | None = None,
        next_meal: str = "lunch",
        meal_type_tags: list[str] | None = None,
        disliked_names: list[str] | None = None,
        min_overlap: int = DEFAULT_MEAL_MIN_OVERLAP,
        nutrient_tolerance: float = DEFAULT_MEAL_MACRO_TOLERANCE,
        alpha: float = DEFAULT_MEAL_ALPHA,
        gamma: float = DEFAULT_MEAL_GAMMA,
        beta: float = DEFAULT_MEAL_BETA,
        top_k_candidates: int = DEFAULT_MEAL_TOP_K_CANDIDATES,
        top_k_final: int = DEFAULT_MEAL_TOP_K_FINAL,
    ) -> list[MealOption]:
        """Embedding-first recipe retrieval with hard requirement filters.

        Hard filters (all must pass; relaxation cascade if zero survive):
          - meal-type tag derived from next_meal (breakfast/lunch/dinner/snack)
          - recipe ingredients do NOT contain any disliked_names (user
            allergies / strong dislikes from PreferenceDB)
          - recipe uses >= min_overlap of the recommended foods
          - recipe's per-recipe macros within ±nutrient_tolerance of targets

        Relaxation cascade (when hard filters return zero):
          1. Drop the macro range filter (keep tag + disliked + overlap)
          2. Drop the overlap minimum (keep tag + disliked)
          3. Last resort: drop disliked too — return tag-matched best by embedding

        Args:
            recommended_foods: FoodOption list (from FoodRecommender.recommend*).
            targets: macro targets (used in q_text and in hard nutrient filter).
            available_fdc_ids: pantry — soft penalty signal only (per user spec).
            next_meal: maps to a meal-type tag (e.g. "dinner" → "dinner-party").
            meal_type_tags: override the automatic mapping. None = derive from next_meal.
            disliked_names: user dislikes/allergies (PreferenceDB descriptions).
                Recipes containing any of these are HARD-excluded (no relaxation
                drops this unless absolutely nothing else passes).
            min_overlap: minimum count of recommended foods that must appear.
            nutrient_tolerance: ±fraction tolerance around macro targets (0.5 = ±50%).
            alpha, gamma, beta: scoring weights for the soft re-rank step.
            top_k_candidates: pull this many from embedding rank before filtering.
            top_k_final: final number returned to caller.
        """
        if not recommended_foods:
            return []
        recommended_fdc_ids = [int(f.fdc_id) for f in recommended_foods]
        recommended_names = [str(f.description) for f in recommended_foods]
        rec_set = set(recommended_fdc_ids)

        # Step 1: query vectors
        q_text = self._build_q_text(recommended_names, targets, next_meal)
        q_gat = self._build_q_gat(recommended_fdc_ids)

        # Step 2: candidate pool = recipes tagged with the meal-type tag.
        # Hard filter via the tag; ~200 recipes per meal type in our DB.
        if meal_type_tags is None:
            meal_type_tags = [_next_meal_to_tag(next_meal)]
        candidate_pool = self._meal_type_candidate_ids(meal_type_tags)

        # Pull enough that hard filters have room — go wide within the tag pool
        # since the tag has already narrowed us to ~200 recipes.
        pull_k = top_k_candidates
        if candidate_pool is not None:
            pull_k = max(pull_k, len(candidate_pool))

        df = hybrid_rank_recipes(
            q_text=q_text,
            q_gat=q_gat,
            candidate_recipe_ids=candidate_pool,
            alpha=alpha,
            k=pull_k,
            db_path=self._db_path,
        )
        if df.empty:
            return []

        # Step 3: post-fetch ingredient + nutrient context for ALL scored recipes
        survivor_ids = df["recipe_id"].astype(int).tolist()
        ctx = self._fetch_recipe_context(survivor_ids)

        # Compute structured signals once, store on each row
        max_missing = 1
        all_rows = []
        for _, row in df.iterrows():
            rid = int(row["recipe_id"])
            c = ctx.get(rid)
            if c is None:
                continue
            ing_fdc = c["ingredient_fdc_ids"]
            overlap = len(ing_fdc & rec_set)
            overlap_ratio = (overlap / len(rec_set)) if rec_set else 0.0
            if available_fdc_ids is not None and ing_fdc:
                missing = len(ing_fdc - set(available_fdc_ids))
                missing_ratio = missing / len(ing_fdc)
            else:
                missing = 0
                missing_ratio = 0.0
            max_missing = max(max_missing, missing)
            all_rows.append({
                "recipe_id": rid,
                "embed_score": float(row["total"]),
                "text_sim": float(row["text_sim"]),
                "gat_sim": float(row["gat_sim"]),
                "overlap_count": overlap,
                "overlap_ratio": overlap_ratio,
                "missing_count": missing,
                "missing_ratio": missing_ratio,
                "context": c,
                # precompute hard-filter predicates so relaxation is cheap
                "_has_disliked": _ingredients_contain_any(c["ingredients"], disliked_names or []),
                "_macros_ok": _macros_in_range(c, targets, nutrient_tolerance),
            })

        # Step 4: apply hard filters with relaxation cascade
        struct_rows = self._filter_with_relaxation(
            all_rows, min_overlap=min_overlap,
        )
        if not struct_rows:
            return []

        # Step 5: re-rank survivors with structured terms (soft scoring)
        for sr in struct_rows:
            sr["final_score"] = (
                sr["embed_score"]
                + gamma * sr["overlap_ratio"]
                - beta * (sr["missing_count"] / max(max_missing, 1))
            )

        struct_rows.sort(key=lambda r: r["final_score"], reverse=True)
        top = struct_rows[:top_k_final]

        return [
            MealOption(
                recipe_id=sr["recipe_id"],
                recipe_name=sr["context"]["recipe_name"],
                ingredients=sr["context"]["ingredients"],
                ingredient_fdc_ids=sr["context"]["ingredient_fdc_ids"],
                nutrients={
                    "calories": sr["context"]["calories"],
                    "protein": sr["context"]["protein"],
                    "carbohydrates": sr["context"]["carbohydrates"],
                    "fat": sr["context"]["fat_total"],
                },
                text_sim=sr["text_sim"],
                gat_sim=sr["gat_sim"],
                embed_score=sr["embed_score"],
                overlap_count=sr["overlap_count"],
                overlap_ratio=sr["overlap_ratio"],
                missing_count=sr["missing_count"],
                missing_ratio=sr["missing_ratio"],
                final_score=sr["final_score"],
            )
            for sr in top
        ]
