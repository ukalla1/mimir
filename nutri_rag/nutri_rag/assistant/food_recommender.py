"""Gap-based food recommendation with GAT embedding neighbor expansion.

Three-step process:
1. DB nutrient query: find foods matching the gap targets
2. GAT neighbor expansion: expand each candidate with similar foods
3. Return ranked candidates with nutrient profiles
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import duckdb
import numpy as np

from nutri_rag.config import (
    DB_PATH,
    FOOD_EMBEDDINGS_PATH,
    GAT_NEIGHBORS_K,
    TEXT_EMBEDDINGS_PATH,
    TEXT_FDC_IDS_PATH,
)
from nutri_rag.search import (
    search_by_nutrient_target,
    get_nutrients,
    hybrid_rank,
)


@dataclass
class FoodOption:
    """A recommended food option with its nutrient profile."""
    fdc_id: int
    description: str
    nutrients: dict[str, float] = field(default_factory=dict)
    gat_similarity: float = 0.0      # cosine similarity from GAT (1.0 = seed itself)
    is_seed: bool = False             # True if this was a direct DB match
    preference_score: float = 0.5     # from preference DB (0-1)


class FoodRecommender:
    """Finds gap-filling foods and expands them via GAT embeddings."""

    def __init__(
        self,
        db_path: str = DB_PATH,
        embeddings_path: str = FOOD_EMBEDDINGS_PATH,
        text_embeddings_path: str = TEXT_EMBEDDINGS_PATH,
        text_fdc_ids_path: str = TEXT_FDC_IDS_PATH,
        neighbor_mode: str | None = None,
    ):
        self._db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None

        # Load GAT embeddings + build food_id_to_idx mapping
        self._embeddings = np.load(embeddings_path)
        self._food_id_to_idx: dict[int, int] | None = None
        self._idx_to_food_id: dict[int, int] | None = None

        # Text embeddings (Phase B): lazy-loaded only when hybrid mode is used.
        # The text embedding matrix may use a different ordering than GAT, so we
        # store an fdc_id → row-index mapping alongside.
        self._text_embeddings_path = text_embeddings_path
        self._text_fdc_ids_path = text_fdc_ids_path
        self._text_embeddings: np.ndarray | None = None
        self._fdc_to_text_idx: dict[int, int] | None = None

        # Food→similar food neighbor mode:
        #   "gat_only" — pure GAT cosine (current default, Phase B keeps this)
        #   "hybrid"   — score-fusion (α·gat + (1-α)·text) over food↔food pairs
        # Override via FOOD_NEIGHBOR_MODE env var or constructor arg.
        self._neighbor_mode = neighbor_mode or os.environ.get(
            "FOOD_NEIGHBOR_MODE", "gat_only"
        )

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(self._db_path, read_only=True)
        return self._con

    def _ensure_id_mapping(self):
        """Build fdc_id <-> GAT index mapping from the database."""
        if self._food_id_to_idx is not None:
            return

        df = self.con.execute("SELECT fdc_id FROM nodes_food").df()
        fdc_ids = df["fdc_id"].astype(int).tolist()
        self._food_id_to_idx = {fid: i for i, fid in enumerate(fdc_ids)}
        self._idx_to_food_id = {i: fid for i, fid in enumerate(fdc_ids)}

    def _ensure_text_embeddings(self):
        """Lazy-load text embeddings + fdc_id mapping for hybrid mode."""
        if self._text_embeddings is not None:
            return
        text_emb = np.load(self._text_embeddings_path)
        # Defensive L2-normalize (file is normally pre-normalized but enforce
        # the invariant so dot products are cosine).
        text_norms = np.linalg.norm(text_emb, axis=1, keepdims=True)
        text_norms = np.where(text_norms == 0, 1.0, text_norms)
        self._text_embeddings = text_emb / text_norms
        text_fdc_ids = np.load(self._text_fdc_ids_path)
        self._fdc_to_text_idx = {
            int(fid): i for i, fid in enumerate(text_fdc_ids)
        }

    def _gat_neighbors(self, fdc_id: int, k: int = GAT_NEIGHBORS_K) -> list[tuple[int, float]]:
        """Find GAT embedding neighbors for a food.

        Returns list of (fdc_id, cosine_similarity) tuples.
        """
        self._ensure_id_mapping()

        idx = self._food_id_to_idx.get(fdc_id)
        if idx is None:
            return []

        # Cosine similarity
        query_emb = self._embeddings[idx]
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)

        if query_norm == 0:
            return []

        similarities = self._embeddings @ query_emb / (norms * query_norm + 1e-8)
        # Get top-k+1 (includes self), then exclude self
        top_indices = np.argsort(similarities)[::-1][:k + 1]

        neighbors = []
        for neighbor_idx in top_indices:
            neighbor_idx = int(neighbor_idx)
            if neighbor_idx == idx:
                continue
            neighbor_fdc_id = self._idx_to_food_id.get(neighbor_idx)
            if neighbor_fdc_id is not None:
                neighbors.append((neighbor_fdc_id, float(similarities[neighbor_idx])))
            if len(neighbors) >= k:
                break

        return neighbors

    def _hybrid_neighbors(
        self,
        fdc_id: int,
        k: int = GAT_NEIGHBORS_K,
        alpha: float = 0.5,
        available_fdc_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Score-fusion food↔food neighbor search (Phase B addition).

        Same scoring shape as HealthyFoodSubs evaluate_hybrid: both query and
        candidate are graph nodes, so q_text and q_gat both exist for the seed.
        Returns list of (neighbor_fdc_id, hybrid_score) sorted by score.
        """
        self._ensure_id_mapping()
        self._ensure_text_embeddings()

        gat_idx = self._food_id_to_idx.get(fdc_id)
        text_idx = self._fdc_to_text_idx.get(fdc_id)
        if gat_idx is None or text_idx is None:
            return []

        # L2-normalize q_gat (self._embeddings is the raw npy file, not
        # normalized — GATIndex inside hybrid_rank IS normalized, so we must
        # normalize the query to make the dot product be true cosine).
        raw_q_gat = self._embeddings[gat_idx]
        q_gat_norm = float(np.linalg.norm(raw_q_gat))
        if q_gat_norm == 0:
            return []
        q_gat = raw_q_gat / q_gat_norm
        q_text = self._text_embeddings[text_idx]
        # Same scoring shape as HealthyFoodSubs evaluate_hybrid
        # (nutri_graph/scripts/eval_food_subs.py:176). HFS hybrid scores
        # predict the quality of this neighbor expansion. See Phase D Gap 2
        # in plans/vectorized-twirling-valley.md.

        # Use the unified primitive so this stays aligned with NutriBench v5
        # and (Phase C) recommend_v2. Excluding self via structured_filter.
        df = hybrid_rank(
            q_text=q_text,
            q_gat=q_gat,
            candidate_fdc_ids=list(available_fdc_ids) if available_fdc_ids else None,
            alpha=alpha,
            structured_filter=(lambda fid: fid != int(fdc_id)),
            k=k,
            db_path=self._db_path,
        )
        return [(int(row["fdc_id"]), float(row["total"])) for _, row in df.iterrows()]

    def recommend(
        self,
        targets: dict[str, float],
        exclude_fdc_ids: set[int] | None = None,
        n_seeds: int = 5,
        n_neighbors: int = GAT_NEIGHBORS_K,
        available_fdc_ids: set[int] | None = None,
        alpha: float = 0.5,
    ) -> list[FoodOption]:
        """Find foods that fill nutritional gaps, expanded via GAT.

        Args:
            targets: Dict with protein_g, fat_g, carb_g, energy_kcal.
            exclude_fdc_ids: Foods already eaten (to avoid recommending).
            n_seeds: Number of seed candidates from DB query.
            n_neighbors: GAT neighbors per seed.
            available_fdc_ids: Hard availability filter. When provided, both
                seeds and expanded neighbors are restricted to this set.
                None (default) disables filtering — current behavior preserved.
            alpha: GAT weight for hybrid neighbor mode (ignored in gat_only).

        Returns:
            List of FoodOption, seeds first then their GAT/hybrid neighbors.
        """
        exclude = exclude_fdc_ids or set()

        # Determine which macro has the biggest gap
        macro_map = {
            "protein_g": ("Protein", targets.get("protein_g", 0)),
            "fat_g": ("Total lipid (fat)", targets.get("fat_g", 0)),
            "carb_g": ("Carbohydrate, by difference", targets.get("carb_g", 0)),
        }

        # Sort by target amount descending — biggest need first
        sorted_macros = sorted(macro_map.items(), key=lambda x: x[1][1], reverse=True)

        all_options: list[FoodOption] = []
        seen_fdc_ids: set[int] = set(exclude)

        # Query DB for the top gap-filling nutrient. Oversample when an
        # availability filter is set so we can still reach n_seeds after
        # dropping unavailable foods.
        primary_nutrient_name, _ = sorted_macros[0][1]
        sql_limit = n_seeds + len(exclude)
        if available_fdc_ids is not None:
            # Generous oversample; SQL ranking by macro amount is cheap.
            sql_limit = max(sql_limit * 10, 50)
        seed_df = search_by_nutrient_target(
            self.con,
            nutrient_name=primary_nutrient_name,
            min_amount=5.0,
            limit=sql_limit,
        )

        # Collect seed candidates (apply availability filter if provided)
        seeds: list[FoodOption] = []
        for _, row in seed_df.iterrows():
            fdc_id = int(row["fdc_id"])
            if fdc_id in seen_fdc_ids:
                continue
            if available_fdc_ids is not None and fdc_id not in available_fdc_ids:
                continue
            seen_fdc_ids.add(fdc_id)

            nutrients = get_nutrients(self.con, fdc_id, key_only=True)
            seeds.append(FoodOption(
                fdc_id=fdc_id,
                description=row["description"],
                nutrients=nutrients,
                gat_similarity=1.0,
                is_seed=True,
            ))
            if len(seeds) >= n_seeds:
                break

        # Expand each seed with food↔food similarity neighbors.
        # Dispatch by self._neighbor_mode (FOOD_NEIGHBOR_MODE env var):
        #   "gat_only" — current behavior, pure GAT cosine
        #   "hybrid"   — score-fusion (α·gat + (1-α)·text), new in Phase B
        for seed in seeds:
            all_options.append(seed)

            if self._neighbor_mode == "hybrid":
                neighbors = self._hybrid_neighbors(
                    seed.fdc_id, k=n_neighbors, alpha=alpha,
                    available_fdc_ids=available_fdc_ids,
                )
            else:
                neighbors = self._gat_neighbors(seed.fdc_id, k=n_neighbors)
                # Availability post-filter for fair comparison with hybrid mode
                if available_fdc_ids is not None:
                    neighbors = [
                        (fid, s) for fid, s in neighbors if fid in available_fdc_ids
                    ]

            for neighbor_fdc_id, similarity in neighbors:
                if neighbor_fdc_id in seen_fdc_ids:
                    continue
                seen_fdc_ids.add(neighbor_fdc_id)

                # Look up description and nutrients
                desc_df = self.con.execute(
                    f"SELECT description FROM nodes_food WHERE fdc_id = {neighbor_fdc_id}"
                ).df()
                if len(desc_df) == 0:
                    continue

                nutrients = get_nutrients(self.con, neighbor_fdc_id, key_only=True)
                all_options.append(FoodOption(
                    fdc_id=neighbor_fdc_id,
                    description=desc_df.iloc[0]["description"],
                    nutrients=nutrients,
                    gat_similarity=similarity,
                    is_seed=False,
                ))

        return all_options

    # ──────────────────────────────────────────────────────────────────
    # Phase C: target-as-query recommendation (additive — does NOT replace
    # recommend() above; dispatched at pipeline level via RECOMMEND_MODE).
    # ──────────────────────────────────────────────────────────────────

    def _load_meal_category_filter(self) -> tuple[set[int], bool] | None:
        """Load curated meal-shape filter, or None to skip filtering.

        Returns (allowed_category_ids, include_uncategorized) or None if no
        filter is configured. Most USDA foods have NULL food_category_id
        (78% incl. Apples, raw) so `include_uncategorized` defaults true.
        """
        import json as _json
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "meal_categories.json",
        )
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        allowed = data.get("allowed_category_ids", [])
        include_unc = bool(data.get("include_uncategorized", True))
        return ({int(x) for x in allowed}, include_unc)

    def recommend_v2(
        self,
        targets: dict[str, float],
        exclude_fdc_ids: set[int] | None = None,
        available_fdc_ids: set[int] | None = None,
        alpha: float = 0.5,
        structured_weight: float = 0.5,
        n_results: int = 10,
    ) -> list[FoodOption]:
        """Target-as-query recommendation (nutrition→Food).

        Replaces v1's "SQL top-K by macro amount" with HealthyFoodSubs-style
        score fusion using nutrient-node GAT embeddings + text descriptor of
        the target + a structured macro_match term:

            score(x) = alpha · cos(q_gat,  GAT[x])
                     + (1-alpha) · cos(q_text, text[x])
                     + structured_weight · macro_match(target, x)

        Crucially, the query is the nutritional target itself — NOT the
        eaten food. So the recommendation is anchored on "what would fill
        this gap" rather than "what's similar to what was eaten."
        """
        from nutri_rag.assistant.target_encoder import encode_target, macro_match

        q_text, q_gat = encode_target(targets)

        # Candidate pool: all foods, with optional category + availability filters.
        cat_filter = self._load_meal_category_filter()
        if cat_filter is not None:
            allowed_ids, include_unc = cat_filter
            clauses = []
            if include_unc:
                clauses.append("food_category_id IS NULL")
            if allowed_ids:
                ids_str = ",".join(str(c) for c in sorted(allowed_ids))
                clauses.append(f"food_category_id IN ({ids_str})")
            where = " OR ".join(clauses) if clauses else "1=1"
            pool_df = self.con.execute(
                f"SELECT fdc_id FROM nodes_food WHERE {where}"
            ).df()
        else:
            pool_df = self.con.execute("SELECT fdc_id FROM nodes_food").df()
        pool = pool_df["fdc_id"].astype(int).tolist()

        if exclude_fdc_ids:
            pool = [f for f in pool if f not in exclude_fdc_ids]
        if available_fdc_ids is not None:
            pool = [f for f in pool if f in available_fdc_ids]

        if not pool:
            return []

        df = hybrid_rank(
            q_text=q_text,
            q_gat=q_gat,
            candidate_fdc_ids=pool,
            alpha=alpha,
            structured_score=lambda fid: macro_match(
                targets, get_nutrients(self.con, fid, key_only=True)
            ),
            structured_weight=structured_weight,
            k=n_results,
            db_path=self._db_path,
        )

        options: list[FoodOption] = []
        for _, row in df.iterrows():
            fid = int(row["fdc_id"])
            options.append(FoodOption(
                fdc_id=fid,
                description=row["description"],
                nutrients=get_nutrients(self.con, fid, key_only=True),
                gat_similarity=float(row["total"]),
                is_seed=True,  # all v2 results are "primary" — no seed/neighbor split
            ))
        return options
