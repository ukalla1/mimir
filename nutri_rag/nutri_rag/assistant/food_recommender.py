"""Gap-based food recommendation with GAT embedding neighbor expansion.

Three-step process:
1. DB nutrient query: find foods matching the gap targets
2. GAT neighbor expansion: expand each candidate with similar foods
3. Return ranked candidates with nutrient profiles
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np

from nutri_rag.config import DB_PATH, FOOD_EMBEDDINGS_PATH, GAT_NEIGHBORS_K
from nutri_rag.search import search_by_nutrient_target, get_nutrients


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
    ):
        self._db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None

        # Load GAT embeddings + build food_id_to_idx mapping
        self._embeddings = np.load(embeddings_path)
        self._food_id_to_idx: dict[int, int] | None = None
        self._idx_to_food_id: dict[int, int] | None = None

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

    def recommend(
        self,
        targets: dict[str, float],
        exclude_fdc_ids: set[int] | None = None,
        n_seeds: int = 5,
        n_neighbors: int = GAT_NEIGHBORS_K,
    ) -> list[FoodOption]:
        """Find foods that fill nutritional gaps, expanded via GAT.

        Args:
            targets: Dict with protein_g, fat_g, carb_g, energy_kcal.
            exclude_fdc_ids: Foods already eaten (to avoid recommending).
            n_seeds: Number of seed candidates from DB query.
            n_neighbors: GAT neighbors per seed.

        Returns:
            List of FoodOption, seeds first then their GAT neighbors.
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

        # Query DB for the top gap-filling nutrient
        primary_nutrient_name, _ = sorted_macros[0][1]
        seed_df = search_by_nutrient_target(
            self.con,
            nutrient_name=primary_nutrient_name,
            min_amount=5.0,
            limit=n_seeds + len(exclude),
        )

        # Collect seed candidates
        seeds: list[FoodOption] = []
        for _, row in seed_df.iterrows():
            fdc_id = int(row["fdc_id"])
            if fdc_id in seen_fdc_ids:
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

        # Expand each seed with GAT neighbors
        for seed in seeds:
            all_options.append(seed)

            neighbors = self._gat_neighbors(seed.fdc_id, k=n_neighbors)
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
