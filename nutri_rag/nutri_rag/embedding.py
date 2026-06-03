"""Text embedding model wrapper and vector index for semantic food search.

Handles two modes:
1. Offline (build_embeddings.py): encode all 74K USDA descriptions, save to disk
2. Online (search time): encode query text, cosine search against pre-computed vectors
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from nutri_rag.config import (
    FOOD_EMBEDDINGS_PATH,
    NODE_EMBEDDINGS_PATH,
    RECIPE_IDS_PATH,
    RECIPE_TEXT_EMBEDDINGS_PATH,
    TEXT_EMBEDDING_DIM,
    TEXT_EMBEDDING_MODEL,
    TEXT_EMBEDDINGS_PATH,
    TEXT_FDC_IDS_PATH,
)


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract the last non-padding token's hidden state (Qwen3-Embedding pooling)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class TextEmbedder:
    """Wrapper around Qwen3-Embedding for encoding text into vectors."""

    def __init__(self, model_name: str = TEXT_EMBEDDING_MODEL, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        task_instruction: str | None = None,
    ) -> np.ndarray:
        """Encode texts into normalized embedding vectors.

        Args:
            texts: list of strings to encode.
            batch_size: encoding batch size.
            task_instruction: optional Qwen3-Embedding instruction prefix.
                For queries, use something like "Given a food name, retrieve
                the matching USDA food description". For documents (USDA
                descriptions), leave as None.

        Returns:
            numpy array of shape (len(texts), TEXT_EMBEDDING_DIM), L2-normalized.
        """
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch_texts = [str(t) if t is not None else "" for t in texts[start : start + batch_size]]

            if task_instruction:
                batch_texts = [
                    f"Instruct: {task_instruction}\nQuery: {t}" for t in batch_texts
                ]

            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**batch_dict)
            embeddings = _last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)


# Query instruction for food term -> USDA description matching
FOOD_SEARCH_INSTRUCTION = (
    "Given a food name or ingredient, retrieve the matching USDA food database entry"
)


class FoodVectorIndex:
    """Pre-computed vector index for cosine similarity search over USDA foods.

    Loads pre-computed embeddings and fdc_ids from disk, then supports
    fast numpy-based cosine search at query time.
    """

    def __init__(
        self,
        embeddings_path: str = TEXT_EMBEDDINGS_PATH,
        fdc_ids_path: str = TEXT_FDC_IDS_PATH,
    ):
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Pre-computed embeddings not found at {embeddings_path}. "
                "Run scripts/build_embeddings.py first."
            )
        self.embeddings = np.load(embeddings_path)  # (N, dim), already L2-normalized
        self.fdc_ids = np.load(fdc_ids_path)          # (N,)

    def search(self, query_vectors: np.ndarray, k: int = 3) -> list[list[tuple[int, float]]]:
        """Find top-k most similar foods for each query vector.

        Args:
            query_vectors: (Q, dim) array of L2-normalized query embeddings.
            k: number of results per query.

        Returns:
            List of Q lists, each containing k tuples of (fdc_id, cosine_similarity).
        """
        # Cosine similarity = dot product when vectors are L2-normalized
        scores = query_vectors @ self.embeddings.T  # (Q, N)

        results = []
        for i in range(scores.shape[0]):
            top_indices = np.argpartition(scores[i], -k)[-k:]
            top_indices = top_indices[np.argsort(scores[i, top_indices])[::-1]]
            results.append([
                (int(self.fdc_ids[idx]), float(scores[i, idx]), int(idx))
                for idx in top_indices
            ])
        return results


class GATIndex:
    """Pre-computed GAT embeddings for nutritional similarity re-ranking.

    These 64-dim embeddings are learned from the food-nutrient bipartite graph
    by nutri_graph's GATv2 model. Foods with similar nutritional profiles are
    close in this space, regardless of their text descriptions.

    The index ordering matches the DuckDB nodes_food table (same as text embeddings).
    """

    def __init__(self, embeddings_path: str = FOOD_EMBEDDINGS_PATH):
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"GAT embeddings not found at {embeddings_path}. "
                "Train nutri_graph first."
            )
        self.embeddings = np.load(embeddings_path)  # (74175, 64)
        # L2-normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms

    def get_vectors(self, indices: list[int]) -> np.ndarray:
        """Get GAT vectors for given array indices. Returns (len(indices), 64)."""
        return self.embeddings[indices]

    def cosine_similarity(self, idx_a: int, idx_b: int) -> float:
        """Cosine similarity between two foods by their array indices."""
        return float(self.embeddings[idx_a] @ self.embeddings[idx_b])

    def neighbors(self, idx: int, k: int = 5) -> list[tuple[int, float]]:
        """Find k nearest GAT neighbors (nutritionally similar foods).

        Returns list of (array_idx, cosine_similarity), excluding self.
        """
        sims = self.embeddings[idx] @ self.embeddings.T  # (N,)
        # Get top-(k+1) to account for self
        top_k = np.argpartition(sims, -(k + 1))[-(k + 1):]
        top_k = top_k[top_k != idx]  # remove self
        top_k = top_k[np.argsort(sims[top_k])[::-1]][:k]
        return [(int(i), float(sims[i])) for i in top_k]


class RecipeVectorIndex:
    """Pre-computed vector index for recipe retrieval (text + GAT embeddings).

    Loads:
    - Recipe text embeddings (Qwen3-Embedding over recipe names)
    - Recipe GAT embeddings (from node_embeddings.npy at RECIPE_OFFSET)
    """

    def __init__(
        self,
        text_embeddings_path: str = RECIPE_TEXT_EMBEDDINGS_PATH,
        recipe_ids_path: str = RECIPE_IDS_PATH,
        node_embeddings_path: str = NODE_EMBEDDINGS_PATH,
        db_path: str | None = None,
    ):
        if not os.path.exists(text_embeddings_path):
            raise FileNotFoundError(
                f"Recipe text embeddings not found at {text_embeddings_path}. "
                "Run scripts/build_recipe_embeddings.py first."
            )
        self.text_embeddings = np.load(text_embeddings_path)  # (N_recipes, 1024)
        self.recipe_ids = np.load(recipe_ids_path)             # (N_recipes,)

        # Build recipe_id → index mapping
        self._id_to_idx = {int(rid): i for i, rid in enumerate(self.recipe_ids)}

        # Load GAT recipe embeddings from node_embeddings
        self.gat_embeddings = None
        if os.path.exists(node_embeddings_path):
            self._load_gat_embeddings(node_embeddings_path, db_path)

    def _load_gat_embeddings(self, node_embeddings_path: str, db_path: str | None):
        """Extract recipe GAT embeddings from full node_embeddings using DB metadata."""
        import torch

        if db_path is None:
            from nutri_rag.config import DB_PATH
            db_path = DB_PATH

        # Get offsets from DB
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        num_foods = con.execute("SELECT COUNT(*) FROM nodes_food").fetchone()[0]
        num_nutrients = con.execute("SELECT COUNT(*) FROM nodes_nutrient").fetchone()[0]
        num_recipes = con.execute("SELECT COUNT(*) FROM nodes_recipe").fetchone()[0]
        con.close()

        recipe_offset = num_foods + num_nutrients

        # Load node embeddings
        if node_embeddings_path.endswith(".pt"):
            all_emb = torch.load(node_embeddings_path, map_location="cpu").numpy()
        else:
            all_emb = np.load(node_embeddings_path)

        if recipe_offset + num_recipes <= all_emb.shape[0]:
            recipe_emb = all_emb[recipe_offset:recipe_offset + num_recipes]
            # L2-normalize
            norms = np.linalg.norm(recipe_emb, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self.gat_embeddings = recipe_emb / norms
        else:
            print(f"[RecipeVectorIndex] Warning: node_embeddings has {all_emb.shape[0]} nodes "
                  f"but recipe_offset+num_recipes={recipe_offset + num_recipes}. "
                  "GAT embeddings not available (retrain GAT with INCLUDE_RECIPES=True).")

    def search_by_ids(
        self,
        query_vector: np.ndarray,
        candidate_recipe_ids: list[int] | None = None,
        k: int = 20,
        lam: float = 0.3,
        mode: str | None = None,
        q_gat: np.ndarray | None = None,
    ) -> list[tuple[int, float, float, float]]:
        """Score and rank candidate recipes by combined text + GAT similarity.

        Args:
            query_vector: (text_dim,) L2-normalized text query embedding.
            candidate_recipe_ids: list of recipe_ids to score. None = score all
                recipes (Phase D embedding-first style, ~4ms for 82k recipes).
            k: max results to return.
            lam: GAT weight in combined score.
            mode: GAT scoring mode (Phase D Gap 1).
                "hybrid"        — query-conditioned via pseudo-anchor (new default):
                                  q_gat* = GAT[text top-1 within pool], then
                                  gat_score = cos(q_gat*, recipe_gat). Same shape
                                  as NutriBench v5 and Phase B/C.
                "external_gat"  — caller supplies a graph-space query vec via
                                  `q_gat` (e.g. mean of food GAT vectors for the
                                  robot meal layer). Recipe-GAT lives in the
                                  same joint node-embedding space as food-GAT.
                "pool_centroid" — LEGACY: gat_score = cos(recipe_gat, mean(pool)).
                                  Pool-typicality, not query-conditioned. Kept
                                  for backward-compat with historical PFoodReq
                                  numbers; opt in via RECIPE_SCORE_MODE.
                None (default)  — reads RECIPE_SCORE_MODE env var, falls back
                                  to "hybrid" if unset.
            q_gat: graph-space query vector (used only when mode="external_gat").

        Returns:
            List of (recipe_id, combined_score, text_score, gat_score),
            sorted by combined_score descending.
        """
        # Default candidate set = all recipes (embedding-first Phase D style)
        if candidate_recipe_ids is None:
            indices = list(range(len(self.recipe_ids)))
            valid_ids = [int(rid) for rid in self.recipe_ids]
        else:
            # Map recipe_ids to indices, skip unknown
            indices = []
            valid_ids = []
            for rid in candidate_recipe_ids:
                idx = self._id_to_idx.get(int(rid))
                if idx is not None:
                    indices.append(idx)
                    valid_ids.append(int(rid))

        if not indices:
            return []

        idx_arr = np.array(indices)

        # Text scores
        text_scores = query_vector @ self.text_embeddings[idx_arr].T  # (N_candidates,)

        # GAT scores — three modes, picked via `mode` arg or env var
        if mode is None:
            mode = os.environ.get("RECIPE_SCORE_MODE", "hybrid")

        if self.gat_embeddings is None or lam <= 0:
            # No GAT embeddings loaded, or GAT weight is zero
            gat_scores = np.zeros(len(indices), dtype=np.float32)
            lam = 0.0
        else:
            gat_vecs = self.gat_embeddings[idx_arr]  # (N_candidates, gat_dim)

            if mode == "external_gat":
                if q_gat is None:
                    raise ValueError("mode='external_gat' requires q_gat to be provided")
                # Normalize q_gat defensively (recipe gat_embeddings are pre-normed)
                q_gat_norm = q_gat / (np.linalg.norm(q_gat) + 1e-8)
                gat_scores = gat_vecs @ q_gat_norm

            elif mode == "pool_centroid":
                # LEGACY: cosine to pool centroid (pool-typicality, not query-conditioned)
                mean_gat = gat_vecs.mean(axis=0)
                mean_gat = mean_gat / (np.linalg.norm(mean_gat) + 1e-8)
                gat_scores = gat_vecs @ mean_gat

            else:  # "hybrid" (new default) — pseudo-anchor via text top-1
                seed_local = int(np.argmax(text_scores))
                q_gat_star = gat_vecs[seed_local]
                q_gat_star = q_gat_star / (np.linalg.norm(q_gat_star) + 1e-8)
                gat_scores = gat_vecs @ q_gat_star

        # Combined score
        combined = (1 - lam) * text_scores + lam * gat_scores

        # Sort and return top-k
        top_k = min(k, len(combined))
        top_indices = np.argpartition(combined, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(combined[top_indices])[::-1]]

        return [
            (valid_ids[i], float(combined[i]), float(text_scores[i]), float(gat_scores[i]))
            for i in top_indices
        ]
