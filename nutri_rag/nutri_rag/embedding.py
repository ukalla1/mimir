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
                (int(self.fdc_ids[idx]), float(scores[i, idx]))
                for idx in top_indices
            ])
        return results
