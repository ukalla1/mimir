#!/usr/bin/env python3
"""Pre-compute text embeddings for all FoodKG recipe names.

One-time offline step. Encodes all ~82K recipe names using
Qwen3-Embedding and saves the vectors + recipe_id mapping to disk.

Usage:
    python scripts/build_recipe_embeddings.py [--batch-size 64] [--device cuda]
"""

import argparse
import os
import sys
import time

import duckdb
import numpy as np

# Ensure nutri_rag is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nutri_rag.config import (
    DB_PATH,
    TEXT_EMBEDDING_MODEL,
    TEXT_EMBEDDINGS_DIR,
    RECIPE_TEXT_EMBEDDINGS_PATH,
    RECIPE_IDS_PATH,
)
from nutri_rag.embedding import TextEmbedder


def load_recipe_names(db_path: str) -> tuple[list[int], list[str]]:
    """Load all recipe_ids and names from DuckDB."""
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        "SELECT recipe_id, recipe_name FROM nodes_recipe ORDER BY recipe_id"
    ).df()
    con.close()
    return df["recipe_id"].tolist(), df["recipe_name"].tolist()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute FoodKG recipe text embeddings")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detected)")
    parser.add_argument("--model", type=str, default=TEXT_EMBEDDING_MODEL, help="Embedding model name")
    args = parser.parse_args()

    print(f"Loading recipe names from {DB_PATH}...")
    recipe_ids, recipe_names = load_recipe_names(DB_PATH)
    print(f"  Found {len(recipe_names)} recipes")

    print(f"\nLoading embedding model: {args.model}")
    embedder = TextEmbedder(model_name=args.model, device=args.device)
    print(f"  Device: {embedder.device}")

    print(f"\nEncoding {len(recipe_names)} recipe names (batch_size={args.batch_size})...")
    t0 = time.time()
    embeddings = embedder.encode(recipe_names, batch_size=args.batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(recipe_names) / elapsed:.0f} texts/sec)")
    print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    os.makedirs(TEXT_EMBEDDINGS_DIR, exist_ok=True)
    np.save(RECIPE_TEXT_EMBEDDINGS_PATH, embeddings)
    np.save(RECIPE_IDS_PATH, np.array(recipe_ids, dtype=np.int64))
    print(f"\nSaved embeddings to {RECIPE_TEXT_EMBEDDINGS_PATH}")
    print(f"Saved recipe_ids to {RECIPE_IDS_PATH}")

    # Quick sanity check
    print("\n--- Sanity Check ---")
    from nutri_rag.embedding import FOOD_SEARCH_INSTRUCTION

    test_queries = ["moose stew", "chicken curry", "chocolate cake", "vegan salad"]
    query_vecs = embedder.encode(test_queries, task_instruction=FOOD_SEARCH_INSTRUCTION)
    scores = query_vecs @ embeddings.T
    for i, query in enumerate(test_queries):
        top5 = np.argsort(scores[i])[-5:][::-1]
        print(f"\n  Query: \"{query}\"")
        for rank, idx in enumerate(top5, 1):
            print(f"    {rank}. [id={recipe_ids[idx]}] {recipe_names[idx]} (sim={scores[i, idx]:.4f})")


if __name__ == "__main__":
    main()
