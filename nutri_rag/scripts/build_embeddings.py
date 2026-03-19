#!/usr/bin/env python3
"""Pre-compute text embeddings for all USDA food descriptions.

One-time offline step. Encodes all 74,175 food descriptions using
Qwen3-Embedding and saves the vectors + fdc_id mapping to disk.

Usage:
    python scripts/build_embeddings.py [--batch-size 64] [--device cuda]
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
    TEXT_EMBEDDINGS_PATH,
    TEXT_FDC_IDS_PATH,
)
from nutri_rag.embedding import TextEmbedder


def load_food_descriptions(db_path: str) -> tuple[list[int], list[str]]:
    """Load all food fdc_ids and descriptions from DuckDB."""
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        "SELECT fdc_id, description FROM nodes_food ORDER BY fdc_id"
    ).df()
    con.close()
    return df["fdc_id"].tolist(), df["description"].tolist()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute USDA food text embeddings")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detected)")
    parser.add_argument("--model", type=str, default=TEXT_EMBEDDING_MODEL, help="Embedding model name")
    args = parser.parse_args()

    print(f"Loading food descriptions from {DB_PATH}...")
    fdc_ids, descriptions = load_food_descriptions(DB_PATH)
    print(f"  Found {len(descriptions)} foods")

    print(f"\nLoading embedding model: {args.model}")
    embedder = TextEmbedder(model_name=args.model, device=args.device)
    print(f"  Device: {embedder.device}")

    print(f"\nEncoding {len(descriptions)} descriptions (batch_size={args.batch_size})...")
    t0 = time.time()
    embeddings = embedder.encode(descriptions, batch_size=args.batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(descriptions) / elapsed:.0f} texts/sec)")
    print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    os.makedirs(TEXT_EMBEDDINGS_DIR, exist_ok=True)
    np.save(TEXT_EMBEDDINGS_PATH, embeddings)
    np.save(TEXT_FDC_IDS_PATH, np.array(fdc_ids, dtype=np.int64))
    print(f"\nSaved embeddings to {TEXT_EMBEDDINGS_PATH}")
    print(f"Saved fdc_ids to {TEXT_FDC_IDS_PATH}")

    # Quick sanity check
    print("\n--- Sanity Check ---")
    from nutri_rag.embedding import FOOD_SEARCH_INSTRUCTION

    test_queries = ["maize flour", "peanut", "orange juice", "chicken breast"]
    query_vecs = embedder.encode(test_queries, task_instruction=FOOD_SEARCH_INSTRUCTION)
    scores = query_vecs @ embeddings.T
    for i, query in enumerate(test_queries):
        top5 = np.argsort(scores[i])[-5:][::-1]
        print(f"\n  Query: \"{query}\"")
        for rank, idx in enumerate(top5, 1):
            print(f"    {rank}. [{fdc_ids[idx]}] {descriptions[idx]} (sim={scores[i, idx]:.4f})")


if __name__ == "__main__":
    main()
