#!/usr/bin/env python3
"""Analyze cosine similarity between NutriBench food terms and USDA entries.

Generates plots showing:
1. Distribution of top-1 cosine similarity scores (per food term)
2. Distribution of best cosine similarity scores (per sample)

Usage:
    python scripts/plot_similarity_analysis.py
    python scripts/plot_similarity_analysis.py --limit 100
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUTRI_RAG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, NUTRI_RAG_ROOT)

from datasets import load_dataset

from nutri_rag.bench.retriever import _extract_food_terms
from nutri_rag.embedding import FOOD_SEARCH_INSTRUCTION, FoodVectorIndex, TextEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=os.path.join(NUTRI_RAG_ROOT, "results"))
    args = parser.parse_args()

    # Load NutriBench
    print("Loading NutriBench v2...")
    ds = load_dataset("dongx1997/NutriBench", "v2", split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"  {len(ds)} samples")

    # Extract food terms from all meals
    print("Extracting food terms...")
    all_terms = []

    for i, row in enumerate(ds):
        meal = row["meal_description"]
        terms = _extract_food_terms(meal)
        for t in terms:
            all_terms.append({"sample_idx": i, "term": t, "gt_carb": row["carb"]})

    print(f"  {len(all_terms)} food terms from {len(ds)} samples")

    # Embed all food terms
    print("Loading embedding model...")
    embedder = TextEmbedder()
    index = FoodVectorIndex()

    term_texts = [t["term"] for t in all_terms]
    print(f"Encoding {len(term_texts)} food terms...")
    query_vecs = embedder.encode(term_texts, task_instruction=FOOD_SEARCH_INSTRUCTION)

    # Search top-1 for each
    print("Searching USDA database...")
    results = index.search(query_vecs, k=1)

    similarities = []
    for i, (term_info, res) in enumerate(zip(all_terms, results)):
        if res:
            fdc_id, sim, _ = res[0]
            term_info["similarity"] = sim
            term_info["fdc_id"] = fdc_id
        else:
            term_info["similarity"] = 0.0
            term_info["fdc_id"] = None
        similarities.append(term_info["similarity"])

    similarities = np.array(similarities)
    df = pd.DataFrame(all_terms)

    # Per-sample: best and mean similarity
    sample_stats = df.groupby("sample_idx").agg(
        best_sim=("similarity", "max"),
        mean_sim=("similarity", "mean"),
        n_terms=("similarity", "count"),
        gt_carb=("gt_carb", "first"),
    ).reset_index()

    # ── Plot 1: Per-Term Similarity Distribution ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("NutriBench \u2194 USDA Cosine Similarity Analysis", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.hist(similarities, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(similarities.mean(), color="red", linestyle="--", label=f"Mean: {similarities.mean():.3f}")
    ax.axvline(np.median(similarities), color="orange", linestyle="--", label=f"Median: {np.median(similarities):.3f}")
    ax.set_xlabel("Cosine Similarity (food term \u2192 top-1 USDA match)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Per-Term Similarity Distribution")
    ax.legend()

    # ── Plot 2: Per-Sample Best Similarity ────────────────────────────
    ax = axes[1]
    ax.hist(sample_stats["best_sim"], bins=30, color="#55A868", edgecolor="white", alpha=0.85)
    ax.axvline(sample_stats["best_sim"].mean(), color="red", linestyle="--",
               label=f"Mean: {sample_stats['best_sim'].mean():.3f}")
    ax.set_xlabel("Best Cosine Similarity per Sample")
    ax.set_ylabel("Count")
    ax.set_title("(b) Per-Sample Best Match Similarity")
    ax.legend()

    plt.tight_layout()
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "similarity_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

    # Print summary stats
    print("\n=== Summary ===")
    print(f"Total food terms: {len(similarities)}")
    print(f"Mean similarity: {similarities.mean():.4f}")
    print(f"Median similarity: {np.median(similarities):.4f}")
    print(f"Min similarity: {similarities.min():.4f}")
    print(f"Max similarity: {similarities.max():.4f}")
    print(f"Terms with sim > 0.5: {(similarities > 0.5).sum()} ({(similarities > 0.5).mean()*100:.1f}%)")
    print(f"Terms with sim > 0.6: {(similarities > 0.6).sum()} ({(similarities > 0.6).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
