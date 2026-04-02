#!/usr/bin/env python3
"""PFoodReq benchmark runner.

Evaluates our RAG pipeline on the PFoodReq test set using:
- Text embedding + GAT scoring for candidate retrieval
- LLM (Qwen3.5-9B) for constraint-based recipe selection

Usage:
    python scripts/run_pfoodreq_bench.py --limit 100
    python scripts/run_pfoodreq_bench.py --split test
    python scripts/run_pfoodreq_bench.py --ablation text_only
    python scripts/run_pfoodreq_bench.py --ablation gat_only
    python scripts/run_pfoodreq_bench.py --ablation no_llm
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

from tqdm import tqdm

# Ensure nutri_rag is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nutri_rag.config import (
    PFOODREQ_DEV_PATH,
    PFOODREQ_LAMBDA,
    PFOODREQ_MAX_TOKENS,
    PFOODREQ_TEST_PATH,
    PFOODREQ_TOP_K,
)
from nutri_rag.embedding import RecipeVectorIndex, TextEmbedder
from nutri_rag.llm import chat_completion
from nutri_rag.pfoodreq.evaluator import aggregate_metrics
from nutri_rag.pfoodreq.prompt import build_prompt
from nutri_rag.pfoodreq.query_parser import load_examples
from nutri_rag.pfoodreq.retriever import RecipeRetriever


def parse_llm_response(response: str) -> list[str]:
    """Extract recipe name list from LLM response.

    Handles JSON arrays, markdown code fences, and plain text lists.
    """
    # Strip think tags
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Try JSON array parse
    try:
        result = json.loads(response)
        if isinstance(result, list):
            return [str(r) for r in result]
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, list):
                return [str(r) for r in result]
        except json.JSONDecodeError:
            pass

    # Try finding first [ ... ] block
    m = re.search(r"\[.*\]", response, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                return [str(r) for r in result]
        except json.JSONDecodeError:
            pass

    # Fallback: split by newlines and extract quoted strings
    names = re.findall(r'"([^"]+)"', response)
    return names


def run_single(
    example: dict,
    retriever: RecipeRetriever,
    top_k: int,
    lam: float,
    max_tokens: int,
    ablation: str | None = None,
) -> dict:
    """Run pipeline for a single PFoodReq example.

    Returns dict with predicted recipes, ground truth, and debug info.
    """
    tag = example["tag"]
    query_text = example["original_query"]

    # Determine lambda based on ablation
    effective_lam = lam
    if ablation == "text_only":
        effective_lam = 0.0
    elif ablation == "gat_only":
        effective_lam = 1.0

    # Retrieve candidates (Config C: tag → filter → GAT re-rank → top-k)
    recipes = retriever.retrieve(
        query_text=query_text,
        tag_name=tag,
        top_k=top_k,
        lam=effective_lam,
        positive_ingredients=example.get("positive_ingredients", []),
        negative_ingredients=example.get("negative_ingredients", []),
        nutrient_constraints=example.get("nutrient_constraints", []),
    )

    if not recipes:
        return {
            "qid": example["qid"],
            "predicted": [],
            "ground_truth": example["ground_truth"],
            "n_candidates": 0,
            "n_retrieved": 0,
            "llm_response": "",
            "error": None,
        }

    # Skip LLM unless explicitly requested
    if ablation != "with_llm":
        predicted = [r["recipe_name"] for r in recipes]
        return {
            "qid": example["qid"],
            "predicted": predicted,
            "ground_truth": example["ground_truth"],
            "n_candidates": len(recipes),
            "n_retrieved": len(predicted),
            "llm_response": "(no_llm ablation)",
            "error": None,
        }

    # Build RAG prompt
    messages = build_prompt(recipes, example)

    # Call LLM
    try:
        response = chat_completion(messages, max_tokens=max_tokens)
        predicted = parse_llm_response(response)
    except Exception as e:
        response = ""
        predicted = []
        return {
            "qid": example["qid"],
            "predicted": predicted,
            "ground_truth": example["ground_truth"],
            "n_candidates": len(recipes),
            "n_retrieved": 0,
            "llm_response": response,
            "error": str(e),
        }

    return {
        "qid": example["qid"],
        "predicted": predicted,
        "ground_truth": example["ground_truth"],
        "n_candidates": len(recipes),
        "n_retrieved": len(predicted),
        "llm_response": response,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(description="PFoodReq benchmark runner")
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev"],
                        help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples to evaluate")
    parser.add_argument("--top-k", type=int, default=PFOODREQ_TOP_K,
                        help="Max candidates to show LLM")
    parser.add_argument("--lam", type=float, default=PFOODREQ_LAMBDA,
                        help="GAT weight: (1-λ)*text + λ*gat")
    parser.add_argument("--max-tokens", type=int, default=PFOODREQ_MAX_TOKENS,
                        help="LLM max output tokens")
    parser.add_argument("--ablation", type=str, default="no_llm",
                        choices=["text_only", "gat_only", "no_llm", "with_llm"],
                        help="Ablation variant (default: no_llm)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    # Load data
    data_path = PFOODREQ_TEST_PATH if args.split == "test" else PFOODREQ_DEV_PATH
    print(f"Loading {args.split} set from {data_path}...")
    examples = load_examples(data_path)
    print(f"  Loaded {len(examples)} examples")

    if args.limit:
        examples = examples[:args.limit]
        print(f"  Limited to {len(examples)} examples")

    # Initialize components
    print("\nInitializing retriever...")
    embedder = TextEmbedder()
    recipe_index = RecipeVectorIndex()
    retriever = RecipeRetriever(embedder=embedder, recipe_index=recipe_index)

    has_gat = recipe_index.gat_embeddings is not None
    print(f"  Text embeddings: {recipe_index.text_embeddings.shape}")
    print(f"  GAT embeddings: {'available' if has_gat else 'NOT available'}")
    print(f"  Lambda: {args.lam}, Top-k: {args.top_k}")
    if args.ablation:
        print(f"  Ablation: {args.ablation}")

    # Run benchmark
    print(f"\nRunning benchmark on {len(examples)} examples...")
    t0 = time.time()
    all_results = []
    n_errors = 0
    n_empty = 0

    for example in tqdm(examples, desc="PFoodReq"):
        result = run_single(
            example=example,
            retriever=retriever,
            top_k=args.top_k,
            lam=args.lam,
            max_tokens=args.max_tokens,
            ablation=args.ablation,
        )
        all_results.append(result)

        if result["error"]:
            n_errors += 1
        if not result["predicted"]:
            n_empty += 1

    elapsed = time.time() - t0

    # Compute metrics
    metrics = aggregate_metrics(all_results)

    print(f"\n{'='*50}")
    print(f"PFoodReq Results ({args.split}, {len(examples)} examples)")
    if args.ablation:
        print(f"Ablation: {args.ablation}")
    print(f"{'='*50}")
    print(f"  MAP:  {metrics['MAP']:.1f}%")
    print(f"  MAR:  {metrics['MAR']:.1f}%")
    print(f"  F1:   {metrics['F1']:.1f}%")
    print(f"  Errors: {n_errors}, Empty predictions: {n_empty}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(examples):.2f}s/query)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    ablation_tag = f"_{args.ablation}" if args.ablation else ""
    result_file = os.path.join(
        args.output_dir,
        f"results_pfoodreq_{args.split}{ablation_tag}_{timestamp}.json",
    )

    output = {
        "benchmark": "pfoodreq",
        "split": args.split,
        "ablation": args.ablation,
        "n_examples": len(examples),
        "n_errors": n_errors,
        "n_empty_predictions": n_empty,
        "top_k": args.top_k,
        "lambda": args.lam,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {result_file}")

    # Save per-query logs
    log_file = result_file.replace(".json", "_logs.jsonl")
    with open(log_file, "w") as f:
        for result in all_results:
            # Don't save full LLM response in logs to save space
            log_entry = {
                "qid": result["qid"],
                "predicted": result["predicted"],
                "ground_truth": result["ground_truth"],
                "n_candidates": result["n_candidates"],
                "n_retrieved": result["n_retrieved"],
                "error": result["error"],
            }
            f.write(json.dumps(log_entry) + "\n")
    print(f"Per-query logs saved to {log_file}")


if __name__ == "__main__":
    main()
