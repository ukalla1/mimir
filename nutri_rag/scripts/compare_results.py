#!/usr/bin/env python3
"""Side-by-side comparison of baseline (CoT) vs RAG benchmark results.

Reads JSONL sample files from both runs and compares accuracy and MAE.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ATLAS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_samples(jsonl_path: str) -> list[dict]:
    """Load per-sample results from a JSONL file."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def compute_metrics(samples: list[dict]) -> dict:
    """Compute accuracy and MAE from sample results."""
    accs = []
    maes = []
    for s in samples:
        if "acc" in s:
            accs.append(s["acc"])
            maes.append(s["mae"])
    return {
        "n_samples": len(accs),
        "accuracy": sum(accs) / len(accs) if accs else 0,
        "mae": sum(maes) / len(maes) if maes else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs RAG results")
    parser.add_argument("--baseline", type=str,
                        help="Path to baseline JSONL samples file")
    parser.add_argument("--rag", type=str,
                        help="Path to RAG JSONL samples file")
    parser.add_argument("--show-improvements", type=int, default=10,
                        help="Number of biggest improvements to show")
    parser.add_argument("--show-regressions", type=int, default=10,
                        help="Number of biggest regressions to show")
    args = parser.parse_args()

    # Auto-detect files if not specified
    if not args.baseline:
        baseline_dir = os.path.join(ATLAS_ROOT, "qwen_test", "results", "qwen3.5-9b")
        candidates = [f for f in os.listdir(baseline_dir) if f.startswith("samples_") and f.endswith(".jsonl")]
        if candidates:
            args.baseline = os.path.join(baseline_dir, sorted(candidates)[-1])
            print(f"Auto-detected baseline: {args.baseline}")

    if not args.rag:
        rag_dir = os.path.join(ATLAS_ROOT, "nutri_rag", "results")
        if os.path.exists(rag_dir):
            candidates = [f for f in os.listdir(rag_dir) if f.startswith("samples_") and f.endswith(".jsonl")]
            if candidates:
                args.rag = os.path.join(rag_dir, sorted(candidates)[-1])
                print(f"Auto-detected RAG: {args.rag}")

    if not args.baseline or not args.rag:
        print("Error: Could not find result files. Specify --baseline and --rag paths.")
        sys.exit(1)

    # Load samples
    baseline_samples = load_samples(args.baseline)
    rag_samples = load_samples(args.rag)

    baseline_metrics = compute_metrics(baseline_samples)
    rag_metrics = compute_metrics(rag_samples)

    # Print comparison
    print("\n" + "=" * 60)
    print("NutriBench v2 — Baseline (CoT) vs RAG Comparison")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Baseline':>12} {'RAG':>12} {'Delta':>12}")
    print("-" * 56)

    acc_delta = rag_metrics["accuracy"] - baseline_metrics["accuracy"]
    mae_delta = rag_metrics["mae"] - baseline_metrics["mae"]

    print(f"{'Samples':<20} {baseline_metrics['n_samples']:>12} {rag_metrics['n_samples']:>12}")
    print(f"{'Accuracy':<20} {baseline_metrics['accuracy']:>11.2%} {rag_metrics['accuracy']:>11.2%} {acc_delta:>+11.2%}")
    print(f"{'MAE (g)':<20} {baseline_metrics['mae']:>12.2f} {rag_metrics['mae']:>12.2f} {mae_delta:>+12.2f}")

    # Per-sample comparison (if same number of samples)
    if len(baseline_samples) == len(rag_samples):
        print(f"\n--- Per-Sample Analysis ---")

        deltas = []
        for b, r in zip(baseline_samples, rag_samples):
            b_mae = b.get("mae", 0)
            r_mae = r.get("mae", 0)
            improvement = b_mae - r_mae  # positive = RAG is better
            query = b.get("doc", {}).get("meal_description", "?")
            deltas.append((improvement, b_mae, r_mae, query))

        deltas.sort(reverse=True)

        if args.show_improvements > 0:
            print(f"\nTop {args.show_improvements} improvements (RAG better):")
            for imp, b_mae, r_mae, query in deltas[:args.show_improvements]:
                if imp > 0:
                    print(f"  MAE: {b_mae:.1f} -> {r_mae:.1f} ({imp:+.1f}g)  |  {query[:60]}")

        if args.show_regressions > 0:
            print(f"\nTop {args.show_regressions} regressions (RAG worse):")
            for imp, b_mae, r_mae, query in deltas[-args.show_regressions:]:
                if imp < 0:
                    print(f"  MAE: {b_mae:.1f} -> {r_mae:.1f} ({imp:+.1f}g)  |  {query[:60]}")

    print()


if __name__ == "__main__":
    main()
