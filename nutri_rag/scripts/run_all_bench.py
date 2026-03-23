#!/usr/bin/env python3
"""Run all NutriBench benchmark combinations: 4 nutrients x 4 modes (baseline + v0/v1/v2).

Each combination runs as a separate subprocess to avoid module caching issues.

Usage:
    # All 16 combinations (full dataset)
    python scripts/run_all_bench.py --modes baseline v0 v1 v2

    # Quick test with 100 samples
    python scripts/run_all_bench.py --limit 100

    # Specific nutrients or modes only
    python scripts/run_all_bench.py --nutrients carb protein --modes v1 v2
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUTRI_RAG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

ALL_NUTRIENTS = ["carb", "protein", "fat", "energy"]
ALL_MODES = ["baseline", "v0", "v1", "v2"]
RAG_MODES = ["v0", "v1", "v2"]


def run_single(mode, nutrient, limit, port):
    """Run a single benchmark as a subprocess."""
    if mode == "baseline":
        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "run_baseline.py"),
            "--nutrient", nutrient,
            "--port", str(port),
        ]
    else:
        cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "run_bench.py"),
            "--mode", mode,
            "--nutrient", nutrient,
            "--port", str(port),
        ]

    if limit:
        cmd += ["--limit", str(limit)]

    print(f"\n{'='*60}")
    print(f"  Running: mode={mode}, nutrient={nutrient}, limit={limit or 'all'}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=NUTRI_RAG_ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all NutriBench benchmark combinations")
    parser.add_argument("--nutrients", nargs="+", choices=ALL_NUTRIENTS, default=ALL_NUTRIENTS,
                        help="Nutrients to evaluate (default: all)")
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=RAG_MODES,
                        help="Modes to run (default: v0 v1 v2). Use 'baseline' for no-RAG.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples per run")
    parser.add_argument("--port", type=int, default=8080,
                        help="llama-server port (default: 8080)")
    args = parser.parse_args()

    total = len(args.nutrients) * len(args.modes)
    print(f"Running {total} benchmark combinations:")
    print(f"  Nutrients: {args.nutrients}")
    print(f"  Modes: {args.modes}")
    print(f"  Limit: {args.limit or 'all'}")
    print()

    results_summary = []
    for nutrient in args.nutrients:
        for mode in args.modes:
            start = datetime.now()
            success = run_single(mode, nutrient, args.limit, args.port)
            elapsed = (datetime.now() - start).total_seconds()
            results_summary.append({
                "mode": mode,
                "nutrient": nutrient,
                "success": success,
                "elapsed_s": round(elapsed, 1),
            })

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<15} {'Nutrient':<10} {'Status':<10} {'Time (s)':<10}")
    print("-" * 45)
    for r in results_summary:
        status = "OK" if r["success"] else "FAILED"
        print(f"{r['mode']:<15} {r['nutrient']:<10} {status:<10} {r['elapsed_s']:<10}")

    # Save summary
    output_dir = os.path.join(NUTRI_RAG_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    summary_file = os.path.join(output_dir, f"run_all_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
