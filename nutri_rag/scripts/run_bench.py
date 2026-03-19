#!/usr/bin/env python3
"""Run NutriBench v2 RAG benchmark against a local llama-server.

This script:
1. Symlinks the RAG task into the lm-evaluation-harness task directory
2. Runs the benchmark using lm_eval
3. Saves results and per-sample logs
"""

import argparse
import json
import os
import sys
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUTRI_RAG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ATLAS_ROOT = os.path.abspath(os.path.join(NUTRI_RAG_ROOT, "..", ".."))  # mimir -> atlas
LM_EVAL_DIR = os.path.join(ATLAS_ROOT, "qwen_test", "lm-evaluation-harness")
TASK_SRC = os.path.join(NUTRI_RAG_ROOT, "tasks", "nutribench_v2_rag")
TASK_DST = os.path.join(LM_EVAL_DIR, "lm_eval", "tasks", "nutribench", "v2", "rag")

# Ensure nutri_rag is importable
sys.path.insert(0, NUTRI_RAG_ROOT)
sys.path.insert(0, LM_EVAL_DIR)


def ensure_task_symlink():
    """Create symlink from harness task dir to our RAG task."""
    if os.path.islink(TASK_DST):
        current_target = os.readlink(TASK_DST)
        if current_target == TASK_SRC:
            return  # already correct
        os.unlink(TASK_DST)
    elif os.path.exists(TASK_DST):
        print(f"Warning: {TASK_DST} exists and is not a symlink. Skipping.")
        return

    os.makedirs(os.path.dirname(TASK_DST), exist_ok=True)
    os.symlink(TASK_SRC, TASK_DST)
    print(f"Symlinked: {TASK_DST} -> {TASK_SRC}")


def main():
    parser = argparse.ArgumentParser(description="Run NutriBench v2 RAG benchmark")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (e.g. 100 for quick test)")
    parser.add_argument("--port", type=int, default=8080,
                        help="llama-server port (default: 8080)")
    parser.add_argument("--concurrent", type=int, default=6,
                        help="Number of concurrent requests (default: 6)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retries per request (default: 2)")
    parser.add_argument("--output", default=os.path.join(NUTRI_RAG_ROOT, "results"),
                        help="Output directory for results")
    args = parser.parse_args()

    # Ensure symlink exists
    ensure_task_symlink()

    import lm_eval

    base_url = f"http://localhost:{args.port}/v1/chat/completions"
    model_args = f"model=qwen3.5-9b,base_url={base_url},num_concurrent={args.concurrent},max_retries={args.max_retries}"

    print(f"==> Running NutriBench v2 RAG benchmark")
    print(f"    Server: {base_url}")
    print(f"    Output: {args.output}")
    print(f"    Limit: {args.limit or 'all'}")
    print()

    # Change to lm-evaluation-harness dir so task imports resolve
    os.chdir(LM_EVAL_DIR)

    # Register RAG task via include_path so the task manager discovers it
    from lm_eval.tasks.manager import TaskManager
    task_manager = TaskManager(include_path=TASK_SRC)

    results = lm_eval.simple_evaluate(
        model="local-chat-completions",
        model_args=model_args,
        tasks=["nutribench_v2_rag"],
        task_manager=task_manager,
        batch_size=1,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42,
        log_samples=True,
        apply_chat_template=True,
        limit=args.limit,
    )

    # Print summary
    print("\n=== Results ===")
    for task_name, metrics in results["results"].items():
        print(f"Task: {task_name}")
        for k, v in metrics.items():
            if k != "alias":
                print(f"  {k}: {v}")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_file = os.path.join(args.output, f"results_rag_{timestamp}.json")
    with open(out_file, "w") as f:
        save_data = {
            "results": results["results"],
            "configs": {k: str(v) for k, v in results.get("configs", {}).items()},
        }
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Save per-sample logs
    if "samples" in results and results["samples"]:
        for task_name, samples in results["samples"].items():
            samples_file = os.path.join(args.output, f"samples_{task_name}_{timestamp}.jsonl")
            with open(samples_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample, default=str) + "\n")
            print(f"Samples saved to {samples_file}")


if __name__ == "__main__":
    main()
