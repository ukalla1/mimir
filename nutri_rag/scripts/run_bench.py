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
TASK_DIRS = {
    "v0": {
        "src": os.path.join(NUTRI_RAG_ROOT, "tasks", "nutribench_v2_rag_bm25"),
        "dst": os.path.join(LM_EVAL_DIR, "lm_eval", "tasks", "nutribench", "v2", "rag_bm25"),
        "task_name": "nutribench_v2_rag_bm25",
    },
    "v1": {
        "src": os.path.join(NUTRI_RAG_ROOT, "tasks", "nutribench_v2_rag"),
        "dst": os.path.join(LM_EVAL_DIR, "lm_eval", "tasks", "nutribench", "v2", "rag"),
        "task_name": "nutribench_v2_rag",
    },
    "v2": {
        "src": os.path.join(NUTRI_RAG_ROOT, "tasks", "nutribench_v2_rag_gat"),
        "dst": os.path.join(LM_EVAL_DIR, "lm_eval", "tasks", "nutribench", "v2", "rag_gat"),
        "task_name": "nutribench_v2_rag_gat",
    },
}
# Backward compat
TASK_SRC = TASK_DIRS["v1"]["src"]
TASK_DST = TASK_DIRS["v1"]["dst"]

# Ensure nutri_rag is importable
sys.path.insert(0, NUTRI_RAG_ROOT)
sys.path.insert(0, LM_EVAL_DIR)


def ensure_task_symlink(task_src: str = TASK_SRC, task_dst: str = TASK_DST):
    """Create symlink from harness task dir to our RAG task."""
    if os.path.islink(task_dst):
        current_target = os.readlink(task_dst)
        if current_target == task_src:
            return  # already correct
        os.unlink(task_dst)
    elif os.path.exists(task_dst):
        print(f"Warning: {task_dst} exists and is not a symlink. Skipping.")
        return

    os.makedirs(os.path.dirname(task_dst), exist_ok=True)
    os.symlink(task_src, task_dst)
    print(f"Symlinked: {task_dst} -> {task_src}")


def main():
    parser = argparse.ArgumentParser(description="Run NutriBench v2 RAG benchmark")
    parser.add_argument("--mode", choices=["v0", "v1", "v2"], default="v1",
                        help="v0 = BM25 keyword, v1 = text embedding, v2 = text + GAT (default: v1)")
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

    # Resolve task config for selected mode
    task_cfg = TASK_DIRS[args.mode]
    task_src = task_cfg["src"]
    task_dst = task_cfg["dst"]
    task_name = task_cfg["task_name"]

    # Ensure symlink exists
    ensure_task_symlink(task_src, task_dst)

    import lm_eval

    base_url = f"http://localhost:{args.port}/v1/chat/completions"
    model_args = f"model=qwen3.5-9b,base_url={base_url},num_concurrent={args.concurrent},max_retries={args.max_retries}"

    mode_labels = {"v0": "V0 (BM25 keyword)", "v1": "V1 (text embedding)", "v2": "V2 (text + GAT)"}
    mode_label = mode_labels[args.mode]
    print(f"==> Running NutriBench v2 RAG benchmark [{mode_label}]")
    print(f"    Server: {base_url}")
    print(f"    Output: {args.output}")
    print(f"    Limit: {args.limit or 'all'}")
    print()

    # Change to lm-evaluation-harness dir so task imports resolve
    os.chdir(LM_EVAL_DIR)

    # Register RAG task via include_path so the task manager discovers it
    from lm_eval.tasks.manager import TaskManager
    task_manager = TaskManager(include_path=task_src)

    results = lm_eval.simple_evaluate(
        model="local-chat-completions",
        model_args=model_args,
        tasks=[task_name],
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
    out_file = os.path.join(args.output, f"results_rag_{args.mode}_{timestamp}.json")
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
