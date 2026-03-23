#!/usr/bin/env python3
"""Run NutriBench v2 RAG benchmark against a local llama-server.

This script:
1. Generates the task YAML template for the selected nutrient
2. Symlinks the RAG task into the lm-evaluation-harness task directory
3. Runs the benchmark using lm_eval
4. Saves results and per-sample logs

Supports all combinations of:
  --mode:     v0 (BM25), v1 (dense retrieval), v2 (dense + GAT)
  --nutrient: carb, protein, fat, energy
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
        "template_name": "_rag_bm25_default_template_yaml",
    },
    "v1": {
        "src": os.path.join(NUTRI_RAG_ROOT, "tasks", "nutribench_v2_rag"),
        "dst": os.path.join(LM_EVAL_DIR, "lm_eval", "tasks", "nutribench", "v2", "rag"),
        "task_name": "nutribench_v2_rag",
        "template_name": "_rag_default_template_yaml",
    },
    "v2": {
        "src": os.path.join(NUTRI_RAG_ROOT, "tasks", "nutribench_v2_rag_gat"),
        "dst": os.path.join(LM_EVAL_DIR, "lm_eval", "tasks", "nutribench", "v2", "rag_gat"),
        "task_name": "nutribench_v2_rag_gat",
        "template_name": "_rag_gat_default_template_yaml",
    },
}

# Ensure nutri_rag is importable
sys.path.insert(0, NUTRI_RAG_ROOT)
sys.path.insert(0, LM_EVAL_DIR)


def generate_template_yaml(task_src: str, template_name: str, nutrient: str):
    """Write the YAML template with the correct system prompt for the nutrient."""
    from nutri_rag.bench.nutrient_prompts import build_system_prompt

    system_prompt = build_system_prompt(nutrient)
    # Indent each line by 2 spaces for YAML block scalar
    indented = "\n".join("  " + line for line in system_prompt.split("\n"))

    yaml_content = f"""dataset_path: dongx1997/NutriBench
dataset_name: v2
test_split: train
output_type: generate_until

description: |
{indented}

doc_to_text:  !function utils.doc_to_text_rag
doc_to_target: 0
process_results: !function utils.process_results

generation_kwargs:
  until: []
  do_sample: false
  temperature: 0.0
  max_gen_toks: 4096

metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: mae
    aggregation: mean
    higher_is_better: true


metadata:
  version: 1.0
"""
    template_path = os.path.join(task_src, template_name)
    with open(template_path, "w") as f:
        f.write(yaml_content)


def ensure_task_symlink(task_src: str, task_dst: str):
    """Create symlink from harness task dir to our RAG task."""
    if os.path.islink(task_dst):
        current_target = os.readlink(task_dst)
        if current_target == task_src:
            return
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
    parser.add_argument("--nutrient", choices=["carb", "protein", "fat", "energy"], default="carb",
                        help="Target nutrient to evaluate (default: carb)")
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

    # Set NUTRI_TARGET env var so task utils pick it up
    os.environ["NUTRI_TARGET"] = args.nutrient

    # Resolve task config for selected mode
    task_cfg = TASK_DIRS[args.mode]
    task_src = task_cfg["src"]
    task_dst = task_cfg["dst"]
    task_name = task_cfg["task_name"]
    template_name = task_cfg["template_name"]

    # Generate YAML template for the target nutrient
    generate_template_yaml(task_src, template_name, args.nutrient)

    # Ensure symlink exists
    ensure_task_symlink(task_src, task_dst)

    import lm_eval

    base_url = f"http://localhost:{args.port}/v1/chat/completions"
    model_args = f"model=qwen3.5-9b,base_url={base_url},num_concurrent={args.concurrent},max_retries={args.max_retries}"

    mode_labels = {"v0": "V0 (BM25)", "v1": "V1 (Dense)", "v2": "V2 (Dense+GAT)"}
    mode_label = mode_labels[args.mode]
    print(f"==> Running NutriBench v2 RAG benchmark [{mode_label}] [nutrient: {args.nutrient}]")
    print(f"    Server: {base_url}")
    print(f"    Output: {args.output}")
    print(f"    Limit: {args.limit or 'all'}")
    print()

    os.chdir(LM_EVAL_DIR)

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
    for tname, metrics in results["results"].items():
        print(f"Task: {tname} | Nutrient: {args.nutrient}")
        for k, v in metrics.items():
            if k != "alias":
                print(f"  {k}: {v}")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_file = os.path.join(args.output, f"results_{args.mode}_{args.nutrient}_{timestamp}.json")
    with open(out_file, "w") as f:
        save_data = {
            "mode": args.mode,
            "nutrient": args.nutrient,
            "results": results["results"],
            "configs": {k: str(v) for k, v in results.get("configs", {}).items()},
        }
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Save per-sample logs
    if "samples" in results and results["samples"]:
        for tname, samples in results["samples"].items():
            samples_file = os.path.join(args.output, f"samples_{tname}_{args.nutrient}_{timestamp}.jsonl")
            with open(samples_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample, default=str) + "\n")
            print(f"Samples saved to {samples_file}")


if __name__ == "__main__":
    main()
