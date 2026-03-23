#!/usr/bin/env python3
"""Run NutriBench v2 baseline (no RAG) benchmark for a specific nutrient.

Usage:
    python scripts/run_baseline.py --nutrient protein --limit 100
"""

import argparse
import json
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUTRI_RAG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ATLAS_ROOT = os.path.abspath(os.path.join(NUTRI_RAG_ROOT, "..", ".."))
LM_EVAL_DIR = os.path.join(ATLAS_ROOT, "qwen_test", "lm-evaluation-harness")

sys.path.insert(0, NUTRI_RAG_ROOT)
sys.path.insert(0, LM_EVAL_DIR)


def main():
    parser = argparse.ArgumentParser(description="Run NutriBench v2 baseline benchmark")
    parser.add_argument("--nutrient", choices=["carb", "protein", "fat", "energy"], default="carb")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--concurrent", type=int, default=6)
    parser.add_argument("--output", default=os.path.join(NUTRI_RAG_ROOT, "results"))
    args = parser.parse_args()

    # Set env var BEFORE any imports that use it
    os.environ["NUTRI_TARGET"] = args.nutrient

    from nutri_rag.bench.nutrient_prompts import build_system_prompt

    # Generate baseline task files
    baseline_task_dir = os.path.join(NUTRI_RAG_ROOT, "tasks", "_baseline_temp")
    os.makedirs(baseline_task_dir, exist_ok=True)

    system_prompt = build_system_prompt(args.nutrient)
    indented = "\n".join("  " + line for line in system_prompt.split("\n"))

    # Write template YAML
    with open(os.path.join(baseline_task_dir, "_default_template_yaml"), "w") as f:
        f.write(f"""dataset_path: dongx1997/NutriBench
dataset_name: v2
test_split: train
output_type: generate_until

description: |
{indented}

doc_to_text:  !function utils.doc_to_text_cot
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
""")

    # Write task YAML
    with open(os.path.join(baseline_task_dir, "baseline.yaml"), "w") as f:
        f.write("task: nutribench_v2_baseline\ninclude: _default_template_yaml\n")

    # Write utils.py
    with open(os.path.join(baseline_task_dir, "utils.py"), "w") as f:
        f.write(f'''"""Baseline (no RAG) NutriBench task utilities — {args.nutrient}."""

import sys
import os

os.environ["NUTRI_TARGET"] = "{args.nutrient}"

_NUTRI_RAG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _NUTRI_RAG_ROOT not in sys.path:
    sys.path.insert(0, _NUTRI_RAG_ROOT)

from nutri_rag.bench.task_utils import process_results, clean_output, agg_mae, is_number  # noqa: F401


def doc_to_text_cot(doc):
    meal = doc["meal_description"]
    return f'Query: "{{meal}}"\\nAnswer: Let\\'s think step by step.'
''')

    # Symlink into lm-evaluation-harness
    dst = os.path.join(LM_EVAL_DIR, "lm_eval", "tasks", "nutribench", "v2", "_baseline_temp")
    if os.path.islink(dst):
        os.unlink(dst)
    elif os.path.exists(dst):
        import shutil
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(baseline_task_dir, dst)

    # Run
    base_url = f"http://localhost:{args.port}/v1/chat/completions"
    model_args = f"model=qwen3.5-9b,base_url={base_url},num_concurrent={args.concurrent},max_retries=2"

    print(f"==> Running NutriBench v2 baseline [nutrient: {args.nutrient}]")
    print(f"    Server: {base_url}")
    print(f"    Limit: {args.limit or 'all'}")
    print()

    os.chdir(LM_EVAL_DIR)

    import lm_eval
    from lm_eval.tasks.manager import TaskManager

    task_manager = TaskManager(include_path=baseline_task_dir)

    results = lm_eval.simple_evaluate(
        model="local-chat-completions",
        model_args=model_args,
        tasks=["nutribench_v2_baseline"],
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
    out_file = os.path.join(args.output, f"results_baseline_{args.nutrient}_{timestamp}.json")
    with open(out_file, "w") as f:
        save_data = {
            "mode": "baseline",
            "nutrient": args.nutrient,
            "results": results["results"],
            "configs": {k: str(v) for k, v in results.get("configs", {}).items()},
        }
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    if "samples" in results and results["samples"]:
        for tname, samples in results["samples"].items():
            samples_file = os.path.join(args.output, f"samples_baseline_{args.nutrient}_{timestamp}.jsonl")
            with open(samples_file, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample, default=str) + "\n")
            print(f"Samples saved to {samples_file}")


if __name__ == "__main__":
    main()
