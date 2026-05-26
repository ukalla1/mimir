#!/usr/bin/env python3
"""Run NutriBench questions against the local llama.cpp chat server on the Jetson.

Reads questions extracted by `nutribench_jetson_questions.py`, builds the exact
same prompt format used by the server-side benchmark, sends each one to the
on-device LLM, parses the predicted nutrient value, compares to ground truth,
and writes both per-question predictions and a summary report.

Modes
-----
- baseline : no RAG. Just the system prompt + few-shot examples + raw query.
             Only requires the chat LLM server (Qwen3.5 via llama.cpp).
- gat      : (future) GAT-only retrieval. Needs food_embeddings.npy + duckdb.
- text     : (future) text-only retrieval. Needs llama.cpp embedding server.
- hybrid   : (future) text + GAT score fusion. Needs both.

This first release implements `baseline` so you can sanity-check the
on-device LLM end-to-end. Other modes will plug into the same loop.

Usage
-----
    # default: baseline, carb, 20 questions, chat server on localhost:8080
    python nutribench_jetson_test.py

    # change nutrient target and questions file
    python nutribench_jetson_test.py --nutrient energy \\
        --questions /path/to/nutribench_jetson_questions.json

    # custom endpoint / model name
    python nutribench_jetson_test.py \\
        --endpoint http://localhost:8080/v1/chat/completions \\
        --model qwen3.5-9b

    # save predictions JSON to a specific path
    python nutribench_jetson_test.py --out results/baseline_carb.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.stderr.write("ERROR: `requests` not installed. Run: pip install requests\n")
    sys.exit(1)


# ── Nutrient configuration (mirrors nutri_rag/bench/nutrient_prompts.py) ──────

NUTRIENT_CONFIG = {
    "carb": {
        "full_name": "carbohydrates",
        "json_key":  "total_carbohydrates",
        "unit":      "grams",
        "gt_column": "carb",
        "acc_threshold": 7.5,
    },
    "protein": {
        "full_name": "protein",
        "json_key":  "total_protein",
        "unit":      "grams",
        "gt_column": "protein",
        "acc_threshold": 7.5,
    },
    "fat": {
        "full_name": "fat",
        "json_key":  "total_fat",
        "unit":      "grams",
        "gt_column": "fat",
        "acc_threshold": 7.5,
    },
    "energy": {
        "full_name": "energy",
        "json_key":  "total_energy",
        "unit":      "kcal",
        "gt_column": "energy",
        "acc_threshold": 50.0,
    },
}

# Few-shot examples per nutrient (lifted verbatim from server-side prompts)
_EXAMPLES = {
    "carb": [
        ("This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
         "The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n"
         "1 cup of oatmeal has 27g carbs.\n"
         "1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.\n"
         "1 glass of orange juice has 26g carbs.\n"
         "So the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5",
         66.5),
        ("I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
         "The meal consists of scrambled eggs made with 2 eggs and 1 toast.\n"
         "Scrambled eggs made with 2 eggs has 2g carbs.\n"
         "1 toast has 13g carbs.\n"
         "So the total grams of carbs in the meal = (2 + 13) = 15",
         15),
        ("Half a peanut butter and jelly sandwich.",
         "The meal consists of 1/2 a peanut butter and jelly sandwich.\n"
         "1 peanut butter and jelly sandwich has 50g carbs so half has (50*(1/2)) = 25g carbs.\n"
         "So the total grams of carbs in the meal = 25",
         25),
    ],
    "protein": [
        ("This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
         "1 cup of oatmeal has 6g protein.\n"
         "1 banana has 1g protein so half has 0.5g.\n"
         "1 glass of orange juice has 2g protein.\n"
         "Total = (6 + 0.5 + 2) = 8.5",
         8.5),
        ("I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
         "2-egg scrambled eggs has 13g protein.\n"
         "1 toast has 3g protein.\n"
         "Total = (13 + 3) = 16",
         16),
    ],
    "fat": [
        ("This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
         "1 cup of oatmeal has 3g fat.\n"
         "1 banana has 0.4g fat so half has 0.2g.\n"
         "1 glass of orange juice has 0.5g fat.\n"
         "Total = (3 + 0.2 + 0.5) = 3.7",
         3.7),
        ("I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
         "2-egg scrambled eggs has 15g fat.\n"
         "1 toast has 1g fat.\n"
         "Total = (15 + 1) = 16",
         16),
    ],
    "energy": [
        ("This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
         "1 cup of oatmeal has 154 kcal.\n"
         "1 banana has 105 kcal so half a banana has (105*(1/2)) = 52.5 kcal.\n"
         "1 glass of orange juice has 112 kcal.\n"
         "So the total energy in the meal = (154 + 52.5 + 112) = 318.5",
         318.5),
        ("I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
         "Scrambled eggs made with 2 eggs has 182 kcal.\n"
         "1 toast has 79 kcal.\n"
         "So the total energy in the meal = (182 + 79) = 261",
         261),
        ("Half a peanut butter and jelly sandwich.",
         "1 peanut butter and jelly sandwich has 376 kcal so half has (376*(1/2)) = 188 kcal.\n"
         "So the total energy in the meal = 188",
         188),
    ],
}


def build_system_prompt(nutrient: str) -> str:
    cfg = NUTRIENT_CONFIG[nutrient]
    full_name, json_key, unit = cfg["full_name"], cfg["json_key"], cfg["unit"]

    lines = [
        "For the given query including a meal description, think step by step as follows:",
        "1. Parse the meal description into discrete food or beverage items along with their serving size. "
        "If the serving size of any item in the meal is not specified, assume it is a single standard serving "
        "based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't "
        "relate to the item name and serving size.",
        f"2. For each food or beverage item in the meal, calculate the amount of {full_name} in {unit} for the specific serving size.",
        f"3. Respond with a dictionary object containing the total {full_name} in {unit} as follows:",
        f'{{"{json_key}": total {unit} of {full_name} for the serving}}',
        f"For the total {full_name}, respond with just the numeric amount without extra text. "
        f'If you don\'t know the answer, set the value of "{json_key}" to -1.',
        "",
        "Follow the format of the following examples when answering",
    ]

    for query, reasoning, val in _EXAMPLES[nutrient]:
        lines.append("")
        lines.append(f'Query: "{query}"')
        lines.append("Answer: Let's think step by step.")
        lines.append(reasoning)
        lines.append(f'Output: {{"{json_key}": {val}}}')

    return "\n".join(lines)


def build_user_message(meal_description: str) -> str:
    """Baseline (no RAG) user message — just the CoT query."""
    return f'Query: "{meal_description}"\nAnswer: Let\'s think step by step.'


# ── Output parsing (mirrors nutri_rag/bench/task_utils.clean_output) ─────────

def parse_prediction(raw_output: str, nutrient: str) -> float:
    """Extract numeric prediction for the target nutrient, or -1 on failure."""
    if not isinstance(raw_output, str):
        raw_output = str(raw_output)

    # Strip Qwen3.5 <think> blocks if present
    raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    # Take the part after the last "Output:" marker if present
    splits = raw_output.split("Output:")
    if len(splits) > 1:
        raw_output = splits[-1]
    raw_output = raw_output.strip()

    patterns = {
        "fat":     r'["\']\s*total_fat["\']: (-?[0-9]+(?:\.[0-9]*)?)',
        "protein": r'["\']\s*total_protein["\']: (-?[0-9]+(?:\.[0-9]*)?)',
        "energy": r'["\']\s*total_energy["\']: (-?[0-9]+(?:\.[0-9]*)?)',
        "carb":   r'["\']\s*total_carbohydrates["\']:\s*["\']?(-?[0-9]+(?:\.[0-9]*)?)["\']?',
    }
    pattern = patterns[nutrient]

    match = re.search(pattern, raw_output)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            return -1.0

    # Last-ditch: maybe the model just emitted a bare number
    try:
        return float(raw_output)
    except ValueError:
        return -1.0


# ── LLM call ──────────────────────────────────────────────────────────────────

def chat_completion(
    endpoint: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int = 300,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ── Main loop ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run NutriBench questions on the Jetson LLM.")
    p.add_argument("--questions", type=Path,
                   default=here.parent / "data" / "nutribench_jetson_questions.json",
                   help="Input questions JSON (from nutribench_jetson_questions.py).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output predictions JSON (default: data/results/<mode>_<nutrient>_<n>.json).")
    p.add_argument("--mode", default="baseline", choices=["baseline"],
                   help="Retrieval mode (baseline = no RAG). Future: gat/text/hybrid.")
    p.add_argument("--nutrient", default="carb", choices=list(NUTRIENT_CONFIG.keys()),
                   help="Target nutrient (default: carb).")
    p.add_argument("--endpoint", default="http://localhost:8080/v1/chat/completions",
                   help="OpenAI-compatible chat endpoint.")
    p.add_argument("--model", default="qwen3.5",
                   help="Model name (must match what the server expects).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of questions (debugging).")
    p.add_argument("--verbose", action="store_true",
                   help="Print each question's prompt and raw response.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.questions.exists():
        sys.exit(f"ERROR: questions file not found: {args.questions}")

    with open(args.questions) as f:
        payload = json.load(f)
    questions = payload["questions"]
    if args.limit:
        questions = questions[: args.limit]

    cfg = NUTRIENT_CONFIG[args.nutrient]
    gt_field = cfg["gt_column"]
    threshold = cfg["acc_threshold"]

    # Sanity: ensure ground truth field is present
    missing = [q["id"] for q in questions if gt_field not in q]
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} questions missing '{gt_field}' field. "
            f"Re-extract questions with --nutrient {args.nutrient} or with no nutrient filter."
        )

    system_prompt = build_system_prompt(args.nutrient)
    print(f"Mode:      {args.mode}")
    print(f"Nutrient:  {args.nutrient}  (gt_field='{gt_field}', threshold={threshold})")
    print(f"Endpoint:  {args.endpoint}")
    print(f"Model:     {args.model}")
    print(f"Questions: {len(questions)}")
    print()

    results = []
    correct = 0
    mae_sum = 0.0
    failures = 0
    t_total_start = time.time()

    for q in questions:
        meal = q["meal_description"]
        gt = float(q[gt_field])
        user_msg = build_user_message(meal)

        if args.verbose:
            print(f"\n--- Q{q['id']} ---")
            print(f"Meal: {meal}")
            print(f"GT  : {gt} {cfg['unit']}")

        t_start = time.time()
        try:
            raw = chat_completion(
                args.endpoint, args.model,
                system_prompt, user_msg,
                temperature=args.temperature, max_tokens=args.max_tokens,
            )
            error = None
        except Exception as e:
            raw = ""
            error = str(e)
        latency = time.time() - t_start

        pred = parse_prediction(raw, args.nutrient) if raw else -1.0
        if pred < 0 or error:
            failures += 1
            mae = float("nan")
            is_correct = False
        else:
            mae = abs(pred - gt)
            is_correct = mae < threshold
            correct += int(is_correct)
            mae_sum += mae

        results.append({
            "id": q["id"],
            "meal_description": meal,
            "gt": gt,
            "pred": pred,
            "mae": mae if mae == mae else None,    # NaN → None for JSON
            "correct": is_correct,
            "latency_s": round(latency, 2),
            "error": error,
            "raw_response": raw,
        })

        marker = "✓" if is_correct else ("✗" if pred >= 0 else "!")
        print(f"[{q['id']:3d}] {marker}  gt={gt:7.2f}  pred={pred:7.2f}  "
              f"mae={mae if mae == mae else float('nan'):6.2f}  ({latency:5.1f}s)  "
              f"{meal[:60]}")

    total_time = time.time() - t_total_start
    scored = len(results) - failures
    acc = correct / scored if scored else 0.0
    mean_mae = mae_sum / scored if scored else float("nan")

    summary = {
        "config": {
            "mode": args.mode,
            "nutrient": args.nutrient,
            "endpoint": args.endpoint,
            "model": args.model,
            "n_questions": len(results),
            "questions_source": str(args.questions),
            "questions_seed": payload.get("seed"),
        },
        "metrics": {
            "n_scored": scored,
            "n_failed": failures,
            "accuracy":    round(acc, 4),
            "mae":         round(mean_mae, 3) if mean_mae == mean_mae else None,
            "total_time_s": round(total_time, 1),
        },
        "predictions": results,
    }

    if args.out is None:
        args.out = (args.questions.parent.parent / "data" / "results"
                    / f"{args.mode}_{args.nutrient}_n{len(results)}.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"  NutriBench Jetson Test — Summary")
    print("=" * 60)
    print(f"  Mode:        {args.mode}")
    print(f"  Nutrient:    {args.nutrient}  (acc threshold = ±{threshold} {cfg['unit']})")
    print(f"  Scored:      {scored} / {len(results)}  ({failures} failed parse / errors)")
    print(f"  Accuracy:    {acc * 100:.1f}%")
    if mean_mae == mean_mae:
        print(f"  Mean MAE:    {mean_mae:.2f} {cfg['unit']}")
    print(f"  Total time:  {total_time:.1f}s   ({total_time / max(len(results), 1):.1f}s / question)")
    print(f"  Saved to:    {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
