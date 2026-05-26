#!/usr/bin/env python3
"""Extract N questions from NutriBench v2 for Jetson Orin testing.

Each sample in NutriBench v2 contains a meal description plus ground-truth
values for carb / protein / fat / energy. This script samples N items and
writes them to a portable JSON file that can be shipped to the Jetson and
replayed against the on-device LLM + RAG pipeline.

Usage
-----
    # default: 20 samples, seed 42, split=train
    python nutribench_jetson_questions.py

    # custom count + split + output path
    python nutribench_jetson_questions.py --n 50 --split train --seed 7 \\
        --out /tmp/nutribench_jetson_50.json

    # restrict to a single nutrient (only includes that gt field in output)
    python nutribench_jetson_questions.py --nutrient carb

Output JSON format
------------------
    {
      "source": "dongx1997/NutriBench",
      "config": "v2",
      "split": "train",
      "n": 20,
      "seed": 42,
      "questions": [
        {
          "id": 0,
          "meal_description": "...",
          "carb": 33.5,
          "protein": 12.1,
          "fat": 4.0,
          "energy": 220
        },
        ...
      ]
    }
"""

import argparse
import json
import random
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    sys.stderr.write(
        "ERROR: `datasets` not installed. Install with:\n"
        "    pip install datasets\n"
    )
    sys.exit(1)


DATASET_PATH = "dongx1997/NutriBench"
DATASET_CONFIG = "v2"
GT_FIELDS = ("carb", "protein", "fat", "energy")

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "nutribench_jetson_questions.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample NutriBench questions for Jetson testing.")
    p.add_argument("--n",        type=int, default=20,            help="Number of questions to sample (default: 20).")
    p.add_argument("--split",    type=str, default="train",       help="Dataset split (default: train).")
    p.add_argument("--seed",     type=int, default=42,            help="Random seed for reproducibility (default: 42).")
    p.add_argument("--nutrient", type=str, default=None, choices=list(GT_FIELDS),
                   help="If set, output only this nutrient's ground truth (default: all four).")
    p.add_argument("--out",      type=Path, default=DEFAULT_OUT,
                   help=f"Output JSON path (default: {DEFAULT_OUT}).")
    return p.parse_args()


def load_nutribench(split: str):
    print(f"Loading {DATASET_PATH} (config={DATASET_CONFIG}, split={split})...")
    ds = load_dataset(DATASET_PATH, DATASET_CONFIG, split=split)
    print(f"  Loaded {len(ds)} samples. Fields: {ds.column_names}")
    return ds


def sample_questions(ds, n: int, seed: int, nutrient: str | None) -> list[dict]:
    if n > len(ds):
        raise ValueError(f"Requested {n} samples but dataset only has {len(ds)}.")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)

    fields = (nutrient,) if nutrient else GT_FIELDS

    questions = []
    for new_id, src_idx in enumerate(indices):
        row = ds[int(src_idx)]
        item = {
            "id": new_id,
            "source_index": int(src_idx),
            "meal_description": row["meal_description"],
        }
        for f in fields:
            if f in row:
                item[f] = row[f]
        questions.append(item)

    return questions


def main() -> None:
    args = parse_args()

    ds = load_nutribench(args.split)
    questions = sample_questions(ds, args.n, args.seed, args.nutrient)

    payload = {
        "source": DATASET_PATH,
        "config": DATASET_CONFIG,
        "split": args.split,
        "n": args.n,
        "seed": args.seed,
        "nutrient": args.nutrient,  # null = all four ground-truth fields included
        "questions": questions,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(questions)} questions to: {args.out}")
    print("\nFirst sample preview:")
    print(json.dumps(questions[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
