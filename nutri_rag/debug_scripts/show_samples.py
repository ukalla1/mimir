#!/usr/bin/env python3
"""Show side-by-side baseline vs RAG prompt and response for specific samples."""
# cd /home/boxun/work/atlas/mimir/nutri_rag && python scripts/show_samples.py \
#   results/samples_baseline_protein_2026-03-23T05-37-09.jsonl \
#   results/samples_nutribench_v2_rag_protein_2026-03-23T07-21-42.jsonl \
#   protein \
#   0,880,939,518 2>&1
import json
import sys


def load_samples(path):
    samples = {}
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            samples[s["doc_id"]] = s
    return samples


def extract_user_prompt(sample):
    """Extract just the user message content."""
    args = sample["arguments"][0][0][0]
    messages = json.loads(args)
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def show_sample(doc_id, baseline, rag, nutrient="protein"):
    b = baseline[doc_id]
    r = rag[doc_id]

    meal = b["doc"]["meal_description"]
    gt = b["doc"].get(nutrient, "?")

    print(f"{'#'*80}")
    print(f"DOC_ID: {doc_id}")
    print(f"MEAL: {meal}")
    print(f"GROUND TRUTH ({nutrient}): {gt}")
    print(f"BASELINE MAE: {b['mae']:.2f} | RAG MAE: {r['mae']:.2f} | DELTA: {r['mae'] - b['mae']:+.2f}")
    print()

    # Baseline prompt
    b_prompt = extract_user_prompt(b)
    print(f"{'='*40}")
    print(f"BASELINE INPUT (user message):")
    print(f"{'='*40}")
    print(b_prompt)
    print()

    # Baseline response
    print(f"{'='*40}")
    print(f"BASELINE OUTPUT:")
    print(f"{'='*40}")
    print(b["filtered_resps"][0])
    print()

    # RAG prompt
    r_prompt = extract_user_prompt(r)
    print(f"{'='*40}")
    print(f"RAG INPUT (user message):")
    print(f"{'='*40}")
    print(r_prompt)
    print()

    # RAG response
    print(f"{'='*40}")
    print(f"RAG OUTPUT:")
    print(f"{'='*40}")
    print(r["filtered_resps"][0])
    print()
    print()


def main():
    baseline_path = sys.argv[1]
    rag_path = sys.argv[2]
    nutrient = sys.argv[3] if len(sys.argv) > 3 else "protein"
    doc_ids = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 2, 880, 518]

    baseline = load_samples(baseline_path)
    rag = load_samples(rag_path)

    for doc_id in doc_ids:
        if doc_id in baseline and doc_id in rag:
            show_sample(doc_id, baseline, rag, nutrient)
        else:
            print(f"doc_id {doc_id} not found in both files")


if __name__ == "__main__":
    main()
