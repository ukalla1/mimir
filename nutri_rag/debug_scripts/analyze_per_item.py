#!/usr/bin/env python3
"""Analyze per-item RAG vs baseline: why is RAG still hurting?

Compares sample-level predictions to understand failure modes.
"""

import json
import sys
import re


def load_samples(path):
    """Load JSONL samples into a dict keyed by doc_id."""
    samples = {}
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            samples[s["doc_id"]] = s
    return samples


def extract_prompt(sample):
    """Extract the user message content from a sample."""
    args = sample["arguments"][0][0][0]
    messages = json.loads(args)
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def count_ref_items(prompt):
    """Count matched and unmatched items in per-item prompt."""
    matched = len(re.findall(r"USDA match →", prompt))
    unmatched = len(re.findall(r"no reliable USDA match", prompt))
    return matched, unmatched


def extract_prediction(sample):
    """Extract the predicted value from model response."""
    resp = sample["filtered_resps"][0]
    # Look for the JSON output pattern
    patterns = [
        r'"total_protein"\s*:\s*([\d.]+)',
        r'"total_carbohydrates"\s*:\s*([\d.]+)',
        r'"total_fat"\s*:\s*([\d.]+)',
        r'"total_energy"\s*:\s*([\d.]+)',
    ]
    for pat in patterns:
        m = re.search(pat, resp)
        if m:
            return float(m.group(1))
    return -1


def main():
    baseline_path = sys.argv[1]
    rag_path = sys.argv[2]
    nutrient = sys.argv[3] if len(sys.argv) > 3 else "protein"

    baseline = load_samples(baseline_path)
    rag = load_samples(rag_path)

    # Match by doc_id
    common_ids = sorted(set(baseline.keys()) & set(rag.keys()))
    print(f"Comparing {len(common_ids)} common samples for {nutrient}")
    print()

    # Categorize
    rag_helped = []   # RAG better than baseline
    rag_hurt = []     # RAG worse than baseline
    rag_same = []     # same result (both correct or both wrong)
    no_ref_samples = []  # RAG had no references at all (fell back to baseline)

    for doc_id in common_ids:
        b = baseline[doc_id]
        r = rag[doc_id]

        b_mae = b["mae"]
        r_mae = r["mae"]
        b_acc = b["acc"]
        r_acc = r["acc"]

        prompt = extract_prompt(r)
        matched, unmatched = count_ref_items(prompt)
        has_ref = "Per-item USDA Reference" in prompt

        entry = {
            "doc_id": doc_id,
            "meal": b["doc"]["meal_description"][:80],
            "gt": b["doc"].get(nutrient, "?"),
            "b_mae": round(b_mae, 2),
            "r_mae": round(r_mae, 2),
            "b_acc": b_acc,
            "r_acc": r_acc,
            "matched": matched,
            "unmatched": unmatched,
            "delta_mae": round(r_mae - b_mae, 2),  # positive = RAG worse
        }

        if not has_ref:
            no_ref_samples.append(entry)
        elif r_mae < b_mae - 0.5:
            rag_helped.append(entry)
        elif r_mae > b_mae + 0.5:
            rag_hurt.append(entry)
        else:
            rag_same.append(entry)

    print(f"=== SUMMARY ===")
    print(f"No USDA refs (pure fallback): {len(no_ref_samples)}")
    print(f"RAG helped (MAE improved >0.5): {len(rag_helped)}")
    print(f"RAG hurt (MAE worsened >0.5):   {len(rag_hurt)}")
    print(f"RAG ~same (delta < 0.5):        {len(rag_same)}")
    print()

    # Average delta
    all_with_ref = rag_helped + rag_hurt + rag_same
    if all_with_ref:
        avg_delta = sum(e["delta_mae"] for e in all_with_ref) / len(all_with_ref)
        print(f"Avg MAE delta (RAG - baseline) for samples WITH refs: {avg_delta:.2f}")
        print(f"  (positive = RAG is worse)")
    print()

    # Breakdown by number of matched items
    print(f"=== BREAKDOWN BY # MATCHED ITEMS ===")
    from collections import defaultdict
    by_matched = defaultdict(list)
    for e in all_with_ref:
        by_matched[e["matched"]].append(e)

    for n_matched in sorted(by_matched.keys()):
        entries = by_matched[n_matched]
        avg_d = sum(e["delta_mae"] for e in entries) / len(entries)
        helped = sum(1 for e in entries if e["delta_mae"] < -0.5)
        hurt = sum(1 for e in entries if e["delta_mae"] > 0.5)
        avg_total = sum(e["matched"] + e["unmatched"] for e in entries) / len(entries)
        print(f"  {n_matched} matched (avg {avg_total:.1f} total items): "
              f"n={len(entries)}, avg_delta={avg_d:+.2f}, helped={helped}, hurt={hurt}")
    print()

    # Show worst RAG-hurt examples
    rag_hurt.sort(key=lambda e: e["delta_mae"], reverse=True)
    print(f"=== TOP 15 WORST RAG-HURT CASES ===")
    for e in rag_hurt[:15]:
        print(f"  doc_id={e['doc_id']}: GT={e['gt']}, baseline_MAE={e['b_mae']}, "
              f"RAG_MAE={e['r_mae']}, delta={e['delta_mae']:+.2f}, "
              f"matched={e['matched']}/{e['matched']+e['unmatched']}")
        print(f"    {e['meal']}")
    print()

    # Show best RAG-helped examples
    rag_helped.sort(key=lambda e: e["delta_mae"])
    print(f"=== TOP 15 BEST RAG-HELPED CASES ===")
    for e in rag_helped[:15]:
        print(f"  doc_id={e['doc_id']}: GT={e['gt']}, baseline_MAE={e['b_mae']}, "
              f"RAG_MAE={e['r_mae']}, delta={e['delta_mae']:+.2f}, "
              f"matched={e['matched']}/{e['matched']+e['unmatched']}")
        print(f"    {e['meal']}")
    print()

    # For worst hurt cases, show the actual prompts and responses
    print(f"=== DETAILED COMPARISON: TOP 3 WORST RAG-HURT ===")
    for e in rag_hurt[:3]:
        doc_id = e["doc_id"]
        print(f"\n{'='*80}")
        print(f"doc_id={doc_id}: {e['meal']}")
        print(f"GT={e['gt']}, baseline_MAE={e['b_mae']}, RAG_MAE={e['r_mae']}")

        # Show RAG prompt
        r_prompt = extract_prompt(rag[doc_id])
        # Extract just the reference block
        ref_match = re.search(r"(=== Per-item.*?===)", r_prompt, re.DOTALL)
        if ref_match:
            print(f"\nRAG References:")
            print(ref_match.group(1))

        print(f"\nBaseline response (excerpt):")
        b_resp = baseline[doc_id]["filtered_resps"][0]
        # Show just the calculation lines
        for line in b_resp.split("\n"):
            if any(kw in line.lower() for kw in ["protein", "total", "output", "calculation"]):
                print(f"  {line.strip()[:120]}")

        print(f"\nRAG response (excerpt):")
        r_resp = rag[doc_id]["filtered_resps"][0]
        for line in r_resp.split("\n"):
            if any(kw in line.lower() for kw in ["protein", "total", "output", "calculation", "usda"]):
                print(f"  {line.strip()[:120]}")


if __name__ == "__main__":
    main()
