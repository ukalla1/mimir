#!/usr/bin/env python3
"""Comprehensive failure mode analysis: baseline vs per-item RAG.

Reads sample-level JSONL files for baseline and RAG (V1), categorizes
every sample where RAG is worse than baseline, and identifies root causes.
"""

import json
import re
import sys
import os
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_samples(path):
    samples = {}
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            samples[s["doc_id"]] = s
    return samples


def extract_user_prompt(sample):
    args = sample["arguments"][0][0][0]
    messages = json.loads(args)
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def parse_per_item_refs(prompt):
    """Parse per-item references from the RAG prompt.
    Returns list of (food_term, matched, usda_desc, similarity_info)"""
    refs = []
    for line in prompt.split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        # Matched: "- sugar: USDA match → "Sugars, granulated" — Carb: 99.6g"
        m = re.match(r'^- (.+?): USDA match → "(.+?)" — (.+)$', line)
        if m:
            refs.append({
                "food_term": m.group(1),
                "matched": True,
                "usda_desc": m.group(2),
                "nutrients": m.group(3),
            })
            continue
        # Unmatched: "- onion: no reliable USDA match — use your own knowledge"
        m = re.match(r'^- (.+?): no reliable USDA match', line)
        if m:
            refs.append({
                "food_term": m.group(1),
                "matched": False,
                "usda_desc": None,
                "nutrients": None,
            })
    return refs


def extract_pred(sample):
    """Extract numeric prediction from the model response."""
    resp = sample.get("filtered_resps", [[""]])[0]
    if isinstance(resp, list):
        resp = resp[0]
    # Look for JSON output pattern
    m = re.search(r'"total_\w+"\s*:\s*(-?\d+(?:\.\d+)?)', resp)
    if m:
        return float(m.group(1))
    return -1


def categorize_failure(rag_prompt, rag_resp, baseline_resp, meal, refs, rag_mae, baseline_mae):
    """Categorize why RAG failed for this sample."""
    reasons = []

    # 1. Check if food term extraction missed items or was broken
    total_refs = len(refs)
    matched_refs = sum(1 for r in refs if r["matched"])
    unmatched_refs = total_refs - matched_refs

    if total_refs == 0:
        reasons.append("NO_TERMS_EXTRACTED")
        return reasons

    # Check for broken food terms (terms that contain numbers or are too long)
    for ref in refs:
        term = ref["food_term"]
        if re.search(r'\d', term):
            reasons.append(f"BROKEN_TERM:{term}")
        if len(term.split()) > 6:
            reasons.append(f"LONG_TERM:{term}")

    # 2. Check for wrong USDA matches
    for ref in refs:
        if not ref["matched"]:
            continue
        term = ref["food_term"].lower()
        usda = ref["usda_desc"].lower()
        # Heuristic: if food_term and usda_desc share no significant words, likely wrong
        term_words = set(re.findall(r'\w{3,}', term))
        usda_words = set(re.findall(r'\w{3,}', usda))
        overlap = term_words & usda_words
        if len(overlap) == 0 and len(term_words) > 0:
            reasons.append(f"WRONG_MATCH:{term}->{ref['usda_desc']}")

    # 3. Check if RAG returned -1 (parse failure)
    rag_pred = extract_pred_from_resp(rag_resp)
    if rag_pred == -1:
        reasons.append("RAG_RETURNED_MINUS1")

    # 4. Check for arithmetic errors in baseline vs RAG
    baseline_pred = extract_pred_from_resp(baseline_resp)
    if baseline_pred == -1:
        reasons.append("BASELINE_RETURNED_MINUS1")

    # 5. All items matched but still worse — model over-trusts references
    if matched_refs == total_refs and not reasons:
        reasons.append("ALL_MATCHED_STILL_WORSE")

    # 6. Mixed (some matched, some not) — partial coverage issue
    if matched_refs > 0 and unmatched_refs > 0 and not reasons:
        reasons.append(f"PARTIAL_COVERAGE:{matched_refs}/{total_refs}")

    # 7. All unmatched but RAG block still injected
    if matched_refs == 0 and not reasons:
        reasons.append("ALL_UNMATCHED")

    if not reasons:
        reasons.append("UNKNOWN")

    return reasons


def extract_pred_from_resp(resp):
    if isinstance(resp, list):
        resp = resp[0] if resp else ""
    m = re.search(r'"total_\w+"\s*:\s*(-?\d+(?:\.\d+)?)', str(resp))
    if m:
        return float(m.group(1))
    return -1


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_failures.py <baseline.jsonl> <rag.jsonl> [nutrient]")
        sys.exit(1)

    baseline_path = sys.argv[1]
    rag_path = sys.argv[2]
    nutrient = sys.argv[3] if len(sys.argv) > 3 else "protein"

    baseline = load_samples(baseline_path)
    rag = load_samples(rag_path)

    common_ids = sorted(set(baseline.keys()) & set(rag.keys()))
    print(f"Loaded {len(common_ids)} common samples\n")

    # Categorize each sample
    rag_better = []
    rag_worse = []
    rag_same = []  # within 0.5g

    failure_reasons = Counter()
    failure_examples = defaultdict(list)

    for doc_id in common_ids:
        b = baseline[doc_id]
        r = rag[doc_id]

        b_mae = b["mae"]
        r_mae = r["mae"]
        delta = r_mae - b_mae

        if delta < -0.5:
            rag_better.append((doc_id, delta, b_mae, r_mae))
        elif delta > 0.5:
            rag_worse.append((doc_id, delta, b_mae, r_mae))

            # Analyze why RAG was worse
            rag_prompt = extract_user_prompt(r)
            refs = parse_per_item_refs(rag_prompt)
            meal = b["doc"]["meal_description"]
            rag_resp = r["filtered_resps"][0]
            baseline_resp = b["filtered_resps"][0]

            reasons = categorize_failure(
                rag_prompt, rag_resp, baseline_resp, meal, refs, r_mae, b_mae
            )
            for reason in reasons:
                # Get the category (before the colon)
                cat = reason.split(":")[0]
                failure_reasons[cat] += 1
                if len(failure_examples[cat]) < 3:
                    failure_examples[cat].append({
                        "doc_id": doc_id,
                        "meal": meal,
                        "gt": b["doc"].get(nutrient, "?"),
                        "baseline_mae": round(b_mae, 2),
                        "rag_mae": round(r_mae, 2),
                        "delta": round(delta, 2),
                        "reason": reason,
                        "refs": refs,
                        "rag_pred": extract_pred_from_resp(rag_resp),
                        "baseline_pred": extract_pred_from_resp(baseline_resp),
                    })
        else:
            rag_same.append((doc_id, delta, b_mae, r_mae))

    # Print overview
    print("=" * 70)
    print("OVERVIEW")
    print("=" * 70)
    print(f"RAG better (delta < -0.5):  {len(rag_better)} samples")
    print(f"RAG same   (|delta| <= 0.5): {len(rag_same)} samples")
    print(f"RAG worse  (delta > 0.5):   {len(rag_worse)} samples")
    print()

    avg_better = sum(d for _, d, _, _ in rag_better) / len(rag_better) if rag_better else 0
    avg_worse = sum(d for _, d, _, _ in rag_worse) / len(rag_worse) if rag_worse else 0
    print(f"When RAG is better: avg improvement = {avg_better:.2f}g MAE")
    print(f"When RAG is worse:  avg degradation = {avg_worse:+.2f}g MAE")

    # Print top RAG wins
    print()
    print("=" * 70)
    print("TOP 10 RAG WINS (biggest improvement)")
    print("=" * 70)
    for doc_id, delta, b_mae, r_mae in sorted(rag_better, key=lambda x: x[1])[:10]:
        meal = baseline[doc_id]["doc"]["meal_description"][:80]
        print(f"  doc={doc_id:4d}  baseline_mae={b_mae:7.2f}  rag_mae={r_mae:7.2f}  delta={delta:+8.2f}")
        print(f"    {meal}")

    # Print top RAG losses
    print()
    print("=" * 70)
    print("TOP 10 RAG LOSSES (biggest degradation)")
    print("=" * 70)
    for doc_id, delta, b_mae, r_mae in sorted(rag_worse, key=lambda x: x[1], reverse=True)[:10]:
        meal = baseline[doc_id]["doc"]["meal_description"][:80]
        gt = baseline[doc_id]["doc"].get(nutrient, "?")
        rag_prompt = extract_user_prompt(rag[doc_id])
        refs = parse_per_item_refs(rag_prompt)
        b_pred = extract_pred_from_resp(baseline[doc_id]["filtered_resps"][0])
        r_pred = extract_pred_from_resp(rag[doc_id]["filtered_resps"][0])
        print(f"  doc={doc_id:4d}  baseline_mae={b_mae:7.2f}  rag_mae={r_mae:7.2f}  delta={delta:+8.2f}")
        print(f"    meal: {meal}")
        print(f"    GT={gt}  baseline_pred={b_pred}  rag_pred={r_pred}")
        for ref in refs:
            status = f'→ "{ref["usda_desc"]}"' if ref["matched"] else "→ no match"
            print(f"    ref: {ref['food_term']} {status}")
        print()

    # Print failure mode breakdown
    print()
    print("=" * 70)
    print("FAILURE MODE BREAKDOWN (samples where RAG is worse)")
    print("=" * 70)
    for reason, count in failure_reasons.most_common():
        print(f"\n  {reason}: {count} samples")
        for ex in failure_examples[reason]:
            print(f"    doc={ex['doc_id']}  meal: {ex['meal'][:70]}")
            print(f"      GT={ex['gt']}  baseline_pred={ex['baseline_pred']}  rag_pred={ex['rag_pred']}")
            print(f"      baseline_mae={ex['baseline_mae']}  rag_mae={ex['rag_mae']}  delta={ex['delta']}")
            for ref in ex["refs"]:
                if ref["matched"]:
                    print(f"      ref: {ref['food_term']} → \"{ref['usda_desc']}\"")
                else:
                    print(f"      ref: {ref['food_term']} → no match")
            print()

    # Additional: check how many samples have broken terms
    print("=" * 70)
    print("FOOD TERM EXTRACTION ANALYSIS (all samples)")
    print("=" * 70)
    total_terms = 0
    broken_terms = 0
    long_terms = 0
    terms_with_numbers = []
    terms_too_long = []

    for doc_id in common_ids:
        r = rag[doc_id]
        rag_prompt = extract_user_prompt(r)
        refs = parse_per_item_refs(rag_prompt)
        for ref in refs:
            total_terms += 1
            term = ref["food_term"]
            if re.search(r'\d', term):
                broken_terms += 1
                if len(terms_with_numbers) < 15:
                    terms_with_numbers.append(f"doc={doc_id}: '{term}'")
            if len(term.split()) > 5:
                long_terms += 1
                if len(terms_too_long) < 15:
                    terms_too_long.append(f"doc={doc_id}: '{term}'")

    print(f"Total food terms extracted: {total_terms}")
    print(f"Terms containing numbers: {broken_terms} ({broken_terms/total_terms*100:.1f}%)")
    print(f"Terms > 5 words: {long_terms} ({long_terms/total_terms*100:.1f}%)")
    print()
    if terms_with_numbers:
        print("Examples of terms with numbers:")
        for t in terms_with_numbers:
            print(f"  {t}")
    print()
    if terms_too_long:
        print("Examples of long terms:")
        for t in terms_too_long:
            print(f"  {t}")

    # Match quality analysis
    print()
    print("=" * 70)
    print("MATCH QUALITY ANALYSIS (all samples)")
    print("=" * 70)
    match_stats = {"total_refs": 0, "matched": 0, "unmatched": 0}
    no_word_overlap = []

    for doc_id in common_ids:
        r = rag[doc_id]
        rag_prompt = extract_user_prompt(r)
        refs = parse_per_item_refs(rag_prompt)
        for ref in refs:
            match_stats["total_refs"] += 1
            if ref["matched"]:
                match_stats["matched"] += 1
                # Check word overlap
                term = ref["food_term"].lower()
                usda = ref["usda_desc"].lower()
                term_words = set(re.findall(r'\w{3,}', term))
                usda_words = set(re.findall(r'\w{3,}', usda))
                overlap = term_words & usda_words
                if len(overlap) == 0 and len(term_words) > 0:
                    if len(no_word_overlap) < 20:
                        no_word_overlap.append(f"doc={doc_id}: '{ref['food_term']}' → '{ref['usda_desc']}'")
            else:
                match_stats["unmatched"] += 1

    print(f"Total references: {match_stats['total_refs']}")
    print(f"Matched (above threshold): {match_stats['matched']} ({match_stats['matched']/match_stats['total_refs']*100:.1f}%)")
    print(f"Unmatched: {match_stats['unmatched']} ({match_stats['unmatched']/match_stats['total_refs']*100:.1f}%)")
    print()
    print(f"Matched but NO word overlap with query (likely wrong matches):")
    for t in no_word_overlap:
        print(f"  {t}")


if __name__ == "__main__":
    main()
