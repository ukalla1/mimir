#!/usr/bin/env python3
"""Show concrete input/output examples for each failure mode.

Usage:
    cd /home/boxun/work/atlas/mimir/nutri_rag && python scripts/show_failure_samples.py \
        results/samples_baseline_protein_2026-03-23T05-37-09.jsonl \
        results/samples_nutribench_v2_rag_protein_2026-03-23T07-21-42.jsonl \
        protein
"""

import json
import re
import sys
import os

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


def extract_system_prompt(sample):
    args = sample["arguments"][0][0][0]
    messages = json.loads(args)
    for m in messages:
        if m["role"] == "system":
            return m["content"]
    return ""


def show(doc_id, baseline, rag, nutrient, label=""):
    b = baseline[doc_id]
    r = rag[doc_id]
    meal = b["doc"]["meal_description"]
    gt = b["doc"].get(nutrient, "?")

    print(f"\n{'#'*80}")
    print(f"FAILURE MODE: {label}")
    print(f"DOC_ID: {doc_id}")
    print(f"MEAL: {meal}")
    print(f"GROUND TRUTH ({nutrient}): {gt}")
    print(f"BASELINE MAE: {b['mae']:.2f} | RAG MAE: {r['mae']:.2f} | DELTA: {r['mae'] - b['mae']:+.2f}")

    # RAG input
    r_prompt = extract_user_prompt(r)
    print(f"\n{'='*40}")
    print(f"RAG INPUT (user message to Qwen3.5):")
    print(f"{'='*40}")
    print(r_prompt)

    # RAG output
    print(f"\n{'='*40}")
    print(f"RAG OUTPUT (Qwen3.5 response):")
    print(f"{'='*40}")
    print(r["filtered_resps"][0])

    # Baseline input
    b_prompt = extract_user_prompt(b)
    print(f"\n{'='*40}")
    print(f"BASELINE INPUT (user message to Qwen3.5):")
    print(f"{'='*40}")
    print(b_prompt)

    # Baseline output
    print(f"\n{'='*40}")
    print(f"BASELINE OUTPUT (Qwen3.5 response):")
    print(f"{'='*40}")
    print(b["filtered_resps"][0])
    print()


def main():
    baseline_path = sys.argv[1]
    rag_path = sys.argv[2]
    nutrient = sys.argv[3] if len(sys.argv) > 3 else "protein"

    baseline = load_samples(baseline_path)
    rag = load_samples(rag_path)

    # ── Problem 0: Empty nutrients (food found but no macros) ──
    # doc=4: maize flour matched with sim=0.634 but has no Protein data
    #        → appears as 0 refs in prompt, essentially becomes baseline
    print("\n" + "█" * 80)
    print("PROBLEM 0: USDA DB — Food matched but NO macronutrient data")
    print("█" * 80)
    print("""
EXPLANATION: The search finds "Corn flour, masa harina" (fdc_id=2712869) with
cosine similarity 0.634 — a GOOD match. But this food entry only has Niacin
in the DB. No Protein, no Carbohydrate, no Fat, no Energy data at all.

The DB has 74,175 foods but only 1,376 (1.9%) have Protein data.
Most entries are sub_sample_food (62K) with only trace minerals.

Here's what happens:
  1. Term "maize flour" extracted ✓
  2. Search finds "Corn flour, masa harina" with sim=0.634 ✓
  3. get_nutrients() returns {} — no macros in DB ✗
  4. _has_target_nutrient() returns False → item filtered out
  5. Prompt shows NO reference for this item

Direct DB evidence:
  fdc_id=2712869 "Corn flour, masa harina, white, dry, raw"
  Nutrients in DB: {Niacin: 1.54}  ← only trace vitamin, no macros!

This affects 42% of samples (417/1000) where ALL food terms get filtered out.
""")
    # Show doc=4 as example
    show(4, baseline, rag, nutrient,
         "PROBLEM 0: Food found with good sim=0.634 but 0 macronutrient data → 0 refs in prompt")

    # Also show doc=10 where ALL items found but none have macros
    show(10, baseline, rag, nutrient,
         "PROBLEM 0: All 5 terms found but NONE have macro data → identical to baseline")

    # ── Problem 1: PARTIAL_COVERAGE ──
    print("\n" + "█" * 80)
    print("PROBLEM 1: PARTIAL COVERAGE — Only some items get references")
    print("█" * 80)
    print("""
EXPLANATION: When RAG only finds references for 1-2 out of 5+ food items,
the model over-anchors on the referenced items and under-estimates the rest.
The per-item format says "use your own knowledge" for unmatched items, but
the model still shifts its reasoning toward the formula-based approach for
matched items and becomes less calibrated overall.
""")
    show(7, baseline, rag, nutrient,
         "PARTIAL COVERAGE: 1/6 items matched (sorghum flour only)")

    show(880, baseline, rag, nutrient,
         "PARTIAL COVERAGE: beans→chickpeas matched, soup unmatched. Model over-anchors on chickpeas.")

    # ── Problem 2: WRONG_MATCH ──
    print("\n" + "█" * 80)
    print("PROBLEM 2: WRONG USDA MATCH — Cosine sim > 0.55 but wrong food")
    print("█" * 80)
    print("""
EXPLANATION: The embedding model gives high similarity to semantically related
but nutritionally different foods. These pass the 0.55 threshold but give
completely wrong nutrient values:
  - "egg" → "Egg, whole, dried" (48g protein vs 13g for fresh egg)
  - "mashed potatoes" → "Flour, potato"
  - "beans" → "Chickpeas, dry" (21g protein vs 5g for baked beans)
  - "apple" → "Pears, raw"
""")
    # doc=988: egg → Egg, whole, dried
    show(988, baseline, rag, nutrient,
         "WRONG MATCH: egg→'Egg, whole, dried' (48g/100g protein vs 13g for normal egg)")

    # doc=667: egg→dried + potato→flour
    show(667, baseline, rag, nutrient,
         "WRONG MATCH: egg→'Egg, whole, dried' AND potato→'Flour, potato'")

    # ── Problem 3: NO_TERMS_EXTRACTED ──
    print("\n" + "█" * 80)
    print("PROBLEM 3: NO FOOD TERMS EXTRACTED — Regex fails on sentence structure")
    print("█" * 80)
    print("""
EXPLANATION: The regex patterns fail to extract food terms from certain
NutriBench sentence structures. When 0 terms are extracted, the RAG block
is empty and the prompt becomes slightly different from baseline (has the
"=== Per-item USDA Reference ===" header or not), causing different model
behavior even though no references are present.

Common missed patterns:
  - "I had 171 grams of boiled fresh groundnuts in their shells"
  - Comma-separated lists: "72g of maize flour, 17g of okra leaves, 5g of onion"
  - "and" connectors parsed incorrectly
""")
    show(29, baseline, rag, nutrient,
         "NO TERMS EXTRACTED: '195g of fritters, 95g of groundnuts, 99g of white maize' → 0 terms")

    show(161, baseline, rag, nutrient,
         "NO TERMS EXTRACTED: '92g of fritters and a 74-gram fresh orange' → 0 terms")

    # ── Problem 4: RAG_RETURNED_MINUS1 ──
    print("\n" + "█" * 80)
    print("PROBLEM 4: MODEL RETURNS -1 — Gives up instead of estimating")
    print("█" * 80)
    print("""
EXPLANATION: When most items show "no reliable USDA match", the model
sometimes gives up entirely and returns {"total_protein": -1} instead
of using its own knowledge. This is catastrophic — even a rough estimate
would be better than -1.

Often triggered when:
  - 6+ items in the meal, only 1 matched
  - The matched item is minor (e.g., tilapia in a meal dominated by flour+veg)
  - Model gets confused by the long list of "no match" items
""")
    show(91, baseline, rag, nutrient,
         "RETURNS -1: 1/6 items matched (tilapia only), model gives up on the rest")

    show(941, baseline, rag, nutrient,
         "RETURNS -1: Complex meal, model can't handle mix of matched/unmatched")

    # ── Problem 5: BROKEN_TERM ──
    print("\n" + "█" * 80)
    print("PROBLEM 5: BROKEN FOOD TERM — Regex grabs too much text")
    print("█" * 80)
    print("""
EXPLANATION: The regex "_PAT_QTY_OF" pattern doesn't stop at "with X grams"
boundaries, so it concatenates multiple food items into one long term:
  - "powdered milk with 13 grams of sugar" → single term (should be 2)
  - "white bread with 100g of chicken" → single term (should be 2)
  - "white rice up with drizzle of 34" → garbled term with numbers

This causes:
  1. Wrong USDA search (query is gibberish)
  2. Missing items (the second food never gets its own reference)
  3. Model confusion (sees weird term in reference list)
""")
    show(257, baseline, rag, nutrient,
         "BROKEN TERM: 'white bread made from Italian flour with 165g of mozzarella cheese' → 1 garbled term")

    show(234, baseline, rag, nutrient,
         "BROKEN TERM: 'white rice up with drizzle of 34' → garbled with number")


if __name__ == "__main__":
    main()
