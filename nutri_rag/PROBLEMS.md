# NutriBench RAG Pipeline — Identified Problems

## Observation

RAG (v0/v1/v2) consistently underperforms the baseline (no RAG) across all four nutrients.

### Benchmark Results (2026-03-23)

| Mode     | Nutrient | Acc@7.5g | MAE    |
|----------|----------|----------|--------|
| baseline | carb     | 0.4580   | 20.47  |
| baseline | energy   | 0.5080   | 154.61 |
| baseline | fat      | 0.6780   | 9.99   |
| baseline | protein  | 0.7350   | 7.00   |
| v0 (BM25)| carb     | 0.4620   | 31.70  |
| v0       | energy   | 0.4360   | 221.44 |
| v0       | fat      | 0.6780   | 10.38  |
| v0       | protein  | 0.6740   | 9.11   |
| v1 (Emb) | carb     | 0.4300   | 28.98  |
| v1       | energy   | 0.4130   | 192.29 |
| v1       | fat      | 0.6310   | 10.69  |
| v1       | protein  | 0.6520   | 8.36   |
| v2 (GAT) | carb     | 0.4290   | 29.14  |
| v2       | energy   | 0.4290   | 188.10 |
| v2       | fat      | 0.6430   | 10.57  |
| v2       | protein  | 0.6550   | 8.45   |

**Key takeaway:** Baseline wins on accuracy for all nutrients except v0-carb (marginal).
Baseline wins on MAE for all nutrients. RAG is actively hurting prediction quality.

---

## Problem 1: Wrong USDA Matches (Critical)

**Impact:** Highest. Wrong matches have multiplicative effect on MAE via the formula.

**What happens:** Text embedding similarity (threshold 0.55) retrieves the wrong form/variant
of a food. The injected formula then amplifies the error.

**Examples:**
- `"egg"` matches "Egg, whole, **dried**" (48g protein/100g) instead of cooked egg (~13g) — **3.7x overestimate**
- `"beans"` matches "Chickpeas, **dry**" (21g protein/100g) instead of baked beans (~5g) — **4x overestimate**
- `"mashed potatoes"` matches "Flour, **potato**" (76g carb/100g) instead of mashed (~15g)
- `"apple"` matches "Pears, raw" (completely different fruit)
- `"pinto beans"` matches "Pupusas con frijoles" (a prepared dish, not the ingredient)

**Root cause:** Cosine similarity cannot distinguish between raw/cooked/dried/flour variants
of the same food. The threshold of 0.55 is too loose.

**Location:** `nutri_rag/config.py:29` (`SIMILARITY_THRESHOLD = 0.55`)

---

## Problem 2: USDA Database Has Almost No Macro Data (High)

**Impact:** High. Even correct matches often return no usable nutrient data.

**What happens:** Only ~1,376 of 74,175 foods (1.9%) in the USDA FoodData Central database
have macronutrient data (Protein, Carbohydrate, Fat, Energy). The vast majority (~62K+) are
`sub_sample_food` entries with only trace minerals (Niacin, Selenium, etc.) — no macros.

**Consequence:**
1. Text embedding correctly matches a food term to a USDA entry
2. `get_nutrients()` queries DuckDB but returns `{}` — no macro data found
3. `_has_target_nutrient()` returns False
4. The reference block is filtered out — no data injected for that item
5. RAG adds processing overhead but provides no value

**Location:** `nutri_rag/bench/prompt.py:47-53` (`_has_target_nutrient()`)

---

## Problem 3: Incomplete Food Term Extraction (High)

**Impact:** High. Missed items create a mixed estimation strategy that confuses the LLM.

**What happens:** The regex-based food term extractor has three patterns:
- Pattern A: `"<qty>g of <food>"` (e.g., "126 grams of maize flour")
- Pattern B: `"<food> weighing <qty>g"` — limited to 1-4 word noun phrases
- Pattern C: `"<food> (<qty>g)"` — limited to 1-4 word noun phrases

When some items match and others don't, the prompt contains a mix of:
- Items WITH USDA reference: "use this formula: carbs = (weight_g / 100) * carbs_per_100g"
- Items WITHOUT match: "no reliable USDA match — use your own knowledge"

This mixed strategy is worse than the baseline's consistent holistic estimation. The LLM
follows the formula strictly for matched items but guesses poorly on unmatched ones,
creating inconsistent predictions.

**Location:** `nutri_rag/bench/retriever.py:31-58` (regex patterns)

---

## Problem 4: Per-Item Format Confuses the LLM (Medium)

**Impact:** Medium. Affects V1/V2 which use per-item format.

**What happens:** V1 and V2 use the per-item prompt format (V3 style):
```
- sugar: USDA match -> "Sugars, granulated" -- Carb: 99.6g
- onion: no reliable USDA match -- use your own knowledge
- caterpillar: no reliable USDA match -- use your own knowledge
```

This format requires the LLM to:
1. Track which items are matched vs unmatched
2. Apply the formula for matched items
3. Switch to guessing for unmatched items
4. Sum everything up

This cognitive load causes more errors than the baseline's simple holistic approach.
The LLM sometimes outputs `-1` as a fallback when too many items are unmatched.

V0 uses the simpler legacy format (global reference block) which is less confusing but
still underperforms baseline due to Problems 1-3.

**Location:** `nutri_rag/bench/prompt.py:100-146` (`_format_per_item()`)

---

## Problem 5: Broken Term Extraction (Low-Medium)

**Impact:** Low-Medium. Affects ~6% of samples.

**What happens:** The regex sometimes grabs too much text, merging multiple food items
into a single search term:
- `"powdered milk with 13 grams of sugar"` → single merged term
- `"white bread with 100g of chicken"` → single term including both items

The merged nonsensical term is then passed to text similarity search, which retrieves
a completely unrelated USDA entry.

**Location:** `nutri_rag/bench/retriever.py:31-58`

---

## Why Baseline Wins

The baseline produces a simple prompt:
```
Query: "126g of maize flour and 27g of raw sugar"
Answer: Let's think step by step.
```

No USDA reference block, no formula, no confusion. The LLM:
1. Parses the meal description holistically
2. Uses its internal knowledge consistently for ALL items
3. Doesn't get confused by partial matches or wrong matches
4. Makes a single coherent estimation without strategy-switching

The LLM's internal nutritional knowledge, while imprecise, is more **consistently**
reliable than the RAG pipeline's mix of correct data, wrong data, and missing data.

---

## Proposed Fixes (Priority Order)

### Fix 1: Filter USDA DB to Macro-Complete Foods Only
**Effort:** Low
**Expected impact:** High

Before building text embeddings, filter `nodes_food` to only include foods that have
at least one macronutrient (Protein, Carb, Fat, or Energy) in `edges_food_contains_nutrient`.
This reduces the search space from 74K to ~1.4K foods, all of which will return useful
nutrient data when matched.

### Fix 2: Raise Similarity Threshold
**Effort:** Low
**Expected impact:** Medium-High

Increase `SIMILARITY_THRESHOLD` from 0.55 to 0.70+ to reduce wrong matches. When the
threshold is higher, more items fall through to "use your own knowledge", which is better
than providing wrong reference data.

### Fix 3: Add Form-Factor Awareness
**Effort:** Medium
**Expected impact:** High

Distinguish raw/cooked/dried/flour variants. Options:
- Add cooking method keywords to the search query
- Pre-filter USDA entries by preparation type
- Use the meal description context to disambiguate (e.g., "mashed potatoes" should
  prefer cooked potato entries over potato flour)

### Fix 4: Replace Regex with LLM-Based Food Extraction
**Effort:** Medium
**Expected impact:** Medium

Use the LLM itself (or a smaller model) to extract structured food items from the meal
description, instead of relying on brittle regex patterns. This would improve coverage
and avoid the merged-term problem.

### Fix 5: Graceful Degradation — Fall Back to Baseline
**Effort:** Low
**Expected impact:** Medium

When the retriever finds no reliable matches (all below threshold), skip the reference
block entirely and use the baseline prompt format. This avoids injecting an empty or
near-empty reference block that only adds confusion.




Loading weights:   0%|          | 0/310 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 310/310 [00:00<00:00, 15698.00it/s]
=== Extracted food terms & candidates ===

  dhal curry: matched=True, candidates=5
    1. Chickpeas, (garbanzo beans, bengal gram), dry (sim=0.480)
    2. Beans, pinto, canned, sodium added, drained and rinsed (sim=0.349)
    3. Nuts, cashew nuts, raw (sim=0.339)
    4. Seeds, pumpkin seeds (pepitas), raw (sim=0.317)
    5. Peanuts, raw (sim=0.305)

  onions: matched=True, candidates=5
    1. Onions, yellow, raw (sim=0.557)
    2. Tomatoes, canned, red, ripe, diced (sim=0.411)
    3. Sauce, salsa, ready-to-serve (sim=0.387)
    4. Apples, fuji, with skin, raw (sim=0.351)
    5. Flour, bread, white, enriched, unbleached (sim=0.328)

  white rice: matched=True, candidates=5
    1. Rice, white, long grain, unenriched, raw (sim=0.622)
    2. Flour, rye (sim=0.517)
    3. Flour, buckwheat (sim=0.494)
    4. Oats, whole grain, steel cut (sim=0.475)
    5. Yogurt, plain, nonfat (sim=0.322)


=== Generated prompt ===
=== USDA Reference (per 100g) ===
These values are for reference only. Use your own judgment — if a match looks
wrong or is for a different form, rely on your
own nutritional knowledge instead. And the reference may only contain partial of
the items mentioned in the query. Just refer to the part if you think some are useful.

- dhal curry:
  1. "Chickpeas, (garbanzo beans, bengal gram), dry" — Carbohydrate: 60.4g | Protein: 21.3g | Fat: 6.3g | Energy: 383.0kcal
  2. "Beans, pinto, canned, sodium added, drained and rinsed" — Carbohydrate: 19.6g | Protein: 6.7g | Fat: 1.3g | Energy: 116.6kcal
  3. "Nuts, cashew nuts, raw" — Carbohydrate: 36.3g | Protein: 17.4g | Fat: 38.9g | Energy: 564.7kcal
  4. "Seeds, pumpkin seeds (pepitas), raw" — Carbohydrate: 18.7g | Protein: 29.9g | Fat: 40.0g | Energy: 554.6kcal
  5. "Peanuts, raw" — Carbohydrate: 26.5g | Protein: 23.2g | Fat: 43.3g | Energy: 588.3kcal

- onions:
  1. "Onions, yellow, raw" — Carbohydrate: 8.6g | Protein: 0.8g | Fat: 0.1g | Energy: 38.0kcal
  2. "Tomatoes, canned, red, ripe, diced" — Carbohydrate: 3.3g | Protein: 0.8g | Fat: 0.5g | Energy: 18.0kcal
  3. "Sauce, salsa, ready-to-serve" — Carbohydrate: 6.7g | Protein: 1.4g | Fat: 0.2g | Energy: 29.0kcal
  4. "Apples, fuji, with skin, raw" — Carbohydrate: 15.7g | Protein: 0.1g | Fat: 0.2g | Energy: 65.0kcal
  5. "Flour, bread, white, enriched, unbleached" — Carbohydrate: 72.8g | Protein: 14.3g | Fat: 1.6g | Energy: 363.0kcal

- white rice:
  1. "Rice, white, long grain, unenriched, raw" — Carbohydrate: 80.3g | Protein: 7.0g | Fat: 1.0g | Energy: 358.7kcal
  2. "Flour, rye" — Carbohydrate: 77.2g | Protein: 8.4g | Fat: 1.9g | Energy: 359.4kcal
  3. "Flour, buckwheat" — Carbohydrate: 75.0g | Protein: 8.9g | Fat: 2.5g | Energy: 358.0kcal
  4. "Oats, whole grain, steel cut" — Carbohydrate: 69.8g | Protein: 12.5g | Fat: 5.8g | Energy: 381.2kcal
  5. "Yogurt, plain, nonfat" — Carbohydrate: 8.1g | Protein: 4.2g | Fat: 0.1g | Energy: 50.0kcal

===

Query: "I've got a breakfast plate with 60 grams of dhal curry, 30 grams of cooked onions, and a hefty serving of 700 grams of cooked white rice."
Answer: Let's think step by step.





==============================
Problems & Solutions Summary
1. Database Deduplication
Before: The USDA FoodData Central DB had 74,175 food entries, but ~84% were duplicate lab sub-samples. For example, "SOY MILK" had 2,272 entries — each was a different lab analysis of the same product from different locations (e.g., "SOY MILK (CA1,CT)", "SOY MILK (FL,MO)").

What it looked like:


fdc_id=325001: SOY MILK (CA1,CT) - NFY1209A1   [sub_sample_food]
fdc_id=325002: SOY MILK (FL,MO) - NFY1209A2   [sub_sample_food]
fdc_id=325003: SOY MILK (IN,NY) - NFY1209A3   [sub_sample_food]
... (2,269 more entries)
These duplicates polluted the search index — searching for any food would return dozens of lab sub-samples instead of one clean entry.

Fix: Deduplication in builder.py — group by description, pick representative fdc_id (most nutrient edges), average nutrient amounts across duplicates (AVG skips NULLs). Reduced 74,175 → 11,739 entries.

2. RAG Prompt Too Authoritative
Before: The prompt said "Use USDA values where provided" with a formula like "carbs = (weight_g / 100) * carbs_per_100g". This forced the LLM to blindly trust wrong USDA matches (e.g., raw rice values for cooked rice).

Result: V2 RAG (Acc=0.655, MAE=8.45) was worse than baseline (Acc=0.735, MAE=7.00).

Fix: Iteratively refined the prompt through multiple versions to the final concise version:


=== USDA Reference (per 100g) ===
Below are possible USDA matches for some ingredients.
Ignore wrong matches and use your own knowledge.
The reference may only cover some of the items — estimate the rest yourself
Keep reasoning brief
3. Multi-Candidate Retrieval (V3) Without Threshold
Before: V3 showed top-5 USDA candidates per food item regardless of match quality. "Tap water" would show 5 candidates including "Beans, Dry, Cranberry" — the LLM used the bean protein value (24g/100g) for 445g of water = 106g phantom protein.

Result: V3 initial (Acc=0.587, MAE=16.14) — much worse than baseline.

Fix: Added similarity threshold (0.6) in prompt.py _format_multi_candidate() — only candidates with cosine similarity >= 0.6 are shown. Bad matches get filtered out, items with no good matches are omitted entirely.

4. LLM Overthinking & Token Truncation
Before: The prompt "Only use a reference value if it is clearly the same food in the same form" made the LLM spend hundreds of tokens debating whether "raw" tilapia reference applies to "boiled smoked" tilapia. Many responses hit the 4096 token limit before producing the Output: {...} line → parsed as -1 (unknown).

Fix: Two changes:

Simplified prompt (removed "same form", "if in doubt", "many may be wrong") to reduce deliberation
Increased max_gen_toks from 4096 to 8192 in run_bench.py as safety net


5. Remaining KB Quality Issues (Identified, Not Yet Fixed)
The DB still has ~3,276 junk entries (28%) with descriptions like "Amino Acids, Chicken..." or "Minerals, Pupusas..." that are lab analysis records, not actual foods. Many common foods (groundnuts, bread/bun, raw onion, soybeans) have no good entries. This limits RAG effectiveness but V3 now handles it gracefully by filtering low-similarity matches.





============================================

Problem A: Junk entries (60.7% of DB)
Even after dedup, 7,121 of 11,739 entries are still junk — lab analysis sub-records with names like "Cholesterol, Chicken, skin, braised..." or "Romaine Lettuce, Region 3, TX2, NFY0102BB". Only 4,618 are usable food descriptions.

Fix: Add a filter in builder.py to exclude entries matching lab prefixes or containing lab codes (NFY, CY, etc.). Simple and safe.

Problem B: Missing common foods (the bigger issue)
Even after cleaning junk, the DB still doesn't have groundnut, chicken wing, bun, maize, or fritter. It only has 343 foundation_food entries with clean names. The rest (market_acquisition, sample_food, etc.) still have messy names like "Beef, top round steak, raw (RS38-R-5)" or "Yogurt, Greek, strawberry, non-fat, FAGE TOTAL (NC1)".

Fix: Add USDA SR Legacy dataset. This is the older, curated USDA database with ~8,000 clean food entries covering common foods like:

"Peanuts, all types, dry-roasted"
"Bread, white, commercially prepared"
"Chicken, broilers or fryers, wing, meat and skin, cooked"
"Sweet potato, cooked, baked in skin"
"Cornmeal, whole-grain, yellow"
"Onions, raw"
"Soybeans, mature seeds, roasted"
SR Legacy has clean, human-readable descriptions and covers cooked/prepared forms that the current dataset lacks. It's freely available from USDA as a single CSV download.

Implementation plan:

Quick fix: Filter junk entries from current DB (builder.py change)
Bigger fix: Download SR Legacy CSV, merge it into the KB builder as an additional data source, prioritize SR Legacy entries for common foods since they have cleaner descriptions