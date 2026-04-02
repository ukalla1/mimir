# NutriBench RAG Pipeline — System Overview

## Pipeline Flow

```
NutriBench sample (meal_description)
  │
  ▼
┌─────────────────────────────────────────────────┐
│  1. SYSTEM PROMPT (description field in YAML)   │
│     Built by nutrient_prompts.build_system_prompt│
│     Contains: task instruction + 3 CoT few-shot │
│     examples with step-by-step reasoning        │
│     (Same for baseline AND RAG modes)           │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  2. USER MESSAGE (doc_to_text)                  │
│                                                 │
│  BASELINE: just the query                       │
│    Query: "126g of maize flour and 27g of sugar"│
│    Answer: Let's think step by step.            │
│                                                 │
│  RAG (v0/v1/v2): reference block + query        │
│    === Per-item USDA Reference (per 100g) ===   │
│    Formula: carbs = (weight_g/100)*carbs_per_100│
│    - maize flour: USDA match → "Corn flour..."  │
│      Carb: 76.8g | Protein: 6.9g | ...         │
│    - sugar: no reliable USDA match — use own    │
│    ===                                          │
│    Query: "126g of maize flour and 27g of sugar"│
│    Answer: Let's think step by step.            │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  3. LLM (Qwen3.5-9B via llama-server)          │
│     Generates CoT reasoning + final answer      │
│     Output: {"total_carb": 123.4}               │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  4. EVALUATION (process_results in task_utils)  │
│     Parse output → extract number → compare     │
│     Acc@7.5g: |pred - gold| <= 7.5              │
│     MAE: |pred - gold|                          │
└─────────────────────────────────────────────────┘
```

---

## RAG Retrieval Pipeline (Step 2 in Detail)

```
meal_description: "126g of maize flour and 27g of raw sugar"
  │
  ├─ Regex extraction (retriever.py)
  │   _PAT_QTY_OF: "126 grams of <food>" → "maize flour"
  │   _PAT_QTY_OF: "27g of <food>" → "sugar"  (after stripping "raw")
  │   Result: ["maize flour", "sugar"]
  │
  ├─ Per-term search (one at a time, k=1):
  │   V0: BM25 keyword search (DuckDB FTS + synonym map)
  │   V1: TextEmbedder → cosine search against 74K pre-computed vectors
  │   V2: Same as V1, then GAT neighbor expansion if top candidates ambiguous
  │
  ├─ Macro-aware ranking (search.py:119-127):
  │   Get top k*5 candidates → count which have carb/protein/fat data
  │   Sort by (macro_count DESC, text_score DESC) → take top k
  │
  ├─ Nutrient lookup (get_nutrients):
  │   Query DuckDB: edges_food_contains_nutrient JOIN nodes_nutrient
  │   Filter to KEY_NUTRIENTS only
  │   Returns: {"Carbohydrate, by difference": 76.8, "Protein": 6.9, ...}
  │
  └─ Prompt formatting (prompt.py):
     V0: Legacy format — global reference block, all matches listed together
     V1/V2: Per-item format — each food paired with match or "no reliable match"
     Threshold gating: similarity_score >= 0.55 required for "reliable match"
```

---

## Key Differences Between Modes

| | Baseline | V0 (BM25) | V1 (Embedding) | V2 (Emb+GAT) |
|---|---|---|---|---|
| **Food extraction** | None | Regex | Regex | Regex |
| **Search method** | None | DuckDB FTS + synonyms | Qwen3-Embedding cosine | Qwen3-Emb + GAT neighbors |
| **Prompt format** | Query only | Legacy (global block) | Per-item (threshold gated) | Per-item (threshold gated) |
| **Reference data** | None | USDA per-100g | USDA per-100g | USDA per-100g |
| **Formula injection** | No | Yes | Yes | Yes |

---

## Key Files

### Benchmark Pipeline

| File | Purpose |
|------|---------|
| `scripts/run_bench.py` | Entry point — generates YAML, symlinks task, runs lm_eval |
| `scripts/run_baseline.py` | Baseline (no RAG) runner |
| `scripts/run_all_bench.py` | Batch runner for all 16 combinations (4 nutrients x 4 modes) |

### Retrieval

| File | Purpose |
|------|---------|
| `nutri_rag/bench/retriever.py` | Regex food term extraction + V1/V2 semantic search |
| `nutri_rag/bench/retriever_bm25.py` | V0 BM25 keyword search retriever |
| `nutri_rag/search.py` | Semantic vector search + GAT expansion + macro-aware ranking |
| `nutri_rag/search_bm25.py` | BM25 full-text search + synonym mapping |
| `nutri_rag/embedding.py` | TextEmbedder (Qwen3-Embedding), FoodVectorIndex, GATIndex |

### Prompt Construction

| File | Purpose |
|------|---------|
| `nutri_rag/bench/prompt.py` | Formats USDA reference block (legacy + per-item formats) |
| `nutri_rag/bench/nutrient_prompts.py` | Builds system prompt with CoT few-shot examples per nutrient |

### lm-eval Task Definitions

| Directory | Mode | doc_to_text |
|-----------|------|-------------|
| `tasks/_baseline_temp/` | Baseline | `doc_to_text_cot` (query only) |
| `tasks/nutribench_v2_rag_bm25/` | V0 | `doc_to_text_rag` (BM25, legacy format) |
| `tasks/nutribench_v2_rag/` | V1 | `doc_to_text_rag` (embedding, per-item format) |
| `tasks/nutribench_v2_rag_gat/` | V2 | `doc_to_text_rag` (embedding+GAT, per-item format) |

Each task directory contains:
- `*.yaml` — task config (includes template)
- `_*_template_yaml` — generated template with system prompt, few-shot examples, metric list
- `utils.py` — Python functions for doc_to_text, process_results

### Configuration

| File | Key Settings |
|------|-------------|
| `nutri_rag/config.py` | `SIMILARITY_THRESHOLD = 0.55`, `TOP_K_FOODS = 3`, `GAT_NEIGHBORS_K = 5` |
| `nutri_rag/config.py` | `DB_PATH`, `FOOD_EMBEDDINGS_PATH`, `TEXT_EMBEDDINGS_PATH` |

---

## Critical Path (Where Errors Compound)

```
Regex extracts term (may miss items or merge multiple items)
  │
  ▼
Search returns USDA match (may return wrong variant: dried/raw/flour)
  │
  ▼
Nutrient lookup (may return empty — 98% of USDA foods lack macros)
  │
  ▼
Prompt injects per-100g values (may be for wrong form factor)
  │
  ▼
LLM applies formula faithfully: weight_g / 100 * per_100g
  │
  ▼
AMPLIFIED ERROR (wrong per-100g value × correct weight = wrong answer)
```

The baseline avoids this entire chain by relying on the LLM's internal knowledge,
which produces more consistent (if imprecise) estimates without catastrophic errors
from wrong USDA matches.

---

## Concrete Example: Same Sample Across All Modes

**Sample:** doc_id=777, nutrient=carb
**Meal:** "I've got a breakfast plate with 60 grams of dhal curry, 30 grams of cooked onions, and a hefty serving of 700 grams of cooked white rice."
**Gold target:** ~207g carbs

### System Prompt (identical for all 4 modes)

```
For the given query including a meal description, think step by step as follows:
1. Parse the meal description into discrete food or beverage items along with their
   serving size. If the serving size of any item in the meal is not specified, assume
   it is a single standard serving based on common nutritional guidelines (e.g., USDA).
   Ignore additional information that doesn't relate to the item name and serving size.
2. For each food or beverage item in the meal, calculate the amount of carbohydrates
   in grams for the specific serving size.
3. Respond with a dictionary object containing the total carbohydrates in grams as follows:
   {"total_carbohydrates": total grams of carbohydrates for the serving}
   For the total carbohydrates, respond with just the numeric amount without extra text.
   If you don't know the answer, set the value of "total_carbohydrates" to -1.

Follow the format of the following examples when answering

Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of
orange juice."
Answer: Let's think step by step.
The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.
1 cup of oatmeal has 27g carbs.
1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.
1 glass of orange juice has 26g carbs.
So the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5
Output: {"total_carbohydrates": 66.5}

Query: "I ate scrambled eggs made with 2 eggs and a toast for breakfast."
Answer: Let's think step by step.
The meal consists of scrambled eggs made with 2 eggs and 1 toast.
Scrambled eggs made with 2 eggs has 2g carbs.
1 toast has 13g carbs.
So the total grams of carbs in the meal = (2 + 13) = 15
Output: {"total_carbohydrates": 15}

Query: "Half a peanut butter and jelly sandwich."
Answer: Let's think step by step.
The meal consists of 1/2 a peanut butter and jelly sandwich.
1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly
sandwich has (50.6*(1/2)) = 25.3g carbs.
So the total grams of carbs in the meal = 25.3
Output: {"total_carbohydrates": 25.3}
```

---

### Baseline | acc=True | MAE=4.0

**USER MESSAGE:**
```
Query: "I've got a breakfast plate with 60 grams of dhal curry, 30 grams of cooked
onions, and a hefty serving of 700 grams of cooked white rice."
Answer: Let's think step by step.
```

**LLM RESPONSE:**
```
The meal consists of 60 grams of dhal curry, 30 grams of cooked onions, and 700 grams
of cooked white rice.

1. 60 grams of dhal curry: Dhal (lentils) is a carbohydrate-rich food. According to
   USDA data, 100 grams of cooked lentils (dhal) contains approximately 20 grams of
   carbohydrates.
   Calculation: 60g * (20g carbs / 100g) = 12g carbs.

2. 30 grams of cooked onions: Onions are vegetables with moderate carbohydrates.
   100 grams of cooked onions contains approximately 9 grams of carbohydrates.
   Calculation: 30g * (9g carbs / 100g) = 2.7g carbs.

3. 700 grams of cooked white rice: White rice is high in carbohydrates. 100 grams
   of cooked white rice contains approximately 28 grams of carbohydrates.
   Calculation: 700g * (28g carbs / 100g) = 196g carbs.

Total carbohydrates = (12 + 2.7 + 196) = 210.7 grams.

Output: {"total_carbohydrates": 210.7}
```

**Analysis:** LLM uses its own knowledge. Correctly identifies all items as *cooked* and
uses cooked nutrient densities (lentils 20g/100g, onions 9g/100g, rice 28g/100g).
Result is very close to gold (MAE=4.0).

---

### V0 (BM25) | acc=False | MAE=392.2

**USER MESSAGE:**
```
=== USDA Nutritional Reference Data (per 100g) ===
These are approximate reference values. Use them if they match your food items.
For foods NOT listed here, use your own nutritional knowledge.
Formula: carbs = (weight_g / 100) * carbs_per_100g

[1] "Lentils, dry" (USDA #2644283)
    Carbohydrate: 62.2g | Protein: 23.6g | Fat: 1.9g | Energy: 360.3kcal

[2] "Onions, red, raw" (USDA #790577)
    Carbohydrate: 9.9g | Protein: 0.9g | Fat: 0.1g | Energy: 44.0kcal

[3] "Flour, rice, white, unenriched" (USDA #790214)
    Carbohydrate: 79.8g | Protein: 6.9g | Fat: 1.3g | Energy: 359.0kcal

===

Query: "I've got a breakfast plate with 60 grams of dhal curry, 30 grams of cooked
onions, and a hefty serving of 700 grams of cooked white rice."
Answer: Let's think step by step.
```

**LLM RESPONSE:**
```
Let's think step by step.
The meal consists of 60 grams of dhal curry (made from lentils), 30 grams of cooked
onions, and 700 grams of cooked white rice.

1. Dhal curry (Lentils, dry):
   The USDA data provided is for "Lentils, dry" with 62.2g carbs per 100g.
   Although the query specifies "dhal curry" and "60 grams", in the context of this
   specific reference data set, we map "dhal" to "Lentils, dry".
   Calculation: (60 / 100) * 62.2 = 0.6 * 62.2 = 37.32g carbs.

2. Cooked onions:
   The USDA data provided is for "Onions, red, raw" with 9.9g carbs per 100g.
   Calculation: (30 / 100) * 9.9 = 0.3 * 9.9 = 2.97g carbs.

3. Cooked white rice:
   The USDA data provided is for "Flour, rice, white, unenriched" with 79.8g carbs
   per 100g. We map "cooked white rice" to the provided "Flour, rice, white" data.
   Calculation: (700 / 100) * 79.8 = 7 * 79.8 = 558.6g carbs.

Total carbohydrates = 37.32 + 2.97 + 558.6 = 598.89g.

Output: {"total_carbohydrates": 598.89}
```

**Analysis:** Three wrong matches:
- "dhal curry" (cooked) matched to "Lentils, **dry**" — 62.2 vs correct 20g (3x error)
- "white rice" (cooked) matched to "**Flour**, rice, white" — 79.8 vs correct 28g (2.8x error)
- LLM obediently applies the formula with wrong values, producing 598g instead of 210g.

---

### V1 (Embedding) | acc=False | MAE=370.6

**USER MESSAGE:**
```
=== Per-item USDA Reference (per 100g) ===
Formula: carbs = (weight_g / 100) * carbs_per_100g
Use USDA values where provided. For items marked 'no reliable match', use your own
knowledge.

- dhal curry: no reliable USDA match — use your own knowledge
- onions: USDA match → "Onions, yellow, raw" — Carbohydrate: 8.6g | Protein: 0.8g |
  Fat: 0.1g | Energy: 38.0kcal
- white rice: USDA match → "Rice, white, long grain, unenriched, raw" — Carbohydrate:
  80.3g | Protein: 7.0g | Fat: 1.0g | Energy: 358.7kcal
===

Query: "I've got a breakfast plate with 60 grams of dhal curry, 30 grams of cooked
onions, and a hefty serving of 700 grams of cooked white rice."
Answer: Let's think step by step.
```

**LLM RESPONSE:**
```
Let's think step by step.
The meal consists of 60 grams of dhal curry, 30 grams of cooked onions, and 700 grams
of cooked white rice.

1. Dhal curry (60g):
   "no reliable USDA match" so we use general nutritional knowledge.
   Cooked lentils typically contain about 20-23g of carbohydrates per 100g.
   Calculation: (60 / 100) * 21 = 12.6g.

2. Cooked onions (30g):
   USDA match: "Onions, yellow, raw" with 8.6g carbs per 100g.
   Calculation: (30 / 100) * 8.6 = 2.58g.

3. Cooked white rice (700g):
   USDA match: "Rice, white, long grain, unenriched, raw" with 80.3g carbs per 100g.
   Strictly following the instruction to "Use USDA values where provided", we must
   calculate based on the provided raw value of 80.3g per 100g.
   Calculation: (700 / 100) * 80.3 = 562.1g carbs.

Total Carbohydrates = 12.6 + 2.58 + 562.1 = 577.28g.

Output: {"total_carbohydrates": 577.28}
```

**Analysis:** The critical failure:
- "white rice" matched to "Rice, white, **raw**" — 80.3g/100g instead of cooked ~28g/100g.
- The LLM **knows** this is wrong (it even notes "raw rice expands when cooked") but
  follows the injected data because the prompt says "Use USDA values where provided."
- Dhal gets correct estimate (12.6g) because it fell through to "use your own knowledge."
- The per-item format creates a mixed strategy: correct for dhal, catastrophically wrong for rice.

---

### V2 (GAT) | acc=False | MAE=207.7

**USER MESSAGE:** (identical to V1)
```
=== Per-item USDA Reference (per 100g) ===
Formula: carbs = (weight_g / 100) * carbs_per_100g
Use USDA values where provided. For items marked 'no reliable match', use your own
knowledge.

- dhal curry: no reliable USDA match — use your own knowledge
- onions: USDA match → "Onions, yellow, raw" — Carbohydrate: 8.6g | Protein: 0.8g |
  Fat: 0.1g | Energy: 38.0kcal
- white rice: USDA match → "Rice, white, long grain, unenriched, raw" — Carbohydrate:
  80.3g | Protein: 7.0g | Fat: 1.0g | Energy: 358.7kcal
===

Query: "I've got a breakfast plate with 60 grams of dhal curry, 30 grams of cooked
onions, and a hefty serving of 700 grams of cooked white rice."
Answer: Let's think step by step.
```

**LLM RESPONSE:** (abbreviated — full response is very long with internal debate)
```
...
3. Cooked white rice (700g):
   The USDA match is "Rice, white, long grain, unenriched, raw" with 80.3g carbs/100g.
   Raw rice absorbs about 2.5 to 3 times its weight in water when cooked.
   700g cooked rice = 700/3 = 233.33g raw rice.
   Calculation: (233.33 / 100) * 80.3 = 187.37g carbs.
...
Total: 12.6 + 2.58 + 196 = ~414g.

Output: {"total_carbohydrates": 414.68}
```

**Analysis:** Same input as V1. The LLM attempts to correct for the raw-vs-cooked mismatch
by converting 700g cooked to ~233g raw equivalent, but the math is inconsistent (uses 187g
in one place, 196g in another, totals to 414g). The internal debate produces a less-wrong
but still inaccurate result. The LLM is fighting against its own instructions.

---

### Summary Table

| Mode | Rice reference injected | LLM rice calculation | Total | MAE |
|------|------------------------|---------------------|-------|-----|
| **Baseline** | None (uses own knowledge) | 700g * 28g/100g (cooked) = 196g | **210.7** | **4.0** |
| **V0 BM25** | "Flour, rice, white" 79.8g | 700g * 79.8g = 558.6g | **598.9** | **392** |
| **V1 Emb** | "Rice, raw" 80.3g | 700g * 80.3g = 562.1g | **577.3** | **371** |
| **V2 GAT** | "Rice, raw" 80.3g | Converts: 233g * 80.3g = 187g | **414.7** | **208** |

### Key Insight

The LLM **knows** cooked rice = ~28g carbs/100g (the baseline proves this). But when RAG
injects "Rice, raw — 80.3g/100g", the prompt instruction "Use USDA values where provided"
overrides the LLM's correct internal knowledge. The fundamental issue is:
**wrong retrieval + authoritative instructions = amplified errors.**
