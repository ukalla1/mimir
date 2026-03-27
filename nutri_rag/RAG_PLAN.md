# RAG Implementation Plan for nutri_rag

## Current Problem

The current retrieval uses regex extraction + BM25 keyword matching, which is **not real RAG**.
It fails on vocabulary mismatches ("groundnut" vs "peanut", "bun" vs "bread roll") and returns
wrong or no results for many food terms. This causes the RAG benchmark to perform **worse**
than the no-RAG baseline (16% vs 45% accuracy).

## Target Architecture

Both versions follow the standard RAG pattern:

```
User Question + Retrieved Knowledge --> Combined Prompt --> LLM --> Answer
```

The difference is **how we retrieve knowledge** — using semantic text embeddings instead
of keyword matching.

---

## Version 1: Standard RAG (Text Embedding Only)

Uses a text embedding model to semantically match food terms to USDA database entries.

### One-Time Setup (Offline)

```
All 74,175 USDA food descriptions
         |
         v
  Text Embedding Model (Qwen3-Embedding)
  encode each description --> vector
         |
         v
  Save to disk: food_text_embeddings.npy (74175, 1024)
```

### Benchmark Mode Pipeline

```
Step 1: Extract food terms from meal description
        "126g of raw maize flour and 27g of raw sugar"
        --> regex patterns --> ["maize flour", "sugar"]

Step 2: Embed each food term
        Qwen3-Embedding("maize flour") --> query vector

Step 3: Vector search (cosine similarity)
        query vector vs 74,175 pre-computed description vectors
        --> top-1 match: "Flour, corn, yellow, fine meal" (fdc_id=790276)

Step 4: DB nutrient lookup
        fdc_id=790276 --> {Carbohydrate: 80.8g, Protein: 6.2g, Fat: 1.7g}

Step 5: Build prompt
        === USDA Nutritional Reference Data (per 100g) ===
        [1] "Flour, corn, yellow, fine meal" -- Carbohydrate: 80.8g | ...
        ===
        Query: "126g of raw maize flour and 27g of raw sugar"
        Answer: Let's think step by step.

Step 6: LLM generation
        Qwen3.5-9B reads reference data + does arithmetic
        --> {"total_carbohydrates": 128.8}
```

### Assistant Mode Pipeline

```
Step 1: Parse eaten foods
        "I ate apple and milk for breakfast"
        --> heuristic parser --> ["apple", "milk"]

Step 2: Embed each food term
        Qwen3-Embedding("apple") --> query vector

Step 3: Vector search
        --> "Apples, honeycrisp, with skin, raw" (fdc_id=1105547)

Step 4: DB nutrient lookup
        --> {Carbohydrate: 14.7g, Protein: 0.3g, ...}

Step 5: LLM Call 1 -- Gap analysis
        "User ate apple (14.7g carb) and milk. What's missing for lunch?"
        --> {"protein_g": 35, "fat_g": 20, "carb_g": 50}

Step 6: DB query for gap-filling foods
        "Find foods high in protein" --> chicken, beans, fish, ...

Step 7: LLM Call 2 -- Recommendation
        Present gap-filling options to LLM
        --> "For lunch, try grilled chicken with brown rice..."
```

### Pros
- Simple, clean pipeline
- Text embedding handles semantic matching (no synonyms needed)
- Easy to implement and debug

### Cons
- Gap-filling options (step 6) are limited to direct DB query results
- No nutritional similarity expansion
- No user preference learning

---

## Version 2: RAG + nutri_graph (Text Embedding + GAT Embedding)

Adds the pre-trained GAT embeddings from nutri_graph on top of the standard RAG pipeline.
The GAT embeddings encode **nutritional similarity** learned from the food-nutrient graph
(74,175 foods x 477 nutrients). They complement the text embeddings which encode
**description similarity**.

### Two Types of Embeddings

| | Text Embeddings (new) | GAT Embeddings (existing) |
|---|---|---|
| Model | Qwen3-Embedding | GATv2 (nutri_graph) |
| Input | Food description strings | Food-nutrient bipartite graph |
| Output shape | (74175, 1024) | (74175, 64) |
| Pre-computed? | Need to create | Already exists (food_embeddings.npy) |
| Purpose | text --> fdc_id matching | fdc_id --> similar foods |
| Runs at query time? | Yes (encode query text) | No (just numpy array lookup) |

### One-Time Setup (Offline)

```
Same as Version 1:
  USDA descriptions --> Qwen3-Embedding --> food_text_embeddings.npy

Already done in nutri_graph:
  Food-nutrient graph --> GATv2 training --> food_embeddings.npy
```

### Benchmark Mode Pipeline

```
Step 1: Extract food terms (same as V1)
        --> ["maize flour", "sugar"]

Step 2: Embed each food term (same as V1)
        Qwen3-Embedding("maize flour") --> query vector

Step 3: Vector search --> top-k candidates (not just top-1)
        --> candidates:
            1. "Flour, corn, yellow, fine meal"    (text_sim=0.93)
            2. "Corn flour, masa, enriched"        (text_sim=0.91)
            3. "Flour, rice, white"                (text_sim=0.78)

Step 4: GAT re-ranking  <-- NEW (not in V1)
        For each candidate, look up its GAT embedding (numpy array index)
        Compare GAT vectors between candidates
        "Flour, corn" and "Corn flour, masa" are GAT-close (both corn-based)
        "Flour, rice, white" is GAT-distant (different nutritional profile)
        --> pick: "Flour, corn, yellow, fine meal" (confirmed)

Step 5: DB nutrient lookup (same as V1)

Step 6: Build prompt (same as V1)

Step 7: LLM generation (same as V1)
```

### Assistant Mode Pipeline

```
Step 1: Parse eaten foods (same as V1)

Step 2: Embed food terms (same as V1)

Step 3: Vector search (same as V1)

Step 4: DB nutrient lookup (same as V1)

Step 5: LLM Call 1 -- Gap analysis (same as V1)
        --> {protein_g: 35, fat_g: 20, carb_g: 50}

Step 6: DB query for gap-filling foods (same as V1)
        --> seed candidates: chicken, beans, fish

Step 7: GAT neighbor expansion  <-- NEW (not in V1)
        For each seed, find cosine-similar foods in GAT embedding space:
          chicken --> turkey, pork tenderloin, salmon
          beans   --> lentils, chickpeas, black beans
        These are nutritionally similar (learned from graph), not just
        textually similar

Step 8: User preference re-ranking  <-- NEW (not in V1)
        Score each option by user's choice history
        (chosen_count / offered_count ratio)
        Boost favorites, demote skipped foods

Step 9: LLM Call 2 -- Recommendation
        Present seeds + GAT neighbors + preferences to LLM
        Rich, diverse, personalized options
        --> "For lunch, try grilled turkey breast with lentil soup..."
```

### Pros
- Everything from V1, plus:
- Benchmark: GAT re-ranking validates text matches using nutritional coherence
- Assistant: GAT expansion provides diverse, nutritionally similar alternatives
- Assistant: User preference tracking enables personalization over time
- Research contribution: novel use of graph-learned food embeddings for RAG

### Cons
- More complex pipeline
- Depends on nutri_graph being trained and embeddings available

---

## Shared Components (Both Versions)

### Text Embedding (needs to be built)
- Model: Qwen3-Embedding (recommended by co-author, same family as Qwen3.5-9B)
- Pre-compute: encode all 74,175 USDA descriptions, save as .npy
- Query time: encode food term, cosine search against pre-computed vectors
- Replaces: current BM25 FTS search in search.py

### What stays the same
- DuckDB knowledge base (nutri_kb.duckdb) -- nutrient lookup
- Prompt format (USDA reference block + CoT)
- lm-evaluation-harness integration (task YAML, utils.py)
- LLM (Qwen3.5-9B via llama-server)
- Benchmark food term extraction (regex patterns in retriever.py)
- Assistant heuristic parser (parse.py)

### What changes
- search.py: replace BM25 FTS with vector similarity search
- New: embedding index module (pre-compute + query text embeddings)
- New: script to pre-compute USDA description embeddings

---

## Version 3: Multi-Candidate RAG (Text + GAT + Threshold Filtering) ✅ IMPLEMENTED

Shows multiple USDA candidates per food item, filtered by cosine similarity threshold,
letting the LLM choose the best match.

### Key Improvements Over V2

- **Multi-candidate display**: Top-5 candidates per food item instead of single best match
- **Similarity threshold (0.60)**: Filter out unreliable candidates before showing to LLM
- **Concise prompt**: Minimal instructions to prevent LLM overthinking
- **Partial coverage note**: "The reference may only cover some of the items — estimate the rest yourself"
- **8192 max tokens**: Prevents response truncation on long chain-of-thought

### Benchmark Mode Pipeline

```
Step 1: Extract food terms (same as V1/V2)
        --> ["maize flour", "sugar"]

Step 2: Embed each food term (same as V1/V2)

Step 3: Vector search --> top-5 candidates per term

Step 4: GAT re-ranking (same as V2)

Step 5: Similarity threshold filter  <-- NEW
        Remove candidates with cosine similarity < 0.60
        Items with no surviving candidates are omitted entirely

Step 6: Format multi-candidate reference block:
        === USDA Reference (per 100g) ===
        - maize flour:
          1. "Corn flour, yellow" — Carbohydrate: 76.7g | ...
          2. "Cornmeal, whole-grain" — Carbohydrate: 73.1g | ...
        - sugar:
          1. "Sugars, granulated" — Carbohydrate: 99.6g | ...
        ===

Step 7: LLM generation (max 8192 tokens)
```

### Results (Protein, 1000 samples)

| Version | Acc@7.5g | MAE |
|---------|----------|-----|
| Baseline (no RAG) | 0.735 | 7.00 |
| V3 (multi-candidate) | **0.763** | **6.04** |

### Lessons Learned

- **Bad references hurt more than no references**: Without threshold filtering, V3 performed
  worse than baseline (0.587 vs 0.735 Acc). A single bad match (tap water → beans) can
  cause the LLM to produce wildly incorrect estimates.
- **Prompt wording matters enormously**: "Many may be wrong" primed distrust; "same form"
  made the LLM reject reasonable matches (raw tilapia for cooked tilapia); "If in doubt,
  ignore it" caused hundreds of tokens of deliberation. The final concise prompt works
  because it gives minimal instructions.
- **Token truncation is silent failure**: When the LLM hits the token limit before producing
  `Output: {...}`, the response is parsed as -1. Bumping to 8192 tokens fixed this.
- **KB quality is foundational**: 60.7% of FDC entries were lab analysis junk with no
  nutrient data. Cleaning these + adding SR Legacy (7,793 curated foods with cooked forms)
  improved retrieval quality significantly.

---

## Implementation Status

| Version | Status | Key Files |
|---------|--------|-----------|
| V0 (BM25) | ✅ Done | `search_bm25.py`, `retriever_bm25.py`, `tasks/nutribench_v2_rag_bm25/` |
| V1 (Dense) | ✅ Done | `search.py`, `embedding.py`, `tasks/nutribench_v2_rag/` |
| V2 (Dense+GAT) | ✅ Done | `search.py` (use_gat=True), `tasks/nutribench_v2_rag_gat/` |
| V3 (Multi-candidate) | ✅ Done | `bench/prompt.py`, `tasks/nutribench_v2_rag_gat_multi/` |

---

## Files Overview

| File | Purpose |
|---|---|
| `nutri_rag/embedding.py` | Text embedding model wrapper + vector index + GAT index |
| `nutri_rag/search.py` | Semantic vector search + optional GAT re-ranking |
| `nutri_rag/search_bm25.py` | V0 BM25 full-text search baseline |
| `nutri_rag/config.py` | Paths, thresholds (SIMILARITY_THRESHOLD=0.60), model config |
| `nutri_rag/bench/retriever.py` | Regex extraction → embedding search → nutrients |
| `nutri_rag/bench/prompt.py` | USDA reference block formatting (legacy, per-item, multi-candidate) |
| `nutri_rag/bench/nutrient_prompts.py` | System prompts + few-shot CoT examples per nutrient |
| `scripts/build_embeddings.py` | One-time USDA description embedding computation |
| `scripts/run_bench.py` | Benchmark runner (V0-V3 via lm-eval) |
| `scripts/run_all_bench.py` | Run all mode/nutrient combinations |
