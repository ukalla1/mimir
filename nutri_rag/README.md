# nutri_rag — RAG Pipeline for Nutrition Estimation & Meal Recommendations

RAG system that combines the [nutri_graph](../nutri_graph/) knowledge base (DuckDB + GAT embeddings) with semantic text embeddings (Qwen3-Embedding) and an LLM (Qwen3.5-9B) for three tasks:
1. **NutriBench Benchmark** — evaluate nutritional content estimation accuracy
2. **PFoodReq Benchmark** — personalized recipe recommendation (no LLM needed)
3. **General Assistant** — personalized meal recommendations

## Retrieval Versions

| Version | Mode | Retrieval Method |
|---------|------|-----------------|
| **V0** | `--mode v0` | BM25 keyword matching (DuckDB FTS) |
| **V1** | `--mode v1` | Dense text embedding (Qwen3-Embedding-0.6B) |
| **V2** | `--mode v2` | Text embedding + GAT re-ranking |
| **V3** | `--mode v3` | Multi-candidate text + GAT (top-5 per food item) |

**V0** uses keyword matching with cross-language synonyms. **V1** replaces it with semantic vector search — handles vocabulary mismatches like "groundnut" → "peanut" without hardcoded synonyms. **V2** adds GAT re-ranking when text similarity is ambiguous (score gap < 0.03). **V3** shows multiple candidates per food item, filtered by a cosine similarity threshold (0.60), letting the LLM choose the best match.

## Key Design Decisions

These were discovered through iterative benchmarking on NutriBench:

- **Similarity threshold (0.60)**: Candidates below this cosine similarity are filtered out. Without this, bad matches (e.g., "tap water" → "beans") actively hurt performance.
- **Concise prompt**: The USDA reference block uses minimal instructions ("Ignore wrong matches and use your own knowledge"). Verbose prompts caused the LLM to overthink and waste tokens debating reference quality.
- **Partial coverage note**: "The reference may only cover some of the items — estimate the rest yourself" prevents the LLM from returning -1 when references don't cover all food items.
- **8192 max tokens**: Long chain-of-thought responses were hitting the 4096 limit before producing output. Bumped to 8192 to prevent truncation.

## NutriBench Results (Protein, 1000 samples)

| Version | Acc@7.5g | MAE |
|---------|----------|-----|
| Baseline (no RAG) | 0.735 | 7.00 |
| V3 (multi-candidate) | **0.763** | **6.04** |

## Pipeline

### NutriBench (V3)

```
Meal: "126g of maize flour and 27g of raw sugar"
  │
  ├─ Regex extraction → ["maize flour", "sugar"]
  │   3 patterns: "Xg of <food>", "<food> weighing Xg", "<food> (Xg)"
  │   Strip cooking words (raw, boiled, fried, peeled, etc.)
  │
  ├─ Qwen3-Embedding cosine search (top-5 per term)
  │
  ├─ GAT re-ranking (when text scores are ambiguous)
  │
  ├─ Similarity threshold filter (≥ 0.60)
  │   Discard unreliable candidates before showing to LLM
  │
  ├─ Format USDA reference block:
  │   === USDA Reference (per 100g) ===
  │   - maize flour:
  │     1. "Corn flour, yellow" — Carbohydrate: 76.7g | ...
  │     2. "Cornmeal, whole-grain" — Carbohydrate: 73.1g | ...
  │   - sugar:
  │     1. "Sugars, granulated" — Carbohydrate: 99.6g | ...
  │
  ├─ CoT prompt: reference block + "Let's think step by step"
  │
  └─ Qwen3.5-9B (greedy, max 8192 tokens) → numerical prediction
```

### General Assistant

```
"I ate an apple and milk for breakfast. What should I eat for lunch?"
  │
  ├─ Parse eaten foods → ["apple", "milk"]
  ├─ Embedding search → USDA matches + nutrient profiles
  │
  ├─ LLM Call 1: Gap Analysis → {"protein_g": 35, "fat_g": 20, ...}
  │
  ├─ DuckDB: find foods high in gap nutrient → 5 seed candidates
  ├─ GAT expansion: 5 neighbors per seed → ~30 total options
  ├─ Preference re-ranking: boost previously chosen foods
  │
  └─ LLM Call 2: Natural-language recommendation
```

## Prerequisites

- **nutri_graph** KB built (`../nutri_graph/data/nutri_kb.duckdb` and `../nutri_graph/outputs/embeddings/food_embeddings.npy`)
- **Pre-computed text embeddings** — generate with `python scripts/build_embeddings.py`
- **Qwen3.5-9B** served via llama-server (OpenAI-compatible API)
- Python 3.9+ with: `duckdb`, `numpy`, `requests`, `torch`, `transformers`, `scikit-learn`

## Setup

The full build order across both subsystems:

```bash
# 1. Build USDA knowledge base (nutri_graph)
cd ~/work/atlas/mimir/nutri_graph
python scripts/build_kb.py

# 2. Build text embeddings for USDA foods (nutri_rag)
#    Must run AFTER build_kb.py (reads nodes_food table)
#    Must run BEFORE build_recipe_kb.py (used for ingredient matching)
cd ~/work/atlas/mimir/nutri_rag
python scripts/build_embeddings.py

# 3. Integrate FoodKG recipes into KB (nutri_graph)
#    Adds recipe tables to nutri_kb.duckdb, does not modify USDA data
cd ~/work/atlas/mimir/nutri_graph
python scripts/build_recipe_kb.py

# 4. Train GAT on full graph (nutri_graph)
python scripts/train_GAT.py
```

`build_embeddings.py` only needs re-running when the USDA food list changes (i.e., after `build_kb.py`). It does **not** need re-running after `build_recipe_kb.py` or `train_GAT.py`.

## Usage

### Start the LLM Server

Both scripts start the same Qwen3.5-9B model on port 8080 with identical config, differing only in concurrency:

```bash
# For assistant / robot usage (parallel=1, single user)
bash scripts/start_server.sh

# For benchmark evaluation (parallel=3, concurrent requests)
bash ../../qwen_test/start_server.sh
```

Note: PFoodReq benchmark (`run_pfoodreq_bench.py`) does not need the LLM server.

### NutriBench Benchmark

```bash
# Run specific mode + nutrient + sample limit
python scripts/run_bench.py --mode v3 --nutrient protein --limit 200 --concurrent 3

# Run all combinations
python scripts/run_all_bench.py --modes v1 v3 --nutrients carb protein --limit 100

# Baseline (no RAG) for comparison
python scripts/run_baseline.py
```

Results saved to `results/results_{mode}_{nutrient}_{timestamp}.json` with per-sample logs in JSONL.

```bash
# Compare results
python scripts/compare_results.py

# Retrieval quality analysis
python scripts/plot_similarity_analysis.py --limit 100

# Interactive retrieval demo
python scripts/demo_bench.py
```

### PFoodReq Benchmark

```bash
# Full test set (2244 queries, no LLM server needed)
python scripts/run_pfoodreq_bench.py

# Quick test
python scripts/run_pfoodreq_bench.py --limit 10

# With LLM verification (optional)
python scripts/run_pfoodreq_bench.py --ablation with_llm
```

### General Assistant

```bash
python scripts/demo_assistant.py
```

## Project Structure

```
nutri_rag/
  nutri_rag/
    config.py                  # Paths, thresholds, model config
    embedding.py               # TextEmbedder (Qwen3-Embedding) + FoodVectorIndex + GATIndex
    search.py                  # V1/V2: embedding search + optional GAT re-ranking
    search_bm25.py             # V0: DuckDB BM25 full-text search
    parse.py                   # Heuristic meal parser
    llm.py                     # OpenAI-compatible chat client

    bench/                     # NutriBench benchmark
      retriever.py             # V1/V2/V3: regex extraction → embedding search → nutrients
      retriever_bm25.py        # V0: regex extraction → BM25 search → nutrients
      prompt.py                # USDA reference block formatting (legacy, per-item, multi-candidate)
      nutrient_prompts.py      # System prompts + few-shot CoT examples per nutrient
      task_utils.py            # Shared lm-eval task utilities

    assistant/                 # General assistant
      gap_analyzer.py          # LLM-driven nutritional gap analysis
      food_recommender.py      # DB query + GAT neighbor expansion
      preference_db.py         # User preference tracking (DuckDB)
      prompt.py                # Recommendation prompt formatting
      pipeline.py              # End-to-end orchestration

  tasks/
    nutribench_v2_rag_bm25/    # V0 task definition
    nutribench_v2_rag/         # V1 task definition
    nutribench_v2_rag_gat/     # V2 task definition
    nutribench_v2_rag_gat_multi/ # V3 task definition

  scripts/
    build_embeddings.py        # Encode all USDA descriptions (one-time)
    run_bench.py               # Single benchmark run (--mode, --nutrient, --limit)
    run_all_bench.py           # Run all mode/nutrient combinations
    run_baseline.py            # No-RAG baseline benchmark
    compare_results.py         # Result comparison
    plot_similarity_analysis.py # Retrieval quality plots
    demo_bench.py              # Interactive retrieval demo
    demo_assistant.py          # Interactive assistant CLI
    start_server.sh            # llama-server launcher

  data/embeddings/             # Pre-computed text embeddings
  results/                     # Benchmark outputs
```

## Key Constants (`config.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `SIMILARITY_THRESHOLD` | 0.60 | Cosine sim below this → filter out candidate |
| `TOP_K_FOODS` | 3 | Max DB matches per parsed food item |
| `GAT_NEIGHBORS_K` | 5 | GAT embedding neighbors per seed |
| `TEXT_EMBEDDING_DIM` | 1024 | Qwen3-Embedding output dimension |
| `LLM_MODEL` | qwen3.5-9b | Local LLM for generation |
