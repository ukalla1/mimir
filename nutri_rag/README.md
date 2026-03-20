# nutri_rag — Embedding-Based RAG for Nutrition Estimation & Personalized Recommendations

RAG (Retrieval-Augmented Generation) system that combines the [nutri_graph](../nutri_graph/) knowledge base (DuckDB + GAT embeddings) with Qwen3-Embedding and Qwen3.5-9B to improve nutrition estimation and provide personalized meal recommendations.

## Three Retrieval Versions

The benchmark supports three retrieval modes, selectable via `--mode`:

| Version | Retrieval Method | Search Module |
|---------|-----------------|---------------|
| **V0** | BM25 keyword matching (DuckDB FTS) | `search_bm25.py` |
| **V1** | Text embedding (Qwen3-Embedding-0.6B) | `search.py` |
| **V2** | Text embedding + GAT re-ranking | `search.py` (with `use_gat=True`) |

V0 is the original manual approach using keyword matching with cross-language synonyms. V1 replaces it with semantic vector search — the embedding model handles vocabulary mismatches like "groundnut" vs "peanut" without hardcoded synonyms. V2 adds GAT nutritional-similarity re-ranking on top of V1, but only when the text embedding is ambiguous (confident matches are left unchanged).

## Two Independent Modes

### Mode 1: NutriBench Benchmark

Improves carbohydrate estimation accuracy by supplying exact USDA per-100g nutrient values to the LLM, instead of letting it guess from memory.

**V1 Pipeline (Text Embedding):**

```
meal_description (e.g. "126g of maize flour and 27g of raw sugar")
    |
    v
[Regex Food Term Extraction] -- 3 patterns: "Xg of <food>", "<food> weighing Xg", "<food> (Xg)"
    |                            strip cooking/prep words (raw, boiled, etc.)
    v
[Qwen3-Embedding Vector Search] -- encode food term -> cosine similarity vs 74K USDA vectors
    |                                no synonyms needed (semantic matching)
    |                                re-rank: prefer entries with macronutrient data
    v
[Nutrient Lookup] -- per-100g values from DuckDB knowledge base
    |                 filter: only include references with carbohydrate data
    v
[Prompt Builder] -- format USDA reference block + CoT prompt
    |                "Use these if they match. For unlisted foods, use your own knowledge."
    v
[Qwen3.5-9B via lm-evaluation-harness] -- single LLM call, greedy decoding
    |
    v
predicted carbohydrates (ACC: |pred - gt| < 7.5g, MAE: |pred - gt|)
```

**V2 adds one step** between vector search and nutrient lookup:

```
[Qwen3-Embedding Vector Search] -- top-k candidates
    |
    v
[GAT Ambiguity Check] -- if text score gap between top unique descriptions < 0.03:
    |                       apply GAT coherence re-ranking (0.7*text + 0.3*gat)
    |                     else: skip GAT, text embedding is confident
    v
[Nutrient Lookup] -- ...
```

**Baseline (no RAG):** 45.08% accuracy, 25.73g MAE (from [qwen_test](../../qwen_test/))

### Mode 2: General Nutrition Assistant

Personalized meal recommendations using LLM-driven gap analysis, GAT embedding neighbor expansion, and user preference tracking.

```
"I ate apple and milk for breakfast, what should I eat for lunch?"
    |
    v
[Heuristic Parser] -- extract eaten foods
    |
    v
[Embedding Vector Search] -- text -> fdc_id -> nutrient profiles
    |
    v
[LLM Call 1: Gap Analysis] -- "what's missing?" -> structured JSON targets
    |
    v
[DuckDB Nutrient Query] -- find foods matching the gap
    |
    v
[GAT Embedding Neighbors] -- expand candidates with nutritionally similar options
    |
    v
[User Preference Re-ranking] -- boost foods the user has chosen before
    |
    v
[LLM Call 2: Recommendation] -- generate personalized meal suggestion
    |
    v
"For lunch, try grilled chicken breast with quinoa and a side salad..."
```

**Key research contribution:** GAT embeddings from nutri_graph provide nutritionally coherent option expansion — not random foods, but structurally related alternatives learned from the food-nutrient graph.

## Prerequisites

- **nutri_graph** knowledge base built (`../nutri_graph/data/nutri_kb.duckdb` and `../nutri_graph/outputs/embeddings/food_embeddings.npy`). See [nutri_graph README](../nutri_graph/README.md) for setup.
- **Pre-computed text embeddings** (`data/embeddings/food_text_embeddings.npy`). Generate with `python scripts/build_embeddings.py` (required for V1/V2, not for V0).
- **Qwen3.5-9B** — follow the [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5#unsloth-studio-guide) to set up llama.cpp and download the GGUF model. See also [qwen_test](../../qwen_test/) for reference.
- **Python 3.9+** with dependencies: `duckdb`, `numpy`, `requests`, `torch`, `transformers`, `scikit-learn`

## Setup

```bash
cd nutri_rag
pip install -e .

# Pre-compute text embeddings (one-time, ~46 seconds on GPU)
python scripts/build_embeddings.py
```

## Usage

### Start the LLM Server

nutri_rag includes its own server script configured for single-user assistant use:

```bash
bash scripts/start_server.sh
```

This launches `llama-server` with `--parallel 1` (single-user, sequential requests). For NutriBench benchmarking, use `qwen_test/start_server.sh` instead (which uses `--parallel 6` for batch throughput).

| | `scripts/start_server.sh` | `qwen_test/start_server.sh` |
|---|---|---|
| **Purpose** | Assistant demo | NutriBench benchmark |
| **Parallel** | 1 (single-user) | 6 (batch throughput) |
| **Use with** | `demo_assistant.py` | `run_bench.py` |

### Mode 1: NutriBench

**Run the NutriBench benchmark:**

```bash
# V0: BM25 keyword matching (original baseline)
python scripts/run_bench.py --mode v0 --limit 100

# V1: Text embedding search (default)
python scripts/run_bench.py --mode v1 --limit 100

# V2: Text embedding + GAT re-ranking
python scripts/run_bench.py --mode v2 --limit 100

# Full benchmark, all 15,617 samples
python scripts/run_bench.py --mode v1
```

Results are saved as `results/results_rag_{v0,v1,v2}_<timestamp>.json` with per-sample logs in JSONL format.

**Run baseline (no RAG) for comparison:**

```bash
bash ../../qwen_test/run_bench.sh nutribench_v2_cot 100
```

**Analyze retrieval quality:**

```bash
# Similarity distribution + ACC/MAE correlation plots
python scripts/plot_similarity_analysis.py --limit 100 \
  --samples-v1 results/samples_nutribench_v2_rag_<timestamp>.jsonl
```

**Compare against baseline:**

```bash
python scripts/compare_results.py
```

**Test retrieval interactively:**

```bash
python scripts/demo_bench.py
```

Type a meal description to see what USDA foods are retrieved and the augmented prompt:

```
Meal: 126 grams of maize flour and 27 grams of raw sugar

=== USDA Nutritional Reference Data (per 100g) ===
[1] "Corn flour, masa harina, white, dry, raw" (USDA #2711103)
    Carbohydrate: 76.7g | Protein: 7.6g | Fat: 4.3g | Energy: 376.1kcal
[2] "Sugars, granulated" (USDA #334247)
    Carbohydrate: 99.6g | Protein: 0.0g | Fat: 0.3g | Energy: 385.0kcal
```

### Mode 2: General Assistant

```bash
python scripts/demo_assistant.py
```

Interactive CLI that analyzes what you've eaten and recommends your next meal. The input parser automatically separates the food description from the question:

```
What did you eat? I ate an apple and a cup of milk for breakfast, what should I eat for lunch?

Analyzing your breakfast...
Recommendation for lunch:
For lunch, try grilled chicken breast (about 150g) with a cup of
brown rice and some steamed vegetables...
```

The assistant learns from your choices over time via the user preference database.

## Project Structure

```
nutri_rag/
  pyproject.toml
  RAG_PLAN.md                  # Implementation plan for V1/V2 RAG
  nutri_rag/
    config.py                  # Paths, constants, LLM settings, embedding model config
    parse.py                   # Heuristic meal parser (regex-based)
    embedding.py               # TextEmbedder (Qwen3-Embedding) + FoodVectorIndex + GATIndex
    search.py                  # V1/V2: Embedding vector search + optional GAT re-ranking
    search_bm25.py             # V0: DuckDB BM25 full-text search (baseline)
    llm.py                     # OpenAI-compatible chat client

    bench/                     # --- Mode 1: NutriBench ---
      retriever.py             # V1/V2: Regex extraction -> embedding search -> nutrients
      retriever_bm25.py        # V0: Regex extraction -> BM25 search -> nutrients
      prompt.py                # Format USDA reference block for CoT

    assistant/                 # --- Mode 2: General Assistant ---
      gap_analyzer.py          # LLM Call 1: meal -> structured JSON targets
      food_recommender.py      # DB nutrient query + GAT neighbor expansion
      preference_db.py         # User choice history (DuckDB)
      prompt.py                # Format recommendations for LLM Call 2
      pipeline.py              # End-to-end orchestration

  tasks/
    nutribench_v2_rag_bm25/    # V0 task definition (BM25)
    nutribench_v2_rag/         # V1 task definition (text embedding)
    nutribench_v2_rag_gat/     # V2 task definition (text + GAT)

  data/
    embeddings/
      food_text_embeddings.npy # Pre-computed Qwen3-Embedding vectors (74175 x 1024)
      food_fdc_ids.npy         # fdc_id array matching embedding rows

  scripts/
    build_embeddings.py        # One-time: encode all USDA descriptions
    start_server.sh            # llama-server for assistant mode (parallel=1)
    demo_bench.py              # Interactive NutriBench retrieval demo
    demo_assistant.py          # Interactive assistant CLI
    run_bench.py               # NutriBench benchmark runner (--mode v0/v1/v2)
    compare_results.py         # Baseline vs RAG comparison
    plot_similarity_analysis.py # Retrieval quality visualization
```

## Module Details

### Shared Core

| Module | Purpose |
|--------|---------|
| `config.py` | Paths to DuckDB, embeddings, LLM endpoint (`localhost:8080`), embedding model name (`Qwen/Qwen3-Embedding-0.6B`), key nutrient names |
| `parse.py` | Splits meal text on commas/and/with, extracts quantities+units via regex, strips filler words. Used by assistant mode. |
| `embedding.py` | `TextEmbedder`: Qwen3-Embedding wrapper with last-token pooling and task instructions. `FoodVectorIndex`: pre-computed cosine search over 74K USDA descriptions. `GATIndex`: nutri_graph GAT embeddings for nutritional similarity. |
| `search.py` | Semantic vector search via `FoodVectorIndex`. Encodes query with task instruction, retrieves candidates, filters by macro data availability. With `use_gat=True`, applies GAT coherence re-ranking when text match is ambiguous. |
| `search_bm25.py` | V0 baseline: DuckDB BM25 full-text search with confidence threshold (`MIN_BM25_SCORE=1.0`), 16 cross-language synonyms, macro-count re-ranking. |
| `llm.py` | Thin `requests`-based client for OpenAI-compatible `/v1/chat/completions`, with `<think>` tag stripping and robust JSON extraction (handles truncated output) |

### Mode 1: NutriBench (`bench/`)

| Module | Purpose |
|--------|---------|
| `retriever.py` | Extracts food terms from meal descriptions via 3 regex patterns (handles "Xg of food", "food weighing Xg", "food (Xg)"), strips cooking/prep words, then does embedding search + gets per-100g nutrients. Supports `use_gat` flag for V2. |
| `retriever_bm25.py` | V0 variant: same regex extraction, but uses BM25 search instead of embeddings. Reuses `FoodContext` and `_extract_food_terms` from `retriever.py`. |
| `prompt.py` | Formats nutrients into a `=== USDA Nutritional Reference Data ===` block. Filters out entries without carb data. Instructs model to use own knowledge for unlisted foods. |

### Mode 2: General Assistant (`assistant/`)

| Module | Purpose |
|--------|---------|
| `gap_analyzer.py` | LLM Call 1: sends eaten foods' nutrient profiles to LLM, gets back structured JSON with reasoning and macro targets (`{"protein_g": 35, "fat_g": 20, ...}`) |
| `food_recommender.py` | Queries DuckDB for gap-filling foods, then expands each candidate via GAT embedding cosine similarity to find nutritionally similar alternatives |
| `preference_db.py` | DuckDB table tracking offered/chosen foods per user, scores by `chosen_count / offered_count` ratio |
| `prompt.py` | Formats gap analysis + ranked options + preference history into LLM Call 2 prompt |
| `pipeline.py` | `NutriAssistant` class orchestrating the full 8-step pipeline |

## How It Works

### Benchmark Food Term Extraction

The benchmark retriever uses regex patterns (not the heuristic parser) to extract food terms from NutriBench meal descriptions:

```
Input:  "126 grams of raw maize flour and 27 grams of raw sugar"
Terms:  ["maize flour", "sugar"]          (Pattern A: "Xg of <food>", strip "raw")

Input:  "a plain bun weighing 126 grams"
Terms:  ["bun"]                            (Pattern B: "<food> weighing Xg", strip "plain")

Input:  "a boiled large onion (1g)"
Terms:  ["onion"]                          (Pattern C: "<food> (Xg)", strip "boiled large")
```

### V0: BM25 Keyword Search

The original approach uses DuckDB's built-in FTS extension with English stemming:

```
"maize flour"  → synonym → "corn flour" → BM25 score 3.2 ✓ → "Flour, corn, yellow"
"groundnuts"   → synonym → "peanuts"    → BM25 score 2.1 ✓ → "Peanuts, raw"
"bun"          → BM25 score < 1.0 ✗     → no match → model uses own knowledge
```

Limitations: requires hardcoded synonyms, fails on vocabulary mismatches not in the synonym list.

### V1: Text Embedding Search

Replaces BM25 with Qwen3-Embedding-0.6B semantic vector search:

```
"maize flour"  → encode → cosine search → "Corn flour, masa harina" (sim=0.634) ✓
"groundnuts"   → encode → cosine search → "Peanuts, raw"            (sim=0.551) ✓
"bun"          → encode → cosine search → "Rolls, hamburger"        (sim=0.547) ✓
```

No synonyms needed — the embedding model understands semantic equivalence. Pre-computed embeddings (74,175 x 1024) enable fast numpy dot-product search.

### V2: GAT Ambiguity-Gated Re-ranking

When text embedding returns ambiguous candidates (small score gap between top unique descriptions), GAT nutritional-similarity re-ranking breaks the tie:

```
"kasepa fish" → text candidates:
  1. "Fish, catfish"  (text_sim=0.556)
  2. "Fish, salmon"   (text_sim=0.552)   gap=0.004 < 0.03 → AMBIGUOUS → use GAT

GAT re-ranking: catfish and salmon have different nutritional profiles.
GAT picks the one more coherent with the overall candidate group.

"sugar" → text candidates:
  1. "Sugars, granulated"               (text_sim=0.611)
  2. "Safeway Fine Granulated Sugar"    (text_sim=0.547)   gap=0.064 > 0.03 → CONFIDENT → skip GAT
```

### Assistant Parser Example

The assistant mode uses the heuristic parser (different from benchmark):

```
Input:  "I ate an apple and milk for breakfast"
Output: [ParsedItem(food_term="apple"), ParsedItem(food_term="milk")]
```

### GAT Embedding Neighbor Expansion

After the DB finds seed candidates that fill the nutritional gap, GAT embeddings provide alternative options that are nutritionally similar:

```
Seed: "Chicken, breast, roasted" (31g protein/100g)
  GAT neighbors:
    - "Turkey, breast, roasted"   (similar lean protein)
    - "Pork, tenderloin, roasted" (similar lean meat)
    - "Salmon, Atlantic, cooked"  (similar protein source)
```

This leverages the learned food-nutrient graph structure from nutri_graph's GATv2 model (74,175 foods x 64-dim embeddings).

### User Preference Re-ranking

The preference DB tracks what the user picks from offered options:

```
Chicken breast:  chosen 4/4 times -> score 1.0 (favorite)
Salmon:          chosen 2/3 times -> score 0.67
Tofu:            chosen 0/2 times -> score 0.0 (skipped)
Pork tenderloin: never offered    -> score 0.5 (neutral)
```

New users start with no history (cold start) — options are ordered by GAT similarity score until preferences are learned.

## Dependencies on nutri_graph

| Component | Source | Used in |
|-----------|--------|---------|
| `nutri_kb.duckdb` | nutri_graph | All modes — food descriptions + nutrient lookup |
| `food_embeddings.npy` | nutri_graph (trained GAT) | V2 benchmark (re-ranking) + Mode 2 assistant (neighbor expansion) |
| `nodes_food` table | nutri_graph | All modes — food descriptions |
| `edges_food_contains_nutrient` table | nutri_graph | All modes — per-100g nutrient values |
| `food_text_embeddings.npy` | `build_embeddings.py` (Qwen3-Embedding) | V1/V2 benchmark + Mode 2 assistant — semantic food search |
