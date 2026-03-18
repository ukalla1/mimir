# nutri_rag — GAT-Embedding RAG for Nutrition Estimation & Personalized Recommendations

RAG (Retrieval-Augmented Generation) system that combines the [nutri_graph](../nutri_graph/) knowledge base (DuckDB + GAT embeddings) with Qwen3.5-9B to improve nutrition estimation and provide personalized meal recommendations.

## Two Independent Modes

### Mode 1: NutriBench Benchmark

Improves carbohydrate estimation accuracy by supplying exact USDA per-100g nutrient values to the LLM, instead of letting it guess from memory.

```
meal_description (e.g. "126g of maize flour and 27g of raw sugar")
    |
    v
[Heuristic Parser] -- split into food items + quantities
    |
    v
[DuckDB Text Search] -- text -> fdc_id -> per-100g nutrient profiles
    |
    v
[Prompt Builder] -- format USDA reference block + CoT prompt
    |
    v
[Qwen3.5-9B] -- single LLM call
    |
    v
predicted carbohydrates
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
[DuckDB Text Search] -- text -> fdc_id -> nutrient profiles
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
- **Qwen3.5-9B** — follow the [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5#unsloth-studio-guide) to set up llama.cpp and download the GGUF model. See also [qwen_test](../../qwen_test/) for reference.
- **Python 3.9+** with dependencies: `duckdb`, `numpy`, `requests`, `torch`, `scikit-learn`

## Setup

```bash
cd nutri_rag
pip install -e .
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

**Test retrieval interactively:**

```bash
python scripts/demo_bench.py
```

Type a meal description to see what USDA foods are retrieved and the augmented prompt:

```
Meal: 126 grams of maize flour and 27 grams of raw sugar

=== USDA Nutritional Reference Data (per 100g) ===
[1] "Corn flour, masa harina" (USDA #2710835)
    Carbohydrate: 76.7g | Protein: 7.6g | Fat: 4.3g | Energy: 376.1kcal
[2] "Sugars, granulated" (USDA #334247)
    Carbohydrate: 99.6g | Protein: 0.0g | Fat: 0.3g | Energy: 385.0kcal
```

**Run the NutriBench benchmark:**

```bash
# Quick test (100 samples)
python scripts/run_bench.py --limit 100

# Full benchmark (all 15,617 samples)
python scripts/run_bench.py
```

This automatically symlinks the RAG task into lm-evaluation-harness and runs it. Requires the llama-server running (use `qwen_test/start_server.sh` for benchmarking).

**Compare against baseline:**

```bash
python scripts/compare_results.py
```

Auto-detects the latest baseline and RAG result files, shows accuracy/MAE comparison and per-sample improvements.

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
  nutri_rag/
    config.py                  # Paths, constants, LLM settings
    parse.py                   # Heuristic meal parser (regex-based)
    search.py                  # DuckDB text search + synonym expansion
    llm.py                     # OpenAI-compatible chat client

    bench/                     # --- Mode 1: NutriBench ---
      retriever.py             # Parse -> search -> nutrient lookup
      prompt.py                # Format USDA reference block for CoT

    assistant/                 # --- Mode 2: General Assistant ---
      gap_analyzer.py          # LLM Call 1: meal -> structured JSON targets
      food_recommender.py      # DB nutrient query + GAT neighbor expansion
      preference_db.py         # User choice history (DuckDB)
      prompt.py                # Format recommendations for LLM Call 2
      pipeline.py              # End-to-end orchestration

  tasks/
    nutribench_v2_rag/         # lm-evaluation-harness task definition
      rag.yaml
      _rag_default_template_yaml
      utils.py                 # doc_to_text_rag() with retrieval

  scripts/
    start_server.sh            # llama-server for assistant mode (parallel=1)
    demo_bench.py              # Interactive NutriBench retrieval demo
    demo_assistant.py          # Interactive assistant CLI
    run_bench.py               # NutriBench benchmark runner
    run_bench.sh               # Shell wrapper
    compare_results.py         # Baseline vs RAG comparison
```

## Module Details

### Shared Core

| Module | Purpose |
|--------|---------|
| `config.py` | Paths to DuckDB, embeddings, LLM endpoint (`localhost:8080`), key nutrient names |
| `parse.py` | Splits meal text on commas/and/with, extracts quantities+units via regex, strips filler words |
| `search.py` | Wraps DuckDB search with synonym expansion (e.g., "maize" -> "corn", "oatmeal" -> "oats") and nutrient-aware ranking (prefers entries with macronutrient data) |
| `llm.py` | Thin `requests`-based client for OpenAI-compatible `/v1/chat/completions`, with `<think>` tag stripping and robust JSON extraction (handles truncated output) |

### Mode 1: NutriBench (`bench/`)

| Module | Purpose |
|--------|---------|
| `retriever.py` | Parses meal -> searches each food in DuckDB -> gets per-100g nutrient profiles. No GAT. |
| `prompt.py` | Formats nutrients into a `=== USDA Nutritional Reference Data ===` block injected before the query |

### Mode 2: General Assistant (`assistant/`)

| Module | Purpose |
|--------|---------|
| `gap_analyzer.py` | LLM Call 1: sends eaten foods' nutrient profiles to LLM, gets back structured JSON with reasoning and macro targets (`{"protein_g": 35, "fat_g": 20, ...}`) |
| `food_recommender.py` | Queries DuckDB for gap-filling foods, then expands each candidate via GAT embedding cosine similarity to find nutritionally similar alternatives |
| `preference_db.py` | DuckDB table tracking offered/chosen foods per user, scores by `chosen_count / offered_count` ratio |
| `prompt.py` | Formats gap analysis + ranked options + preference history into LLM Call 2 prompt |
| `pipeline.py` | `NutriAssistant` class orchestrating the full 8-step pipeline |

## How It Works

### Parser Example

```
Input:  "I ate an apple and milk for breakfast"
Output: [ParsedItem(food_term="apple"), ParsedItem(food_term="milk")]

Input:  "126 grams of maize flour and 27 grams of raw sugar"
Output: [ParsedItem(food_term="maize flour", quantity=126, unit="g"),
         ParsedItem(food_term="raw sugar", quantity=27, unit="g")]
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
| `nutri_kb.duckdb` | nutri_graph | Both modes — text-to-fdc_id bridge + nutrient lookup |
| `food_embeddings.npy` | nutri_graph (trained GAT) | Mode 2 only — cosine KNN for neighbor expansion |
| `nodes_food` table | nutri_graph | Both modes — food descriptions |
| `edges_food_contains_nutrient` table | nutri_graph | Both modes — per-100g nutrient values |
| `food_id_to_idx` mapping | Built from `nodes_food` row order | Mode 2 only — fdc_id to embedding array index |
