# Mimir: Graph-Augmented Retrieval for Nutritional Intelligence

Mimir is a nutrition intelligence system that combines **Graph Attention Networks (GAT)** with **Retrieval-Augmented Generation (RAG)** to achieve accurate nutritional content estimation and personalized meal recommendations. The system learns structured food-nutrient relationships from the USDA FoodData Central database and leverages them alongside semantic text embeddings and LLM reasoning.

## System Architecture

Mimir consists of two complementary subsystems:

```
                        ┌──────────────────────────────────────────┐
                        │              nutri_graph                 │
                        │  Knowledge Base + GAT Embedding Training │
                        │                                          │
                        │  USDA CSVs ──► DuckDB ──► PyG Graph     │
                        │                             │            │
                        │                         GATv2 Training   │
                        │                             │            │
                        │                    Food Embeddings (64d) │
                        └──────────┬───────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
    ┌──────────────────────────┐   ┌──────────────────────────────┐
    │    NutriBench Benchmark  │   │     General Assistant         │
    │    (Nutrient Estimation) │   │  (Meal Recommendations)      │
    │                          │   │                              │
    │  Meal ──► Regex Extract  │   │  Eaten Foods ──► Gap Analysis│
    │  ──► Embedding Search    │   │  ──► DB Query + GAT Expand   │
    │  ──► (±GAT Re-rank)     │   │  ──► Preference Re-rank      │
    │  ──► CoT Prompt ──► LLM │   │  ──► LLM Recommendation      │
    └──────────────────────────┘   └──────────────────────────────┘
```

---

## 1. nutri_graph — Knowledge Base and GAT Embedding Training

### Purpose

Constructs a structured food-nutrient knowledge base from USDA FoodData Central and trains a Graph Attention Network to produce food embeddings that encode nutritional relationships.

### Features

- **Bipartite Knowledge Graph Construction**: Builds a food-nutrient bipartite graph from USDA Foundation Foods data (74,175 foods, 153 nutrients) stored in DuckDB with `nodes_food`, `nodes_nutrient`, and `edges_food_contains_nutrient` tables.

- **GATv2 Architecture with Dual Decoders**: A 2-layer GATv2 model with 4 attention heads (layer 1) and 1 head (layer 2), using residual connections and layer normalization. The model has two decoders — a **nutrient existence decoder** (BCE loss with negative sampling) predicting whether a food contains a nutrient, and a **nutrient amount regression decoder** (smooth L1 loss on standardized log1p amounts) predicting how much.

- **Nutritionally-Aware Embeddings**: Produces 64-dimensional food embeddings where proximity in the embedding space reflects nutritional similarity rather than textual similarity. Foods with similar macro/micronutrient profiles cluster together regardless of their names.

- **Supervised Edge-Level Training**: 85/7/8 train/val/test split on food-nutrient edges with bipartite negative sampling, ReduceLROnPlateau scheduling, gradient clipping, and best-checkpoint tracking by validation RMSE.

- **Embedding Visualization Pipeline**: Epoch-by-epoch UMAP progression tracking, MiniBatchKMeans clustering analysis, and interactive Plotly training curves for loss, MAE, RMSE, and AUC.

### Key Technical Details

| Component         | Specification                                     |
|--------------------|---------------------------------------------------|
| Embedding dimension | 64                                               |
| GATv2 layers       | 2 (4 heads → 1 head)                             |
| Edge attributes     | log1p-normalized nutrient amounts                |
| Regression target   | Standardized log1p(amount) with per-split stats  |
| Existence loss      | BCE with bipartite negative sampling             |
| Amount loss         | Smooth L1 (Huber)                                |
| Combined loss       | `loss_amt + 0.4 * loss_exist`                    |
| Best model criterion | Validation RMSE                                 |

---

## 2. nutri_rag — Retrieval-Augmented Generation Pipeline

### Purpose

Applies the knowledge base and GAT embeddings to two downstream tasks: benchmarking nutritional content estimation accuracy (NutriBench) and providing personalized meal recommendations (General Assistant).

### Shared Core Features

- **Semantic Text Embedding Search**: Pre-computed 1024-dimensional embeddings for all 74,175 USDA foods using Qwen3-Embedding-0.6B with last-token pooling and L2 normalization. Supports task-instruction tuning (e.g., "Given a food name, retrieve the matching USDA food database entry") for asymmetric query-document retrieval.

- **Cross-Language Vocabulary Handling**: Semantic embeddings naturally resolve vocabulary mismatches such as "groundnut" vs. "peanut", "maize flour" vs. "corn flour", and "aubergine" vs. "eggplant" — eliminating the need for manually curated synonym lists.

- **GAT Nutritional Coherence Index**: A secondary retrieval layer using pre-trained 64-dim GAT embeddings. When text embedding scores are ambiguous (gap < 0.03 between top candidates), GAT cosine similarity is used to re-rank by nutritional profile coherence.

- **Hybrid Retrieval with Macro-Aware Filtering**: Candidates are ranked by a combination of text similarity and macro data availability (carb/protein/fat coverage), ensuring retrieved foods have complete nutritional profiles for downstream computation.

- **Local LLM Inference**: Qwen3.5-9B served via llama-server (OpenAI-compatible API) with greedy decoding (temperature=0.0) for reproducible predictions.

---

### Mode A: NutriBench Benchmark (Nutrient Estimation)

#### Purpose

Evaluates how accurately the system can estimate nutritional content (carbohydrate, protein, fat, energy) from natural-language meal descriptions, using the NutriBench benchmark.

#### Features

- **Regex-Based Food Term Extraction**: Three complementary patterns extract food terms from NutriBench's structured meal descriptions:
  - Pattern A: `"<qty>g of <food>"` (e.g., "126 grams of maize flour")
  - Pattern B: `"<food> weighing <qty>g"` (e.g., "a plain bun weighing 126 grams")
  - Pattern C: `"<food> (<qty>g)"` (e.g., "boiled onion (1g)")

  Extracted terms are cleaned by stripping cooking methods and preparation adjectives (raw, boiled, fried, peeled, etc.).

- **Three Retrieval Versions for Ablation**:
  - **V0 (BM25 Baseline)**: DuckDB full-text search with English stemming and 16 hardcoded cross-language synonyms. Serves as the keyword-matching baseline.
  - **V1 (Text Embedding)**: Qwen3-Embedding semantic search over pre-computed vectors. Handles vocabulary mismatches that BM25 cannot.
  - **V2 (Text + GAT Re-ranking)**: V1 search augmented with GAT neighbor expansion — when text embedding candidates are ambiguous, nutritionally similar alternatives are surfaced via GAT cosine similarity and merged into the candidate pool.

- **Chain-of-Thought Prompting with USDA Reference Data**: Retrieved nutrient profiles (per 100g) are formatted into a structured reference block injected before the query. The LLM is instructed: "Use these reference values if they match; for unlisted foods, use your own knowledge." Includes per-nutrient formulas (e.g., `carbs = (weight_g / 100) * carbs_per_100g`).

- **Per-Item Threshold Gating (V3 Format)**: An advanced prompt format that pairs each extracted food item individually with either a reliable USDA match (similarity >= 0.55) or an explicit "no reliable match — use your own knowledge" instruction, giving the LLM fine-grained context awareness.

- **lm-evaluation-harness Integration**: Task definitions for V0, V1, and V2 are packaged as standard lm-eval tasks, enabling reproducible benchmarking with standardized metrics (accuracy at 7.5g tolerance, MAE).

#### Benchmark Data Flow

```
Meal Description ("126g of maize flour and 27g of raw sugar")
  │
  ├─ Regex extraction ──► ["maize flour", "sugar"]
  │
  ├─ V0: BM25 keyword search (DuckDB FTS + synonyms)
  ├─ V1: Qwen3-Embedding cosine search (74K pre-computed vectors)
  └─ V2: V1 + GAT neighbor expansion (when top-k gap < 0.03)
         │
         ▼
  Nutrient Lookup (per 100g from DuckDB)
         │
         ▼
  CoT Prompt: reference block + formula + "Let's think step by step"
         │
         ▼
  LLM (Qwen3.5-9B, greedy) ──► Numerical prediction
```

---

### Mode B: General Assistant (Personalized Meal Recommendations)

#### Purpose

An interactive nutrition assistant that analyzes what the user has eaten, identifies nutritional gaps, and recommends foods for their next meal — personalized through GAT-based food expansion and user preference learning.

#### Features

- **LLM-Driven Nutritional Gap Analysis**: The first LLM call takes the user's meal history (with per-100g nutrient profiles and serving sizes) and outputs structured JSON targets: `{protein_g, fat_g, carb_g, energy_kcal}` for the next meal, along with reasoning about what's missing.

- **Gap-Based Database Querying**: Identifies which macronutrient has the largest deficit and queries DuckDB for foods rich in that nutrient, yielding seed candidates that directly address the nutritional gap.

- **GAT Neighbor Expansion**: Each seed candidate is expanded with k=5 nutritionally similar alternatives found via GAT embedding cosine similarity. This diversifies recommendations beyond exact database matches — if the top hit is "chicken breast", GAT neighbors might surface "turkey breast" or "lean pork loin" as nutritionally equivalent options.

- **User Preference Learning and Re-ranking**: A DuckDB-backed preference database tracks which foods the user has been offered and which they actually chose. Over time, chosen/offered ratios produce preference scores (0–1) used to re-rank recommendations — foods the user has selected before are boosted, while consistently rejected options are deprioritized.

- **Two-Stage LLM Pipeline**: The second LLM call receives the gap analysis reasoning, ranked food options (seeds first by preference, neighbors by GAT similarity), and preference history to generate natural-language meal suggestions tailored to the user.

- **Heuristic Meal Parser**: Splits free-text meal descriptions on separators (commas, "and", semicolons), extracts quantities and units, and maps each item to the USDA database via semantic search.

#### Assistant Data Flow

```
User: "I had oatmeal and a banana for breakfast. What should I eat for lunch?"
  │
  ├─ Parse eaten foods ──► ["oatmeal", "banana"]
  ├─ Semantic search ──► USDA matches with nutrient profiles
  │
  ▼
  LLM Call 1: Gap Analysis
  ──► {"reasoning": "...", "targets": {"protein_g": 35, "fat_g": 20, ...}}
  │
  ▼
  DuckDB: Find foods high in top-gap nutrient (e.g., protein)
  ──► 5 seed candidates
  │
  ▼
  GAT Expansion: 5 neighbors per seed ──► up to 30 total options
  │
  ▼
  Preference Re-ranking: boost previously chosen foods
  │
  ▼
  LLM Call 2: Natural-language recommendation
  ──► "For lunch, consider grilled chicken breast with quinoa and..."
```

---

## Key Contributions

1. **Graph-Learned Nutritional Representations**: Training a GATv2 on the food-nutrient bipartite graph produces embeddings where nutritional similarity is encoded structurally, complementing text-based semantic similarity. This dual-representation approach captures relationships invisible to text embeddings alone (e.g., "coconut oil" and "palm oil" are nutritionally similar but textually distant from each other).

2. **Hybrid Retrieval with GAT Re-ranking**: A novel retrieval strategy that uses text embeddings for initial semantic matching and falls back to GAT-based nutritional coherence scoring when text similarity is ambiguous — combining the best of semantic understanding and domain-specific nutritional knowledge.

3. **GAT-Expanded Recommendation Diversity**: Using GAT neighbors to expand database query results surfaces nutritionally equivalent alternatives that keyword or text searches would miss, enabling more diverse and personalized meal suggestions.

4. **Progressive Retrieval Ablation (V0 → V1 → V2)**: A systematic comparison framework with three retrieval strategies — BM25 keywords, semantic embeddings, and hybrid text+GAT — integrated into lm-evaluation-harness for reproducible benchmarking against NutriBench.

5. **Preference-Aware Personalization Loop**: A closed-loop system where user food choices are tracked and fed back into recommendation ranking, allowing the assistant to learn individual dietary preferences over time without retraining the underlying models.

---

## Tech Stack

| Layer               | Technology                                          |
|----------------------|-----------------------------------------------------|
| Graph ML             | PyTorch + PyTorch Geometric (GATv2Conv)             |
| Text Embeddings      | Qwen3-Embedding-0.6B (HuggingFace Transformers)    |
| LLM Inference        | Qwen3.5-9B via llama.cpp / llama-server             |
| Database             | DuckDB (columnar storage + full-text search)        |
| Data Source          | USDA FoodData Central (74,175 foods, 153 nutrients) |
| Evaluation Framework | lm-evaluation-harness                               |
| Visualization        | Plotly, UMAP, scikit-learn (KMeans clustering)      |

## Project Structure

```
mimir/
├── nutri_graph/                    # Knowledge base + GAT model
│   ├── nutri_graph/
│   │   ├── kb/                     # DuckDB builder (CSV → structured DB)
│   │   ├── graph/                  # PyG graph construction + negative sampling
│   │   ├── models/                 # GATv2 architecture (dual decoders)
│   │   ├── training/               # Training loop (Colab-faithful)
│   │   └── visualization/          # UMAP, clustering, training curves
│   ├── scripts/                    # download_data, build_kb, train_GAT
│   ├── data/                       # USDA CSVs + nutri_kb.duckdb
│   └── outputs/                    # Embeddings, checkpoints, plots
│
├── nutri_rag/                      # RAG pipeline
│   ├── nutri_rag/
│   │   ├── embedding.py            # TextEmbedder, FoodVectorIndex, GATIndex
│   │   ├── search.py               # Semantic + GAT hybrid search
│   │   ├── search_bm25.py          # BM25 baseline search
│   │   ├── parse.py                # Heuristic meal parser
│   │   ├── llm.py                  # OpenAI-compatible LLM client
│   │   ├── bench/                  # NutriBench benchmark mode
│   │   │   ├── retriever.py        # Regex extraction + V1/V2 search
│   │   │   ├── retriever_bm25.py   # V0 BM25 retriever
│   │   │   └── prompt.py           # CoT prompt formatting
│   │   └── assistant/              # General assistant mode
│   │       ├── gap_analyzer.py     # LLM-driven gap analysis
│   │       ├── food_recommender.py # DB query + GAT expansion
│   │       ├── preference_db.py    # User preference tracking
│   │       └── pipeline.py         # End-to-end orchestration
│   ├── tasks/                      # lm-eval task definitions (V0/V1/V2)
│   ├── scripts/                    # build_embeddings, run_bench, demos
│   └── results/                    # Benchmark outputs
│
└── README.md
```
