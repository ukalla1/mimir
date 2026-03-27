# Mimir: Graph-Augmented RAG for Nutritional Intelligence

Mimir combines **Graph Attention Networks (GAT)** with **Retrieval-Augmented Generation (RAG)** to estimate nutritional content from natural-language meal descriptions and provide personalized meal recommendations. The system learns food-nutrient relationships from USDA databases and uses them alongside semantic text embeddings and LLM reasoning.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                nutri_graph                    │
                    │    Knowledge Base + GAT Embedding Training    │
                    │                                              │
                    │  USDA FDC + SR Legacy                        │
                    │    → Junk filtering (remove lab samples)     │
                    │    → Deduplication (avg nutrients)            │
                    │    → DuckDB KB                               │
                    │    → GATv2 Training → Food Embeddings (64d)  │
                    └──────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
    ┌──────────────────────────┐   ┌──────────────────────────────┐
    │    NutriBench Benchmark  │   │     General Assistant         │
    │    (Nutrient Estimation) │   │  (Meal Recommendations)      │
    │                          │   │                              │
    │  Meal → Regex Extract    │   │  Eaten Foods → Gap Analysis  │
    │  → Embedding Search      │   │  → DB Query + GAT Expand     │
    │  → GAT Re-rank/Filter    │   │  → Preference Re-rank        │
    │  → CoT Prompt → LLM     │   │  → LLM Recommendation        │
    └──────────────────────────┘   └──────────────────────────────┘
```

## Subsystems

### [nutri_graph](nutri_graph/) — Knowledge Base + GAT Training

Builds a cleaned DuckDB KB from two USDA sources:
- **FoodData Central** (Foundation Foods): ~74K raw entries, cleaned to ~4.6K after removing lab analysis junk (60.7% of entries had no usable nutrient data)
- **SR Legacy**: 7,793 curated entries with cooked/prepared forms (chicken roasted, sweet potato boiled, tilapia cooked, etc.)

After merging and deduplication, trains a GATv2 on the food-nutrient bipartite graph to produce 64-dim food embeddings encoding nutritional similarity.

### [nutri_rag](nutri_rag/) — RAG Pipeline

Four retrieval versions for ablation:

| Version | Method | Description |
|---------|--------|-------------|
| V0 | BM25 | Keyword matching with cross-language synonyms |
| V1 | Dense | Qwen3-Embedding semantic search |
| V2 | Dense + GAT | V1 + GAT re-ranking when text is ambiguous |
| V3 | Multi-candidate | Top-5 candidates per food, filtered by similarity threshold (0.60) |

Two modes:
- **NutriBench Benchmark**: Evaluate carb/protein/fat/energy estimation accuracy using lm-evaluation-harness
- **General Assistant**: Interactive meal recommendations with GAT neighbor expansion and user preference learning

## Results (Protein, 1000 samples)

| Version | Acc@7.5g | MAE |
|---------|----------|-----|
| Baseline (no RAG) | 0.735 | 7.00 |
| V3 (multi-candidate + GAT) | **0.763** | **6.04** |

## Quick Start

```bash
# 1. Build knowledge base (FDC + SR Legacy + cleaning)
cd nutri_graph && python scripts/build_kb.py

# 2. Train GAT embeddings
cd nutri_graph && python scripts/train_GAT.py

# 3. Build text embeddings for RAG
cd nutri_rag && python scripts/build_embeddings.py

# 4. Start LLM server
bash nutri_rag/scripts/start_server.sh  # or qwen_test/start_server.sh for benchmarks

# 5. Run benchmark
cd nutri_rag && python scripts/run_bench.py --mode v3 --nutrient protein --limit 200

# 6. Run assistant
cd nutri_rag && python scripts/demo_assistant.py

# 7. (Optional) Export KB for fine-tuning
cd nutri_graph && python scripts/export_kb.py
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Graph ML | PyTorch + PyTorch Geometric (GATv2Conv) |
| Text Embeddings | Qwen3-Embedding-0.6B (1024d, last-token pooling) |
| LLM Inference | Qwen3.5-9B via llama.cpp / llama-server |
| Database | DuckDB (columnar storage + full-text search) |
| Data Sources | USDA FoodData Central + USDA SR Legacy |
| Evaluation | lm-evaluation-harness |
| Visualization | Plotly, UMAP, scikit-learn |

## Project Structure

```
mimir/
├── nutri_graph/                    # Knowledge base + GAT model
│   ├── nutri_graph/
│   │   ├── kb/builder.py          # KB builder (FDC + SR Legacy + cleaning + dedup)
│   │   ├── graph/                 # PyG graph construction + negative sampling
│   │   ├── models/gat_model.py    # GATv2 (dual decoders)
│   │   ├── training/trainer.py    # Training loop
│   │   └── visualization/         # UMAP, clustering, training curves
│   ├── scripts/
│   │   ├── build_kb.py            # Build DuckDB KB
│   │   ├── train_GAT.py           # Train GAT + save embeddings
│   │   └── export_kb.py           # Export KB to JSON
│   ├── data/                      # USDA CSVs + SR Legacy + DuckDB
│   └── outputs/                   # Embeddings, checkpoints, plots
│
├── nutri_rag/                     # RAG pipeline
│   ├── nutri_rag/
│   │   ├── config.py              # Thresholds, model config
│   │   ├── embedding.py           # TextEmbedder + FoodVectorIndex + GATIndex
│   │   ├── search.py              # Semantic + GAT hybrid search
│   │   ├── bench/                 # NutriBench: retriever, prompt, task utils
│   │   └── assistant/             # Assistant: gap analysis, recommendations
│   ├── tasks/                     # lm-eval task definitions (V0/V1/V2/V3)
│   ├── scripts/                   # build_embeddings, run_bench, demos
│   └── results/                   # Benchmark outputs
│
└── README.md
```
