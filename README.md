# Mimir: Graph-Augmented RAG for Nutritional Intelligence

Mimir combines **Graph Attention Networks (GAT)** with **Retrieval-Augmented Generation (RAG)** to estimate nutritional content from natural-language meal descriptions and provide personalized meal recommendations. The system learns food-nutrient-recipe relationships from USDA databases and FoodKG, and uses them alongside semantic text embeddings and LLM reasoning.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │                nutri_graph                    │
                    │    Knowledge Base + GAT Embedding Training    │
                    │                                              │
                    │  USDA FDC + SR Legacy + FoodKG               │
                    │    → Junk filtering (remove lab samples)     │
                    │    → Deduplication (avg nutrients)            │
                    │    → Recipe-ingredient-tag integration        │
                    │    → DuckDB KB                               │
                    │    → Heterogeneous GATv2 Training             │
                    │      (food, nutrient, recipe, tag)            │
                    │    → Food Embeddings (64d)                   │
                    └──────────────┬───────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                           ▼
┌────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  NutriBench        │ │  General Assistant   │ │  PFoodReq Benchmark  │
│  (Nutrient Est.)   │ │  (Meal Recommend.)   │ │  (Recipe Recommend.) │
│                    │ │                      │ │                      │
│ Meal → Parse       │ │ Eaten Foods → Parse  │ │ Tag → All Recipes    │
│ → Embedding Search │ │ → Gap Analysis (LLM) │ │ → Constraint Filter  │
│ → GAT Re-rank      │ │ → DB Query + GAT     │ │ → GAT Re-rank        │
│ → CoT Prompt → LLM│ │ → Preference Re-rank │ │ → Return matches     │
│                    │ │ → Recommend (LLM)    │ │ (no LLM needed)      │
└────────────────────┘ └──────────┬───────────┘ └──────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │      nutri-atlas          │
                    │  (Robot Integration)      │
                    │                           │
                    │  User ─► Qwen Agent       │
                    │  ─► Tool calls            │
                    │    ├─ Navigation (ZMQ)    │
                    │    ├─ Object detection    │
                    │    └─ Meal recommendation │
                    │       (via nutri_rag)     │
                    └──────────────────────────┘
```

## Subsystems

### [nutri_graph](nutri_graph/) — Knowledge Base + GAT Training

Builds a cleaned DuckDB KB from three data sources:
- **USDA FoodData Central** (Foundation Foods): ~74K raw entries, cleaned to ~4.6K after removing lab analysis junk
- **USDA SR Legacy**: 7,793 curated entries with cooked/prepared forms
- **FoodKG** (via PFoodReq): ~82K recipes with ingredients, nutrients, and cuisine tags — ingredients matched to USDA fdc_ids via Qwen3-Embedding

Trains a heterogeneous GATv2 on a 4-node-type graph (food, nutrient, recipe, tag) to produce 64-dim embeddings encoding nutritional and relational similarity.

### [nutri_rag](nutri_rag/) — RAG Pipeline

Four retrieval versions for ablation:

| Version | Method | Description |
|---------|--------|-------------|
| V0 | BM25 | Keyword matching with cross-language synonyms |
| V1 | Dense | Qwen3-Embedding semantic search |
| V2 | Dense + GAT | V1 + GAT re-ranking when text is ambiguous |
| V3 | Multi-candidate | Top-5 candidates per food, filtered by similarity threshold (0.60) |

Three modes:
- **NutriBench Benchmark**: Evaluate carb/protein/fat/energy estimation accuracy using lm-evaluation-harness
- **General Assistant**: Interactive meal recommendations with GAT neighbor expansion, gap analysis, and user preference learning
- **PFoodReq Benchmark**: Personalized food recommendation using deterministic constraint filtering + GAT re-ranking

### [nutri-atlas](nutri-atlas/) — Robot Integration

Connects the nutrition assistant to a Clearpath Go2 robot via Qwen Agent tool-calling. The robot assistant can navigate, detect objects, and provide meal recommendations — all through natural language.

- **Navigation tools**: go to landmarks, spin, detect objects via camera
- **Nutrition tool**: `get_meal_recommendation` wraps `nutri_rag`'s full pipeline (parse → gap analysis → GAT expansion → recommendation)
- **LLM**: Qwen3.5-9B as the orchestration agent, deciding when to call navigation vs nutrition tools
- **Communication**: ZMQ bridge to ROS2 on the robot

Example: *"I ate an apple and milk for breakfast, what should I eat for lunch?"* → the agent calls `get_meal_recommendation` → returns a personalized suggestion based on nutritional gap analysis.

## Results

### NutriBench (Protein, 1000 samples)

| Version | Acc@7.5g | MAE |
|---------|----------|-----|
| Baseline (no RAG) | 0.735 | 7.00 |
| V3 (multi-candidate + GAT) | **0.763** | **6.04** |

### PFoodReq (Full test set, 2244 queries)

| Method | MAP | MAR | F1 |
|--------|-----|-----|-----|
| BAMnet (PFoodReq paper, WSDM 2021) | 62.7 | 61.8 | 63.7 |
| **Ours (Config C: filter + GAT, no LLM)** | **78.7** | **82.9** | **77.5** |

Config C pipeline: tag lookup → deterministic constraint filter (ingredient inclusion/exclusion + nutrient ranges) → GAT re-ranking. The deterministic filter leverages our clean augmented KB with accurate ingredient-nutrient mappings, outperforming BAMnet's learned embedding approach without requiring an LLM.

## Quick Start

```bash
# 1. Build knowledge base (USDA FDC + SR Legacy + cleaning)
cd nutri_graph && python scripts/build_kb.py

# 2. Build text embeddings for RAG (must run before recipe KB)
cd nutri_rag && python scripts/build_embeddings.py

# 3. Integrate FoodKG recipes into KB
cd nutri_graph && python scripts/build_recipe_kb.py

# 4. Train GAT embeddings (heterogeneous graph: food, nutrient, recipe, tag)
cd nutri_graph && python scripts/train_GAT.py

# 5. Build recipe text embeddings (for PFoodReq)
cd nutri_rag && python scripts/build_recipe_embeddings.py

# 6. Start LLM server (port 8080 for nutri_rag)
bash nutri_rag/scripts/start_server.sh

# 7. Run NutriBench
cd nutri_rag && python scripts/run_bench.py --mode v3 --nutrient protein --limit 200

# 8. Run PFoodReq benchmark (no LLM server needed)
cd nutri_rag && python scripts/run_pfoodreq_bench.py

# 9. Run interactive assistant
cd nutri_rag && python scripts/demo_assistant.py

# 10. Run robot assistant with nutrition (requires both LLM servers)
#     Terminal 1: bash nutri_rag/scripts/start_server.sh     (port 8080, for nutrition)
#     Terminal 2:
cd nutri-atlas/robot_control && python robot_assistant.py
```

Note: Steps 7, 9, 10 require the LLM server on port 8080 (step 6). Step 10 additionally requires the robot agent LLM server on port 8001. PFoodReq (step 8) runs without any LLM server.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Graph ML | PyTorch + PyTorch Geometric (GATv2Conv) |
| Text Embeddings | Qwen3-Embedding-0.6B (1024d, last-token pooling) |
| LLM Inference | Qwen3.5-9B via llama.cpp / llama-server |
| Agent Framework | Qwen Agent (tool-calling orchestration) |
| Robot Communication | ZMQ → ROS2 (Clearpath Go2) |
| Database | DuckDB (columnar storage + full-text search) |
| Data Sources | USDA FoodData Central + USDA SR Legacy + FoodKG |
| Evaluation | lm-evaluation-harness |
| Visualization | Plotly, UMAP, scikit-learn |

## Project Structure

```
mimir/
├── nutri_graph/                    # Knowledge base + GAT model
│   ├── nutri_graph/
│   │   ├── kb/builder.py          # KB builder (FDC + SR Legacy + cleaning + dedup)
│   │   ├── kb/recipe_builder.py   # FoodKG recipe integration
│   │   ├── graph/                 # PyG graph construction + negative sampling
│   │   ├── models/gat_model.py    # GATv2 (dual decoders, 4 node types)
│   │   ├── training/trainer.py    # Training loop
│   │   └── visualization/         # UMAP, clustering, training curves
│   ├── scripts/
│   │   ├── build_kb.py            # Build DuckDB KB
│   │   ├── build_recipe_kb.py     # Integrate FoodKG recipes
│   │   ├── train_GAT.py           # Train GAT + save embeddings
│   │   └── export_kb.py           # Export KB to JSON
│   ├── data/                      # USDA CSVs + SR Legacy + DuckDB
│   └── outputs/                   # Embeddings, checkpoints, plots
│
├── nutri_rag/                     # RAG pipeline
│   ├── nutri_rag/
│   │   ├── config.py              # Thresholds, model config
│   │   ├── embedding.py           # TextEmbedder + FoodVectorIndex + GATIndex + RecipeVectorIndex
│   │   ├── search.py              # Semantic + GAT hybrid search
│   │   ├── bench/                 # NutriBench: retriever, prompt, task utils
│   │   ├── assistant/             # Assistant: gap analysis, recommendations
│   │   └── pfoodreq/             # PFoodReq: query parser, retriever, evaluator, prompt
│   ├── tasks/                     # lm-eval task definitions (V0/V1/V2/V3)
│   ├── scripts/                   # build_embeddings, run_bench, run_pfoodreq_bench, demos
│   └── results/                   # Benchmark outputs
│
├── nutri-atlas/                   # Robot integration
│   ├── robot_control/
│   │   ├── robot_assistant.py     # Main chat loop + Qwen Agent
│   │   ├── tools/
│   │   │   ├── navigate_tool.py   # Navigate to landmarks / coordinates
│   │   │   ├── object_tool.py     # Camera object detection queries
│   │   │   ├── motion_tool.py     # Spin and move primitives
│   │   │   ├── nutrition_tool.py  # Meal recommendation (wraps nutri_rag)
│   │   │   └── zmq_client.py      # ZMQ transport to robot
│   │   └── config/landmarks.yaml  # Named room positions
│   ├── scripts/start_server.sh    # LLM server for robot agent (port 8001)
│   └── benchmark_tasks.yaml       # 30+ graded robot benchmark tasks
│
├── PFoodReq/                      # PFoodReq benchmark data (WSDM 2021)
│   └── data/kbqa_data/            # test/dev/train splits + dish_info_map
│
└── README.md
```
