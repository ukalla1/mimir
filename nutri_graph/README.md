# nutri_graph — USDA Food-Nutrient Knowledge Base + GAT Embeddings

Builds a cleaned **DuckDB knowledge base** from USDA FoodData Central (Foundation Foods) and USDA SR Legacy, then trains a **Graph Attention Network (GATv2)** on a bipartite Food↔Nutrient graph to produce **nutritionally-aware food embeddings**.

## Data Pipeline

### Data Sources

| Source | Description | Foods |
|--------|-------------|-------|
| **USDA FoodData Central** | Foundation Foods — lab-analyzed nutrient profiles | ~74K raw entries |
| **USDA SR Legacy** | Curated food database with cooked/prepared forms | 7,793 entries |

### KB Build Process

```
USDA FDC CSVs ──► Raw ingest (74K foods + nutrient edges)
                      │
                      ▼
                  Junk filtering
                  - Remove foods with no nutrient edges (lab samples)
                  - Remove lab analysis prefix entries (e.g., "Amino Acids, Chicken...")
                  - Remove lab tracking codes (NFY, CY suffixes)
                  → ~4.6K clean FDC foods
                      │
                      ▼
                  SR Legacy merge (7,793 foods with 644K nutrient edges)
                  - Offset fdc_ids by 1,000,000 to avoid collision
                  - Parse tilde-delimited ASCII format
                  - Merge nutrient definitions
                      │
                      ▼
                  Deduplication
                  - Group by description
                  - Pick representative fdc_id (most nutrient edges)
                  - Average nutrient amounts across duplicates
                      │
                      ▼
                  DuckDB KB (nodes_food, nodes_nutrient, edges_food_contains_nutrient)
```

### KB Quality

Before cleaning, 60.7% of FDC entries were junk — lab analysis records (`sub_sample_food`, `market_acquisition`) with no usable nutrient data. These polluted retrieval results (e.g., querying "onion" returned "Sugars, total..."). The cleaning pipeline removes these and adds SR Legacy's curated entries, which include common cooked/prepared forms often missing from FDC (e.g., chicken thigh roasted, sweet potato boiled, tilapia cooked).

## GAT Model

### Architecture

A 2-layer GATv2 with dual decoders trained on the food-nutrient bipartite graph:

| Component | Specification |
|-----------|---------------|
| Embedding dimension | 64 |
| GATv2 layers | 2 (4 heads → 1 head) |
| Edge attributes | log1p-normalized nutrient amounts |
| Existence decoder | BCE with bipartite negative sampling |
| Amount decoder | Smooth L1 (Huber) on standardized log1p(amount) |
| Combined loss | `loss_amt + 0.4 * loss_exist` |
| Best model criterion | Validation RMSE |
| Train/Val/Test split | 85/7/8 on food-nutrient edges |

### What the Embeddings Encode

The 64-dim food embeddings capture **nutritional similarity** rather than textual similarity. Foods with similar macro/micronutrient profiles cluster together regardless of their names (e.g., "coconut oil" near "palm oil"). These embeddings are used downstream by nutri_rag for:
- **GAT re-ranking**: break ties when text embedding is ambiguous
- **Neighbor expansion**: surface nutritionally similar alternatives for meal recommendations

## End-to-End Pipeline

```bash
# 1. Download USDA FDC data
python scripts/download_data.py

# 2. Build KB (FDC + SR Legacy + cleaning + dedup)
python scripts/build_kb.py

# 3. Train GAT + save embeddings
python scripts/train_GAT.py

# 4. (Optional) Export cleaned KB to JSON
python scripts/export_kb.py

# 5. (Optional) Generate UMAP visualizations
python scripts/visualize_umap_progression.py
```

### Outputs

| Artifact | Path |
|----------|------|
| Knowledge base | `data/nutri_kb.duckdb` |
| KB export (JSON) | `data/nutri_kb_export.json` |
| Best GAT checkpoint | `models/gat_checkpoint.pt` |
| Food embeddings | `outputs/embeddings/food_embeddings.npy` |
| Node embeddings | `outputs/embeddings/node_embeddings.{pt,npy}` |
| UMAP snapshots | `outputs/snapshots/food_emb_epoch_*.npy` |
| Training curves | `outputs/training/*.html` |

## Project Structure

```
nutri_graph/
  nutri_graph/
    config.py               # Paths, constants
    kb/
      builder.py            # DuckDB KB builder (FDC + SR Legacy + cleaning + dedup + export)
    graph/                   # PyG graph construction + negative sampling
    models/
      gat_model.py          # GATv2 architecture (dual decoders)
    training/
      trainer.py            # Training loop with best-checkpoint tracking
    visualization/          # UMAP, clustering, Plotly training curves
    retrevial/              # Direct DB retrieval utilities

  scripts/
    download_data.py        # Download USDA FDC from Kaggle
    build_kb.py             # Build DuckDB KB
    train_GAT.py            # Train GAT + save embeddings + snapshots
    export_kb.py            # Export KB to JSON for fine-tuning
    build_recipe_kb.py      # Recipe graph builder
    demo_retrival.py        # Interactive retrieval demo
    visualize_umap_progression.py
    generate_umap.py

  data/
    raw/                    # USDA FDC CSVs
    SR-Leg_ASC/             # SR Legacy tilde-delimited files
    nutri_kb.duckdb         # Built knowledge base
    nutri_kb_export.json    # JSON export

  outputs/                  # Embeddings, checkpoints, plots
  models/                   # GAT checkpoints
```

## Dependencies

- Python 3.9+
- PyTorch + PyTorch Geometric (GATv2Conv)
- DuckDB
- Kaggle API token (for dataset download)
- Optional: kaleido (PNG export), umap-learn, plotly
