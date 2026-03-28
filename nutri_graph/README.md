# nutri_graph — Food-Nutrient Knowledge Base + GAT Embeddings

Builds a cleaned **DuckDB knowledge base** from USDA FoodData Central (Foundation Foods) and USDA SR Legacy, optionally integrates **FoodKG recipe data** (from PFoodReq), then trains a **Graph Attention Network (GATv2)** on a heterogeneous Food↔Nutrient↔Recipe↔Tag graph to produce **nutritionally-aware embeddings** for foods, recipes, and cuisine tags.

## Data Pipeline

### Data Sources

| Source | Description | Entries |
|--------|-------------|---------|
| **USDA FoodData Central** | Foundation Foods — lab-analyzed nutrient profiles | ~74K raw entries |
| **USDA SR Legacy** | Curated food database with cooked/prepared forms | 7,793 entries |
| **FoodKG** (via PFoodReq) | Recipe knowledge graph with ingredients, nutrients, cuisine tags | ~82K recipes |

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

### FoodKG Recipe Integration (Optional)

After building the USDA KB, `build_recipe_kb.py` **adds** recipe-level tables without touching the existing USDA data:

```
PFoodReq recipe_kg.json (~82K recipes)
        │
        ▼
    Parse recipes, ingredients, cuisine tags
        │
        ▼
    Batch-match ingredient names → USDA fdc_ids
    (using Qwen3-Embedding cosine similarity, threshold=0.45)
        │
        ▼
    Insert into existing nutri_kb.duckdb:
      + nodes_recipe              (recipe metadata + per-recipe nutrients)
      + nodes_tag                 (cuisine/category tags)
      + edges_recipe_uses_food    (recipe → USDA fdc_id)
      + edges_recipe_has_tag      (recipe → tag)
```

This bridges FoodKG recipes to USDA foods, enabling the GAT to learn over a heterogeneous graph.

### KB Quality

Before cleaning, 60.7% of FDC entries were junk — lab analysis records (`sub_sample_food`, `market_acquisition`) with no usable nutrient data. These polluted retrieval results (e.g., querying "onion" returned "Sugars, total..."). The cleaning pipeline removes these and adds SR Legacy's curated entries, which include common cooked/prepared forms often missing from FDC (e.g., chicken thigh roasted, sweet potato boiled, tilapia cooked).

## GAT Model

### Graph Structure

When `INCLUDE_RECIPES = True` (default), the GAT trains on a heterogeneous graph with 4 node types:

```
                    food ←→ nutrient          (USDA food-nutrient edges)
                    recipe ←→ food            (FoodKG ingredient links)
                    recipe ←→ tag             (FoodKG cuisine/category tags)
```

| Node Type | ID | Source |
|-----------|----|--------|
| food | 0 | USDA FDC + SR Legacy |
| nutrient | 1 | USDA nutrient definitions |
| recipe | 2 | FoodKG (via build_recipe_kb.py) |
| tag | 3 | FoodKG cuisine/category tags |

Without recipe tables, the graph falls back to the original bipartite food↔nutrient graph (2 node types).

### Architecture

A 2-layer GATv2 with dual decoders:

| Component | Specification |
|-----------|---------------|
| Embedding dimension | 64 |
| GATv2 layers | 2 (4 heads → 1 head) |
| Node type embeddings | Per-type learned embedding (food/nutrient/recipe/tag) |
| Edge attributes | log1p-normalized nutrient amounts (food↔nutrient), 1.0 (recipe edges) |
| Existence decoder | BCE with bipartite negative sampling |
| Amount decoder | Smooth L1 (Huber) on standardized log1p(amount) |
| Combined loss | `loss_amt + 0.4 * loss_exist` |
| Best model criterion | Validation RMSE |
| Train/Val/Test split | 85/7/8 on food-nutrient edges |

The supervised task (food→nutrient prediction) is unchanged — recipe and tag edges only participate in **message passing**, enriching food embeddings with recipe-level context.

### What the Embeddings Encode

The 64-dim embeddings capture **nutritional similarity** learned from the graph structure:
- **Food embeddings**: foods with similar nutrient profiles cluster together (e.g., "coconut oil" near "palm oil"), enriched by recipe co-occurrence when recipes are included
- **Recipe embeddings**: recipes with similar ingredients/nutritional profiles cluster together
- **Tag embeddings**: cuisine tags aggregate the nutritional patterns of their recipes

These embeddings are used downstream by nutri_rag for:
- **GAT re-ranking**: break ties when text embedding is ambiguous
- **Neighbor expansion**: surface nutritionally similar alternatives for meal recommendations
- **Recipe retrieval**: match queries to nutritionally appropriate recipes (PFoodReq benchmark)

## End-to-End Pipeline

```bash
# 1. Download USDA FDC data
cd ~/work/atlas/mimir/nutri_graph
python scripts/download_data.py

# 2. Build KB (FDC + SR Legacy + cleaning + dedup)
python scripts/build_kb.py

# 3. Build text embeddings for USDA foods (needed by step 4)
cd ~/work/atlas/mimir/nutri_rag
python scripts/build_embeddings.py

# 4. Integrate FoodKG recipes into KB (adds tables, does not modify USDA data)
cd ~/work/atlas/mimir/nutri_graph
python scripts/build_recipe_kb.py

# 5. Train GAT on heterogeneous graph (food + nutrient + recipe + tag)
python scripts/train_GAT.py

# 6. (Optional) Export cleaned KB to JSON
python scripts/export_kb.py

# 7. (Optional) Generate UMAP visualizations
python scripts/visualize_umap_progression.py
```

Note: Steps 3-4 are needed for the heterogeneous graph. If you skip them, `train_GAT.py` falls back to the bipartite food↔nutrient graph only.

### When to Re-run Each Step

| Script | Trigger to re-run | Depends on |
|--------|-------------------|------------|
| `build_kb.py` | USDA data sources change | Raw USDA CSVs + SR Legacy |
| `build_embeddings.py` (nutri_rag) | `nodes_food` table changes | `build_kb.py` output |
| `build_recipe_kb.py` | FoodKG data changes | `build_kb.py` + `build_embeddings.py` |
| `train_GAT.py` | Any KB table changes | `build_kb.py` + `build_recipe_kb.py` |

### Outputs

| Artifact | Path |
|----------|------|
| Knowledge base | `data/nutri_kb.duckdb` |
| KB export (JSON) | `data/nutri_kb_export.json` |
| Best GAT checkpoint | `models/gat_checkpoint.pt` |
| Food embeddings | `outputs/embeddings/food_embeddings.npy` |
| Node embeddings (all types) | `outputs/embeddings/node_embeddings.{pt,npy}` |
| UMAP snapshots | `outputs/snapshots/food_emb_epoch_*.npy` |
| Training curves | `outputs/training/*.html` |

## Project Structure

```
nutri_graph/
  nutri_graph/
    config.py               # Paths, constants
    kb/
      builder.py            # DuckDB KB builder (FDC + SR Legacy + cleaning + dedup + export)
      recipe_builder.py     # FoodKG recipe integration (adds recipe/tag tables to KB)
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
