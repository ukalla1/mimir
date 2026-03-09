# NutriGraph — USDA Food → Nutrient Knowledge Base + GAT Embeddings

NutriGraph builds a compact **DuckDB knowledge base** from the USDA FoodData Central (Foundation Foods) dataset, then trains a **Graph Attention Network (GATv2)** on a bipartite Food↔Nutrient graph to learn **food embeddings**. The repo also generates:

- **training plots** (loss / AUC / LR)
- **embedding snapshots** at selected epochs (for UMAP progression)
- **UMAP cluster progression** plots

This codebase is a structured port of an original Colab pipeline. The intent is to reproduce the same sequence of steps and artifacts locally.

---

## Repository Layout (high level)

```
nutri_graph/
  kb/                 # build DuckDB KB from CSVs
  graph/              # construct PyG graph + metadata from DuckDB
  models/             # GATv2 model
  training/           # Colab-faithful training loop + metrics
  visualization/      # Plotly paper-style plots, UMAP utilities
scripts/
  download_data.py
  build_kb.py
  train_GAT.py
  visualize_umap_progression.py
data/
outputs/
models/
```

---

## Requirements

- Python 3.9+ recommended
- Kaggle account + Kaggle API token (for dataset download)
- PyTorch + PyTorch Geometric (PyG)

> Note: Installing PyG can vary by OS / CUDA. If `torch_geometric` import fails, follow the official installation instructions.

---

## Setup

### 1) Create and activate a virtual environment

### 2) Install dependencies

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

## End-to-End Pipeline

Run these commands from the **repo root**.

### Step 1 — Download dataset

```bash
python scripts/download_data.py
```

Expected output:
- `data/raw/` populated with USDA CSV files.

---

### Step 2 — Build DuckDB KB

```bash
python scripts/build_kb.py
```

Expected output:
- `data/nutri_kb.duckdb`

Expected tables inside DuckDB:
- `nodes_food`
- `nodes_nutrient`
- `edges_food_contains_nutrient`
- `food_index`

---

### Step 3 — Train GAT + save embeddings + save snapshots + training plots

```bash
python scripts/train_GAT.py
```

What this produces:

**Model**
- `models/gat_checkpoint.pt` (best checkpoint by val RMSE)

**Embeddings**
- `outputs/embeddings/node_embeddings.(pt|npy)`
- `outputs/embeddings/food_embeddings.npy`

**Snapshots (for UMAP progression)**
- `outputs/snapshots/vis_idx.npy`
- `outputs/snapshots/food_emb_epoch_1.npy`
- `outputs/snapshots/food_emb_epoch_30.npy`
- `outputs/snapshots/food_emb_epoch_60.npy`
- `outputs/snapshots/food_emb_epoch_90.npy`

**Training curves (paper style Plotly)**
- `outputs/training/train_loss.html` (+ `.png` if Kaleido works)
- `outputs/training/validation_regression_log1pamount.html` (+ `.png`)
- `outputs/training/validation_existence_auc.html` (+ `.png`)
- `outputs/training/learning_rate.html` (+ `.png`)
- `outputs/training/history.json`

> PNG export uses `kaleido`. HTML is always saved; PNG is best-effort.

---

### Step 4 — Generate vivid UMAP cluster progression plots (Plotly)

```bash
python scripts/visualize_umap_progression.py
```

This script mirrors the Colab visualization logic:
- UMAP is **fit once** on the reference epoch (`REF_EPOCH = max(snapshot_epochs)`)
- KMeans (MiniBatchKMeans) is **fit once** on that reference snapshot
- each epoch snapshot is transformed using the same UMAP model, then clustered using the same KMeans model

Outputs:
- `outputs/umap/umap_clusters_vivid_epoch_1.html` (+ `.png`)
- `outputs/umap/umap_clusters_vivid_epoch_30.html` (+ `.png`)
- `outputs/umap/umap_clusters_vivid_epoch_60.html` (+ `.png`)
- `outputs/umap/umap_clusters_vivid_epoch_90.html` (+ `.png`)

Plots use vivid palettes like the original Colab code.

---

## Notes on Reproducibility

You may notice embeddings/UMAP visuals shift between full runs. This is usually due to training nondeterminism (dropout, GPU kernels, random train/val/test split). Even if final metrics are similar, UMAP can change noticeably because it is sensitive to small embedding differences.

If you want strict run-to-run reproducibility, add explicit seeding + deterministic flags (planned as a follow-up improvement).

---

## Acknowledgements
- USDA FoodData Central (Foundation Foods)
- DuckDB
- PyTorch Geometric
- UMAP-learn
- Plotly
