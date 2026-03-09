import json
from pathlib import Path

import numpy as np
import torch

from nutri_graph.config import Config
from nutri_graph.graph.dataset import build_graph_from_db
from nutri_graph.models.gat_model import GATFrontEnd
from nutri_graph.training.trainer import Trainer
from nutri_graph.visualization.snapshots import SnapshotManager
from nutri_graph.visualization.training_plots import make_training_plots


SNAPSHOT_EPOCHS = [1, 30, 60, 90]   # match Colab


def ensure_dirs():
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/embeddings").mkdir(parents=True, exist_ok=True)
    Path("outputs/snapshots").mkdir(parents=True, exist_ok=True)
    Path("outputs/training").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()

    data, meta = build_graph_from_db("data/nutri_kb.duckdb")

    model = GATFrontEnd(
        num_nodes=data.num_nodes,
        num_types=2,
        emb_dim=Config.EMB_DIM,
        hidden=Config.HIDDEN,
        heads=Config.HEADS,
        dropout=Config.DROPOUT,
    )

    # ---- VIS_IDX exactly like Colab ----
    rng_vis = np.random.default_rng(0)
    VIS_N = min(6000, int(meta["NUM_FOODS"]))
    vis_idx = rng_vis.choice(int(meta["NUM_FOODS"]), size=VIS_N, replace=False)

    snapshot_mgr = SnapshotManager(vis_idx, out_dir="outputs/snapshots")

    trainer = Trainer(
        model=model,
        data=data,
        meta=meta,
        config=Config,
        snapshot_mgr=snapshot_mgr,
        snapshot_epochs=SNAPSHOT_EPOCHS,
        use_contrastive=False,  # matches your current notebook (contrastive commented out)
    )

    results = trainer.train()

    # Save history + test metrics (so plots/viz scripts can use them)
    with open("outputs/training/history.json", "w") as f:
        json.dump(results, f, indent=2)

    make_training_plots(results["history"], out_dir="outputs/training", write_png=True)

    # Save best checkpoint (val_rmse) — mirrors Colab best_state load
    torch.save(model.state_dict(), "models/gat_checkpoint.pt")

    # Save final embeddings (after best_state loaded)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        node_ids = torch.arange(data.num_nodes, device=device, dtype=torch.long)
        h = model.encode(node_ids, data.node_type, data.edge_index, data.edge_attr).detach().cpu().numpy()

    # node embeddings
    torch.save(torch.from_numpy(h), "outputs/embeddings/node_embeddings.pt")
    np.save("outputs/embeddings/node_embeddings.npy", h)

    # food embeddings only (for retrieval)
    food_emb = h[: int(meta["NUM_FOODS"])]
    np.save("outputs/embeddings/food_embeddings.npy", food_emb)

    print("Training complete.")
    print("Saved:")
    print(" - models/gat_checkpoint.pt")
    print(" - outputs/embeddings/node_embeddings.(pt|npy)")
    print(" - outputs/embeddings/food_embeddings.npy")
    print(" - outputs/snapshots/vis_idx.npy + food_emb_epoch_*.npy")
    print(" - outputs/training/history.json")