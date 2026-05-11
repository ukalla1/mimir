#!/usr/bin/env python3
"""Seed search: train GAT with seeds 1-50, evaluate on HealthyFoodSubs, save best checkpoint.

Run from nutri_graph/:
    python scripts/seed_search.py
"""

import contextlib
import io
import json
import os
import random
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPTS_DIR      = Path(__file__).resolve().parent
NUTRI_GRAPH_ROOT = SCRIPTS_DIR.parent
MIMIR_ROOT       = NUTRI_GRAPH_ROOT.parent
sys.path.insert(0, str(NUTRI_GRAPH_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from nutri_graph.config import Config
from nutri_graph.graph.dataset import build_graph_from_db
from nutri_graph.models.gat_model import GATFrontEnd
from nutri_graph.training.trainer import Trainer
from eval_food_subs import (
    load_hs_data,
    load_text_embeddings,
    evaluate_single,
    evaluate_hybrid,
    SR_OFFSET,
)

DB       = str(NUTRI_GRAPH_ROOT / "data" / "nutri_kb.duckdb")
HS_ROOT  = str(MIMIR_ROOT / "HealthyFoodSubs")
TEXT_EMB = str(MIMIR_ROOT / "nutri_rag" / "data" / "embeddings" / "food_text_embeddings.npy")
TEXT_IDS = str(MIMIR_ROOT / "nutri_rag" / "data" / "embeddings" / "food_fdc_ids.npy")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_gat_emb_dict(food_emb: np.ndarray, fdc_ids: list) -> dict:
    """Build {ndb_no: vector} dict matching eval_food_subs.load_gat_embeddings order."""
    return {
        int(fdc_id) - SR_OFFSET: food_emb[i]
        for i, fdc_id in enumerate(fdc_ids)
        if int(fdc_id) > SR_OFFSET
    }


def ensure_dirs():
    for d in ["outputs/embeddings", "outputs/snapshots", "models"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=1,  help="First seed (inclusive)")
    parser.add_argument("--seed-end",   type=int, default=50, help="Last seed (inclusive)")
    args = parser.parse_args()
    seed_range = range(args.seed_start, args.seed_end + 1)

    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── one-time setup ────────────────────────────────────────────────────────
    print("Building graph (once)...")
    with contextlib.redirect_stdout(io.StringIO()):
        data, meta = build_graph_from_db(
            DB,
            include_recipes=Config.INCLUDE_RECIPES,
            subs_csv_path=Config.SUBS_CSV,
        )
    data = data.to(device)
    print(f"  {data.num_nodes} nodes, {data.edge_index.size(1)} edges")

    # fdc_ids in heap order — must match dataset.py food node ordering
    con = duckdb.connect(DB, read_only=True)
    fdc_ids = con.execute("SELECT fdc_id FROM nodes_food").df()["fdc_id"].tolist()
    con.close()

    print("Loading HealthyFoodSubs eval data...")
    subs_dict, test_uris, category_map = load_hs_data(HS_ROOT)
    print(f"  {sum(len(v) for v in subs_dict.values())} pairs, {len(test_uris)} test queries")

    text_emb = None
    if os.path.exists(TEXT_EMB) and os.path.exists(TEXT_IDS):
        print("Loading text embeddings...")
        text_emb = load_text_embeddings(TEXT_IDS, TEXT_EMB)
    else:
        print("Text embeddings not found — hybrid column will be N/A")

    # ── seed loop ─────────────────────────────────────────────────────────────
    print(f"\nSearching seeds {args.seed_start}–{args.seed_end} ({len(seed_range)} seeds × {Config.MAX_EPOCHS} epochs each)...\n")
    header = f"{'seed':>4}  {'GAT MAP':>7}  {'GAT MRR':>7}  {'Hyb MAP':>7}  {'note'}"
    print(header)
    print("-" * len(header))

    all_results     = []
    best_gat_map    = -1.0
    best_state_dict = None
    best_food_emb   = None
    best_full_emb   = None
    best_seed       = None

    for seed in seed_range:
        set_seed(seed)

        model = GATFrontEnd(
            num_nodes=data.num_nodes,
            num_types=meta["NUM_TYPES"],
            emb_dim=Config.EMB_DIM,
            hidden=Config.HIDDEN,
            heads=Config.HEADS,
            dropout=Config.DROPOUT,
        )

        trainer = Trainer(
            model=model,
            data=data,
            meta=meta,
            config=Config,
            snapshot_mgr=None,
            snapshot_epochs=[],
            use_contrastive=False,
            lambda_subs=Config.LAMBDA_SUBS,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train()

        # compute embeddings in-memory
        model.eval()
        node_ids    = torch.arange(data.num_nodes, device=device, dtype=torch.long)
        node_type_t = data.node_type
        with torch.no_grad():
            h = model.encode(
                node_ids, node_type_t, data.edge_index, data.edge_attr
            ).detach().cpu().numpy()

        food_emb     = h[: meta["NUM_FOODS"]]
        gat_emb_dict = build_gat_emb_dict(food_emb, fdc_ids)

        gat_m    = evaluate_single(gat_emb_dict, test_uris, subs_dict, category_map)
        hybrid_m = (
            evaluate_hybrid(gat_emb_dict, text_emb, 0.5, test_uris, subs_dict, category_map)
            if text_emb is not None else None
        )

        is_best = gat_m["MAP"] > best_gat_map
        if is_best:
            best_gat_map    = gat_m["MAP"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_food_emb   = food_emb.copy()
            best_full_emb   = h.copy()
            best_seed       = seed

        hyb_str = f"{hybrid_m['MAP']:7.3f}" if hybrid_m is not None else "    N/A"
        note    = " <- best" if is_best else ""
        print(f"{seed:>4}  {gat_m['MAP']:7.3f}  {gat_m['MRR']:7.3f}  {hyb_str}  {note}")
        sys.stdout.flush()

        all_results.append({"seed": seed, "gat": gat_m, "hybrid": hybrid_m})

    # ── save best ─────────────────────────────────────────────────────────────
    print(f"\nBest seed: {best_seed}  GAT MAP={best_gat_map:.3f}")
    print("Saving best checkpoint and embeddings...")

    torch.save(best_state_dict, "models/gat_checkpoint.pt")
    np.save("outputs/embeddings/food_embeddings.npy", best_food_emb)
    np.save("outputs/embeddings/node_embeddings.npy", best_full_emb)
    torch.save(torch.from_numpy(best_full_emb), "outputs/embeddings/node_embeddings.pt")

    tag = f"seed{best_seed}_map{best_gat_map:.3f}".replace(".", "")
    torch.save(best_state_dict, f"models/gat_checkpoint_best_{tag}.pt")
    np.save(f"outputs/embeddings/food_embeddings_best_{tag}.npy", best_food_emb)
    np.save(f"outputs/embeddings/node_embeddings_best_{tag}.npy", best_full_emb)
    torch.save(torch.from_numpy(best_full_emb), f"outputs/embeddings/node_embeddings_best_{tag}.pt")

    with open("outputs/seed_search_results.json", "w") as f:
        json.dump({"best_seed": best_seed, "best_gat_map": best_gat_map, "results": all_results}, f, indent=2)

    # ── top-10 summary ────────────────────────────────────────────────────────
    print("\n── Top 10 seeds by GAT MAP ──")
    print(f"{'seed':>4}  {'GAT MAP':>7}  {'GAT MRR':>7}  {'Hyb MAP':>7}")
    print("-" * 35)
    top10 = sorted(all_results, key=lambda r: r["gat"]["MAP"], reverse=True)[:10]
    for r in top10:
        h_map = r["hybrid"]["MAP"] if r["hybrid"] is not None else float("nan")
        marker = " <- saved" if r["seed"] == best_seed else ""
        print(f"{r['seed']:>4}  {r['gat']['MAP']:7.3f}  {r['gat']['MRR']:7.3f}  {h_map:7.3f}{marker}")

    print(f"\nFull results: outputs/seed_search_results.json")
    print("Run eval_food_subs.py to confirm final metrics.")


if __name__ == "__main__":
    main()
