#!/usr/bin/env python3
"""HealthyFoodSubs benchmark on the Jetson (no LLM, no embedding server needed).

Replicates the substitution evaluation protocol from Loesch et al. (2024):
  - For each test query food, rank all ground-truth substitutes against other foods
    in the same food category, then compute MAP / MRR / RR@5 / RR@10.

Three retrieval modes are evaluated against the paper baseline:
  GAT    — cosine similarity in 64-dim GAT embedding space
  Text   — cosine similarity in 1024-dim Qwen3-Embedding text space
  Hybrid — alpha * GAT_cosine + (1-alpha) * Text_cosine

Unlike the NutriBench test, this test does NOT call the chat LLM or the embedding
server — it operates entirely on precomputed `.npy` files plus the 3 HealthyFoodSubs
CSV files. Only numpy + pandas + duckdb are required.

Usage on Jetson
---------------
    python healthyfoodsubs_jetson_test.py \\
        --gat-emb   ~/nutri/data/food_embeddings.npy \\
        --text-emb  ~/nutri/data/food_text_embeddings.npy \\
        --text-ids  ~/nutri/data/food_fdc_ids.npy \\
        --db        ~/nutri/data/nutri_kb.duckdb \\
        --hs-root   ~/nutri/data/HealthyFoodSubs
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    sys.stderr.write(f"ERROR: missing dep ({e}). Run: pip install numpy pandas duckdb\n")
    sys.exit(1)


# ── Defaults aligned with the Jetson deployment layout ────────────────────────
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data"

SR_OFFSET = 1_000_000  # fdc_id = SR_OFFSET + NDB_No for SR Legacy foods


# ── Helpers ───────────────────────────────────────────────────────────────────

def uri_to_ndb(uri: str) -> int:
    return int(uri.split("#")[1])


def cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-10)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m @ q


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_hs_data(hs_root: Path):
    """Load substitution pairs, test query URIs, and category map."""
    subs_df = pd.read_csv(hs_root / "Input Data" / "final_substitution.csv", sep=";")
    test_df = pd.read_csv(hs_root / "Output" / "GAT_foods_2_test.csv")
    cat_df  = pd.read_csv(hs_root / "Input Data" / "food_category.csv")

    subs_dict: dict[str, set[str]] = defaultdict(set)
    for _, row in subs_df.iterrows():
        subs_dict[row["Food id"]].add(row["Substitution id"])

    test_uris = test_df["id"].tolist()
    category_map = dict(zip(cat_df["NDB_No"].astype(int), cat_df["FdGrp_Desc"]))
    return subs_dict, test_uris, category_map


def load_gat_embeddings(db_path: Path, emb_path: Path) -> dict:
    """Return {ndb_no: gat_vector} for SR Legacy foods.

    GAT embeddings use the DuckDB heap order (no ORDER BY), matching dataset.py.
    """
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    fdc_ids = con.execute("SELECT fdc_id FROM nodes_food").df()["fdc_id"].tolist()
    con.close()

    emb = np.load(emb_path).astype(np.float32)
    if len(fdc_ids) != len(emb):
        raise ValueError(
            f"GAT embedding count mismatch: {len(fdc_ids)} fdc_ids vs {emb.shape[0]} rows"
        )

    return {
        int(fdc_id) - SR_OFFSET: emb[i]
        for i, fdc_id in enumerate(fdc_ids)
        if int(fdc_id) > SR_OFFSET
    }


def load_text_embeddings(ids_path: Path, emb_path: Path) -> dict:
    """Return {ndb_no: text_vector} for SR Legacy foods."""
    fdc_ids = np.load(ids_path)
    emb     = np.load(emb_path).astype(np.float32)
    return {
        int(fdc_id) - SR_OFFSET: emb[i]
        for i, fdc_id in enumerate(fdc_ids)
        if int(fdc_id) > SR_OFFSET
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def _get_rank(query_emb, sub_ndb, all_ndbs, emb_matrix, sub_cat, category_map, use_filter):
    """Return 1-based rank of sub_ndb by cosine similarity to query_emb."""
    if use_filter:
        mask = np.array([category_map.get(n) == sub_cat for n in all_ndbs])
    else:
        mask = np.ones(len(all_ndbs), dtype=bool)

    if not mask.any():
        return len(all_ndbs) + 1

    cand_ndbs = [n for n, m in zip(all_ndbs, mask) if m]
    sims      = cosine_sim(query_emb, emb_matrix[mask])
    order     = np.argsort(-sims)
    for rank, idx in enumerate(order, start=1):
        if cand_ndbs[idx] == sub_ndb:
            return rank
    return len(cand_ndbs) + 1


def _compute_metrics(ranks_per_query: list) -> dict:
    maps, mrrs, rr5s, rr10s = [], [], [], []
    for ranks in ranks_per_query:
        if not ranks:
            continue
        min_rank = min(ranks)
        mrrs.append(1.0 / min_rank)
        sorted_ranks = sorted(ranks)
        precisions   = [(i + 1) / r for i, r in enumerate(sorted_ranks)]
        maps.append(float(np.mean(precisions)))
        rr5s.append(1 if min_rank <= 5 else 0)
        rr10s.append(1 if min_rank <= 10 else 0)
    return {
        "MAP":   float(np.mean(maps))  if maps  else 0.0,
        "MRR":   float(np.mean(mrrs))  if mrrs  else 0.0,
        "RR@5":  float(np.mean(rr5s))  if rr5s  else 0.0,
        "RR@10": float(np.mean(rr10s)) if rr10s else 0.0,
        "n_queries": len(maps),
    }


def evaluate_single(
    emb_dict: dict,
    test_uris: list,
    subs_dict: dict,
    category_map: dict,
    use_filter: bool = True,
) -> dict:
    all_ndbs   = sorted(emb_dict)
    emb_matrix = np.stack([emb_dict[n] for n in all_ndbs])

    ranks_per_query = []
    for query_uri in test_uris:
        q_ndb = uri_to_ndb(query_uri)
        if q_ndb not in emb_dict:
            continue
        valid_subs = [uri_to_ndb(s) for s in subs_dict.get(query_uri, set())
                      if uri_to_ndb(s) in emb_dict]
        if not valid_subs:
            continue

        query_emb = emb_dict[q_ndb]
        ranks = []
        for sub_ndb in valid_subs:
            sub_cat = category_map.get(sub_ndb)
            if sub_cat is None:
                continue
            ranks.append(_get_rank(query_emb, sub_ndb, all_ndbs, emb_matrix,
                                   sub_cat, category_map, use_filter))
        ranks_per_query.append(ranks)

    return _compute_metrics(ranks_per_query)


def evaluate_hybrid(
    gat_dict: dict,
    text_dict: dict,
    alpha: float,
    test_uris: list,
    subs_dict: dict,
    category_map: dict,
    use_filter: bool = True,
) -> dict:
    """alpha * GAT_cosine + (1-alpha) * Text_cosine."""
    shared    = sorted(set(gat_dict) & set(text_dict))
    gat_norm  = np.stack([gat_dict[n]  / (np.linalg.norm(gat_dict[n])  + 1e-10) for n in shared])
    text_norm = np.stack([text_dict[n] / (np.linalg.norm(text_dict[n]) + 1e-10) for n in shared])

    ranks_per_query = []
    for query_uri in test_uris:
        q_ndb = uri_to_ndb(query_uri)
        if q_ndb not in gat_dict or q_ndb not in text_dict:
            continue
        valid_subs = [uri_to_ndb(s) for s in subs_dict.get(query_uri, set())
                      if uri_to_ndb(s) in gat_dict and uri_to_ndb(s) in text_dict]
        if not valid_subs:
            continue

        q_gat  = gat_dict[q_ndb]  / (np.linalg.norm(gat_dict[q_ndb])  + 1e-10)
        q_text = text_dict[q_ndb] / (np.linalg.norm(text_dict[q_ndb]) + 1e-10)

        ranks = []
        for sub_ndb in valid_subs:
            sub_cat = category_map.get(sub_ndb)
            if sub_cat is None:
                continue
            if use_filter:
                mask = np.array([category_map.get(n) == sub_cat for n in shared])
            else:
                mask = np.ones(len(shared), dtype=bool)
            if not mask.any():
                continue

            cand_ndbs = [n for n, m in zip(shared, mask) if m]
            combined  = alpha * (gat_norm[mask] @ q_gat) + (1 - alpha) * (text_norm[mask] @ q_text)
            order     = np.argsort(-combined)
            for rank, idx in enumerate(order, start=1):
                if cand_ndbs[idx] == sub_ndb:
                    ranks.append(rank)
                    break
            else:
                ranks.append(len(cand_ndbs) + 1)
        ranks_per_query.append(ranks)

    return _compute_metrics(ranks_per_query)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HealthyFoodSubs benchmark on Jetson.")
    p.add_argument("--hs-root",  type=Path,  default=DEFAULT_DATA / "HealthyFoodSubs",
                   help="HealthyFoodSubs root containing 'Input Data/' and 'Output/'.")
    p.add_argument("--db",       type=Path,  default=DEFAULT_DATA / "nutri_kb.duckdb")
    p.add_argument("--gat-emb",  type=Path,  default=DEFAULT_DATA / "food_embeddings.npy")
    p.add_argument("--text-emb", type=Path,  default=DEFAULT_DATA / "food_text_embeddings.npy")
    p.add_argument("--text-ids", type=Path,  default=DEFAULT_DATA / "food_fdc_ids.npy")
    p.add_argument("--alpha",    type=float, default=0.5,
                   help="GAT weight in hybrid (default: 0.5).")
    p.add_argument("--no-text",   action="store_true",
                   help="Skip text/hybrid evaluation (GAT only).")
    p.add_argument("--no-filter", action="store_true",
                   help="Disable food category filtering (debug).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path (default: data/results/healthyfoodsubs.json).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Prerequisite checks
    required = [
        (args.db,      "nutri_kb.duckdb — build with nutri_graph/scripts/build_kb.py on dev machine"),
        (args.gat_emb, "food_embeddings.npy — produced by nutri_graph/scripts/train_GAT.py"),
        (args.hs_root / "Input Data" / "final_substitution.csv",
         "HealthyFoodSubs/Input Data/final_substitution.csv — clone the HealthyFoodSubs repo"),
        (args.hs_root / "Input Data" / "food_category.csv",
         "HealthyFoodSubs/Input Data/food_category.csv"),
        (args.hs_root / "Output" / "GAT_foods_2_test.csv",
         "HealthyFoodSubs/Output/GAT_foods_2_test.csv"),
    ]
    for path, hint in required:
        if not path.exists():
            sys.exit(f"ERROR: {path}\n  Hint: {hint}")

    use_text = not args.no_text
    if use_text:
        for path, hint in [(args.text_emb, "build_embeddings.py"),
                            (args.text_ids, "build_embeddings.py")]:
            if not path.exists():
                print(f"Warning: {path} not found ({hint}) — skipping text/hybrid modes")
                use_text = False
                break

    # Load ground truth
    print("Loading HealthyFoodSubs data...")
    subs_dict, test_uris, category_map = load_hs_data(args.hs_root)
    all_hs_uris = set(subs_dict) | {s for v in subs_dict.values() for s in v}
    n_pairs = sum(len(v) for v in subs_dict.values())
    print(f"  Ground truth pairs:  {n_pairs}")
    print(f"  Unique foods:        {len(all_hs_uris)}")
    print(f"  Test query foods:    {len(test_uris)}")

    # Load embeddings
    print("\nLoading GAT embeddings...")
    gat_emb = load_gat_embeddings(args.db, args.gat_emb)
    hs_ndbs = {uri_to_ndb(u) for u in all_hs_uris}
    gat_cov = len(hs_ndbs & set(gat_emb))
    print(f"  SR Legacy in GAT KB:      {len(gat_emb)}")
    print(f"  HealthyFoodSubs coverage: {gat_cov} / {len(hs_ndbs)} "
          f"({100 * gat_cov / len(hs_ndbs):.1f}%)")

    text_emb = None
    if use_text:
        print("\nLoading text embeddings...")
        text_emb = load_text_embeddings(args.text_ids, args.text_emb)
        txt_cov = len(hs_ndbs & set(text_emb))
        print(f"  SR Legacy in text KB:     {len(text_emb)}")
        print(f"  HealthyFoodSubs coverage: {txt_cov} / {len(hs_ndbs)} "
              f"({100 * txt_cov / len(hs_ndbs):.1f}%)")

    # Run evaluation (category filter on by default; --no-filter disables for debug)
    filter_flags = [True] if not args.no_filter else [False]
    runs = []
    for use_filter in filter_flags:
        tag = "+cat" if use_filter else "no filter"

        print(f"\nEvaluating GAT ({tag})...")
        runs.append((f"GAT    ({tag})", evaluate_single(
            gat_emb, test_uris, subs_dict, category_map, use_filter)))

        if text_emb is not None:
            print(f"Evaluating Text ({tag})...")
            runs.append((f"Text   ({tag})", evaluate_single(
                text_emb, test_uris, subs_dict, category_map, use_filter)))

            print(f"Evaluating Hybrid α={args.alpha} ({tag})...")
            runs.append((f"Hybrid ({tag})", evaluate_hybrid(
                gat_emb, text_emb, args.alpha,
                test_uris, subs_dict, category_map, use_filter)))

    # Results table
    W = 26
    print(f"\n{'=' * 70}")
    print("  FOOD SUBSTITUTION EVALUATION  vs HealthyFoodSubs (Loesch et al., 2024)")
    print(f"{'=' * 70}")
    print(f"  {'Method':<{W}}  {'MAP':>6}  {'MRR':>6}  {'RR@5':>6}  {'RR@10':>6}  {'n':>5}")
    print(f"  {'-' * 64}")
    paper_n = 183
    print(f"  {'Paper GAT (+cat) [paper]':<{W}}  "
          f"{0.345:>6.3f}  {0.549:>6.3f}  {0.680:>6.3f}  {0.757:>6.3f}  {paper_n:>5}")
    print(f"  {'-' * 64}")
    for label, m in runs:
        print(f"  {f'Ours {label}':<{W}}  "
              f"{m['MAP']:>6.3f}  {m['MRR']:>6.3f}  "
              f"{m['RR@5']:>6.3f}  {m['RR@10']:>6.3f}  {m['n_queries']:>5}")
    print(f"{'=' * 70}")

    # Save JSON results
    if args.out is None:
        args.out = DEFAULT_DATA / "results" / "healthyfoodsubs.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "hs_root": str(args.hs_root),
            "db": str(args.db),
            "gat_emb": str(args.gat_emb),
            "text_emb": str(args.text_emb) if use_text else None,
            "alpha": args.alpha,
            "use_filter": not args.no_filter,
        },
        "paper_baseline": {
            "MAP": 0.345, "MRR": 0.549, "RR@5": 0.680, "RR@10": 0.757, "n_queries": 183,
        },
        "runs": [{"method": label, "metrics": m} for label, m in runs],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
