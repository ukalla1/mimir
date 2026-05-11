#!/usr/bin/env python3
"""Evaluate Mimir embeddings on the HealthyFoodSubs food substitution benchmark.

Replicates the evaluation protocol from Loesch et al. (2024):
  - Rank each ground-truth substitute within foods sharing its food category
  - Metrics: MAP, MRR, RR@5, RR@10

Three retrieval modes are compared against the paper baseline:
  GAT    — 64-dim GATv2 trained on nutrient amount regression (zero-shot transfer)
  Text   — 1024-dim Qwen3-Embedding
  Hybrid — alpha * GAT_cosine + (1-alpha) * Text_cosine

Usage:
    python scripts/eval_food_subs.py
    python scripts/eval_food_subs.py --no-text          # GAT only
    python scripts/eval_food_subs.py --alpha 0.3        # custom hybrid weight
    python scripts/eval_food_subs.py --no-filter        # skip category filtering
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
NUTRI_GRAPH_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MIMIR_ROOT       = os.path.abspath(os.path.join(NUTRI_GRAPH_ROOT, ".."))

DEFAULT_HS_ROOT  = os.path.join(MIMIR_ROOT, "HealthyFoodSubs")
DEFAULT_DB       = os.path.join(NUTRI_GRAPH_ROOT, "data", "nutri_kb.duckdb")
DEFAULT_GAT_EMB  = os.path.join(NUTRI_GRAPH_ROOT, "outputs", "embeddings", "food_embeddings.npy")
DEFAULT_TEXT_EMB = os.path.join(MIMIR_ROOT, "nutri_rag", "data", "embeddings", "food_text_embeddings.npy")
DEFAULT_TEXT_IDS = os.path.join(MIMIR_ROOT, "nutri_rag", "data", "embeddings", "food_fdc_ids.npy")

SR_OFFSET = 1_000_000  # fdc_id = SR_OFFSET + NDB_No for SR Legacy foods


# ── Helpers ───────────────────────────────────────────────────────────────────

def uri_to_ndb(uri: str) -> int:
    return int(uri.split("#")[1])


def cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-10)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m @ q


# ── Data loading ──────────────────────────────────────────────────────────────

def load_hs_data(hs_root: str):
    subs_df = pd.read_csv(os.path.join(hs_root, "Input Data", "final_substitution.csv"), sep=";")
    test_df = pd.read_csv(os.path.join(hs_root, "Output", "GAT_foods_2_test.csv"))
    cat_df  = pd.read_csv(os.path.join(hs_root, "Input Data", "food_category.csv"))

    subs_dict: dict[str, set[str]] = defaultdict(set)
    for _, row in subs_df.iterrows():
        subs_dict[row["Food id"]].add(row["Substitution id"])

    test_uris    = test_df["id"].tolist()
    category_map = dict(zip(cat_df["NDB_No"].astype(int), cat_df["FdGrp_Desc"]))
    return subs_dict, test_uris, category_map


def load_gat_embeddings(db_path: str, emb_path: str) -> dict[int, np.ndarray]:
    """Return {ndb_no: gat_vector} for SR Legacy foods.

    GAT embeddings use the DuckDB heap order (no ORDER BY), matching dataset.py.
    """
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    fdc_ids = con.execute("SELECT fdc_id FROM nodes_food").df()["fdc_id"].tolist()
    con.close()

    emb = np.load(emb_path).astype(np.float32)
    if len(fdc_ids) != len(emb):
        raise ValueError(f"GAT embedding count mismatch: {len(fdc_ids)} fdc_ids vs {emb.shape[0]} rows")

    return {
        int(fdc_id) - SR_OFFSET: emb[i]
        for i, fdc_id in enumerate(fdc_ids)
        if int(fdc_id) > SR_OFFSET
    }


def load_text_embeddings(ids_path: str, emb_path: str) -> dict[int, np.ndarray]:
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


def _compute_metrics(ranks_per_query: list[list[int]]) -> dict:
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
    emb_dict: dict[int, np.ndarray],
    test_uris: list[str],
    subs_dict: dict[str, set[str]],
    category_map: dict[int, str],
    use_filter: bool = True,
) -> dict:
    all_ndbs   = sorted(emb_dict)
    emb_matrix = np.stack([emb_dict[n] for n in all_ndbs])

    ranks_per_query = []
    for query_uri in test_uris:
        q_ndb = uri_to_ndb(query_uri)
        if q_ndb not in emb_dict:
            continue
        valid_subs = [uri_to_ndb(s) for s in subs_dict.get(query_uri, set()) if uri_to_ndb(s) in emb_dict]
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
    gat_dict: dict[int, np.ndarray],
    text_dict: dict[int, np.ndarray],
    alpha: float,
    test_uris: list[str],
    subs_dict: dict[str, set[str]],
    category_map: dict[int, str],
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


def evaluate_v3_style(
    gat_dict: dict[int, np.ndarray],
    text_dict: dict[int, np.ndarray],
    top_k_text: int,
    gat_neighbors: int,
    test_uris: list[str],
    subs_dict: dict[str, set[str]],
    category_map: dict[int, str],
    use_filter: bool = True,
) -> dict:
    """V3-style: text retrieval → GAT expansion → re-rank by text (mirrors nutri_rag pipeline).

    1. Score same-category foods by text cosine similarity, keep top_k_text
    2. For each, find gat_neighbors nearest GAT neighbors in the same category
    3. Merge into expanded pool (deduplicated)
    4. Re-rank entire pool by text cosine similarity only
    5. Find rank of each valid substitute in this re-ranked list
    """
    shared     = sorted(set(gat_dict) & set(text_dict))
    gat_norm   = np.stack([gat_dict[n]  / (np.linalg.norm(gat_dict[n])  + 1e-10) for n in shared])
    text_norm  = np.stack([text_dict[n] / (np.linalg.norm(text_dict[n]) + 1e-10) for n in shared])
    ndb_to_idx = {n: i for i, n in enumerate(shared)}

    ranks_per_query = []
    for query_uri in test_uris:
        q_ndb = uri_to_ndb(query_uri)
        if q_ndb not in gat_dict or q_ndb not in text_dict:
            continue
        valid_subs = [uri_to_ndb(s) for s in subs_dict.get(query_uri, set())
                      if uri_to_ndb(s) in gat_dict and uri_to_ndb(s) in text_dict]
        if not valid_subs:
            continue

        q_gat  = gat_norm[ndb_to_idx[q_ndb]]
        q_text = text_norm[ndb_to_idx[q_ndb]]

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

            cat_ndbs   = [n for n, m in zip(shared, mask) if m]
            cat_text   = text_norm[mask]
            cat_gat    = gat_norm[mask]

            # Step 1 — text retrieval: top_k_text by text similarity
            text_sims  = cat_text @ q_text
            top_k      = min(top_k_text, len(cat_ndbs))
            top_idx    = np.argpartition(-text_sims, top_k - 1)[:top_k]
            seed_set   = set(top_idx.tolist())

            # Step 2 — GAT expansion: find gat_neighbors for each seed
            expanded   = set(seed_set)
            for si in seed_set:
                gat_sims   = cat_gat @ cat_gat[si]
                k_neigh    = min(gat_neighbors + 1, len(cat_ndbs))
                neigh_idx  = np.argpartition(-gat_sims, k_neigh - 1)[:k_neigh]
                expanded.update(neigh_idx.tolist())

            # Step 3 — re-rank expanded pool by text similarity only
            exp_list   = sorted(expanded)
            exp_ndbs   = [cat_ndbs[i] for i in exp_list]
            exp_text   = cat_text[exp_list]
            exp_scores = exp_text @ q_text
            order      = np.argsort(-exp_scores)

            for rank, idx in enumerate(order, start=1):
                if exp_ndbs[idx] == sub_ndb:
                    ranks.append(rank)
                    break
            else:
                # substitute not in expanded pool — assign worst rank
                ranks.append(len(cat_ndbs) + 1)

        ranks_per_query.append(ranks)

    return _compute_metrics(ranks_per_query)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mimir on HealthyFoodSubs benchmark")
    parser.add_argument("--hs-root",   default=DEFAULT_HS_ROOT,   help="HealthyFoodSubs root dir")
    parser.add_argument("--db",        default=DEFAULT_DB,         help="Path to nutri_kb.duckdb")
    parser.add_argument("--gat-emb",   default=DEFAULT_GAT_EMB,    help="GAT food_embeddings.npy")
    parser.add_argument("--text-emb",  default=DEFAULT_TEXT_EMB,   help="Text food_text_embeddings.npy")
    parser.add_argument("--text-ids",  default=DEFAULT_TEXT_IDS,   help="Text food_fdc_ids.npy")
    parser.add_argument("--alpha",     type=float, default=0.5,    help="GAT weight in hybrid (default: 0.5)")
    parser.add_argument("--no-text",      action="store_true",      help="Skip text/hybrid evaluation")
    parser.add_argument("--no-filter",    action="store_true",      help="Disable food category filtering")
    parser.add_argument("--top-k-text",   type=int, default=20,     help="Initial text candidates for V3-style (default: 20)")
    parser.add_argument("--gat-neighbors",type=int, default=5,      help="GAT neighbors per candidate for V3-style (default: 5)")
    args = parser.parse_args()

    # Prerequisite checks
    required = [
        (args.db,      "nutri_kb.duckdb — run nutri_graph/scripts/build_kb.py first"),
        (args.gat_emb, "food_embeddings.npy — run nutri_graph/scripts/train_GAT.py first"),
        (os.path.join(args.hs_root, "Input Data", "final_substitution.csv"),
         "HealthyFoodSubs/Input Data/final_substitution.csv — clone the HealthyFoodSubs repo"),
    ]
    for path, hint in required:
        if not os.path.exists(path):
            print(f"ERROR: {path}\n  Hint: {hint}")
            sys.exit(1)

    use_text = not args.no_text
    if use_text:
        for path, hint in [(args.text_emb, "build_embeddings.py"), (args.text_ids, "build_embeddings.py")]:
            if not os.path.exists(path):
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
    print(f"  HealthyFoodSubs coverage: {gat_cov} / {len(hs_ndbs)} ({100*gat_cov/len(hs_ndbs):.1f}%)")

    text_emb = None
    if use_text:
        print("\nLoading text embeddings...")
        text_emb = load_text_embeddings(args.text_ids, args.text_emb)
        txt_cov = len(hs_ndbs & set(text_emb))
        print(f"  SR Legacy in text KB:     {len(text_emb)}")
        print(f"  HealthyFoodSubs coverage: {txt_cov} / {len(hs_ndbs)} ({100*txt_cov/len(hs_ndbs):.1f}%)")

    # Build evaluation runs (category filter always on; --no-filter disables for debugging)
    filter_flags = [True] if not args.no_filter else [False]
    runs: list[tuple[str, dict]] = []

    for use_filter in filter_flags:
        tag = "+cat" if use_filter else "no filter"
        print(f"\nEvaluating GAT ({tag})...")
        runs.append((f"GAT    ({tag})", evaluate_single(gat_emb, test_uris, subs_dict, category_map, use_filter)))

        if text_emb is not None:
            print(f"Evaluating Text ({tag})...")
            runs.append((f"Text   ({tag})", evaluate_single(text_emb, test_uris, subs_dict, category_map, use_filter)))

            print(f"Evaluating Hybrid α={args.alpha} ({tag})...")
            runs.append((f"Hybrid ({tag})", evaluate_hybrid(gat_emb, text_emb, args.alpha,
                                                             test_uris, subs_dict, category_map, use_filter)))


    # Results table
    W = 26
    print(f"\n{'='*70}")
    print("  FOOD SUBSTITUTION EVALUATION  vs HealthyFoodSubs (Loesch et al., 2024)")
    print(f"{'='*70}")
    print(f"  {'Method':<{W}}  {'MAP':>6}  {'MRR':>6}  {'RR@5':>6}  {'RR@10':>6}  {'n':>5}")
    print(f"  {'-'*64}")
    # Paper baseline (Table 3b, 5-fold CV, GAT + category filter)
    paper_n = 183
    print(f"  {'Paper GAT (+cat) [paper]':<{W}}  {0.345:>6.3f}  {0.549:>6.3f}  {0.680:>6.3f}  {0.757:>6.3f}  {paper_n:>5}")
    print(f"  {'-'*64}")
    for label, m in runs:
        print(f"  {f'Ours {label}':<{W}}  {m['MAP']:>6.3f}  {m['MRR']:>6.3f}  "
              f"{m['RR@5']:>6.3f}  {m['RR@10']:>6.3f}  {m['n_queries']:>5}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
