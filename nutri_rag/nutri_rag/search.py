"""Semantic vector search + nutrient lookup for RAG retrieval.

Uses pre-computed Qwen3-Embedding vectors for cosine similarity search
instead of BM25 keyword matching. This handles vocabulary mismatches
like "groundnut" vs "peanut", "maize flour" vs "corn flour", etc.
"""

from __future__ import annotations

from typing import Callable

import duckdb
import numpy as np
import pandas as pd

from nutri_rag.config import DB_PATH, KEY_NUTRIENTS, TOP_K_FOODS
from nutri_rag.embedding import (
    FOOD_SEARCH_INSTRUCTION,
    FoodVectorIndex,
    GATIndex,
    TextEmbedder,
)

# ── Module-level singletons (lazy init) ──────────────────────────────

_embedder: TextEmbedder | None = None
_index: FoodVectorIndex | None = None
_gat_index: GATIndex | None = None
_kb_con: duckdb.DuckDBPyConnection | None = None


def _get_embedder() -> TextEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedder()
    return _embedder


def _get_index() -> FoodVectorIndex:
    global _index
    if _index is None:
        _index = FoodVectorIndex()
    return _index


def _get_gat_index() -> GATIndex:
    global _gat_index
    if _gat_index is None:
        _gat_index = GATIndex()
    return _gat_index


def _get_kb(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    global _kb_con
    if _kb_con is None:
        _kb_con = duckdb.connect(db_path, read_only=True)
    return _kb_con


def _get_description(fdc_id: int, db_path: str = DB_PATH) -> str:
    """Look up the food description for an fdc_id."""
    con = _get_kb(db_path)
    result = con.execute(
        f"SELECT description FROM nodes_food WHERE fdc_id = {int(fdc_id)}"
    ).fetchone()
    return result[0] if result else ""


def search_food(
    con: duckdb.DuckDBPyConnection | None,
    query: str,
    k: int = TOP_K_FOODS,
    db_path: str = DB_PATH,
    use_gat: bool = False,
) -> pd.DataFrame:
    """Search for foods matching a text query using semantic vector search.

    Returns DataFrame with columns [fdc_id, description].

    Args:
        use_gat: If True, apply GAT expansion — find nutritionally similar
                 neighbors of text candidates to expand the search pool (V2).

    The `con` parameter is accepted for backward compatibility but
    the vector search uses its own singletons internally.
    """
    embedder = _get_embedder()
    index = _get_index()

    # Encode query with task instruction for better retrieval
    query_vec = embedder.encode(
        [query], task_instruction=FOOD_SEARCH_INSTRUCTION
    )  # (1, dim)

    # Retrieve more candidates than needed, then filter by macro data
    n_candidates = k * 5
    results = index.search(query_vec, k=n_candidates)

    if not results or not results[0]:
        return pd.DataFrame(columns=["fdc_id", "description"])

    candidates = results[0]  # list of (fdc_id, score, array_idx)

    # Build result dataframe
    rows = []
    for fdc_id, score, arr_idx in candidates:
        desc = _get_description(fdc_id, db_path)
        rows.append({
            "fdc_id": fdc_id, "description": desc,
            "text_score": score, "arr_idx": arr_idx,
        })
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=["fdc_id", "description"])

    if use_gat and len(df) > 1:
        df = _gat_expand(df, query_vec, index, db_path)

    # Look up macro counts for all candidates in current pool
    all_fdc_ids = df["fdc_id"].tolist()
    macro_counts = _get_macro_counts(all_fdc_ids, db_path)
    df = df.merge(macro_counts, on="fdc_id", how="left")
    df["macro_count"] = df["macro_count"].fillna(0)

    # Prefer entries with macros, then by text score
    df = df.sort_values(["macro_count", "text_score"], ascending=[False, False])
    df = df.head(k)
    return df[["fdc_id", "description", "text_score"]].reset_index(drop=True)


def _get_macro_counts(fdc_ids: list[int], db_path: str = DB_PATH) -> pd.DataFrame:
    """Count how many macronutrients (carb/protein/fat) each food has."""
    if not fdc_ids:
        return pd.DataFrame(columns=["fdc_id", "macro_count"])
    kb = _get_kb(db_path)
    placeholders = ", ".join(str(int(fid)) for fid in fdc_ids)
    return kb.execute(f"""
        SELECT e.fdc_id, COUNT(*) AS macro_count
        FROM edges_food_contains_nutrient e
        JOIN nodes_nutrient n USING(nutrient_id)
        WHERE e.fdc_id IN ({placeholders})
          AND n.nutrient_name IN (
              'Carbohydrate, by difference', 'Protein', 'Total lipid (fat)'
          )
        GROUP BY e.fdc_id
    """).df()


# ── GAT Expansion parameters ─────────────────────────────────────────

GAT_N_UNIQUE = 5          # number of unique text candidates to expand
GAT_NEIGHBORS_PER = 5     # GAT neighbors per unique candidate


def _gat_expand(
    df: pd.DataFrame,
    query_vec: np.ndarray,
    index: FoodVectorIndex,
    db_path: str = DB_PATH,
    n_unique: int = GAT_N_UNIQUE,
    gat_neighbors: int = GAT_NEIGHBORS_PER,
) -> pd.DataFrame:
    """Expand text candidates with GAT nutritional neighbors, then re-score.

    Pipeline:
    1. Dedup text candidates by description → top-N unique
    2. For each unique candidate, find M GAT neighbors
    3. Collect all neighbors into expanded pool (keep all, no dedup)
    4. Re-score neighbors against original query using text embedding
    5. Return combined pool (original + neighbors) with text scores
    """
    gat = _get_gat_index()

    # Step 1: dedup by description, keep top-N unique
    unique_df = df.sort_values("text_score", ascending=False).drop_duplicates(
        subset="description"
    ).head(n_unique)

    # Step 2: for each unique candidate, find GAT neighbors
    existing_indices = set(df["arr_idx"].tolist())
    neighbor_rows = []

    for _, row in unique_df.iterrows():
        neighbors = gat.neighbors(int(row["arr_idx"]), k=gat_neighbors)
        for neigh_idx, gat_sim in neighbors:
            if neigh_idx not in existing_indices:
                neighbor_rows.append({
                    "arr_idx": neigh_idx,
                    "gat_sim": gat_sim,
                    "parent_desc": row["description"],
                })

    if not neighbor_rows:
        # No new neighbors found — return original df unchanged
        return df

    # Step 3: look up fdc_id and description for each neighbor
    for nr in neighbor_rows:
        nr["fdc_id"] = int(index.fdc_ids[nr["arr_idx"]])
        nr["description"] = _get_description(nr["fdc_id"], db_path)

    # Step 4: compute text similarity for neighbors against original query
    neighbor_indices = [nr["arr_idx"] for nr in neighbor_rows]
    neighbor_vecs = index.embeddings[neighbor_indices]  # (M, dim)
    text_scores = (query_vec @ neighbor_vecs.T).flatten()  # (M,)

    for nr, score in zip(neighbor_rows, text_scores):
        nr["text_score"] = float(score)

    # Step 5: combine original candidates + neighbors
    neighbor_df = pd.DataFrame(neighbor_rows)[["fdc_id", "description", "text_score", "arr_idx"]]
    combined = pd.concat([df[["fdc_id", "description", "text_score", "arr_idx"]], neighbor_df],
                         ignore_index=True)

    return combined


# ── Unified hybrid retrieval primitive (used by NutriBench v4/v5,
#    robot eaten-side, robot food-neighbor expansion, robot recommend_v2) ──

_fdc_to_arr_idx: dict[int, int] | None = None


def _get_fdc_to_arr_idx() -> dict[int, int]:
    """fdc_id → arr_idx mapping. FoodVectorIndex.fdc_ids is arr_idx → fdc_id.

    FoodVectorIndex and GATIndex share the same arr_idx ordering, so the
    same dict works for both text and GAT lookups.
    """
    global _fdc_to_arr_idx
    if _fdc_to_arr_idx is None:
        index = _get_index()
        _fdc_to_arr_idx = {int(fid): i for i, fid in enumerate(index.fdc_ids)}
    return _fdc_to_arr_idx


def hybrid_rank(
    q_text: np.ndarray | None = None,
    q_gat: np.ndarray | None = None,
    candidate_fdc_ids: list[int] | None = None,
    alpha: float = 0.5,
    structured_filter: Callable[[int], bool] | None = None,
    structured_score: Callable[[int], float] | None = None,
    structured_weight: float = 0.0,
    k: int = 5,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Unified score-fusion retrieval over the food index.

    Score per candidate x:
        s(x) = alpha · cos(q_gat,  x_gat)        (if q_gat  is not None)
             + (1-alpha) · cos(q_text, x_text)   (if q_text is not None)
             + structured_weight · structured_score(x)   (if provided)

    When only one of (q_text, q_gat) is provided, alpha is overridden so the
    available vector dominates:
        q_text only → effective alpha = 0  (pure text)
        q_gat  only → effective alpha = 1  (pure GAT)

    Args:
        q_text: 1-d text query vector (1024-d, L2-normalized) or None.
        q_gat:  1-d GAT  query vector (64-d,   L2-normalized) or None.
        candidate_fdc_ids: optional restriction of the candidate pool.
        alpha: GAT weight in [0, 1]; ignored when only one of q_* is given.
        structured_filter: returns False to exclude a candidate.
        structured_score:  additive structured term (range ~[0, 1] expected).
        structured_weight: weight on the structured term (default 0 → unused).
        k: number of top-k candidates to return.

    Returns DataFrame with columns [fdc_id, description, text_sim, gat_sim,
    struct, total], sorted by total desc, k rows.
    """
    if q_text is None and q_gat is None:
        raise ValueError("hybrid_rank requires at least one of q_text or q_gat")

    text_index = _get_index() if q_text is not None else None
    gat_index = _get_gat_index() if q_gat is not None else None

    # Determine candidate arr_idx set
    if candidate_fdc_ids is not None:
        fdc_to_arr = _get_fdc_to_arr_idx()
        arr_idxs = np.array(
            [fdc_to_arr[int(fid)] for fid in candidate_fdc_ids if int(fid) in fdc_to_arr],
            dtype=np.int64,
        )
        if arr_idxs.size == 0:
            return pd.DataFrame(columns=["fdc_id", "description", "text_sim", "gat_sim", "struct", "total"])
    else:
        ref = text_index if text_index is not None else gat_index
        arr_idxs = np.arange(ref.embeddings.shape[0], dtype=np.int64)

    # Compute per-space similarities
    if q_text is not None:
        text_sims = text_index.embeddings[arr_idxs] @ q_text  # (M,)
    else:
        text_sims = np.zeros(arr_idxs.shape, dtype=np.float32)

    if q_gat is not None:
        gat_sims = gat_index.embeddings[arr_idxs] @ q_gat     # (M,)
    else:
        gat_sims = np.zeros(arr_idxs.shape, dtype=np.float32)

    # Effective alpha when one vector is missing
    if q_text is None:
        eff_alpha = 1.0
    elif q_gat is None:
        eff_alpha = 0.0
    else:
        eff_alpha = float(alpha)

    sim_scores = eff_alpha * gat_sims + (1.0 - eff_alpha) * text_sims

    # Resolve fdc_id per arr_idx. Both FoodVectorIndex and GATIndex share
    # the same arr_idx ordering (nodes_food heap order), but only
    # FoodVectorIndex stores fdc_ids — load it as the canonical mapping.
    fdc_ids = np.asarray(_get_index().fdc_ids)[arr_idxs].astype(np.int64)

    # Structured filter as hard mask
    if structured_filter is not None:
        mask = np.array([bool(structured_filter(int(fid))) for fid in fdc_ids])
        if not mask.any():
            return pd.DataFrame(columns=["fdc_id", "description", "text_sim", "gat_sim", "struct", "total"])
        arr_idxs = arr_idxs[mask]
        fdc_ids = fdc_ids[mask]
        text_sims = text_sims[mask]
        gat_sims = gat_sims[mask]
        sim_scores = sim_scores[mask]

    # Structured additive score
    if structured_score is not None and structured_weight != 0.0:
        struct = np.array([float(structured_score(int(fid))) for fid in fdc_ids])
    else:
        struct = np.zeros(fdc_ids.shape, dtype=np.float32)

    total = sim_scores + structured_weight * struct

    # Top-k
    k = int(min(k, len(total)))
    if k <= 0:
        return pd.DataFrame(columns=["fdc_id", "description", "text_sim", "gat_sim", "struct", "total"])

    top_local = np.argpartition(total, -k)[-k:]
    top_local = top_local[np.argsort(total[top_local])[::-1]]

    rows = []
    for j in top_local:
        fid = int(fdc_ids[j])
        rows.append({
            "fdc_id": fid,
            "description": _get_description(fid, db_path),
            "text_sim": float(text_sims[j]),
            "gat_sim": float(gat_sims[j]),
            "struct": float(struct[j]),
            "total": float(total[j]),
        })
    return pd.DataFrame(rows)


def search_food_v2(
    query: str,
    mode: str = "hybrid",
    k: int = TOP_K_FOODS,
    alpha: float = 0.5,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Food→nutrition retrieval with unified text / gat / hybrid modes.

    Uses a text-bootstrapped pseudo-anchor for gat and hybrid modes, since
    the query is free text and has no native GAT vector:

        q_text = embed(query)
        seed   = text top-1 candidate
        q_gat* = GAT[seed]

    Modes:
      "text"   — text cosine only           (NutriBench v1 equivalent)
      "gat"    — pure GAT via pseudo-anchor (NutriBench v4, new)
      "hybrid" — alpha·gat + (1-alpha)·text via pseudo-anchor (NutriBench v5, new)

    Returns a DataFrame with the same shape as search_food (fdc_id,
    description, text_score) for downstream compatibility — the text_score
    column is the fused score for gat/hybrid modes.
    """
    embedder = _get_embedder()
    q_text = embedder.encode([query], task_instruction=FOOD_SEARCH_INSTRUCTION)[0]  # (dim,)

    if mode == "text":
        df = hybrid_rank(q_text=q_text, q_gat=None, alpha=0.0, k=k, db_path=db_path)
    else:
        # Pseudo-anchor: pick text top-1 to obtain a GAT query vector
        text_index = _get_index()
        gat_index = _get_gat_index()
        seed_results = text_index.search(q_text[None, :], k=1)
        if not seed_results or not seed_results[0]:
            # Fall back to text mode if no candidate found
            df = hybrid_rank(q_text=q_text, q_gat=None, alpha=0.0, k=k, db_path=db_path)
        else:
            _, _, seed_arr_idx = seed_results[0][0]
            q_gat_star = gat_index.embeddings[seed_arr_idx]
            if mode == "gat":
                df = hybrid_rank(q_text=None, q_gat=q_gat_star, alpha=1.0, k=k, db_path=db_path)
            elif mode == "hybrid":
                df = hybrid_rank(q_text=q_text, q_gat=q_gat_star, alpha=alpha, k=k, db_path=db_path)
            else:
                raise ValueError(f"unknown mode: {mode!r} (expected text/gat/hybrid)")

    if df.empty:
        return pd.DataFrame(columns=["fdc_id", "description", "text_score"])

    # Match search_food's output schema for downstream call sites
    out = df[["fdc_id", "description", "total"]].rename(columns={"total": "text_score"})
    return out.reset_index(drop=True)


# ── Recipe-side hybrid rank (Phase D) ─────────────────────────────────
#
# Mirrors hybrid_rank's role but over the RecipeVectorIndex. Used by:
#   - assistant/meal_recommender.MealRecommender (robot pipeline)
#   - scripts/run_pfoodreq_bench.py --retrieval-style embedding_first (testing)
# Both call sites share this function so a regression here shows up in both
# the robot's meal suggestions and PFoodReq scores.

_recipe_index = None


def _get_recipe_index():
    """Lazy-init RecipeVectorIndex singleton (deferred until first recipe call)."""
    global _recipe_index
    if _recipe_index is None:
        # Import here to avoid loading recipe embeddings unless needed
        from nutri_rag.embedding import RecipeVectorIndex
        _recipe_index = RecipeVectorIndex()
    return _recipe_index


def hybrid_rank_recipes(
    q_text: np.ndarray,
    q_gat: np.ndarray | None = None,
    candidate_recipe_ids: list[int] | None = None,
    alpha: float = 0.5,
    structured_score: Callable[[int], float] | None = None,
    structured_weight: float = 0.0,
    structured_filter: Callable[[int], bool] | None = None,
    k: int = 30,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Recipe-store analog of hybrid_rank.

    Same algorithmic shape as Phase A's search_food_v2, Phase B's
    _hybrid_neighbors, and Phase C's recommend_v2 — just over the
    RecipeVectorIndex instead of the food index.

    Args:
        q_text: (text_dim,) text query vector (L2-normalized).
        q_gat:  optional graph-space query vector. When provided, uses the
                'external_gat' mode of RecipeVectorIndex.search_by_ids —
                recipe-GAT is in the same node-embedding space as food-GAT,
                so callers can pass mean(GAT_food[recommended_fdc_ids])
                directly. When None, falls back to the pseudo-anchor 'hybrid'
                mode (text-top-1 within pool → q_gat*).
        candidate_recipe_ids: optional restriction. None = score all recipes.
        alpha: GAT weight; (1-alpha) is the text weight.
        structured_score: optional per-recipe additive term in [-1, 1].
        structured_weight: weight on the structured term (default 0 = unused).
        structured_filter: optional hard mask (returns False to exclude).
        k: top-k to return.

    Returns DataFrame: [recipe_id, text_sim, gat_sim, struct, total].
    """
    index = _get_recipe_index()

    # Step 1: candidate pool
    if candidate_recipe_ids is None:
        valid_ids = [int(rid) for rid in index.recipe_ids]
    else:
        valid_ids = [int(rid) for rid in candidate_recipe_ids
                     if int(rid) in index._id_to_idx]
    if not valid_ids:
        return pd.DataFrame(columns=["recipe_id", "text_sim", "gat_sim", "struct", "total"])

    # Step 2: hard filter
    if structured_filter is not None:
        valid_ids = [rid for rid in valid_ids if structured_filter(rid)]
        if not valid_ids:
            return pd.DataFrame(columns=["recipe_id", "text_sim", "gat_sim", "struct", "total"])

    # Step 3: delegate scoring to RecipeVectorIndex.search_by_ids
    # We pull more than k from the scorer so structured_score can re-rank.
    fetch_k = max(k * 3, k) if structured_score is not None else k
    mode = "external_gat" if q_gat is not None else "hybrid"
    raw = index.search_by_ids(
        query_vector=q_text,
        candidate_recipe_ids=valid_ids,
        k=fetch_k,
        lam=alpha,
        mode=mode,
        q_gat=q_gat,
    )
    # raw is list of (recipe_id, combined, text, gat)

    rows = []
    for rid, combined, text_sim, gat_sim in raw:
        struct = float(structured_score(rid)) if structured_score is not None else 0.0
        total = combined + structured_weight * struct
        rows.append({
            "recipe_id": int(rid),
            "text_sim": float(text_sim),
            "gat_sim": float(gat_sim),
            "struct": struct,
            "total": total,
        })

    df = pd.DataFrame(rows).sort_values("total", ascending=False).head(k)
    return df.reset_index(drop=True)


def get_nutrients(
    con: duckdb.DuckDBPyConnection | None,
    fdc_id: int,
    key_only: bool = True,
    db_path: str = DB_PATH,
) -> dict[str, float]:
    """Get per-100g nutrient values for a food.

    Returns dict like {"Carbohydrate, by difference": 13.8, ...}.
    """
    kb = _get_kb(db_path)
    df = kb.execute(f"""
        SELECT n.nutrient_name, e.amount
        FROM edges_food_contains_nutrient e
        JOIN nodes_nutrient n USING(nutrient_id)
        WHERE e.fdc_id = {int(fdc_id)}
        ORDER BY e.amount DESC
    """).df()

    if key_only:
        df = df[df["nutrient_name"].isin(KEY_NUTRIENTS)]

    return dict(zip(df["nutrient_name"], df["amount"]))


def search_by_nutrient_target(
    con: duckdb.DuckDBPyConnection | None,
    nutrient_name: str,
    min_amount: float = 0.0,
    limit: int = 20,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Find foods high in a specific nutrient.

    Returns DataFrame with [fdc_id, description, amount_per_100g].
    Used by assistant mode to find gap-filling foods.
    """
    kb = _get_kb(db_path)
    nutrient_name_escaped = nutrient_name.replace("'", "''")
    df = kb.execute(f"""
        SELECT f.fdc_id, f.description,
               e.amount AS amount_per_100g
        FROM nodes_food f
        JOIN edges_food_contains_nutrient e ON f.fdc_id = e.fdc_id
        JOIN nodes_nutrient n ON e.nutrient_id = n.nutrient_id
        WHERE n.nutrient_name = '{nutrient_name_escaped}'
          AND e.amount >= {min_amount}
        ORDER BY e.amount DESC
        LIMIT {limit}
    """).df()

    return df
