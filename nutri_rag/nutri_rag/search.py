"""Semantic vector search + nutrient lookup for RAG retrieval.

Uses pre-computed Qwen3-Embedding vectors for cosine similarity search
instead of BM25 keyword matching. This handles vocabulary mismatches
like "groundnut" vs "peanut", "maize flour" vs "corn flour", etc.
"""

from __future__ import annotations

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
