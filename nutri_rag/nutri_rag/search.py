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
        use_gat: If True, apply GAT nutritional-similarity re-ranking on top
                 of text embedding results (V2 pipeline).

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
    fdc_ids = [fdc_id for fdc_id, _, _ in candidates]

    # Check which candidates have macronutrient data
    kb = _get_kb(db_path)
    placeholders = ", ".join(str(int(fid)) for fid in fdc_ids)
    macro_counts = kb.execute(f"""
        SELECT e.fdc_id, COUNT(*) AS macro_count
        FROM edges_food_contains_nutrient e
        JOIN nodes_nutrient n USING(nutrient_id)
        WHERE e.fdc_id IN ({placeholders})
          AND n.nutrient_name IN (
              'Carbohydrate, by difference', 'Protein', 'Total lipid (fat)'
          )
        GROUP BY e.fdc_id
    """).df()

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

    df = df.merge(macro_counts, on="fdc_id", how="left")
    df["macro_count"] = df["macro_count"].fillna(0)

    if use_gat and len(df) > 1:
        df = _gat_rerank(df)

    # Prefer entries with macros, then by combined score
    score_col = "combined_score" if "combined_score" in df.columns else "text_score"
    df = df.sort_values(["macro_count", score_col], ascending=[False, False])
    df = df.head(k)
    return df[["fdc_id", "description"]].reset_index(drop=True)


def _gat_rerank(df: pd.DataFrame, text_weight: float = 0.7) -> pd.DataFrame:
    """Re-rank text embedding candidates using GAT nutritional similarity.

    Strategy: compute each candidate's average GAT cosine similarity to all
    other candidates. Candidates that are nutritionally coherent with the
    group score higher. Combine with text score:
        combined = text_weight * text_score + (1 - text_weight) * gat_coherence
    """
    gat = _get_gat_index()
    indices = df["arr_idx"].tolist()
    vecs = gat.get_vectors(indices)  # (n, 64)

    # Pairwise cosine similarity matrix (vectors are already L2-normalized)
    sim_matrix = vecs @ vecs.T  # (n, n)
    # Average similarity to all OTHER candidates (exclude self=1.0)
    n = len(indices)
    np.fill_diagonal(sim_matrix, 0.0)
    gat_coherence = sim_matrix.sum(axis=1) / max(n - 1, 1)

    # Normalize gat_coherence to [0, 1] range
    gat_min, gat_max = gat_coherence.min(), gat_coherence.max()
    if gat_max > gat_min:
        gat_coherence = (gat_coherence - gat_min) / (gat_max - gat_min)
    else:
        gat_coherence = np.ones(n) * 0.5

    df = df.copy()
    df["gat_coherence"] = gat_coherence
    df["combined_score"] = (
        text_weight * df["text_score"] + (1 - text_weight) * df["gat_coherence"]
    )
    return df


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
