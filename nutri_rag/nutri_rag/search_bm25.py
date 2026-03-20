"""DuckDB Full-Text Search (BM25) bridge + nutrient lookup.

This is the original V0 manual retrieval approach, kept as a baseline
for comparison against the embedding-based V1/V2 pipelines.

Uses DuckDB's built-in FTS extension for proper tokenized matching
instead of naive LIKE substring search.  Includes confidence filtering
to reject low-quality matches — it's better to provide no reference
than a wrong one.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from nutri_rag.config import DB_PATH, KEY_NUTRIENTS, TOP_K_FOODS

# ── Cross-language synonyms ───────────────────────────────────────────
SYNONYMS: dict[str, str] = {
    "groundnut": "peanut",
    "groundnuts": "peanuts",
    "maize": "corn",
    "capsicum": "pepper",
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "coriander": "cilantro",
    "spring onion": "green onion",
    "rocket": "arugula",
    "prawns": "shrimp",
    "nshima": "cornmeal",
    "ugali": "cornmeal",
    "fufu": "cassava",
    "dhal": "lentils",
    "dal": "lentils",
    "porridge": "oatmeal",
}

MIN_BM25_SCORE = 1.0


class FoodSearcher:
    """Full-text search over the USDA food database."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._mem: duckdb.DuckDBPyConnection | None = None
        self._kb: duckdb.DuckDBPyConnection | None = None

    @property
    def kb(self) -> duckdb.DuckDBPyConnection:
        if self._kb is None:
            self._kb = duckdb.connect(self._db_path, read_only=True)
        return self._kb

    @property
    def mem(self) -> duckdb.DuckDBPyConnection:
        if self._mem is None:
            self._mem = duckdb.connect(":memory:")
            self._mem.execute("INSTALL fts; LOAD fts;")
            self._mem.execute(f"""
                ATTACH '{self._db_path}' AS kb (READ_ONLY);
                CREATE TABLE foods AS
                    SELECT fdc_id, description FROM kb.nodes_food;
                DETACH kb;
            """)
            self._mem.execute("""
                PRAGMA create_fts_index(
                    'foods', 'fdc_id', 'description',
                    stemmer='english', lower=true
                )
            """)
        return self._mem


def _apply_synonyms(term: str) -> str:
    lower = term.lower().strip()
    for alias in sorted(SYNONYMS, key=len, reverse=True):
        if alias in lower:
            lower = lower.replace(alias, SYNONYMS[alias], 1)
            break
    return lower


_searcher: FoodSearcher | None = None


def _get_searcher(db_path: str = DB_PATH) -> FoodSearcher:
    global _searcher
    if _searcher is None:
        _searcher = FoodSearcher(db_path)
    return _searcher


def search_food(
    con: duckdb.DuckDBPyConnection | None,
    query: str,
    k: int = TOP_K_FOODS,
    db_path: str = DB_PATH,
    use_gat: bool = False,  # accepted but ignored for API compatibility
) -> pd.DataFrame:
    """Search for foods matching a text query using BM25 full-text search."""
    query = _apply_synonyms(query)
    searcher = _get_searcher(db_path)
    q = query.replace("'", "''")

    try:
        df = searcher.mem.execute(f"""
            SELECT fdc_id, description,
                   fts_main_foods.match_bm25(fdc_id, '{q}') AS score
            FROM foods
            WHERE score IS NOT NULL
              AND fts_main_foods.match_bm25(fdc_id, '{q}') >= {MIN_BM25_SCORE}
            ORDER BY score DESC
        """).df()
    except Exception:
        return pd.DataFrame(columns=["fdc_id", "description"])

    if df.empty:
        return pd.DataFrame(columns=["fdc_id", "description"])

    fdc_ids = df["fdc_id"].tolist()
    placeholders = ", ".join(str(int(fid)) for fid in fdc_ids)

    macro_counts = searcher.kb.execute(f"""
        SELECT e.fdc_id, COUNT(*) AS macro_count
        FROM edges_food_contains_nutrient e
        JOIN nodes_nutrient n USING(nutrient_id)
        WHERE e.fdc_id IN ({placeholders})
          AND n.nutrient_name IN (
              'Carbohydrate, by difference', 'Protein', 'Total lipid (fat)'
          )
        GROUP BY e.fdc_id
    """).df()

    df = df.merge(macro_counts, on="fdc_id", how="left")
    df["macro_count"] = df["macro_count"].fillna(0)
    df = df.sort_values(["macro_count", "score"], ascending=[False, False])
    df = df.head(k)
    return df[["fdc_id", "description"]].reset_index(drop=True)


def get_nutrients(
    con: duckdb.DuckDBPyConnection | None,
    fdc_id: int,
    key_only: bool = True,
    db_path: str = DB_PATH,
) -> dict[str, float]:
    """Get per-100g nutrient values for a food."""
    searcher = _get_searcher(db_path)
    df = searcher.kb.execute(f"""
        SELECT n.nutrient_name, e.amount
        FROM edges_food_contains_nutrient e
        JOIN nodes_nutrient n USING(nutrient_id)
        WHERE e.fdc_id = {int(fdc_id)}
        ORDER BY e.amount DESC
    """).df()

    if key_only:
        df = df[df["nutrient_name"].isin(KEY_NUTRIENTS)]

    return dict(zip(df["nutrient_name"], df["amount"]))
