"""DuckDB Full-Text Search (BM25) bridge + nutrient lookup.

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
# Only genuine vocabulary mappings where different English dialects or
# regions use completely different words for the same food.
# NOT search hints like "apple" -> "apples, raw".
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

# Minimum BM25 score to accept a match.  Below this, we assume the
# match is unreliable and return nothing (model falls back to its own
# knowledge, which is the baseline behavior).
MIN_BM25_SCORE = 1.0


class FoodSearcher:
    """Full-text search over the USDA food database.

    Creates an in-memory FTS index on first use, then reuses it for all
    subsequent queries.  The original DB is attached read-only for
    nutrient lookups.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._mem: duckdb.DuckDBPyConnection | None = None
        self._kb: duckdb.DuckDBPyConnection | None = None

    @property
    def kb(self) -> duckdb.DuckDBPyConnection:
        """Read-only connection to the original knowledge base."""
        if self._kb is None:
            self._kb = duckdb.connect(self._db_path, read_only=True)
        return self._kb

    @property
    def mem(self) -> duckdb.DuckDBPyConnection:
        """In-memory connection with FTS index for search."""
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

    def close(self):
        if self._mem is not None:
            self._mem.close()
            self._mem = None
        if self._kb is not None:
            self._kb.close()
            self._kb = None


def _apply_synonyms(term: str) -> str:
    """Replace known cross-language aliases."""
    lower = term.lower().strip()
    # Longest-prefix match so multi-word synonyms win over single-word
    for alias in sorted(SYNONYMS, key=len, reverse=True):
        if alias in lower:
            lower = lower.replace(alias, SYNONYMS[alias], 1)
            break
    return lower


# Module-level singleton searcher (lazy init)
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
) -> pd.DataFrame:
    """Search for foods matching a text query using BM25 full-text search.

    Returns DataFrame with columns [fdc_id, description].
    Only returns results above the confidence threshold.

    The `con` parameter is accepted for backward compatibility but the
    FTS searcher uses its own connections internally.
    """
    query = _apply_synonyms(query)
    searcher = _get_searcher(db_path)

    # Escape single quotes for SQL
    q = query.replace("'", "''")

    # Step 1: Get ALL matches above threshold (no hard limit yet)
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

    # Step 2: Among all confident matches, find those with macronutrient data
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
    # Prefer entries with macros, then by BM25 score, then shorter names
    df = df.sort_values(
        ["macro_count", "score"], ascending=[False, False]
    )

    df = df.head(k)
    return df[["fdc_id", "description"]].reset_index(drop=True)


def get_nutrients(
    con: duckdb.DuckDBPyConnection | None,
    fdc_id: int,
    key_only: bool = True,
    db_path: str = DB_PATH,
) -> dict[str, float]:
    """Get per-100g nutrient values for a food.

    Returns dict like {"Carbohydrate, by difference": 13.8, ...}.
    """
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
    searcher = _get_searcher(db_path)
    nutrient_name_escaped = nutrient_name.replace("'", "''")
    df = searcher.kb.execute(f"""
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
