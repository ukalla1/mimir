"""DuckDB text search bridge + nutrient lookup.

Wraps nutri_graph's search_food() with synonym expansion and multi-word
fallback.  Also provides nutrient target queries for the assistant mode.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from nutri_rag.config import DB_PATH, KEY_NUTRIENTS, TOP_K_FOODS

# ── Synonym dictionary ────────────────────────────────────────────────
SYNONYMS: dict[str, str] = {
    "oj": "orange juice",
    "pb": "peanut butter",
    "groundnuts": "peanuts",
    "groundnut": "peanut",
    "maize": "corn",
    "maize flour": "corn flour",
    "capsicum": "pepper",
    "aubergine": "eggplant",
    "courgette": "zucchini",
    "coriander": "cilantro",
    "spring onion": "green onion",
    "rocket": "arugula",
    "prawns": "shrimp",
    "mince": "ground beef",
    "chips": "french fries",
    "crisps": "potato chips",
    "porridge": "oatmeal",
    "semolina": "semolina",
    "nshima": "corn flour",
    "ugali": "corn flour",
    "fufu": "cassava flour",
    "chapati": "flatbread",
    "roti": "flatbread",
    "dhal": "lentils",
    "dal": "lentils",
    "raw sugar": "sugars, granulated",
    "sugar": "sugars, granulated",
    "white sugar": "sugars, granulated",
    "brown sugar": "sugars, brown",
    "oatmeal": "oats",
    "oat": "oats",
    "orange juice": "orange juice",
    "rice": "rice, white",
    "bread": "bread, wheat",
    "egg": "egg, whole",
    "eggs": "egg, whole",
    "milk": "milk, whole",
    "chicken": "chicken, breast",
    "beef": "beef, ground",
    "butter": "butter, salted",
    "cheese": "cheese, cheddar",
    "banana": "bananas, raw",
    "apple": "apples, raw",
    "orange": "oranges, raw",
    "potato": "potatoes",
    "tomato": "tomatoes, raw",
    "onion": "onions, raw",
    "garlic": "garlic, raw",
    "peanut butter": "peanut butter",
    "yogurt": "yogurt",
    "salmon": "salmon",
    "tuna": "tuna",
    "shrimp": "shrimp",
    "tofu": "tofu",
    "lentil": "lentils",
    "lentils": "lentils",
    "pasta": "pasta",
    "spaghetti": "spaghetti",
    "flour": "flour, wheat",
    "wheat flour": "flour, wheat",
    "corn oil": "oil, corn",
    "olive oil": "oil, olive",
    "vegetable oil": "oil, vegetable",
}


def _get_connection(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection."""
    return duckdb.connect(db_path, read_only=True)


def _apply_synonyms(term: str) -> str:
    """Replace known aliases with canonical USDA terms."""
    lower = term.lower().strip()
    # Try exact match first, then check if term starts with a synonym
    if lower in SYNONYMS:
        return SYNONYMS[lower]
    for alias, canonical in SYNONYMS.items():
        if lower.startswith(alias):
            return lower.replace(alias, canonical, 1)
    return lower


def search_food(
    con: duckdb.DuckDBPyConnection,
    query: str,
    k: int = TOP_K_FOODS,
) -> pd.DataFrame:
    """Search for foods matching a text query.

    Returns DataFrame with columns [fdc_id, description].
    Uses synonym expansion + multi-word OR fallback.
    """
    query = _apply_synonyms(query)
    q = query.lower().replace("'", "''")

    # Primary: full-term LIKE search, prefer entries with key macronutrient data
    # Many DB entries are sub-samples without standard macros, so we prioritize
    # entries that have "Carbohydrate, by difference" data
    df = con.execute(f"""
        SELECT f.fdc_id, f.description
        FROM nodes_food f
        WHERE lower(f.description) LIKE '%{q}%'
        ORDER BY (
            SELECT COUNT(*) FROM edges_food_contains_nutrient e
            JOIN nodes_nutrient n USING(nutrient_id)
            WHERE e.fdc_id = f.fdc_id
              AND n.nutrient_name IN ('Carbohydrate, by difference', 'Protein', 'Total lipid (fat)')
        ) DESC, length(f.description) ASC
        LIMIT {k}
    """).df()

    if len(df) > 0:
        return df

    # Fallback: split into words, OR-style LIKE, score by match count
    # Require min 4 chars to avoid false substring matches (e.g., "eat" in "GREAT")
    words = [w for w in q.split() if len(w) >= 4]
    if not words:
        return pd.DataFrame(columns=["fdc_id", "description"])

    like_clauses = " + ".join(
        f"CASE WHEN lower(description) LIKE '%{w}%' THEN 1 ELSE 0 END"
        for w in words
    )
    df = con.execute(f"""
        SELECT f.fdc_id, f.description, ({like_clauses}) AS match_score
        FROM nodes_food f
        WHERE ({like_clauses}) > 0
        ORDER BY match_score DESC,
            (SELECT COUNT(*) FROM edges_food_contains_nutrient e
             JOIN nodes_nutrient n USING(nutrient_id)
             WHERE e.fdc_id = f.fdc_id
               AND n.nutrient_name IN ('Carbohydrate, by difference', 'Protein', 'Total lipid (fat)')
            ) DESC,
            length(f.description) ASC
        LIMIT {k}
    """).df()

    if "match_score" in df.columns:
        df = df.drop(columns=["match_score"])

    return df


def get_nutrients(
    con: duckdb.DuckDBPyConnection,
    fdc_id: int,
    key_only: bool = True,
) -> dict[str, float]:
    """Get per-100g nutrient values for a food.

    Returns dict like {"Carbohydrate, by difference": 13.8, ...}.
    If key_only=True, filters to KEY_NUTRIENTS only.
    """
    df = con.execute(f"""
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
    con: duckdb.DuckDBPyConnection,
    nutrient_name: str,
    min_amount: float = 0.0,
    limit: int = 20,
) -> pd.DataFrame:
    """Find foods high in a specific nutrient.

    Returns DataFrame with [fdc_id, description, amount_per_100g].
    Used by assistant mode to find gap-filling foods.
    """
    nutrient_name_escaped = nutrient_name.replace("'", "''")
    df = con.execute(f"""
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
