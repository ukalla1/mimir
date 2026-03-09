import duckdb
import glob
import os
from pathlib import Path


def build_kb(dataset_path: str, db_path: str):
    """
    Faithful port of the Colab KB-build section:
      - Loads all CSVs as DuckDB lazy views
      - Robustly picks FOOD table
      - Robustly picks NUTRIENT dictionary table (avoids measure_unit etc.)
      - Robustly picks LINK table by join coverage
      - Builds:
          nodes_food
          nodes_nutrient
          edges_food_contains_nutrient
          food_index
    """

    dataset_path = Path(dataset_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[KB] dataset_path={dataset_path}")
    print(f"[KB] db_path={db_path}")

    con = duckdb.connect(str(db_path))

    # -------------------------
    # 1) Load into DuckDB as lazy views
    # -------------------------
    csvs = sorted(glob.glob(os.path.join(dataset_path, "**/*.csv"), recursive=True))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSVs found under: {dataset_path}")

    for i, f in enumerate(csvs):
        view = f"t{i}"
        # Escape single quotes in paths just in case
        f_sql = str(f).replace("'", "''")
        con.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_csv_auto('{f_sql}', ALL_VARCHAR=TRUE);"
        )

    def cols(view: str):
        return [r[0].lower() for r in con.execute(f"DESCRIBE {view}").fetchall()]

    # Collect candidates with column sets
    candidates = []
    for (view,) in con.execute("SHOW TABLES").fetchall():
        candidates.append((view, set(cols(view))))

    # -------------------------
    # 2) Pick FOOD table robustly
    # -------------------------
    food_views = [v for v, c in candidates if ("fdc_id" in c) and ("description" in c)]
    if len(food_views) == 0:
        raise RuntimeError("No food table found (needs fdc_id + description).")

    def score_food(v: str) -> float:
        c = set(cols(v))
        s = 0.0
        s += 3.0 if "data_type" in c else 0.0
        s += 2.0 if "publication_date" in c else 0.0
        s += 1.0 if "food_category_id" in c else 0.0
        # Prefer larger tables slightly
        nrows = con.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0]
        s += 0.001 * float(nrows)
        return s

    FOOD_T = sorted(food_views, key=score_food, reverse=True)[0]
    print("[KB] Selected FOOD_T =", FOOD_T)

    # -------------------------
    # 3) Pick NUTRIENT dictionary table robustly
    # -------------------------
    nutr_views = [
        v for v, c in candidates
        if (("id" in c) or ("nutrient_id" in c)) and (("name" in c) or ("nutrient_name" in c))
    ]
    if len(nutr_views) == 0:
        raise RuntimeError("No nutrient-like tables found (needs id/nutrient_id + name/nutrient_name).")

    def pick_name_col(v: str):
        c = set(cols(v))
        if "nutrient_name" in c:
            return "nutrient_name"
        if "name" in c:
            return "name"
        return None

    def score_nutrient_table(v: str) -> float:
        c = set(cols(v))
        name_col = pick_name_col(v)
        if name_col is None:
            return -1.0

        # Reject tables that include fdc_id (usually per-food, not dictionary)
        if "fdc_id" in c:
            return -1.0

        nrows = con.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0]

        # keyword hits: does it contain real nutrient strings?
        kw_hits = con.execute(f"""
            SELECT SUM(
                CASE WHEN lower(trim({name_col})) LIKE '%protein%'
                   OR lower(trim({name_col})) LIKE '%fat%'
                   OR lower(trim({name_col})) LIKE '%carbo%'
                   OR lower(trim({name_col})) LIKE '%fiber%'
                   OR lower(trim({name_col})) LIKE '%energy%'
                   OR lower(trim({name_col})) LIKE '%calcium%'
                   OR lower(trim({name_col})) LIKE '%iron%'
                   OR lower(trim({name_col})) LIKE '%vitamin%'
                THEN 1 ELSE 0 END
            )
            FROM {v}
        """).fetchone()[0]

        bonus = 0.0
        bonus += 4.0 if "unit_name" in c else 0.0
        bonus += 3.0 if "nutrient_nbr" in c else 0.0
        bonus += 2.0 if "rank" in c else 0.0
        bonus += 1.0 if "nutrient_id" in c else 0.0

        # prefer a few hundred rows
        size_bonus = 2.0 if (200 <= nrows <= 1000) else (1.0 if nrows > 1000 else 0.0)

        return float(kw_hits) * 100.0 + bonus + size_bonus

    ranked_nutr = sorted(nutr_views, key=score_nutrient_table, reverse=True)
    NUTR_T = ranked_nutr[0]
    print("[KB] Selected NUTR_T =", NUTR_T, "score=", score_nutrient_table(NUTR_T))

    nutr_cols = set(cols(NUTR_T))
    nutr_id_col = "nutrient_id" if "nutrient_id" in nutr_cols else "id"
    name_col = "nutrient_name" if "nutrient_name" in nutr_cols else "name"
    unit_col = "unit_name" if "unit_name" in nutr_cols else ("unit" if "unit" in nutr_cols else None)

    # Build nodes_nutrient (IMPORTANT: do NOT reference SELECT aliases in WHERE)
    con.execute(f"""
        CREATE OR REPLACE TABLE nodes_nutrient AS
        SELECT
          CAST({nutr_id_col} AS BIGINT) AS nutrient_id,
          TRIM({name_col}) AS nutrient_name,
          {f"TRIM({unit_col})" if unit_col else "NULL"} AS unit_name
        FROM {NUTR_T}
        WHERE {nutr_id_col} IS NOT NULL;
    """)

    # -------------------------
    # 4) Pick LINK table robustly by join coverage with nodes_nutrient
    # -------------------------
    link_views = [
        v for v, c in candidates
        if ("fdc_id" in c) and ("nutrient_id" in c) and (("amount" in c) or ("value" in c))
    ]
    if len(link_views) == 0:
        raise RuntimeError("No link (food_nutrient) table found (needs fdc_id + nutrient_id + amount/value).")

    def link_join_coverage(v: str) -> float:
        c = set(cols(v))
        amt = "amount" if "amount" in c else "value"

        # Use aliases + qualify columns to avoid the DuckDB binder issue
        return con.execute(f"""
            SELECT 1.0 - (SUM(CASE WHEN n.nutrient_id IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*))
            FROM (
              SELECT
                CAST(t.fdc_id AS BIGINT) AS fdc_id,
                CAST(t.nutrient_id AS BIGINT) AS nutrient_id,
                TRY_CAST(t.{amt} AS DOUBLE) AS amount
              FROM {v} t
              WHERE t.fdc_id IS NOT NULL AND t.nutrient_id IS NOT NULL
            ) e
            LEFT JOIN nodes_nutrient n
              ON e.nutrient_id = n.nutrient_id
        """).fetchone()[0]

    ranked_links = sorted(link_views, key=link_join_coverage, reverse=True)
    LINK_T = ranked_links[0]
    print("[KB] Selected LINK_T =", LINK_T, "coverage=", link_join_coverage(LINK_T))

    # -------------------------
    # 5) Build nodes_food + edges
    # -------------------------
    food_cols = set(cols(FOOD_T))
    food_select = ["CAST(fdc_id AS BIGINT) AS fdc_id", "description"]
    if "data_type" in food_cols:
        food_select.append("data_type")
    if "food_category_id" in food_cols:
        food_select.append("CAST(food_category_id AS BIGINT) AS food_category_id")
    if "publication_date" in food_cols:
        food_select.append("publication_date")

    con.execute(f"""
        CREATE OR REPLACE TABLE nodes_food AS
        SELECT
          {", ".join(food_select)}
        FROM {FOOD_T}
        WHERE fdc_id IS NOT NULL;
    """)

    link_cols = set(cols(LINK_T))
    amt_col = "amount" if "amount" in link_cols else "value"

    # IMPORTANT: qualify columns in WHERE to avoid binder alias issue
    con.execute(f"""
        CREATE OR REPLACE TABLE edges_food_contains_nutrient AS
        SELECT
          CAST(t.fdc_id AS BIGINT) AS fdc_id,
          CAST(t.nutrient_id AS BIGINT) AS nutrient_id,
          TRY_CAST(t.{amt_col} AS DOUBLE) AS amount
        FROM {LINK_T} t
        WHERE t.fdc_id IS NOT NULL AND t.nutrient_id IS NOT NULL;
    """)

    # -------------------------
    # 6) Build food search index (same as Colab)
    # -------------------------
    con.execute("""
        CREATE OR REPLACE TABLE food_index AS
        SELECT
          fdc_id,
          description,
          lower(description) AS description_lc,
          data_type
        FROM nodes_food
        WHERE description IS NOT NULL;
    """)

    # Basic stats (same as notebook sanity prints)
    foods_n = con.execute("SELECT COUNT(*) FROM nodes_food").fetchone()[0]
    nutrs_n = con.execute("SELECT COUNT(*) FROM nodes_nutrient").fetchone()[0]
    edges_n = con.execute("SELECT COUNT(*) FROM edges_food_contains_nutrient").fetchone()[0]
    print(f"[KB] Foods={foods_n}, Nutrients={nutrs_n}, Edges={edges_n}")

    con.close()
    print("[KB] Saved KB to", db_path)