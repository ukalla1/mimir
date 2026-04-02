import duckdb
import glob
import os
from pathlib import Path


def _load_sr_legacy(con, sr_legacy_path: str):
    """Load USDA SR Legacy data (tilde-delimited ASCII) into temp tables.

    Creates:
      _sr_food(fdc_id BIGINT, description VARCHAR, data_type VARCHAR)
      _sr_edges(fdc_id BIGINT, nutrient_id BIGINT, amount DOUBLE)
      _sr_nutrient(nutrient_id BIGINT, nutrient_name VARCHAR, unit_name VARCHAR)
    """
    sr_path = Path(sr_legacy_path)
    food_file = sr_path / "FOOD_DES.txt"
    nut_data_file = sr_path / "NUT_DATA.txt"
    nutr_def_file = sr_path / "NUTR_DEF.txt"

    if not food_file.exists():
        print(f"[KB] SR Legacy not found at {sr_path}, skipping")
        return False

    def parse_sr(filepath):
        rows = []
        with open(filepath, "r", encoding="latin-1") as f:
            for line in f:
                fields = [field.strip("~") for field in line.strip().split("^")]
                rows.append(fields)
        return rows

    # Parse foods: NDB_No, FdGrp_Cd, Long_Desc, ...
    foods = parse_sr(str(food_file))
    # Use NDB_No as fdc_id (offset by 1_000_000 to avoid collision with FDC ids)
    SR_OFFSET = 1_000_000
    con.execute("CREATE OR REPLACE TEMP TABLE _sr_food (fdc_id BIGINT, description VARCHAR, data_type VARCHAR)")
    for r in foods:
        ndb = int(r[0])
        desc = r[2].replace("'", "''")
        con.execute(f"INSERT INTO _sr_food VALUES ({SR_OFFSET + ndb}, '{desc}', 'sr_legacy')")
    sr_food_n = con.execute("SELECT COUNT(*) FROM _sr_food").fetchone()[0]
    print(f"[KB] SR Legacy: loaded {sr_food_n} foods")

    # Parse nutrient definitions: Nutr_No, Units, Tagname, NutrDesc
    nutr_defs = parse_sr(str(nutr_def_file))
    con.execute("CREATE OR REPLACE TEMP TABLE _sr_nutrient (nutrient_id BIGINT, nutrient_name VARCHAR, unit_name VARCHAR)")
    for r in nutr_defs:
        nutr_id = int(r[0])
        name = r[3].replace("'", "''")
        unit = r[1].replace("'", "''")
        con.execute(f"INSERT INTO _sr_nutrient VALUES ({nutr_id}, '{name}', '{unit}')")

    # Parse nutrient data: NDB_No, Nutr_No, Nutr_Val, ...
    nut_data = parse_sr(str(nut_data_file))
    con.execute("CREATE OR REPLACE TEMP TABLE _sr_edges (fdc_id BIGINT, nutrient_id BIGINT, amount DOUBLE)")
    # Batch insert for speed
    batch = []
    for r in nut_data:
        ndb = int(r[0])
        nutr_id = int(r[1])
        try:
            amount = float(r[2]) if r[2] else None
        except ValueError:
            continue
        if amount is not None:
            batch.append((SR_OFFSET + ndb, nutr_id, amount))
        if len(batch) >= 10000:
            con.executemany("INSERT INTO _sr_edges VALUES (?, ?, ?)", batch)
            batch = []
    if batch:
        con.executemany("INSERT INTO _sr_edges VALUES (?, ?, ?)", batch)

    sr_edges_n = con.execute("SELECT COUNT(*) FROM _sr_edges").fetchone()[0]
    print(f"[KB] SR Legacy: loaded {sr_edges_n} nutrient edges")
    return True


def build_kb(dataset_path: str, db_path: str, sr_legacy_path: str | None = None):
    """
    Build the nutrition knowledge base from USDA FoodData Central CSVs,
    optionally augmented with SR Legacy data.

      - Loads all CSVs as DuckDB lazy views
      - Optionally loads SR Legacy (tilde-delimited ASCII)
      - Robustly picks FOOD table
      - Robustly picks NUTRIENT dictionary table (avoids measure_unit etc.)
      - Robustly picks LINK table by join coverage
      - Filters junk entries (lab records, entries with no nutrients)
      - Deduplicates by description, averages nutrient amounts
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
    # 5) Build raw nodes_food + edges (before deduplication)
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
        CREATE OR REPLACE TEMP TABLE _raw_food AS
        SELECT
          {", ".join(food_select)}
        FROM {FOOD_T}
        WHERE fdc_id IS NOT NULL;
    """)

    link_cols = set(cols(LINK_T))
    amt_col = "amount" if "amount" in link_cols else "value"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _raw_edges AS
        SELECT
          CAST(t.fdc_id AS BIGINT) AS fdc_id,
          CAST(t.nutrient_id AS BIGINT) AS nutrient_id,
          TRY_CAST(t.{amt_col} AS DOUBLE) AS amount
        FROM {LINK_T} t
        WHERE t.fdc_id IS NOT NULL AND t.nutrient_id IS NOT NULL;
    """)

    raw_foods = con.execute("SELECT COUNT(*) FROM _raw_food").fetchone()[0]
    raw_unique = con.execute("SELECT COUNT(DISTINCT description) FROM _raw_food").fetchone()[0]
    print(f"[KB] Raw foods: {raw_foods}, unique descriptions: {raw_unique}")

    # -------------------------
    # 5a) Remove junk entries: lab analysis records and entries with no nutrients
    # -------------------------
    # Remove foods with no nutrient edges (lab sub-samples that store data elsewhere)
    no_nutr = con.execute("""
        DELETE FROM _raw_food
        WHERE fdc_id NOT IN (SELECT DISTINCT fdc_id FROM _raw_edges)
    """).fetchone()
    after_no_nutr = con.execute("SELECT COUNT(*) FROM _raw_food").fetchone()[0]
    print(f"[KB] After removing foods with no nutrients: {after_no_nutr} (removed {raw_foods - after_no_nutr})")

    # Remove lab analysis prefix entries (e.g. "Amino Acids, Chicken...")
    con.execute("""
        DELETE FROM _raw_food
        WHERE REGEXP_MATCHES(description,
            '^(Amino Acids|Minerals|Fatty Acids|Proximates|Starch|Sugars,|TDF|'
            'Thiamin|Vitamin|Riboflavin|Selenium|Pantothenic|Choline|Niacin|'
            'Fat,|Carotenoids|Folate|Cholesterol)')
    """)
    # Remove entries with lab tracking codes
    con.execute("""
        DELETE FROM _raw_food
        WHERE description LIKE '%- NFY%'
           OR description LIKE '%- CY%'
    """)
    after_junk = con.execute("SELECT COUNT(*) FROM _raw_food").fetchone()[0]
    after_junk_unique = con.execute("SELECT COUNT(DISTINCT description) FROM _raw_food").fetchone()[0]
    print(f"[KB] After removing junk entries: {after_junk} ({after_junk_unique} unique descriptions)")

    # Also clean up _raw_edges to only keep edges for surviving foods
    con.execute("""
        DELETE FROM _raw_edges
        WHERE fdc_id NOT IN (SELECT fdc_id FROM _raw_food)
    """)

    # -------------------------
    # 5a-2) Merge SR Legacy data if available
    # -------------------------
    if sr_legacy_path and _load_sr_legacy(con, sr_legacy_path):
        # Merge SR Legacy nutrient definitions into nodes_nutrient
        # (add any new nutrients not already in the table)
        con.execute("""
            INSERT INTO nodes_nutrient (nutrient_id, nutrient_name, unit_name)
            SELECT s.nutrient_id, s.nutrient_name, s.unit_name
            FROM _sr_nutrient s
            WHERE s.nutrient_id NOT IN (SELECT nutrient_id FROM nodes_nutrient)
        """)

        # Add SR Legacy foods to _raw_food
        # _raw_food may have columns like food_category_id, publication_date
        # SR Legacy only has fdc_id, description, data_type — insert with NULLs for extra cols
        raw_cols = [r[0] for r in con.execute("DESCRIBE _raw_food").fetchall()]
        sr_select = ["s.fdc_id", "s.description"]
        if "data_type" in raw_cols:
            sr_select.append("s.data_type")
        for col in raw_cols:
            if col not in ("fdc_id", "description", "data_type"):
                sr_select.append(f"NULL AS {col}")

        con.execute(f"""
            INSERT INTO _raw_food
            SELECT {', '.join(sr_select)}
            FROM _sr_food s
        """)

        # Add SR Legacy edges to _raw_edges
        con.execute("""
            INSERT INTO _raw_edges
            SELECT fdc_id, nutrient_id, amount
            FROM _sr_edges
        """)

        sr_added = con.execute("SELECT COUNT(*) FROM _sr_food").fetchone()[0]
        total_after = con.execute("SELECT COUNT(*) FROM _raw_food").fetchone()[0]
        print(f"[KB] After merging SR Legacy: {total_after} foods (+{sr_added} from SR Legacy)")

        # Clean up SR Legacy temp tables
        con.execute("DROP TABLE IF EXISTS _sr_food")
        con.execute("DROP TABLE IF EXISTS _sr_edges")
        con.execute("DROP TABLE IF EXISTS _sr_nutrient")

    # -------------------------
    # 5b) Deduplicate: group by description, pick representative fdc_id,
    #     average nutrient amounts (skip missing, don't count as 0)
    # -------------------------

    # For each unique description, pick the fdc_id with the most nutrient edges
    con.execute("""
        CREATE OR REPLACE TEMP TABLE _repr AS
        SELECT f.description, f.fdc_id AS repr_fdc_id
        FROM _raw_food f
        JOIN (
            SELECT f2.description,
                   f2.fdc_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY f2.description
                       ORDER BY cnt DESC, f2.fdc_id
                   ) AS rn
            FROM _raw_food f2
            LEFT JOIN (
                SELECT fdc_id, COUNT(*) AS cnt
                FROM _raw_edges
                GROUP BY fdc_id
            ) ec ON f2.fdc_id = ec.fdc_id
        ) ranked ON f.fdc_id = ranked.fdc_id AND ranked.rn = 1
    """)

    # Build deduplicated nodes_food using representative fdc_ids
    con.execute("""
        CREATE OR REPLACE TABLE nodes_food AS
        SELECT f.*
        FROM _raw_food f
        JOIN _repr r ON f.fdc_id = r.repr_fdc_id
    """)

    # Build deduplicated edges: average amounts across all fdc_ids sharing
    # the same description, then assign to the representative fdc_id.
    # AVG naturally skips NULLs — missing nutrients are not counted as 0.
    con.execute("""
        CREATE OR REPLACE TABLE edges_food_contains_nutrient AS
        SELECT
            r.repr_fdc_id AS fdc_id,
            e.nutrient_id,
            AVG(e.amount) AS amount
        FROM _raw_edges e
        JOIN _raw_food f ON e.fdc_id = f.fdc_id
        JOIN _repr r ON f.description = r.description
        WHERE e.amount IS NOT NULL
        GROUP BY r.repr_fdc_id, e.nutrient_id
    """)

    # Clean up temp tables
    con.execute("DROP TABLE IF EXISTS _raw_food")
    con.execute("DROP TABLE IF EXISTS _raw_edges")
    con.execute("DROP TABLE IF EXISTS _repr")

    # -------------------------
    # 6) Build food search index
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

    # Basic stats
    foods_n = con.execute("SELECT COUNT(*) FROM nodes_food").fetchone()[0]
    nutrs_n = con.execute("SELECT COUNT(*) FROM nodes_nutrient").fetchone()[0]
    edges_n = con.execute("SELECT COUNT(*) FROM edges_food_contains_nutrient").fetchone()[0]
    foods_with_edges = con.execute("""
        SELECT COUNT(DISTINCT fdc_id) FROM edges_food_contains_nutrient
    """).fetchone()[0]
    print(f"[KB] Foods={foods_n}, Nutrients={nutrs_n}, Edges={edges_n}")
    print(f"[KB] Foods with ≥1 nutrient: {foods_with_edges}/{foods_n}")

    con.close()
    print("[KB] Saved KB to", db_path)


def export_kb(db_path: str, output_path: str):
    """Export the KB to a JSON file for sharing/fine-tuning.

    Each entry contains:
      - fdc_id, description, data_type
      - nutrients: dict of nutrient_name -> {amount, unit}
    """
    import json

    db_path = Path(db_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)

    # Get all foods
    foods = con.execute("""
        SELECT fdc_id, description, data_type FROM nodes_food
        ORDER BY fdc_id
    """).fetchall()

    # Get all edges with nutrient info
    edges = con.execute("""
        SELECT e.fdc_id, n.nutrient_name, e.amount, n.unit_name
        FROM edges_food_contains_nutrient e
        JOIN nodes_nutrient n ON e.nutrient_id = n.nutrient_id
        ORDER BY e.fdc_id, n.nutrient_name
    """).fetchall()

    # Build nutrient lookup by fdc_id
    nutr_by_fdc = {}
    for fdc_id, name, amount, unit in edges:
        nutr_by_fdc.setdefault(fdc_id, {})[name] = {
            "amount": round(amount, 4) if amount is not None else None,
            "unit": unit,
        }

    # Build export list
    records = []
    for fdc_id, description, data_type in foods:
        records.append({
            "fdc_id": fdc_id,
            "description": description,
            "data_type": data_type,
            "nutrients": nutr_by_fdc.get(fdc_id, {}),
        })

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    con.close()
    print(f"[KB] Exported {len(records)} foods to {output_path}")