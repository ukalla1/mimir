"""Ingest PFoodReq's recipe_kg.json into the existing nutri_kb.duckdb.

Creates four new tables:
  - nodes_recipe:            recipe metadata + nutrition
  - nodes_tag:               cuisine/category tags
  - edges_recipe_uses_food:  recipe → USDA food (matched by text embedding)
  - edges_recipe_has_tag:    recipe → tag

Ingredient names are batch-matched to USDA fdc_ids using the same Qwen3-Embedding
model that nutri_rag uses for semantic search.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NUTR_KEYS = [
    "calories",
    "protein",
    "carbohydrates",
    "saturated fat",
    "monounsaturated fat",
    "polyunsaturated fat",
]


def _safe_float(val) -> float | None:
    """Parse a nutrition value that may be a string or list."""
    if isinstance(val, list):
        val = val[0] if val else None
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_recipe_kg(path: str):
    """First pass: extract all tags, recipes, ingredients, and edges.

    Yields nothing — populates and returns three containers:
      tags:        {tag_uri: tag_name}
      recipes:     {recipe_uri: {name, calories, protein, ...}}
      ingredients: set of unique ingredient names
      recipe_ingredients: {recipe_uri: [ingredient_name, ...]}
      recipe_tags:        {recipe_uri: [tag_uri, ...]}
    """
    tags: dict[str, str] = {}
    recipes: dict[str, dict] = {}
    ingredients: set[str] = set()
    recipe_ingredients: dict[str, list[str]] = defaultdict(list)
    recipe_tags: dict[str, list[str]] = defaultdict(list)

    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            for tag_uri, tag_data in obj.items():
                tag_name = tag_data.get("name", [""])[0]
                tags[tag_uri] = tag_name

                neighbors = tag_data.get("neighbors", {})
                if not neighbors:
                    continue

                for dish_obj in neighbors.get("tagged_dishes", []):
                    for recipe_uri, recipe_data in dish_obj.items():
                        recipe_name = recipe_data.get("name", [""])[0]

                        # Nutrition
                        rn = recipe_data.get("neighbors", {})
                        recipe_info = {
                            "name": recipe_name,
                            "calories": _safe_float(rn.get("calories")),
                            "protein": _safe_float(rn.get("protein")),
                            "carbohydrates": _safe_float(rn.get("carbohydrates")),
                            "saturated_fat": _safe_float(rn.get("saturated fat")),
                            "monounsaturated_fat": _safe_float(rn.get("monounsaturated fat")),
                            "polyunsaturated_fat": _safe_float(rn.get("polyunsaturated fat")),
                        }

                        # Only store first occurrence (recipes can appear under multiple tags)
                        if recipe_uri not in recipes:
                            recipes[recipe_uri] = recipe_info

                        # Tag edge
                        recipe_tags[recipe_uri].append(tag_uri)

                        # Ingredients
                        for ing_obj in rn.get("contains_ingredients", []):
                            for ing_uri, ing_data in ing_obj.items():
                                ing_name = ing_data.get("name", [""])[0]
                                if ing_name:
                                    ingredients.add(ing_name)
                                    recipe_ingredients[recipe_uri].append(ing_name)

            if line_no % 50 == 0:
                print(f"  parsed {line_no} lines ...", file=sys.stderr)

    print(f"[recipe_builder] Parsed: {len(tags)} tags, {len(recipes)} recipes, "
          f"{len(ingredients)} unique ingredients")

    return tags, recipes, ingredients, recipe_ingredients, recipe_tags


# ---------------------------------------------------------------------------
# Ingredient matching
# ---------------------------------------------------------------------------

def match_ingredients_batch(
    ingredient_names: list[str],
    threshold: float = 0.45,
) -> dict[str, tuple[int, float]]:
    """Batch-match ingredient names to USDA fdc_ids using text embeddings.

    Returns {ingredient_name: (fdc_id, similarity_score)} for matches above threshold.
    """
    # Import here to avoid circular / heavy imports when just parsing
    from nutri_rag.embedding import TextEmbedder, FoodVectorIndex, FOOD_SEARCH_INSTRUCTION

    print(f"[recipe_builder] Loading text embedder and food index ...")
    embedder = TextEmbedder()
    index = FoodVectorIndex()

    print(f"[recipe_builder] Encoding {len(ingredient_names)} ingredient names ...")
    query_vecs = embedder.encode(
        ingredient_names,
        batch_size=128,
        task_instruction=FOOD_SEARCH_INSTRUCTION,
    )

    # Cosine similarity: query_vecs is already L2-normalized, index.embeddings too
    print(f"[recipe_builder] Computing similarity matrix ({len(ingredient_names)} x {len(index.fdc_ids)}) ...")
    scores = query_vecs @ index.embeddings.T  # (N_ing, N_foods)

    best_indices = scores.argmax(axis=1)
    best_scores = scores[np.arange(len(scores)), best_indices]

    result = {}
    for i, name in enumerate(ingredient_names):
        if best_scores[i] >= threshold:
            result[name] = (int(index.fdc_ids[best_indices[i]]), float(best_scores[i]))

    matched = len(result)
    total = len(ingredient_names)
    print(f"[recipe_builder] Matched {matched}/{total} ingredients "
          f"({100*matched/total:.1f}%) at threshold={threshold}")

    return result


# ---------------------------------------------------------------------------
# DuckDB insertion
# ---------------------------------------------------------------------------

def build_recipe_kb(
    recipe_kg_path: str,
    db_path: str,
    threshold: float = 0.45,
):
    """Parse recipe_kg.json and insert recipe/tag tables into nutri_kb.duckdb.

    Steps:
    1. Parse recipe_kg.json to extract tags, recipes, ingredients, edges
    2. Batch-encode ingredient names and match to USDA fdc_ids
    3. Insert new tables into the existing DuckDB
    """
    # Step 1: Parse
    print(f"[recipe_builder] Parsing {recipe_kg_path} ...")
    tags, recipes, ingredients, recipe_ingredients, recipe_tags = parse_recipe_kg(recipe_kg_path)

    # Step 2: Match ingredients
    ingredient_list = sorted(ingredients)
    ing_to_fdc = match_ingredients_batch(ingredient_list, threshold=threshold)

    # Step 3: Insert into DuckDB
    print(f"[recipe_builder] Inserting into {db_path} ...")
    con = duckdb.connect(db_path)

    # -- Create tables --
    con.execute("DROP TABLE IF EXISTS edges_recipe_has_tag")
    con.execute("DROP TABLE IF EXISTS edges_recipe_uses_food")
    con.execute("DROP TABLE IF EXISTS nodes_recipe")
    con.execute("DROP TABLE IF EXISTS nodes_tag")

    con.execute("""
        CREATE TABLE nodes_tag (
            tag_id          BIGINT PRIMARY KEY,
            tag_name        VARCHAR,
            source_uri      VARCHAR
        )
    """)

    con.execute("""
        CREATE TABLE nodes_recipe (
            recipe_id           BIGINT PRIMARY KEY,
            recipe_name         VARCHAR,
            source_uri          VARCHAR,
            calories            DOUBLE,
            protein             DOUBLE,
            carbohydrates       DOUBLE,
            saturated_fat       DOUBLE,
            monounsaturated_fat DOUBLE,
            polyunsaturated_fat DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE edges_recipe_uses_food (
            recipe_id        BIGINT,
            fdc_id           BIGINT,
            ingredient_name  VARCHAR,
            similarity_score DOUBLE
        )
    """)

    con.execute("""
        CREATE TABLE edges_recipe_has_tag (
            recipe_id BIGINT,
            tag_id    BIGINT
        )
    """)

    # -- Build ID mappings --
    tag_uri_to_id: dict[str, int] = {}
    for i, uri in enumerate(sorted(tags.keys())):
        tag_uri_to_id[uri] = i

    recipe_uri_to_id: dict[str, int] = {}
    for i, uri in enumerate(recipes.keys()):
        recipe_uri_to_id[uri] = i

    # -- Batch insert tags --
    print(f"[recipe_builder] Inserting {len(tags)} tags ...")
    tag_rows = [(i, tags[uri], uri) for uri, i in tag_uri_to_id.items()]
    con.executemany("INSERT INTO nodes_tag VALUES (?, ?, ?)", tag_rows)

    # -- Batch insert recipes --
    print(f"[recipe_builder] Inserting {len(recipes)} recipes ...")
    recipe_rows = []
    for uri, info in tqdm(recipes.items(), desc="  preparing recipes"):
        rid = recipe_uri_to_id[uri]
        recipe_rows.append((
            rid, info["name"], uri,
            info["calories"], info["protein"], info["carbohydrates"],
            info["saturated_fat"], info["monounsaturated_fat"], info["polyunsaturated_fat"],
        ))
    con.executemany("INSERT INTO nodes_recipe VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", recipe_rows)

    # -- Batch insert recipe → food edges --
    print("[recipe_builder] Building recipe→food edges ...")
    food_edge_rows = []
    for recipe_uri, ing_names in tqdm(recipe_ingredients.items(), desc="  recipe→food edges"):
        recipe_id = recipe_uri_to_id.get(recipe_uri)
        if recipe_id is None:
            continue
        seen = set()
        for ing_name in ing_names:
            if ing_name in seen:
                continue
            seen.add(ing_name)
            match = ing_to_fdc.get(ing_name)
            if match is not None:
                fdc_id, score = match
                food_edge_rows.append((recipe_id, fdc_id, ing_name, score))

    edge_count_food = len(food_edge_rows)
    print(f"[recipe_builder] Inserting {edge_count_food} recipe→food edges ...")
    con.executemany("INSERT INTO edges_recipe_uses_food VALUES (?, ?, ?, ?)", food_edge_rows)

    # -- Batch insert recipe → tag edges --
    print("[recipe_builder] Building recipe→tag edges ...")
    tag_edge_rows = []
    for recipe_uri, tag_uris in tqdm(recipe_tags.items(), desc="  recipe→tag edges"):
        recipe_id = recipe_uri_to_id.get(recipe_uri)
        if recipe_id is None:
            continue
        seen_tags = set()
        for tag_uri in tag_uris:
            tag_id = tag_uri_to_id.get(tag_uri)
            if tag_id is not None and tag_id not in seen_tags:
                seen_tags.add(tag_id)
                tag_edge_rows.append((recipe_id, tag_id))

    edge_count_tag = len(tag_edge_rows)
    print(f"[recipe_builder] Inserting {edge_count_tag} recipe→tag edges ...")
    con.executemany("INSERT INTO edges_recipe_has_tag VALUES (?, ?)", tag_edge_rows)

    con.close()

    print(f"\n[recipe_builder] Done!")
    print(f"  nodes_tag:              {len(tags)}")
    print(f"  nodes_recipe:           {len(recipes)}")
    print(f"  edges_recipe_uses_food: {edge_count_food}")
    print(f"  edges_recipe_has_tag:   {edge_count_tag}")
    print(f"  ingredient match rate:  {len(ing_to_fdc)}/{len(ingredient_list)} "
          f"({100*len(ing_to_fdc)/max(len(ingredient_list),1):.1f}%)")


def print_quality_report(db_path: str):
    """Print quality validation queries after building recipe tables."""
    con = duckdb.connect(db_path, read_only=True)

    print("\n=== Recipe KB Quality Report ===\n")

    # Table counts
    for table in ["nodes_tag", "nodes_recipe", "edges_recipe_uses_food", "edges_recipe_has_tag"]:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n} rows")

    # Match quality distribution
    print("\n  Ingredient match quality distribution:")
    rows = con.execute("""
        SELECT
            CASE
                WHEN similarity_score >= 0.7 THEN 'high (>=0.7)'
                WHEN similarity_score >= 0.55 THEN 'medium (0.55-0.7)'
                ELSE 'low (<0.55)'
            END AS quality,
            COUNT(*) AS cnt
        FROM edges_recipe_uses_food
        GROUP BY 1
        ORDER BY 1
    """).fetchall()
    for quality, cnt in rows:
        print(f"    {quality}: {cnt}")

    # Recipes with zero matched ingredients
    orphan = con.execute("""
        SELECT COUNT(*) FROM nodes_recipe r
        WHERE NOT EXISTS (
            SELECT 1 FROM edges_recipe_uses_food e WHERE e.recipe_id = r.recipe_id
        )
    """).fetchone()[0]
    total_recipes = con.execute("SELECT COUNT(*) FROM nodes_recipe").fetchone()[0]
    print(f"\n  Recipes with zero matched ingredients: {orphan}/{total_recipes}")

    # Average ingredients per recipe
    avg = con.execute("""
        SELECT AVG(cnt) FROM (
            SELECT recipe_id, COUNT(*) as cnt
            FROM edges_recipe_uses_food
            GROUP BY recipe_id
        )
    """).fetchone()[0]
    print(f"  Average matched ingredients per recipe: {avg:.1f}")

    con.close()
