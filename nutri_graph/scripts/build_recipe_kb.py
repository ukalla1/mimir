"""Build recipe/tag tables in nutri_kb.duckdb from PFoodReq's recipe_kg.json.

Usage:
    cd mimir/nutri_graph
    python scripts/build_recipe_kb.py

Requires:
    - data/nutri_kb.duckdb (built by scripts/build_kb.py)
    - ../PFoodReq/data/recipe_kg/recipe_kg.json
    - nutri_rag's pre-computed text embeddings (built by nutri_rag/scripts/build_embeddings.py)
"""

import sys
from pathlib import Path

# Add project roots to path so we can import nutri_graph and nutri_rag
_base = Path(__file__).resolve().parent.parent          # nutri_graph/
_mimir = _base.parent                                    # mimir/
sys.path.insert(0, str(_base))
sys.path.insert(0, str(_mimir / "nutri_rag"))

from nutri_graph.kb.recipe_builder import build_recipe_kb, print_quality_report


if __name__ == "__main__":
    db_path = str(_base / "data" / "nutri_kb.duckdb")
    recipe_kg_path = str(_mimir / "PFoodReq" / "data" / "recipe_kg" / "recipe_kg.json")

    # Check files exist
    if not Path(db_path).exists():
        print(f"ERROR: {db_path} not found. Run scripts/build_kb.py first.")
        sys.exit(1)
    if not Path(recipe_kg_path).exists():
        print(f"ERROR: {recipe_kg_path} not found.")
        print("Download recipe_kg.json from PFoodReq and place it at the path above.")
        sys.exit(1)

    build_recipe_kb(
        recipe_kg_path=recipe_kg_path,
        db_path=db_path,
        threshold=0.45,
    )

    print_quality_report(db_path)
