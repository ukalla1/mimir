#!/usr/bin/env python3
"""Build nutrient_id → row_idx mapping for node_embeddings.npy.

The trained GAT model in nutri_graph saves node_embeddings.npy where:
  - rows 0..NUM_FOODS-1            = food nodes
  - rows NUM_FOODS..NUM_FOODS+NUM_NUTRIENTS-1  = nutrient nodes
  - rows after that                 = recipes, tags, etc.

Food ordering comes from `SELECT fdc_id FROM nodes_food` (heap order).
Nutrient ordering comes from `SELECT nutrient_id FROM nodes_nutrient`.
See nutri_graph/nutri_graph/graph/dataset.py:62-63 for the mapping.

This script reconstructs that mapping from the database and writes a JSON
file that the robot assistant's target_encoder reads at runtime.

Output: nutri_graph/data/nutrient_emb_index.json
    {
        "num_foods": 9991,
        "nutrient_id_to_row": {
            "1003": 9992,   # Protein
            "1004": 9993,   # Total lipid (fat)
            ...
        },
        "nutrient_name_to_id": {
            "Protein": 1003,
            "Total lipid (fat)": 1004,
            ...
        }
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import duckdb


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUTRI_GRAPH_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_DB = os.path.join(NUTRI_GRAPH_ROOT, "data", "nutri_kb.duckdb")
DEFAULT_OUTPUT = os.path.join(NUTRI_GRAPH_ROOT, "data", "nutrient_emb_index.json")


def build_index(db_path: str) -> dict:
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Match dataset.py exactly: same SELECT, same ordering
        foods = con.execute("SELECT fdc_id FROM nodes_food").df()
        nutrs = con.execute(
            "SELECT nutrient_id, nutrient_name FROM nodes_nutrient"
        ).df()
    finally:
        con.close()

    num_foods = int(len(foods))
    nutrient_id_to_row = {
        str(int(row["nutrient_id"])): num_foods + i
        for i, row in nutrs.iterrows()
    }
    nutrient_name_to_id = {
        str(row["nutrient_name"]): int(row["nutrient_id"])
        for _, row in nutrs.iterrows()
    }
    return {
        "num_foods": num_foods,
        "num_nutrients": int(len(nutrs)),
        "nutrient_id_to_row": nutrient_id_to_row,
        "nutrient_name_to_id": nutrient_name_to_id,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="DuckDB path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: DuckDB not found at {args.db}", file=sys.stderr)
        sys.exit(1)

    idx = build_index(args.db)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(idx, f, indent=2)

    print(f"Wrote nutrient embedding index to {args.output}")
    print(f"  num_foods    = {idx['num_foods']}")
    print(f"  num_nutrients = {idx['num_nutrients']}")
    print(f"  total rows used 0..{idx['num_foods'] + idx['num_nutrients'] - 1}")
    # Sanity check on the key macros
    for name in ["Protein", "Total lipid (fat)", "Carbohydrate, by difference", "Energy"]:
        nid = idx["nutrient_name_to_id"].get(name)
        row = idx["nutrient_id_to_row"].get(str(nid)) if nid else None
        print(f'  "{name}" -> nutrient_id={nid}, row_idx={row}')


if __name__ == "__main__":
    main()
