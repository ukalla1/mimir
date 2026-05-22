import os

import duckdb
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _table_exists(con, table_name: str) -> bool:
    """Check if a table exists in the connected DuckDB database."""
    result = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return result[0] > 0


def build_graph_from_db(db_path: str, include_recipes: bool = False, subs_csv_path: str = None):
    """Build a PyG Data object from the nutri_kb DuckDB.

    Parameters
    ----------
    db_path : str
        Path to nutri_kb.duckdb.
    include_recipes : bool
        If True, add recipe (type=2) and tag (type=3) nodes plus their edges
        from the nodes_recipe / nodes_tag / edges_recipe_* tables.
        The supervised task (food→nutrient prediction) is unchanged — recipe
        edges only participate in message passing.
    """
    con = duckdb.connect(db_path, read_only=True)

    # Pull the same columns your Colab code later relies on
    foods = con.execute("""
        SELECT fdc_id, description, food_category_id
        FROM nodes_food
    """).df()

    nutrs = con.execute("""
        SELECT nutrient_id, nutrient_name, unit_name
        FROM nodes_nutrient
    """).df()

    edges = con.execute("""
        SELECT fdc_id, nutrient_id, amount
        FROM edges_food_contains_nutrient
        WHERE amount IS NOT NULL
    """).df()

    # Match Colab dtype conversions
    foods["fdc_id"] = foods["fdc_id"].astype(int)
    nutrs["nutrient_id"] = nutrs["nutrient_id"].astype(int)
    edges["fdc_id"] = edges["fdc_id"].astype(int)
    edges["nutrient_id"] = edges["nutrient_id"].astype(int)
    edges["amount"] = edges["amount"].astype(float)

    NUM_FOODS = len(foods)
    NUM_NUTRIENTS = len(nutrs)

    # id -> contiguous idx
    food_id_to_idx = {int(fid): i for i, fid in enumerate(foods["fdc_id"].tolist())}
    nutr_id_to_idx = {int(nid): i for i, nid in enumerate(nutrs["nutrient_id"].tolist())}

    # src_food in [0..NUM_FOODS-1], dst_nutr in [0..NUM_NUTRIENTS-1]
    src_food = edges["fdc_id"].map(food_id_to_idx).to_numpy(dtype=np.int64)
    dst_nutr = edges["nutrient_id"].map(nutr_id_to_idx).to_numpy(dtype=np.int64)

    # supervised direction: food -> nutrient (GLOBAL indexing for nutrient nodes)
    pos_edge_index = torch.tensor(
        np.vstack([src_food, dst_nutr + NUM_FOODS]),
        dtype=torch.long
    )

    # reverse edges for message passing
    rev_edge_index = torch.tensor(
        np.vstack([dst_nutr + NUM_FOODS, src_food]),
        dtype=torch.long
    )

    # edge_attr: log1p(amount) for positive edges; duplicated for reverse edges
    edge_amount = np.log1p(edges["amount"].to_numpy(dtype=np.float32))
    edge_attr_pos = torch.tensor(edge_amount, dtype=torch.float32).view(-1, 1)

    # Collect all edge segments for the full propagation graph
    edge_segments = [pos_edge_index, rev_edge_index]
    attr_segments = [edge_attr_pos, edge_attr_pos]

    # ── Recipe / Tag nodes and edges (Phase 2) ──────────────────────────
    NUM_RECIPES = 0
    NUM_TAGS = 0
    RECIPE_OFFSET = NUM_FOODS + NUM_NUTRIENTS
    TAG_OFFSET = RECIPE_OFFSET

    if include_recipes and _table_exists(con, "nodes_recipe"):
        recipe_df = con.execute("SELECT recipe_id FROM nodes_recipe ORDER BY recipe_id").df()
        tag_df = con.execute("SELECT tag_id FROM nodes_tag ORDER BY tag_id").df()
        recipe_food_df = con.execute("SELECT recipe_id, fdc_id FROM edges_recipe_uses_food").df()
        recipe_tag_df = con.execute("SELECT recipe_id, tag_id FROM edges_recipe_has_tag").df()

        NUM_RECIPES = len(recipe_df)
        NUM_TAGS = len(tag_df)
        RECIPE_OFFSET = NUM_FOODS + NUM_NUTRIENTS
        TAG_OFFSET = RECIPE_OFFSET + NUM_RECIPES

        # Build index mappings for recipe/tag IDs → contiguous indices
        recipe_id_to_idx = {int(rid): i for i, rid in enumerate(recipe_df["recipe_id"].tolist())}
        tag_id_to_idx = {int(tid): i for i, tid in enumerate(tag_df["tag_id"].tolist())}

        # ── recipe ↔ food edges ──
        # Filter to only fdc_ids that exist in our food index
        rf_recipe_ids = recipe_food_df["recipe_id"].astype(int).to_numpy()
        rf_fdc_ids = recipe_food_df["fdc_id"].astype(int).to_numpy()

        # Vectorized mapping with NaN filtering for unmatched fdc_ids
        rf_recipe_mapped = np.array([recipe_id_to_idx.get(int(r), -1) for r in rf_recipe_ids])
        rf_food_mapped = np.array([food_id_to_idx.get(int(f), -1) for f in rf_fdc_ids])
        valid_rf = (rf_recipe_mapped >= 0) & (rf_food_mapped >= 0)

        if valid_rf.any():
            src_recipe = rf_recipe_mapped[valid_rf] + RECIPE_OFFSET
            dst_food = rf_food_mapped[valid_rf]

            recipe_food_ei = torch.tensor(np.vstack([src_recipe, dst_food]), dtype=torch.long)
            food_recipe_ei = torch.tensor(np.vstack([dst_food, src_recipe]), dtype=torch.long)
            recipe_food_attr = torch.ones(recipe_food_ei.size(1), 1)

            edge_segments.extend([recipe_food_ei, food_recipe_ei])
            attr_segments.extend([recipe_food_attr, recipe_food_attr])

            print(f"[dataset] Added {recipe_food_ei.size(1)} recipe↔food edges (bidirectional)")

        # ── recipe ↔ tag edges ──
        rt_recipe_ids = recipe_tag_df["recipe_id"].astype(int).to_numpy()
        rt_tag_ids = recipe_tag_df["tag_id"].astype(int).to_numpy()

        rt_recipe_mapped = np.array([recipe_id_to_idx.get(int(r), -1) for r in rt_recipe_ids])
        rt_tag_mapped = np.array([tag_id_to_idx.get(int(t), -1) for t in rt_tag_ids])
        valid_rt = (rt_recipe_mapped >= 0) & (rt_tag_mapped >= 0)

        if valid_rt.any():
            src_recipe_t = rt_recipe_mapped[valid_rt] + RECIPE_OFFSET
            dst_tag = rt_tag_mapped[valid_rt] + TAG_OFFSET

            recipe_tag_ei = torch.tensor(np.vstack([src_recipe_t, dst_tag]), dtype=torch.long)
            tag_recipe_ei = torch.tensor(np.vstack([dst_tag, src_recipe_t]), dtype=torch.long)
            recipe_tag_attr = torch.ones(recipe_tag_ei.size(1), 1)

            edge_segments.extend([recipe_tag_ei, tag_recipe_ei])
            attr_segments.extend([recipe_tag_attr, recipe_tag_attr])

            print(f"[dataset] Added {recipe_tag_ei.size(1)} recipe↔tag edges (bidirectional)")

        print(f"[dataset] Recipe graph: {NUM_RECIPES} recipes, {NUM_TAGS} tags")
    elif include_recipes:
        print("[dataset] include_recipes=True but nodes_recipe table not found, skipping")

    con.close()

    # ── Substitution edges (optional) ──────────────────────────────────────
    subs_pos_edge_index = None
    if subs_csv_path and os.path.exists(subs_csv_path):
        subs_df = pd.read_csv(subs_csv_path, sep=";")

        def uri_to_ndb(u): return int(u.split("#")[1])

        src_list, dst_list = [], []
        for _, row in subs_df.iterrows():
            src_fdc = 1_000_000 + uri_to_ndb(row["Food id"])
            dst_fdc = 1_000_000 + uri_to_ndb(row["Substitution id"])
            if src_fdc in food_id_to_idx and dst_fdc in food_id_to_idx:
                src_list.append(food_id_to_idx[src_fdc])
                dst_list.append(food_id_to_idx[dst_fdc])

        if src_list:
            subs_fwd  = torch.tensor([src_list, dst_list], dtype=torch.long)
            subs_rev  = torch.tensor([dst_list, src_list], dtype=torch.long)
            subs_attr = torch.ones(len(src_list), 1)
            edge_segments.extend([subs_fwd, subs_rev])
            attr_segments.extend([subs_attr, subs_attr])
            subs_pos_edge_index = subs_fwd
            print(f"[dataset] Added {len(src_list)} substitution pairs ({len(src_list)*2} bidirectional edges, {subs_pos_edge_index.size(1)} supervised)")
        else:
            print("[dataset] Substitution CSV loaded but no pairs matched our food KB")
    elif subs_csv_path:
        print(f"[dataset] subs_csv_path not found: {subs_csv_path}")

    # ── Assemble full graph ─────────────────────────────────────────────
    TOTAL_NODES = NUM_FOODS + NUM_NUTRIENTS + NUM_RECIPES + NUM_TAGS

    edge_index_all = torch.cat(edge_segments, dim=1)
    edge_attr_all = torch.cat(attr_segments, dim=0)

    # node_type: 0=food, 1=nutrient, 2=recipe, 3=tag
    node_type = torch.zeros(TOTAL_NODES, dtype=torch.long)
    node_type[NUM_FOODS:NUM_FOODS + NUM_NUTRIENTS] = 1
    if NUM_RECIPES > 0:
        node_type[RECIPE_OFFSET:RECIPE_OFFSET + NUM_RECIPES] = 2
    if NUM_TAGS > 0:
        node_type[TAG_OFFSET:TAG_OFFSET + NUM_TAGS] = 3

    data = Data(
        edge_index=edge_index_all,
        edge_attr=edge_attr_all,
        node_type=node_type,
        num_nodes=TOTAL_NODES,
    )

    # Attach supervised tensors (trainer expects these like Colab)
    data.pos_edge_index = pos_edge_index
    data.edge_attr_pos = edge_attr_pos
    if subs_pos_edge_index is not None:
        data.subs_pos_edge_index = subs_pos_edge_index

    # Build food_to_nutrs exactly like Colab (dst_nutr WITHOUT offset)
    food_to_nutrs = [set() for _ in range(NUM_FOODS)]
    for f_idx, n_idx in zip(src_food.tolist(), dst_nutr.tolist()):
        food_to_nutrs[f_idx].add(n_idx)

    num_types = 4 if (NUM_RECIPES > 0 or NUM_TAGS > 0) else 2

    meta = {
        "NUM_FOODS": NUM_FOODS,
        "NUM_NUTRIENTS": NUM_NUTRIENTS,
        "NUM_RECIPES": NUM_RECIPES,
        "NUM_TAGS": NUM_TAGS,
        "RECIPE_OFFSET": RECIPE_OFFSET,
        "TAG_OFFSET": TAG_OFFSET,
        "NUM_TYPES": num_types,
        "foods": foods,                       # used later for descriptions/category labels
        "nutrs": nutrs,
        "food_id_to_idx": food_id_to_idx,
        "nutr_id_to_idx": nutr_id_to_idx,
        "food_to_nutrs": food_to_nutrs,       # required by negative sampler
    }

    print(f"[dataset] Total graph: {TOTAL_NODES} nodes, "
          f"{edge_index_all.size(1)} edges, {num_types} node types")

    return data, meta