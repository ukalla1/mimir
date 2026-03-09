import duckdb
import numpy as np
import torch
from torch_geometric.data import Data


def build_graph_from_db(db_path: str):
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

    con.close()

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

    # full propagation graph
    edge_index_all = torch.cat([pos_edge_index, rev_edge_index], dim=1)

    # edge_attr: log1p(amount) for positive edges; duplicated for reverse edges
    edge_amount = np.log1p(edges["amount"].to_numpy(dtype=np.float32))
    edge_attr_pos = torch.tensor(edge_amount, dtype=torch.float32).view(-1, 1)
    edge_attr_all = torch.cat([edge_attr_pos, edge_attr_pos], dim=0)

    # node_type: 0 food, 1 nutrient
    node_type = torch.zeros(NUM_FOODS + NUM_NUTRIENTS, dtype=torch.long)
    node_type[NUM_FOODS:] = 1

    data = Data(
        edge_index=edge_index_all,
        edge_attr=edge_attr_all,
        node_type=node_type,
        num_nodes=(NUM_FOODS + NUM_NUTRIENTS),
    )

    # Attach supervised tensors (trainer expects these like Colab)
    data.pos_edge_index = pos_edge_index
    data.edge_attr_pos = edge_attr_pos

    # Build food_to_nutrs exactly like Colab (dst_nutr WITHOUT offset)
    food_to_nutrs = [set() for _ in range(NUM_FOODS)]
    for f_idx, n_idx in zip(src_food.tolist(), dst_nutr.tolist()):
        food_to_nutrs[f_idx].add(n_idx)

    meta = {
        "NUM_FOODS": NUM_FOODS,
        "NUM_NUTRIENTS": NUM_NUTRIENTS,
        "foods": foods,                       # used later for descriptions/category labels
        "nutrs": nutrs,
        "food_id_to_idx": food_id_to_idx,
        "nutr_id_to_idx": nutr_id_to_idx,
        "food_to_nutrs": food_to_nutrs,       # required by negative sampler
    }

    return data, meta