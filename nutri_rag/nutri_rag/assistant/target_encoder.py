"""Encode a structured nutritional target into query vectors (Phase C).

For the nutrition→Food sub-problem, the "query" is a structured target dict
like {protein_g: 30, fat_g: 20, carb_g: 60, energy_kcal: 500} — not a known
food. To do score-fusion hybrid retrieval the way HealthyFoodSubs does, both
q_text and q_gat must exist.

Construction:
    q_text = TextEmbedder.encode(prose_description_of_target)
    q_gat  = weighted_mean(GAT[nutrient_node])  for each macro,
             weights = normalized target magnitudes.

The nutrient node GAT embeddings live in node_embeddings.npy at offsets that
depend on the database ordering; the mapping is built one-off by
nutri_graph/scripts/build_nutrient_emb_index.py and read here as JSON.

This gives us the HealthyFoodSubs precondition: both q_text and q_gat exist
before fusion (because nutrient nodes ARE graph nodes), so we can plug
straight into hybrid_rank() with no pseudo-anchor needed.
"""
from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

# Default location of the index produced by build_nutrient_emb_index.py
DEFAULT_INDEX_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "nutri_graph", "data", "nutrient_emb_index.json",
)
DEFAULT_NODE_EMB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "nutri_graph", "outputs", "embeddings", "node_embeddings.npy",
)

# Target dict key → DB nutrient name (matches gap_analyzer output keys)
_KEY_TO_NUTRIENT = {
    "protein_g":   "Protein",
    "fat_g":       "Total lipid (fat)",
    "carb_g":      "Carbohydrate, by difference",
    "energy_kcal": "Energy",
}

# Module-level cache for the loaded artifacts.
_index_cache: dict | None = None
_nutrient_emb_cache: np.ndarray | None = None  # only the nutrient slice, L2-normalized


def _load_index(path: str = DEFAULT_INDEX_JSON) -> dict:
    global _index_cache
    if _index_cache is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"nutrient embedding index not found at {path}. "
                "Run: python nutri_graph/scripts/build_nutrient_emb_index.py"
            )
        with open(path) as f:
            _index_cache = json.load(f)
    return _index_cache


def _load_nutrient_embeddings(node_emb_path: str = DEFAULT_NODE_EMB) -> np.ndarray:
    """Load just the nutrient slice of node_embeddings.npy, L2-normalized."""
    global _nutrient_emb_cache
    if _nutrient_emb_cache is not None:
        return _nutrient_emb_cache
    idx = _load_index()
    if not os.path.exists(node_emb_path):
        raise FileNotFoundError(
            f"node embeddings not found at {node_emb_path}. "
            "Train nutri_graph first."
        )
    node_emb = np.load(node_emb_path)
    nf = int(idx["num_foods"])
    nn = int(idx["num_nutrients"])
    if node_emb.shape[0] < nf + nn:
        raise ValueError(
            f"node_embeddings has only {node_emb.shape[0]} rows but expected "
            f"≥ {nf + nn} (num_foods + num_nutrients)"
        )
    slice_ = node_emb[nf:nf + nn]
    norms = np.linalg.norm(slice_, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    _nutrient_emb_cache = (slice_ / norms).astype(np.float32)
    return _nutrient_emb_cache


def _nutrient_gat_vec(nutrient_name: str) -> np.ndarray | None:
    """Return the L2-normalized GAT embedding for a named nutrient, or None."""
    idx = _load_index()
    nid = idx["nutrient_name_to_id"].get(nutrient_name)
    if nid is None:
        return None
    row_idx = idx["nutrient_id_to_row"].get(str(nid))
    if row_idx is None:
        return None
    nutrient_emb = _load_nutrient_embeddings()
    # row_idx is absolute in node_embeddings — convert to slice-relative
    rel = int(row_idx) - int(idx["num_foods"])
    if rel < 0 or rel >= nutrient_emb.shape[0]:
        return None
    return nutrient_emb[rel]


def _format_target_description(targets: dict[str, float]) -> str:
    """Prose description of the nutritional target for text encoding."""
    return (
        f"A balanced meal with approximately "
        f"{targets.get('protein_g', 0):.0f}g protein, "
        f"{targets.get('fat_g', 0):.0f}g fat, "
        f"{targets.get('carb_g', 0):.0f}g carbohydrate, "
        f"and {targets.get('energy_kcal', 0):.0f} kcal per serving."
    )


def encode_target(
    targets: dict[str, float],
    keys: Sequence[str] = ("protein_g", "fat_g", "carb_g"),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_text, q_gat) for a structured nutritional target.

    Args:
        targets: dict with keys like protein_g, fat_g, carb_g, energy_kcal
                 (matches the output of gap_analyzer.analyze_gap).
        keys: which target keys to use as anchors for the q_gat blend. The
              default uses the three macro keys (energy is implicit in them).

    Both vectors are L2-normalized.

    q_text comes from the same TextEmbedder used everywhere else in the
    pipeline. q_gat is a magnitude-weighted blend of the nutrient nodes'
    GAT embeddings — heavier on whichever macro the gap analyzer wants most.
    """
    # Lazy import to avoid loading TextEmbedder unless this path is used
    from nutri_rag.search import _get_embedder
    from nutri_rag.embedding import FOOD_SEARCH_INSTRUCTION

    embedder = _get_embedder()
    desc = _format_target_description(targets)
    q_text = embedder.encode([desc], task_instruction=FOOD_SEARCH_INSTRUCTION)[0]
    # Defensive normalize
    n = float(np.linalg.norm(q_text))
    if n > 0:
        q_text = q_text / n

    # q_gat: weighted mean of nutrient node embeddings
    weights = []
    vecs = []
    total = sum(max(float(targets.get(k, 0)), 0.0) for k in keys)
    if total <= 0:
        # All zero target — fall back to uniform weights
        total = 1.0
        weights_per_key = {k: 1.0 / len(keys) for k in keys}
    else:
        weights_per_key = {
            k: max(float(targets.get(k, 0)), 0.0) / total for k in keys
        }
    for k, w in weights_per_key.items():
        if w <= 0:
            continue
        nutrient_name = _KEY_TO_NUTRIENT.get(k)
        if nutrient_name is None:
            continue
        v = _nutrient_gat_vec(nutrient_name)
        if v is None:
            continue
        weights.append(w)
        vecs.append(v)

    if not vecs:
        # No nutrient GAT vectors found — return a zero q_gat. The caller
        # (hybrid_rank with alpha=0) will gracefully degrade to text-only.
        idx = _load_index()
        # Shape inferred from any nutrient embedding (load once just for dim)
        dim = _load_nutrient_embeddings().shape[1]
        q_gat = np.zeros(dim, dtype=np.float32)
    else:
        q_gat = np.average(np.stack(vecs), axis=0, weights=weights)
        n = float(np.linalg.norm(q_gat))
        if n > 0:
            q_gat = q_gat / n
    return q_text.astype(np.float32), q_gat.astype(np.float32)


def macro_match(
    targets: dict[str, float],
    nutrients: dict[str, float],
) -> float:
    """Structured fit term in [0, 1]: how close the food's per-100g macros
    are to the per-100g target proportions.

    Compares ratios rather than absolute amounts since `nutrients` is per-100g
    while `targets` is per-serving. Uses negative L1 distance over relative
    macro shares, mapped to [0, 1] via 1 - dist/2.
    """
    # Targets share (within macros)
    macros = ["protein_g", "fat_g", "carb_g"]
    t_vals = [max(float(targets.get(k, 0)), 0.0) for k in macros]
    t_total = sum(t_vals) or 1.0
    t_share = [v / t_total for v in t_vals]

    # Food share — match key names to nutrient names
    name_map = {
        "protein_g": "Protein",
        "fat_g":     "Total lipid (fat)",
        "carb_g":    "Carbohydrate, by difference",
    }
    f_vals = [max(float(nutrients.get(name_map[k], 0)), 0.0) for k in macros]
    f_total = sum(f_vals)
    if f_total <= 0:
        return 0.0
    f_share = [v / f_total for v in f_vals]

    # L1 distance is in [0, 2]; map to similarity in [0, 1]
    l1 = sum(abs(a - b) for a, b in zip(t_share, f_share))
    return max(0.0, min(1.0, 1.0 - l1 / 2.0))
