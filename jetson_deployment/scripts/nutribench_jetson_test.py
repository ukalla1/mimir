#!/usr/bin/env python3
"""Run NutriBench questions against the local llama.cpp chat server on the Jetson.

Reads questions extracted by `nutribench_jetson_questions.py`, builds the exact
same prompt format used by the server-side benchmark (with optional USDA reference
block from RAG retrieval), sends each one to the on-device LLM, parses the
predicted nutrient value, compares to ground truth, and writes both per-question
predictions and a summary report.

Modes
-----
- baseline : no RAG. System prompt + few-shot examples + raw query.
             Only requires the chat LLM server (Qwen3.5 via llama.cpp).
- text     : V1 RAG. Text embedding cosine search per food item.
             Requires chat server + llama.cpp embedding server.
- gat      : V2 RAG. Text retrieval + GAT-neighbor expansion (single best per item).
             Requires chat + embedding server + GAT embeddings + DuckDB.
- hybrid   : V3 RAG. Same as gat but multi-candidate per item (top-K, threshold gated).
             Requires chat + embedding server + GAT embeddings + DuckDB.

Usage
-----
    # baseline (no RAG)
    python nutribench_jetson_test.py --mode baseline

    # text-only RAG
    python nutribench_jetson_test.py --mode text \\
        --text-embeddings ../data/food_text_embeddings.npy \\
        --text-fdc-ids    ../data/food_fdc_ids.npy \\
        --duckdb          ../data/nutri_kb.duckdb

    # gat or hybrid (also needs GAT embeddings)
    python nutribench_jetson_test.py --mode hybrid \\
        --text-embeddings ../data/food_text_embeddings.npy \\
        --text-fdc-ids    ../data/food_fdc_ids.npy \\
        --gat-embeddings  ../data/food_embeddings.npy \\
        --duckdb          ../data/nutri_kb.duckdb
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.stderr.write("ERROR: `requests` not installed. Run: pip install requests\n")
    sys.exit(1)


# ── Nutrient configuration (mirrors nutri_rag/bench/nutrient_prompts.py) ──────

NUTRIENT_CONFIG = {
    "carb": {
        "full_name": "carbohydrates",
        "json_key":  "total_carbohydrates",
        "unit":      "grams",
        "gt_column": "carb",
        "acc_threshold": 7.5,
    },
    "protein": {
        "full_name": "protein",
        "json_key":  "total_protein",
        "unit":      "grams",
        "gt_column": "protein",
        "acc_threshold": 7.5,
    },
    "fat": {
        "full_name": "fat",
        "json_key":  "total_fat",
        "unit":      "grams",
        "gt_column": "fat",
        "acc_threshold": 7.5,
    },
    "energy": {
        "full_name": "energy",
        "json_key":  "total_energy",
        "unit":      "kcal",
        "gt_column": "energy",
        "acc_threshold": 50.0,
    },
}

_EXAMPLES = {
    "carb": [
        ("This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
         "The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n"
         "1 cup of oatmeal has 27g carbs.\n"
         "1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.\n"
         "1 glass of orange juice has 26g carbs.\n"
         "So the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5",
         66.5),
        ("I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
         "The meal consists of scrambled eggs made with 2 eggs and 1 toast.\n"
         "Scrambled eggs made with 2 eggs has 2g carbs.\n"
         "1 toast has 13g carbs.\n"
         "So the total grams of carbs in the meal = (2 + 13) = 15",
         15),
        ("Half a peanut butter and jelly sandwich.",
         "The meal consists of 1/2 a peanut butter and jelly sandwich.\n"
         "1 peanut butter and jelly sandwich has 50g carbs so half has (50*(1/2)) = 25g carbs.\n"
         "So the total grams of carbs in the meal = 25",
         25),
    ],
    "protein": [
        ("This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
         "1 cup of oatmeal has 6g protein.\n"
         "1 banana has 1g protein so half has 0.5g.\n"
         "1 glass of orange juice has 2g protein.\n"
         "Total = (6 + 0.5 + 2) = 8.5",
         8.5),
        ("I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
         "2-egg scrambled eggs has 13g protein.\n"
         "1 toast has 3g protein.\n"
         "Total = (13 + 3) = 16",
         16),
    ],
    "fat": [
        ("This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
         "1 cup of oatmeal has 3g fat.\n"
         "1 banana has 0.4g fat so half has 0.2g.\n"
         "1 glass of orange juice has 0.5g fat.\n"
         "Total = (3 + 0.2 + 0.5) = 3.7",
         3.7),
        ("I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
         "2-egg scrambled eggs has 15g fat.\n"
         "1 toast has 1g fat.\n"
         "Total = (15 + 1) = 16",
         16),
    ],
    "energy": [
        ("This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
         "1 cup of oatmeal has 154 kcal.\n"
         "1 banana has 105 kcal so half a banana has (105*(1/2)) = 52.5 kcal.\n"
         "1 glass of orange juice has 112 kcal.\n"
         "So the total energy in the meal = (154 + 52.5 + 112) = 318.5",
         318.5),
        ("I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
         "Scrambled eggs made with 2 eggs has 182 kcal.\n"
         "1 toast has 79 kcal.\n"
         "So the total energy in the meal = (182 + 79) = 261",
         261),
        ("Half a peanut butter and jelly sandwich.",
         "1 peanut butter and jelly sandwich has 376 kcal so half has (376*(1/2)) = 188 kcal.\n"
         "So the total energy in the meal = 188",
         188),
    ],
}


def build_system_prompt(nutrient: str) -> str:
    cfg = NUTRIENT_CONFIG[nutrient]
    full_name, json_key, unit = cfg["full_name"], cfg["json_key"], cfg["unit"]

    lines = [
        "For the given query including a meal description, think step by step as follows:",
        "1. Parse the meal description into discrete food or beverage items along with their serving size. "
        "If the serving size of any item in the meal is not specified, assume it is a single standard serving "
        "based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't "
        "relate to the item name and serving size.",
        f"2. For each food or beverage item in the meal, calculate the amount of {full_name} in {unit} for the specific serving size.",
        f"3. Respond with a dictionary object containing the total {full_name} in {unit} as follows:",
        f'{{"{json_key}": total {unit} of {full_name} for the serving}}',
        f"For the total {full_name}, respond with just the numeric amount without extra text. "
        f'If you don\'t know the answer, set the value of "{json_key}" to -1.',
        "",
        "Follow the format of the following examples when answering",
    ]

    for query, reasoning, val in _EXAMPLES[nutrient]:
        lines.append("")
        lines.append(f'Query: "{query}"')
        lines.append("Answer: Let's think step by step.")
        lines.append(reasoning)
        lines.append(f'Output: {{"{json_key}": {val}}}')

    return "\n".join(lines)


def build_cot_query(meal_description: str) -> str:
    return f'Query: "{meal_description}"\nAnswer: Let\'s think step by step.'


# ── Output parsing (mirrors nutri_rag/bench/task_utils.clean_output) ─────────

def parse_prediction(raw_output: str, nutrient: str) -> float:
    if not isinstance(raw_output, str):
        raw_output = str(raw_output)

    # Strip Qwen3.5 <think> blocks if present
    raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    splits = raw_output.split("Output:")
    if len(splits) > 1:
        raw_output = splits[-1]
    raw_output = raw_output.strip()

    patterns = {
        "fat":     r'["\']\s*total_fat["\']: (-?[0-9]+(?:\.[0-9]*)?)',
        "protein": r'["\']\s*total_protein["\']: (-?[0-9]+(?:\.[0-9]*)?)',
        "energy":  r'["\']\s*total_energy["\']: (-?[0-9]+(?:\.[0-9]*)?)',
        "carb":    r'["\']\s*total_carbohydrates["\']:\s*["\']?(-?[0-9]+(?:\.[0-9]*)?)["\']?',
    }
    pattern = patterns[nutrient]

    match = re.search(pattern, raw_output)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            return -1.0

    try:
        return float(raw_output)
    except ValueError:
        return -1.0


# ═══════════════════════════════════════════════════════════════════════════
#   RAG retrieval — text / gat / hybrid modes
# ═══════════════════════════════════════════════════════════════════════════

# Lazy-init singletons (populated on first call when RAG mode is active)
_text_emb_matrix = None   # (N, dim)
_text_fdc_ids    = None   # (N,)
_fdc_to_arr_idx  = None   # dict[int, int]
_gat_emb_matrix  = None   # (N, 64)
_kb_con          = None
_embedding_endpoint = None
_food_search_instruction = (
    "Given a food or beverage description from a user's meal log, "
    "retrieve the most likely USDA food entries that match it."
)

KEY_NUTRIENTS = [
    "Carbohydrate, by difference",
    "Protein",
    "Total lipid (fat)",
    "Energy",
    "Energy (Atwater General Factors)",
    "Energy (Atwater Specific Factors)",
]


def _l2_normalize(mat):
    import numpy as np
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def _load_text_index(emb_path: Path, ids_path: Path) -> None:
    global _text_emb_matrix, _text_fdc_ids, _fdc_to_arr_idx
    if _text_emb_matrix is not None:
        return
    import numpy as np
    if not emb_path.exists():
        sys.exit(f"ERROR: text embeddings not found: {emb_path}")
    if not ids_path.exists():
        sys.exit(f"ERROR: text fdc_ids not found: {ids_path}")
    mat = np.load(emb_path).astype(np.float32)
    ids = np.load(ids_path).astype(np.int64)
    # Defensive normalization (text embeddings are usually already normalized)
    mat = _l2_normalize(mat)
    _text_emb_matrix = mat
    _text_fdc_ids    = ids
    _fdc_to_arr_idx  = {int(f): int(i) for i, f in enumerate(ids)}
    print(f"  Loaded text index: {mat.shape[0]} foods × {mat.shape[1]}-dim")


def _load_gat_index(emb_path: Path) -> None:
    global _gat_emb_matrix
    if _gat_emb_matrix is not None:
        return
    import numpy as np
    if not emb_path.exists():
        sys.exit(f"ERROR: GAT embeddings not found: {emb_path}")
    mat = np.load(emb_path).astype(np.float32)
    mat = _l2_normalize(mat)
    _gat_emb_matrix = mat
    print(f"  Loaded GAT index:  {mat.shape[0]} foods × {mat.shape[1]}-dim")


def _get_kb(db_path: Path):
    global _kb_con
    if _kb_con is None:
        import duckdb
        if not db_path.exists():
            sys.exit(f"ERROR: DuckDB not found: {db_path}")
        _kb_con = duckdb.connect(str(db_path), read_only=True)
    return _kb_con


def _get_description(fdc_id: int, db_path: Path) -> str:
    con = _get_kb(db_path)
    row = con.execute(
        "SELECT description FROM nodes_food WHERE fdc_id = ?", [int(fdc_id)]
    ).fetchone()
    return row[0] if row else ""


def _get_nutrients(fdc_id: int, db_path: Path) -> dict:
    con = _get_kb(db_path)
    df = con.execute(
        "SELECT n.nutrient_name, e.amount "
        "FROM edges_food_contains_nutrient e "
        "JOIN nodes_nutrient n USING(nutrient_id) "
        "WHERE e.fdc_id = ? ORDER BY e.amount DESC",
        [int(fdc_id)]
    ).fetchall()
    return {name: float(amt) for name, amt in df if name in KEY_NUTRIENTS}


def _embed_query(text: str, with_instruction: bool = True) -> "np.ndarray":
    """Embed text via llama.cpp embedding server. Returns L2-normalized vector."""
    import numpy as np
    if with_instruction:
        payload_text = f"Instruct: {_food_search_instruction}\nQuery: {text}"
    else:
        payload_text = text
    try:
        r = requests.post(_embedding_endpoint, json={"content": payload_text}, timeout=60)
        r.raise_for_status()
    except Exception as e:
        sys.exit(f"ERROR calling embedding endpoint {_embedding_endpoint}: {e}")
    data = r.json()
    # llama.cpp returns {"embedding": [...]} or {"embedding": [[...]]}
    emb = data["embedding"] if "embedding" in data else data[0]["embedding"]
    arr = np.array(emb, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    arr = arr / (np.linalg.norm(arr) + 1e-10)
    return arr


# ── Food term extraction (ported from nutri_rag/bench/retriever.py) ──────────

_PAT_QTY_OF = re.compile(
    r'\d+(?:\.\d+)?\s*(?:g(?:rams?)?)\s+(?:of\s+)'
    r'([\w\s,\'-]+?)(?=\s*(?:,|\band\b|\balong\b|\.|$))',
    re.IGNORECASE,
)
_PAT_WEIGHING = re.compile(
    r'((?:\w+\s+){0,3}\w+)\s+weighing\s+\d+',
    re.IGNORECASE,
)
_PAT_INLINE = re.compile(
    r'((?:\w+\s+){0,3}\w+)\s*\(\d+(?:\.\d+)?\s*g\)',
    re.IGNORECASE,
)
_STRIP_WORDS = re.compile(
    r'\b(?:raw|boiled|fried|baked|cooked|roasted|steamed|grilled|dried|fresh|'
    r'ripe|unripe|large|small|medium|plain|whole|chopped|sliced|diced|'
    r'peeled|unpeeled|skinless|boneless|without\s+skin|in\s+their\s+shells?|'
    r'weighing|along|the|a|an|ate|had|consumed|got|sprinkled)\b',
    re.IGNORECASE,
)


def extract_food_terms(meal: str) -> list:
    raw_matches = []
    raw_matches.extend(_PAT_QTY_OF.findall(meal))
    raw_matches.extend(_PAT_WEIGHING.findall(meal))
    raw_matches.extend(_PAT_INLINE.findall(meal))

    terms = []
    seen = set()
    for raw in raw_matches:
        cleaned = _STRIP_WORDS.sub(' ', raw)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().rstrip('.')
        if len(cleaned) < 2:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            terms.append(cleaned)

    # Fallback: if no terms extracted, use the meal description itself as one term
    if not terms:
        terms = [meal.strip()]
    return terms


# ── Text search with macro filtering (ported from nutri_rag/search.py) ───────

def _get_macro_counts(fdc_ids: list, db_path: Path) -> dict:
    """Return {fdc_id: macro_count}. Higher count = food has more macro data."""
    if not fdc_ids:
        return {}
    con = _get_kb(db_path)
    placeholders = ", ".join(str(int(f)) for f in fdc_ids)
    rows = con.execute(f"""
        SELECT e.fdc_id, COUNT(*) AS macro_count
        FROM edges_food_contains_nutrient e
        JOIN nodes_nutrient n USING(nutrient_id)
        WHERE e.fdc_id IN ({placeholders})
          AND n.nutrient_name IN (
              'Carbohydrate, by difference', 'Protein', 'Total lipid (fat)'
          )
        GROUP BY e.fdc_id
    """).fetchall()
    return {int(fid): int(cnt) for fid, cnt in rows}


def text_search_with_vec(q_vec, k: int, db_path: Path) -> list:
    """Top-K text matches with 5x oversampling and macro-count re-ranking."""
    import numpy as np
    n_candidates = k * 5
    sims = _text_emb_matrix @ q_vec        # (N,)
    # Get top-n_candidates indices
    top_n = min(n_candidates, len(sims))
    top_idx = np.argpartition(-sims, top_n - 1)[:top_n]
    top_idx = top_idx[np.argsort(-sims[top_idx])]   # sorted desc

    candidates = []
    for arr_idx in top_idx:
        fdc_id = int(_text_fdc_ids[arr_idx])
        candidates.append({
            "fdc_id": fdc_id,
            "description": _get_description(fdc_id, db_path),
            "text_score": float(sims[arr_idx]),
            "arr_idx": int(arr_idx),
        })

    # Re-rank: prefer entries that have macro data, then by text score
    fdc_ids = [c["fdc_id"] for c in candidates]
    macro_counts = _get_macro_counts(fdc_ids, db_path)
    for c in candidates:
        c["macro_count"] = macro_counts.get(c["fdc_id"], 0)
    candidates.sort(key=lambda c: (-c["macro_count"], -c["text_score"]))
    return candidates[:k]


# ── GAT expansion (ported from nutri_rag/search.py:_gat_expand) ──────────────

def _gat_neighbors(arr_idx: int, k: int) -> list:
    """Find k nearest GAT neighbors. Returns [(arr_idx, gat_sim), ...]."""
    import numpy as np
    sims = _gat_emb_matrix[arr_idx] @ _gat_emb_matrix.T   # (N,)
    top_k = np.argpartition(sims, -(k + 1))[-(k + 1):]
    top_k = top_k[top_k != arr_idx]
    top_k = top_k[np.argsort(sims[top_k])[::-1]][:k]
    return [(int(i), float(sims[i])) for i in top_k]


def gat_expand(text_cands: list, q_vec, n_unique: int, gat_neighbors: int,
               db_path: Path) -> list:
    """Expand text candidates with GAT neighbors, re-score by text similarity."""
    import numpy as np

    # Step 1: dedup by description, keep top n_unique by text score
    seen_desc = set()
    unique_seeds = []
    for c in sorted(text_cands, key=lambda x: -x["text_score"]):
        if c["description"] in seen_desc:
            continue
        seen_desc.add(c["description"])
        unique_seeds.append(c)
        if len(unique_seeds) >= n_unique:
            break

    # Step 2: find GAT neighbors for each seed
    existing_indices = {c["arr_idx"] for c in text_cands}
    neighbor_rows = []
    for seed in unique_seeds:
        for neigh_idx, gat_sim in _gat_neighbors(seed["arr_idx"], gat_neighbors):
            if neigh_idx in existing_indices:
                continue
            existing_indices.add(neigh_idx)
            neighbor_rows.append({"arr_idx": neigh_idx, "gat_sim": gat_sim})

    if not neighbor_rows:
        return text_cands

    # Step 3: fetch fdc_id + description for each neighbor
    for nr in neighbor_rows:
        fdc_id = int(_text_fdc_ids[nr["arr_idx"]])
        nr["fdc_id"] = fdc_id
        nr["description"] = _get_description(fdc_id, db_path)

    # Step 4: re-score neighbors by text similarity to original query
    neigh_indices = [nr["arr_idx"] for nr in neighbor_rows]
    neigh_vecs = _text_emb_matrix[neigh_indices]      # (M, dim)
    text_scores = (neigh_vecs @ q_vec).flatten()      # (M,)
    for nr, score in zip(neighbor_rows, text_scores):
        nr["text_score"] = float(score)

    # Step 5: combine original + expanded
    combined = list(text_cands) + neighbor_rows
    combined.sort(key=lambda c: -c["text_score"])
    return combined


# ── USDA reference block builders (ported from nutri_rag/bench/prompt.py) ────

_DISPLAY = {
    "Carbohydrate, by difference": "Carbohydrate",
    "Protein": "Protein",
    "Total lipid (fat)": "Fat",
}
_ENERGY_KEYS = ["Energy", "Energy (Atwater General Factors)", "Energy (Atwater Specific Factors)"]
_REQUIRED_NUTRIENT = {
    "carb": "Carbohydrate, by difference",
    "protein": "Protein",
    "fat": "Total lipid (fat)",
    "energy": None,
}


def _format_nutrient_values(nutrients: dict) -> str:
    parts = []
    for key, name in _DISPLAY.items():
        v = nutrients.get(key)
        if v is not None:
            parts.append(f"{name}: {v:.1f}g")
    for ek in _ENERGY_KEYS:
        v = nutrients.get(ek)
        if v is not None:
            parts.append(f"Energy: {v:.1f}kcal")
            break
    return " | ".join(parts)


def _has_target_nutrient(nutrients: dict, nutrient: str) -> bool:
    req = _REQUIRED_NUTRIENT.get(nutrient)
    if req:
        return nutrients.get(req) is not None
    return any(nutrients.get(ek) is not None for ek in _ENERGY_KEYS)


def _format_per_item(contexts: list, nutrient: str, threshold: float) -> str:
    """Per-item format: one best USDA match per food term (V1/V2)."""
    lines = [
        "=== USDA Reference (per 100g) ===",
        "Below are possible USDA matches for some ingredients.",
        "Ignore wrong matches and use your own knowledge.",
        "The reference may only cover some of the items — estimate the rest yourself",
        "Keep reasoning brief",
        "",
    ]
    has_any = False
    for ctx in contexts:
        term = ctx["food_term"]
        cands = ctx["candidates"]
        if not cands:
            lines.append(f"- {term}: no reliable USDA match — use your own knowledge")
            continue

        best = cands[0]
        reliable = (
            best.get("text_score", 0.0) >= threshold
            and _has_target_nutrient(best.get("nutrients", {}), nutrient)
        )
        if reliable:
            has_any = True
            nutr = _format_nutrient_values(best["nutrients"])
            lines.append(f'- {term}: USDA match → "{best["description"]}" — {nutr}')
        else:
            lines.append(f"- {term}: no reliable USDA match — use your own knowledge")
    lines.append("===")
    return "\n".join(lines) if has_any else ""


def _format_multi_candidate(contexts: list, nutrient: str, threshold: float) -> str:
    """Multi-candidate format: top-K USDA matches per food term (V3)."""
    lines = [
        "=== USDA Reference (per 100g) ===",
        "Below are possible USDA matches for some ingredients.",
        "Ignore wrong matches and use your own knowledge.",
        "The reference may only cover some of the items — estimate the rest yourself",
        "Keep reasoning brief",
        "",
    ]
    has_any = False
    for ctx in contexts:
        term = ctx["food_term"]
        valid = [
            c for c in ctx["candidates"]
            if c.get("text_score", 0.0) >= threshold
               and _has_target_nutrient(c.get("nutrients", {}), nutrient)
        ]
        if not valid:
            continue
        has_any = True
        lines.append(f"- {term}:")
        for i, c in enumerate(valid, 1):
            nutr = _format_nutrient_values(c["nutrients"])
            lines.append(f'  {i}. "{c["description"]}" — {nutr}')
        lines.append("")
    lines.append("===")
    return "\n".join(lines) if has_any else ""


def build_usda_block(contexts: list, nutrient: str, mode: str, threshold: float) -> str:
    if not contexts:
        return ""
    if mode == "hybrid":
        return _format_multi_candidate(contexts, nutrient, threshold)
    else:  # text or gat
        return _format_per_item(contexts, nutrient, threshold)


# ── Full RAG retrieval per question ──────────────────────────────────────────

def retrieve_contexts(meal: str, mode: str, args) -> list:
    """Return list of {food_term, candidates: [{fdc_id, description, text_score, nutrients}]}."""
    terms = extract_food_terms(meal)
    contexts = []
    db_path = args.duckdb

    for term in terms:
        q_vec = _embed_query(term, with_instruction=True)
        cands = text_search_with_vec(q_vec, k=args.top_k_foods, db_path=db_path)

        if mode in ("gat", "hybrid"):
            cands = gat_expand(cands, q_vec,
                               n_unique=args.gat_n_unique,
                               gat_neighbors=args.gat_neighbors,
                               db_path=db_path)
            cands = cands[: max(args.top_k_foods, args.gat_n_unique)]

        # Attach nutrient profiles
        for c in cands:
            c["nutrients"] = _get_nutrients(c["fdc_id"], db_path)
        contexts.append({"food_term": term, "candidates": cands})
    return contexts


# ── LLM call ──────────────────────────────────────────────────────────────────

def chat_completion(endpoint, model, system_prompt, user_message,
                    temperature=0.0, max_tokens=4096, timeout=300):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_data = here.parent / "data"
    p = argparse.ArgumentParser(description="Run NutriBench questions on the Jetson LLM.")

    p.add_argument("--questions", type=Path,
                   default=default_data / "nutribench_jetson_questions.json")
    p.add_argument("--out", type=Path, default=None,
                   help="Output predictions JSON (default: data/results/<mode>_<nutrient>_n<N>.json).")

    p.add_argument("--mode", default="baseline",
                   choices=["baseline", "text", "gat", "hybrid"],
                   help="Retrieval mode.")
    p.add_argument("--nutrient", default="carb", choices=list(NUTRIENT_CONFIG.keys()))

    p.add_argument("--endpoint", default="http://localhost:8080/v1/chat/completions")
    p.add_argument("--model", default="qwen3.5")
    p.add_argument("--embedding-endpoint", default="http://localhost:8081/embedding",
                   help="llama.cpp embedding server URL (used by text/gat/hybrid).")

    # RAG resource paths
    p.add_argument("--text-embeddings", type=Path,
                   default=default_data / "food_text_embeddings.npy")
    p.add_argument("--text-fdc-ids",    type=Path,
                   default=default_data / "food_fdc_ids.npy")
    p.add_argument("--gat-embeddings",  type=Path,
                   default=default_data / "food_embeddings.npy")
    p.add_argument("--duckdb",          type=Path,
                   default=default_data / "nutri_kb.duckdb")

    # RAG hyperparams
    p.add_argument("--top-k-foods",   type=int,   default=3)
    p.add_argument("--gat-neighbors", type=int,   default=5)
    p.add_argument("--gat-n-unique",  type=int,   default=5)
    p.add_argument("--sim-threshold", type=float, default=0.60)

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens",  type=int,   default=4096)
    p.add_argument("--limit",       type=int,   default=None)
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def setup_rag(args) -> None:
    """Load embedding indices + DuckDB based on the mode."""
    global _embedding_endpoint
    if args.mode == "baseline":
        return
    _embedding_endpoint = args.embedding_endpoint
    print(f"Setting up RAG mode='{args.mode}'...")
    _load_text_index(args.text_embeddings, args.text_fdc_ids)
    if args.mode in ("gat", "hybrid"):
        _load_gat_index(args.gat_embeddings)
    # Trigger DuckDB connection
    _get_kb(args.duckdb)
    print(f"  DuckDB:    {args.duckdb}")
    print(f"  Embedding endpoint: {_embedding_endpoint}")
    # Quick sanity-check the embedding endpoint
    try:
        v = _embed_query("test", with_instruction=False)
        print(f"  Embedding probe: dim={len(v)} ✓")
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"ERROR: embedding endpoint probe failed: {e}")


def main() -> None:
    args = parse_args()

    if not args.questions.exists():
        sys.exit(f"ERROR: questions file not found: {args.questions}")

    with open(args.questions) as f:
        payload = json.load(f)
    questions = payload["questions"]
    if args.limit:
        questions = questions[: args.limit]

    cfg = NUTRIENT_CONFIG[args.nutrient]
    gt_field = cfg["gt_column"]
    threshold_score = cfg["acc_threshold"]

    missing = [q["id"] for q in questions if gt_field not in q]
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} questions missing '{gt_field}' field. "
            f"Re-extract questions with --nutrient {args.nutrient} or with no nutrient filter."
        )

    setup_rag(args)

    system_prompt = build_system_prompt(args.nutrient)
    print()
    print(f"Mode:      {args.mode}")
    print(f"Nutrient:  {args.nutrient}  (gt_field='{gt_field}', threshold=±{threshold_score} {cfg['unit']})")
    print(f"Endpoint:  {args.endpoint}")
    print(f"Model:     {args.model}")
    print(f"Questions: {len(questions)}")
    print()

    results = []
    correct = 0
    mae_sum = 0.0
    failures = 0
    t_total_start = time.time()

    for q in questions:
        meal = q["meal_description"]
        gt = float(q[gt_field])
        t_q_start = time.time()

        # Build user message (with or without USDA block)
        usda_block = ""
        contexts_summary = None
        if args.mode == "baseline":
            user_msg = build_cot_query(meal)
        else:
            try:
                contexts = retrieve_contexts(meal, args.mode, args)
                usda_block = build_usda_block(contexts, args.nutrient,
                                              args.mode, args.sim_threshold)
                contexts_summary = [
                    {
                        "food_term": ctx["food_term"],
                        "n_candidates": len(ctx["candidates"]),
                        "best": (ctx["candidates"][0]["description"]
                                 if ctx["candidates"] else None),
                    }
                    for ctx in contexts
                ]
            except SystemExit:
                raise
            except Exception as e:
                usda_block = ""
                contexts_summary = [{"error": str(e)}]
            user_msg = (f"{usda_block}\n\n" if usda_block else "") + build_cot_query(meal)

        if args.verbose:
            print(f"\n--- Q{q['id']} ---")
            print(f"Meal: {meal}")
            print(f"GT  : {gt} {cfg['unit']}")
            if usda_block:
                print(f"USDA block:\n{usda_block}\n")

        try:
            raw = chat_completion(
                args.endpoint, args.model,
                system_prompt, user_msg,
                temperature=args.temperature, max_tokens=args.max_tokens,
            )
            error = None
        except Exception as e:
            raw = ""
            error = str(e)
        latency = time.time() - t_q_start

        pred = parse_prediction(raw, args.nutrient) if raw else -1.0
        if pred < 0 or error:
            failures += 1
            mae = float("nan")
            is_correct = False
        else:
            mae = abs(pred - gt)
            is_correct = mae < threshold_score
            correct += int(is_correct)
            mae_sum += mae

        results.append({
            "id": q["id"],
            "meal_description": meal,
            "gt": gt,
            "pred": pred,
            "mae": mae if mae == mae else None,
            "correct": is_correct,
            "latency_s": round(latency, 2),
            "error": error,
            "raw_response": raw,
            "contexts": contexts_summary,
        })

        marker = "✓" if is_correct else ("✗" if pred >= 0 else "!")
        print(f"[{q['id']:3d}] {marker}  gt={gt:7.2f}  pred={pred:7.2f}  "
              f"mae={mae if mae == mae else float('nan'):6.2f}  ({latency:5.1f}s)  "
              f"{meal[:60]}")

    total_time = time.time() - t_total_start
    scored = len(results) - failures
    acc = correct / scored if scored else 0.0
    mean_mae = mae_sum / scored if scored else float("nan")

    summary = {
        "config": {
            "mode": args.mode,
            "nutrient": args.nutrient,
            "endpoint": args.endpoint,
            "model": args.model,
            "embedding_endpoint": args.embedding_endpoint if args.mode != "baseline" else None,
            "n_questions": len(results),
            "questions_source": str(args.questions),
            "questions_seed": payload.get("seed"),
            "rag_params": ({
                "top_k_foods": args.top_k_foods,
                "gat_neighbors": args.gat_neighbors,
                "gat_n_unique": args.gat_n_unique,
                "sim_threshold": args.sim_threshold,
            } if args.mode != "baseline" else None),
        },
        "metrics": {
            "n_scored": scored,
            "n_failed": failures,
            "accuracy": round(acc, 4),
            "mae": round(mean_mae, 3) if mean_mae == mean_mae else None,
            "total_time_s": round(total_time, 1),
        },
        "predictions": results,
    }

    if args.out is None:
        args.out = (args.questions.parent.parent / "data" / "results"
                    / f"{args.mode}_{args.nutrient}_n{len(results)}.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"  NutriBench Jetson Test — Summary")
    print("=" * 60)
    print(f"  Mode:        {args.mode}")
    print(f"  Nutrient:    {args.nutrient}  (acc threshold = ±{threshold_score} {cfg['unit']})")
    print(f"  Scored:      {scored} / {len(results)}  ({failures} failed parse / errors)")
    print(f"  Accuracy:    {acc * 100:.1f}%")
    if mean_mae == mean_mae:
        print(f"  Mean MAE:    {mean_mae:.2f} {cfg['unit']}")
    print(f"  Total time:  {total_time:.1f}s   ({total_time / max(len(results), 1):.1f}s / question)")
    print(f"  Saved to:    {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
