"""NutriBench v2 RAG task utilities — V0 BM25 baseline.

Uses the original BM25 keyword-matching retrieval (search_bm25.py)
instead of embedding-based search. Kept as a baseline for comparison.
"""

import re
import sys
import os

_NUTRI_RAG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _NUTRI_RAG_ROOT not in sys.path:
    sys.path.insert(0, _NUTRI_RAG_ROOT)

from nutri_rag.bench.retriever_bm25 import BenchRetrieverBM25
from nutri_rag.bench.prompt import build_rag_doc_to_text

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = BenchRetrieverBM25()
    return _retriever


def doc_to_text_rag(doc):
    """Build RAG-augmented prompt using BM25 retrieval."""
    retriever = _get_retriever()
    meal = doc["meal_description"]
    contexts = retriever.retrieve(meal)
    return build_rag_doc_to_text(meal, contexts)


# ── Below: copied verbatim from CoT utils.py ──────────────────────────

def process_results(doc, results):
    candidates = results[0]
    pred = clean_output(candidates, doc["meal_description"], "cot", "carb")
    gt = doc["carb"]
    mae = abs(pred - gt)

    results = {
        "acc": mae < 7.5,
        "mae": mae,
    }
    return results


def agg_mae(items):
    mae = sum(items) / len(items)
    return mae


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def clean_output(raw_output, query, method_name, nutrition_name):
    if isinstance(raw_output, list):
        raw_output = raw_output[0] if raw_output else ""
    if not isinstance(raw_output, str):
        raw_output = str(raw_output)

    if "cot" in method_name:
        splits = raw_output.split("Output:")
        if len(splits) > 1:
            raw_output = splits[1]

    raw_output = raw_output.strip()

    if nutrition_name == 'fat':
        pattern = r'["\']\s*total_fat["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'protein':
        pattern = r'["\']\s*total_protein["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'energy':
        pattern = r'["\']\s*total_energy["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'carb':
        pattern = r'["\']\s*total_carbohydrates["\']:\s*(?:["\']?(-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)["\']?|\[(-?[0-9]+(?:\.[0-9]*)?(?:,\s*-?[0-9]+(?:\.[0-9]*)?)*)\])'
    else:
        raise NotImplementedError

    match = re.search(pattern, raw_output)
    if match:
        if match.group(1):
            pred_carbs = match.group(1)
            if is_number(pred_carbs):
                return float(pred_carbs)
            else:
                pred_carbs_list = pred_carbs.split('-')
                if len(pred_carbs_list) == 2 and is_number(pred_carbs_list[0]) and is_number(pred_carbs_list[1]):
                    p0 = float(pred_carbs_list[0])
                    p1 = float(pred_carbs_list[1])
                    return (p0 + p1) / 2.0
                else:
                    print(f"EXCEPTION AFTER MATCHING")
                    print(f"Matched output: {raw_output}")
                    print(f"Query: {query}")
                    return -1
        elif match.group(2):
            try:
                pred_carbs_list = match.group(2).split(',')
                p0 = float(pred_carbs_list[0])
                p1 = float(pred_carbs_list[1])
                return (p0 + p1) / 2.0
            except:
                print(f"EXCEPTION AFTER MATCHING")
                print(f"Matched output: {raw_output}")
                print(f"Query: {query}")
                return -1
    else:
        if is_number(raw_output):
            return float(raw_output)
        else:
            print(f"EXCEPTION")
            print(f"Matched output: {raw_output}")
            print(f"Query: {query}")
            return -1
