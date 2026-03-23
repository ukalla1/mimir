"""Baseline (no RAG) NutriBench task utilities — carb."""

import sys
import os

os.environ["NUTRI_TARGET"] = "carb"

_NUTRI_RAG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _NUTRI_RAG_ROOT not in sys.path:
    sys.path.insert(0, _NUTRI_RAG_ROOT)

from nutri_rag.bench.task_utils import process_results, clean_output, agg_mae, is_number  # noqa: F401


def doc_to_text_cot(doc):
    meal = doc["meal_description"]
    return f'Query: "{meal}"\nAnswer: Let\'s think step by step.'
