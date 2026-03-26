"""NutriBench v2 RAG + GAT multi-candidate task utilities (V3 pipeline).

Retrieves top-5 USDA candidates per food item and lets the LLM pick the best.
Items with no matches are omitted to save tokens.

Nutrient target is controlled by NUTRI_TARGET env var (default: carb).
"""

import sys
import os

_NUTRI_RAG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _NUTRI_RAG_ROOT not in sys.path:
    sys.path.insert(0, _NUTRI_RAG_ROOT)

from nutri_rag.bench.retriever import BenchRetriever
from nutri_rag.bench.prompt import build_rag_doc_to_text
from nutri_rag.bench.nutrient_prompts import build_system_prompt
from nutri_rag.bench.task_utils import process_results, clean_output, agg_mae, is_number  # noqa: F401

_retriever = None

TOP_K_CANDIDATES = 5


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = BenchRetriever(use_gat=True)
    return _retriever


def doc_to_text_rag(doc):
    """Build RAG+GAT multi-candidate prompt for a NutriBench sample."""
    retriever = _get_retriever()
    meal = doc["meal_description"]
    contexts = retriever.retrieve(meal, top_k=TOP_K_CANDIDATES)
    return build_rag_doc_to_text(meal, contexts, multi_candidate=True)
