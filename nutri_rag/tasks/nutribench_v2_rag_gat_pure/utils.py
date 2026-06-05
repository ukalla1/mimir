"""NutriBench v2 RAG pure-GAT task utilities (V4 — new in Phase A of the
unify-RAG plan).

Pure GAT scoring via text-bootstrapped pseudo-anchor:

    q_text  = embed(food_term)
    seed    = argmax_x cos(q_text, x_text)         # text top-1
    q_gat*  = GAT[seed]                            # pseudo-anchor
    score(x) = cos(q_gat*, x_gat)                  # pure GAT

This is the symmetric "GAT-only" mode of NutriBench, matching the GAT-only
column in HealthyFoodSubs' eval_food_subs.evaluate_single.

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


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = BenchRetriever(mode="gat_pure")
    return _retriever


def doc_to_text_rag(doc):
    """Build pure-GAT RAG prompt for a NutriBench sample."""
    retriever = _get_retriever()
    meal = doc["meal_description"]
    contexts = retriever.retrieve(meal)
    return build_rag_doc_to_text(meal, contexts, per_item=True)
