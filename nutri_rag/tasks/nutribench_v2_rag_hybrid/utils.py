"""NutriBench v2 RAG hybrid (score-fusion) task utilities (V5 — new in
Phase A of the unify-RAG plan).

Score-fusion hybrid retrieval via text-bootstrapped pseudo-anchor:

    q_text  = embed(food_term)
    seed    = argmax_x cos(q_text, x_text)                 # text top-1
    q_gat*  = GAT[seed]                                    # pseudo-anchor
    score(x) = alpha · cos(q_gat*, x_gat)
             + (1-alpha) · cos(q_text, x_text)

This matches the Hybrid column in HealthyFoodSubs' eval_food_subs.evaluate_hybrid
(line 216 of nutri_graph/scripts/eval_food_subs.py). For NutriBench the query
isn't a graph node, so q_gat is bootstrapped from the top text candidate.

Uses multi-candidate output (top-K per food item) since hybrid scoring
benefits from showing the LLM the top alternatives to disambiguate.

Nutrient target is controlled by NUTRI_TARGET env var (default: carb).
Hybrid weight is controlled by NUTRI_HYBRID_ALPHA env var (default: 0.5).
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
        alpha = float(os.environ.get("NUTRI_HYBRID_ALPHA", "0.5"))
        _retriever = BenchRetriever(mode="hybrid", alpha=alpha)
    return _retriever


def doc_to_text_rag(doc):
    """Build hybrid RAG multi-candidate prompt for a NutriBench sample."""
    retriever = _get_retriever()
    meal = doc["meal_description"]
    contexts = retriever.retrieve(meal, top_k=TOP_K_CANDIDATES)
    return build_rag_doc_to_text(meal, contexts, multi_candidate=True)
