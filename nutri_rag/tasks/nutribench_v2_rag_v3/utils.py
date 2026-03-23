"""NutriBench v2 RAG task utilities — V3 (per-item threshold-gated prompt).

Uses V1 dense retrieval but with per-item prompt format:
- Items with cosine similarity >= threshold get USDA references
- Items below threshold are marked "no reliable match — use your own knowledge"

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
        _retriever = BenchRetriever()
    return _retriever


def doc_to_text_rag(doc):
    """Build RAG-augmented prompt with per-item threshold gating."""
    retriever = _get_retriever()
    meal = doc["meal_description"]
    contexts = retriever.retrieve(meal)
    return build_rag_doc_to_text(meal, contexts, per_item=True)
