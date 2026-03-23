"""BM25-based retriever for NutriBench benchmark (V0 baseline).

Same food term extraction as the embedding retriever, but uses
BM25 keyword search instead of vector similarity.
"""

from __future__ import annotations

from nutri_rag.config import DB_PATH
from nutri_rag.bench.retriever import FoodContext, _extract_food_terms
from nutri_rag.search_bm25 import search_food, get_nutrients


class BenchRetrieverBM25:
    """V0 retriever using BM25 full-text search."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path

    def retrieve(self, meal_description: str) -> list[FoodContext]:
        food_terms = _extract_food_terms(meal_description)
        contexts: list[FoodContext] = []
        seen_fdc_ids: set[int] = set()

        for term in food_terms:
            ctx = FoodContext(food_term=term)

            df = search_food(None, term, k=1, db_path=self._db_path)

            if len(df) > 0:
                fdc_id = int(df.iloc[0]["fdc_id"])
                if fdc_id in seen_fdc_ids:
                    continue
                seen_fdc_ids.add(fdc_id)

                ctx.fdc_id = fdc_id
                ctx.description = df.iloc[0]["description"]
                ctx.nutrients = get_nutrients(
                    None, ctx.fdc_id, key_only=True, db_path=self._db_path
                )
                ctx.matched = True

            contexts.append(ctx)

        return contexts
