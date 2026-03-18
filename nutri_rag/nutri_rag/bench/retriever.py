"""NutriBench retriever: parse meal → search DB → get nutrient profiles.

No GAT embeddings, no gap analysis — just exact nutrient lookup for
augmenting the LLM prompt with per-100g USDA values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from nutri_rag.config import DB_PATH, TOP_K_FOODS
from nutri_rag.parse import ParsedItem, parse_meal
from nutri_rag.search import search_food, get_nutrients


@dataclass
class FoodContext:
    """Retrieved context for a single parsed food item."""
    parsed: ParsedItem
    fdc_id: int | None = None
    description: str | None = None
    nutrients: dict[str, float] = field(default_factory=dict)
    matched: bool = False


class BenchRetriever:
    """Singleton-friendly retriever for NutriBench mode.

    Usage::

        retriever = BenchRetriever()
        contexts = retriever.retrieve("126g of maize flour and 27g of raw sugar")
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._con: duckdb.DuckDBPyConnection | None = None

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(self._db_path, read_only=True)
        return self._con

    def retrieve(self, meal_description: str) -> list[FoodContext]:
        """Parse a meal description and retrieve nutrient context."""
        items = parse_meal(meal_description)
        contexts: list[FoodContext] = []

        for item in items:
            ctx = FoodContext(parsed=item)

            # Search for the food in DuckDB
            df = search_food(self.con, item.food_term, k=1)

            if len(df) > 0:
                ctx.fdc_id = int(df.iloc[0]["fdc_id"])
                ctx.description = df.iloc[0]["description"]
                ctx.nutrients = get_nutrients(self.con, ctx.fdc_id, key_only=True)
                ctx.matched = True

            contexts.append(ctx)

        return contexts
