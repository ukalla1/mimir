"""NutriBench retriever: extract food terms → search DB → get nutrient profiles.

Uses regex to extract food terms directly from NutriBench meal descriptions,
which follow predictable patterns like "126g of maize flour" or
"171 grams of boiled fresh groundnuts".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import duckdb

from nutri_rag.config import DB_PATH, TOP_K_FOODS
from nutri_rag.search import search_food, get_nutrients


@dataclass
class FoodContext:
    """Retrieved context for a single parsed food item."""
    food_term: str
    fdc_id: int | None = None
    description: str | None = None
    nutrients: dict[str, float] = field(default_factory=dict)
    matched: bool = False


# Pattern A: "<qty> g/grams of <food>"  — most common NutriBench format
_PAT_QTY_OF = re.compile(
    r'\d+(?:\.\d+)?\s*(?:g(?:rams?)?)\s+(?:of\s+)'
    r'([\w\s,\'-]+?)(?=\s*(?:,|\band\b|\balong\b|\.|$))',
    re.IGNORECASE,
)

# Pattern B: "<food> weighing <qty> grams"
# Only capture 1-4 words before "weighing" to avoid grabbing the whole sentence
_PAT_WEIGHING = re.compile(
    r'((?:\w+\s+){0,3}\w+)\s+weighing\s+\d+',
    re.IGNORECASE,
)

# Pattern C: "<food> (<qty>g)" — inline weight notation
# Only capture 1-4 words before the parenthetical
_PAT_INLINE = re.compile(
    r'((?:\w+\s+){0,3}\w+)\s*\(\d+(?:\.\d+)?\s*g\)',
    re.IGNORECASE,
)

# Words to strip from extracted food terms — cooking methods, adjectives, etc.
_STRIP_WORDS = re.compile(
    r'\b(?:raw|boiled|fried|baked|cooked|roasted|steamed|grilled|dried|fresh|'
    r'ripe|unripe|large|small|medium|plain|whole|chopped|sliced|diced|'
    r'peeled|unpeeled|skinless|boneless|without\s+skin|in\s+their\s+shells?|'
    r'weighing|along|the|a|an|ate|had|consumed|got|sprinkled)\b',
    re.IGNORECASE,
)


def _extract_food_terms(meal_description: str) -> list[str]:
    """Extract food terms from a NutriBench meal description.

    Handles multiple formats:
    - "126 grams of maize flour"
    - "a plain bun weighing 126 grams"
    - "a boiled large onion (1g)"

    Returns deduplicated, cleaned food terms.
    """
    raw_matches: list[str] = []

    # Collect from all patterns
    raw_matches.extend(_PAT_QTY_OF.findall(meal_description))
    raw_matches.extend(_PAT_WEIGHING.findall(meal_description))
    raw_matches.extend(_PAT_INLINE.findall(meal_description))

    terms = []
    seen = set()

    for raw_term in raw_matches:
        # Strip cooking/preparation words
        cleaned = _STRIP_WORDS.sub(' ', raw_term)
        # Collapse whitespace and strip
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().rstrip('.')
        # Skip empty or very short terms
        if len(cleaned) < 2:
            continue
        # Deduplicate
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            terms.append(cleaned)

    return terms


class BenchRetriever:
    """Retriever for NutriBench benchmark mode.

    Extracts food terms from meal descriptions using regex patterns,
    then searches the USDA database for matching nutrient profiles.

    Usage::

        retriever = BenchRetriever()
        contexts = retriever.retrieve("126g of maize flour and 27g of raw sugar")
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path

    def retrieve(self, meal_description: str) -> list[FoodContext]:
        """Extract food terms and retrieve nutrient context."""
        food_terms = _extract_food_terms(meal_description)
        contexts: list[FoodContext] = []
        seen_fdc_ids: set[int] = set()

        for term in food_terms:
            ctx = FoodContext(food_term=term)

            df = search_food(None, term, k=1, db_path=self._db_path)

            if len(df) > 0:
                fdc_id = int(df.iloc[0]["fdc_id"])
                # Skip duplicates
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
