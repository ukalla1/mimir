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
from nutri_rag.search import search_food, search_food_v2, get_nutrients


@dataclass
class FoodCandidate:
    """A single USDA candidate for a food item."""
    fdc_id: int
    description: str
    nutrients: dict[str, float] = field(default_factory=dict)
    similarity_score: float = 0.0


@dataclass
class FoodContext:
    """Retrieved context for a single parsed food item."""
    food_term: str
    fdc_id: int | None = None
    description: str | None = None
    nutrients: dict[str, float] = field(default_factory=dict)
    matched: bool = False
    similarity_score: float = 0.0
    # Top-k candidates for multi-candidate mode
    candidates: list[FoodCandidate] = field(default_factory=list)


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

    Args:
        mode: retrieval mode.
            "text"       — text cosine search only (V1 equivalent).
            "gat_expand" — text top-K + GAT-neighbor expansion, text re-rank (V2/V3 legacy).
            "gat_pure"   — pure GAT via text-bootstrapped pseudo-anchor (V4 new).
            "hybrid"     — score-fusion hybrid via pseudo-anchor (V5 new).
        alpha: GAT weight in [0, 1] for "hybrid" mode (ignored otherwise).
        use_gat: deprecated. True ↔ mode="gat_expand"; False ↔ mode="text". Kept
                 for back-compat with existing task utils. Ignored if `mode`
                 is set explicitly.

    Usage::

        retriever = BenchRetriever()                      # V1: text only
        retriever = BenchRetriever(mode="gat_expand")     # V2/V3 legacy expansion
        retriever = BenchRetriever(mode="gat_pure")       # V4: pure GAT (new)
        retriever = BenchRetriever(mode="hybrid", alpha=0.5)  # V5: hybrid (new)
        contexts = retriever.retrieve("126g of maize flour and 27g of raw sugar")
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        use_gat: bool | None = None,
        mode: str | None = None,
        alpha: float = 0.5,
    ):
        self._db_path = db_path
        self._alpha = float(alpha)
        if mode is not None:
            self._mode = mode
        elif use_gat is True:
            self._mode = "gat_expand"
        else:
            self._mode = "text"
        if self._mode not in {"text", "gat_expand", "gat_pure", "hybrid"}:
            raise ValueError(f"unknown BenchRetriever mode: {self._mode!r}")
        # Back-compat field referenced by some legacy code paths
        self._use_gat = self._mode == "gat_expand"

    def retrieve(
        self, meal_description: str, top_k: int = 1,
    ) -> list[FoodContext]:
        """Extract food terms and retrieve nutrient context.

        Args:
            top_k: Number of USDA candidates to retrieve per food item.
                   When top_k > 1, candidates are stored in ctx.candidates.
        """
        food_terms = _extract_food_terms(meal_description)
        contexts: list[FoodContext] = []
        seen_fdc_ids: set[int] = set()

        for term in food_terms:
            ctx = FoodContext(food_term=term)

            if self._mode in {"gat_pure", "hybrid"}:
                # New score-fusion path (pseudo-anchor inside search_food_v2)
                v2_mode = "gat" if self._mode == "gat_pure" else "hybrid"
                df = search_food_v2(
                    term, mode=v2_mode, k=top_k, alpha=self._alpha,
                    db_path=self._db_path,
                )
            else:
                # Legacy paths: text-only or text+GAT-expansion
                df = search_food(
                    None, term, k=top_k, db_path=self._db_path,
                    use_gat=(self._mode == "gat_expand"),
                )

            if len(df) > 0:
                # Primary match (first row) for backward compatibility
                fdc_id = int(df.iloc[0]["fdc_id"])
                if fdc_id in seen_fdc_ids:
                    continue
                seen_fdc_ids.add(fdc_id)

                ctx.fdc_id = fdc_id
                ctx.description = df.iloc[0]["description"]
                ctx.similarity_score = float(df.iloc[0].get("text_score", 0.0))
                ctx.nutrients = get_nutrients(
                    None, ctx.fdc_id, key_only=True, db_path=self._db_path
                )
                ctx.matched = True

                # Store all candidates when top_k > 1
                if top_k > 1:
                    for _, row in df.iterrows():
                        cand_fdc_id = int(row["fdc_id"])
                        cand = FoodCandidate(
                            fdc_id=cand_fdc_id,
                            description=row["description"],
                            similarity_score=float(row.get("text_score", 0.0)),
                            nutrients=get_nutrients(
                                None, cand_fdc_id, key_only=True, db_path=self._db_path
                            ),
                        )
                        ctx.candidates.append(cand)

            contexts.append(ctx)

        return contexts
