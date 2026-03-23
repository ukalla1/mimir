"""Heuristic meal-description parser.

Splits free-text meal descriptions into individual food items with
optional quantity and unit.  Handles both metric NutriBench-style inputs
("126 grams of maize flour") and natural language ("a cup of oatmeal").
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedItem:
    raw_text: str
    food_term: str
    quantity: float | None
    unit: str | None


# ── Quantity + unit regex ──────────────────────────────────────────────
_UNIT_PATTERN = (
    r"(g|grams?|gram|oz|ounces?|cup|cups|tbsp|tablespoons?|tsp|teaspoons?"
    r"|slice|slices|ml|milliliters?|l|liters?|litres?|lb|pounds?|kg|kilograms?"
    r"|piece|pieces|serving|servings|bowl|bowls|glass|glasses|can|cans)"
)

_QTY_UNIT_RE = re.compile(
    rf"(\d+\.?\d*)\s*{_UNIT_PATTERN}\b",
    re.IGNORECASE,
)

# Fraction patterns like "1/2", "3/4"
_FRACTION_RE = re.compile(
    rf"(\d+)\s*/\s*(\d+)\s*(?:of\s+a\s+)?{_UNIT_PATTERN}?\b",
    re.IGNORECASE,
)

# Words like "a", "an", "half a", "one", etc. preceding a food term
_ARTICLE_RE = re.compile(
    r"\b(a|an|one|two|three|four|five|half\s+a|half)\b",
    re.IGNORECASE,
)

# Context words to strip (meal timing, filler)
_CONTEXT_RE = re.compile(
    r"\b(for\s+)?(breakfast|lunch|dinner|snack|brunch|dessert|supper|"
    r"this\s+morning|today|yesterday|"
    r"i\s+ate|i\s+had|i\s+drank|i\s+eat|i\s+have|"
    r"we\s+had|we\s+ate|"
    r"ate|eat|eaten|had|have|drank|drinking|"
    r"then|also|some|with)\b",
    re.IGNORECASE,
)

# Word-to-number map for written quantities
_WORD_NUMS = {
    "half": 0.5, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _split_items(text: str) -> list[str]:
    """Split on commas, semicolons, and words like 'and', 'with'."""
    parts = re.split(r",|;|\band\b|\bwith\b", text)
    return [p.strip() for p in parts if p.strip()]


def _extract_quantity(text: str) -> tuple[float | None, str | None, str]:
    """Extract (quantity, unit, remaining_text) from a chunk."""
    # Try explicit quantity + unit  ("126 grams of maize flour")
    m = _QTY_UNIT_RE.search(text)
    if m:
        qty = float(m.group(1))
        unit = m.group(2).lower().rstrip("s")  # normalize plural
        # Map common unit aliases
        unit = {"gram": "g", "ounce": "oz", "milliliter": "ml",
                "liter": "l", "litre": "l", "kilogram": "kg",
                "pound": "lb", "tablespoon": "tbsp", "teaspoon": "tsp",
                "glass": "glass", "bowl": "bowl", "can": "can",
                "piece": "piece", "serving": "serving", "slice": "slice",
                "cup": "cup"}.get(unit, unit)
        remaining = text[:m.start()] + text[m.end():]
        return qty, unit, remaining

    # Try fraction  ("1/2 a banana")
    fm = _FRACTION_RE.search(text)
    if fm:
        qty = float(fm.group(1)) / float(fm.group(2))
        unit = fm.group(3).lower().rstrip("s") if fm.group(3) else None
        remaining = text[:fm.start()] + text[fm.end():]
        return qty, unit, remaining

    # Try "a/an <unit>" pattern  ("a cup of oatmeal", "a glass of milk")
    a_unit_re = re.compile(
        rf"\b(?:a|an|one)\s+{_UNIT_PATTERN}\b",
        re.IGNORECASE,
    )
    am = a_unit_re.search(text)
    if am:
        unit = am.group(1).lower().rstrip("s")
        unit = {"gram": "g", "ounce": "oz", "milliliter": "ml",
                "liter": "l", "litre": "l", "kilogram": "kg",
                "pound": "lb", "tablespoon": "tbsp", "teaspoon": "tsp",
                "glass": "glass", "bowl": "bowl", "can": "can",
                "piece": "piece", "serving": "serving", "slice": "slice",
                "cup": "cup"}.get(unit, unit)
        remaining = text[:am.start()] + text[am.end():]
        return 1.0, unit, remaining

    # Try word-number  ("half a banana")
    for word, num in _WORD_NUMS.items():
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        if pattern.search(text):
            remaining = pattern.sub("", text, count=1)
            return num, None, remaining

    return None, None, text


def _clean_food_term(text: str) -> str:
    """Strip articles, context words, and extra whitespace."""
    text = _CONTEXT_RE.sub(" ", text)
    text = _ARTICLE_RE.sub(" ", text)
    # Remove "of" at the start (left over from "126g of ...")
    text = re.sub(r"^\s*of\s+", "", text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_meal(description: str) -> list[ParsedItem]:
    """Parse a meal description into a list of ParsedItems."""
    items: list[ParsedItem] = []
    for chunk in _split_items(description):
        qty, unit, remaining = _extract_quantity(chunk)
        food_term = _clean_food_term(remaining)
        if not food_term:
            continue
        items.append(ParsedItem(
            raw_text=chunk.strip(),
            food_term=food_term,
            quantity=qty,
            unit=unit,
        ))
    return items
