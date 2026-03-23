"""Format retrieved USDA nutrient context for NutriBench prompts.

The reference block is injected before the Query line in the user message,
keeping the original CoT few-shot examples and system prompt intact.

Two formats:
- Legacy (V0/V1/V2): global USDA reference block, all matches listed together
- Per-item (V3): each food item explicitly paired with USDA reference or
  "no reliable match — use your own knowledge"
"""

from __future__ import annotations

import os

from nutri_rag.bench.retriever import FoodContext
from nutri_rag.config import SIMILARITY_THRESHOLD


# Friendly display names for nutrient keys
_DISPLAY = {
    "Carbohydrate, by difference": "Carbohydrate",
    "Protein": "Protein",
    "Total lipid (fat)": "Fat",
}

# Energy keys in preference order (DB stores under different names)
_ENERGY_KEYS = ["Energy", "Energy (Atwater General Factors)", "Energy (Atwater Specific Factors)"]

# Which DB nutrient key is required for each target nutrient
_REQUIRED_NUTRIENT = {
    "carb": "Carbohydrate, by difference",
    "protein": "Protein",
    "fat": "Total lipid (fat)",
    "energy": None,  # energy uses a separate lookup
}

# Formula line per nutrient
_FORMULA = {
    "carb": "Formula: carbs = (weight_g / 100) * carbs_per_100g",
    "protein": "Formula: protein = (weight_g / 100) * protein_per_100g",
    "fat": "Formula: fat = (weight_g / 100) * fat_per_100g",
    "energy": "Formula: energy = (weight_g / 100) * energy_per_100g",
}


def _has_target_nutrient(ctx: FoodContext, nutrient: str) -> bool:
    """Check if a context has the required nutrient data."""
    required_key = _REQUIRED_NUTRIENT.get(nutrient)
    if required_key:
        return ctx.nutrients.get(required_key) is not None
    # For energy, check any energy key
    return any(ctx.nutrients.get(ek) is not None for ek in _ENERGY_KEYS)


def _format_nutrient_values(ctx: FoodContext) -> str:
    """Format nutrient values for a single food context."""
    parts = []
    for nutrient_key, display_name in _DISPLAY.items():
        val = ctx.nutrients.get(nutrient_key)
        if val is not None:
            parts.append(f"{display_name}: {val:.1f}g")

    for ekey in _ENERGY_KEYS:
        energy_val = ctx.nutrients.get(ekey)
        if energy_val is not None:
            parts.append(f"Energy: {energy_val:.1f}kcal")
            break

    return " | ".join(parts)


# ── Legacy format (V0/V1/V2) ─────────────────────────────────────────

def _format_legacy(contexts: list[FoodContext], nutrient: str) -> str:
    """Build the legacy global USDA reference block."""
    matched = [c for c in contexts if c.matched and _has_target_nutrient(c, nutrient)]
    if not matched:
        return ""

    lines = [
        "=== USDA Nutritional Reference Data (per 100g) ===",
        "These are approximate reference values. Use them if they match your food items.",
        "For foods NOT listed here, use your own nutritional knowledge.",
        _FORMULA.get(nutrient, _FORMULA["carb"]),
        "",
    ]

    for i, ctx in enumerate(matched, 1):
        lines.append(f'[{i}] "{ctx.description}" (USDA #{ctx.fdc_id})')
        lines.append("    " + _format_nutrient_values(ctx))
        lines.append("")

    lines.append("===")
    return "\n".join(lines)


# ── Per-item format (V3) ─────────────────────────────────────────────

def _format_per_item(
    contexts: list[FoodContext],
    nutrient: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> str:
    """Build per-item USDA reference block with threshold gating.

    Each food item is explicitly labeled as either:
    - A reliable USDA match (similarity >= threshold) with nutrient data
    - No reliable match — model should use its own knowledge
    """
    if not contexts:
        return ""

    lines = [
        "=== Per-item USDA Reference (per 100g) ===",
        _FORMULA.get(nutrient, _FORMULA["carb"]),
        "Use USDA values where provided. For items marked 'no reliable match', use your own knowledge.",
        "",
    ]

    has_any_reference = False
    for ctx in contexts:
        is_reliable = (
            ctx.matched
            and ctx.similarity_score >= threshold
            and _has_target_nutrient(ctx, nutrient)
        )

        if is_reliable:
            has_any_reference = True
            nutrient_str = _format_nutrient_values(ctx)
            lines.append(
                f'- {ctx.food_term}: USDA match \u2192 "{ctx.description}" \u2014 {nutrient_str}'
            )
        else:
            lines.append(
                f"- {ctx.food_term}: no reliable USDA match \u2014 use your own knowledge"
            )

    lines.append("===")

    if not has_any_reference:
        # All items are unmatched — don't inject the block at all
        return ""

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────

def format_nutrient_block(
    contexts: list[FoodContext],
    nutrient: str | None = None,
    per_item: bool = False,
    threshold: float = SIMILARITY_THRESHOLD,
) -> str:
    """Build the USDA reference block from retrieved food contexts.

    Parameters
    ----------
    contexts : list of FoodContext
    nutrient : target nutrient (default: NUTRI_TARGET env var or "carb")
    per_item : if True, use V3 per-item format with threshold gating
    threshold : cosine similarity threshold (only used when per_item=True)
    """
    nutrient = nutrient or os.environ.get("NUTRI_TARGET", "carb")

    if per_item:
        return _format_per_item(contexts, nutrient, threshold)
    else:
        return _format_legacy(contexts, nutrient)


def build_rag_doc_to_text(
    meal_description: str,
    contexts: list[FoodContext],
    per_item: bool = False,
) -> str:
    """Build the full RAG-augmented user message for NutriBench.

    Combines the USDA reference block with the original CoT query format.
    """
    block = format_nutrient_block(contexts, per_item=per_item)

    if block:
        return (
            f"{block}\n\n"
            f'Query: "{meal_description}"\n'
            f"Answer: Let's think step by step."
        )
    else:
        # No matches — fall back to plain CoT format
        return (
            f'Query: "{meal_description}"\n'
            f"Answer: Let's think step by step."
        )
