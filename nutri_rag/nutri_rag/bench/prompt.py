"""Format retrieved USDA nutrient context for NutriBench prompts.

The reference block is injected before the Query line in the user message,
keeping the original CoT few-shot examples and system prompt intact.
"""

from __future__ import annotations

import os

from nutri_rag.bench.retriever import FoodContext


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


def format_nutrient_block(contexts: list[FoodContext], nutrient: str | None = None) -> str:
    """Build the USDA reference block from retrieved food contexts.

    Parameters
    ----------
    contexts : list of FoodContext
    nutrient : target nutrient (default: NUTRI_TARGET env var or "carb")
    """
    nutrient = nutrient or os.environ.get("NUTRI_TARGET", "carb")

    # Filter: only include references that have the target nutrient data
    required_key = _REQUIRED_NUTRIENT.get(nutrient)

    def _has_target(ctx):
        if not ctx.matched:
            return False
        if required_key:
            return ctx.nutrients.get(required_key) is not None
        # For energy, check any energy key
        return any(ctx.nutrients.get(ek) is not None for ek in _ENERGY_KEYS)

    matched = [c for c in contexts if _has_target(c)]
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

        parts = []
        for nutrient_key, display_name in _DISPLAY.items():
            val = ctx.nutrients.get(nutrient_key)
            if val is not None:
                parts.append(f"{display_name}: {val:.1f}g")

        # Find energy value (stored under different names in DB)
        for ekey in _ENERGY_KEYS:
            energy_val = ctx.nutrients.get(ekey)
            if energy_val is not None:
                parts.append(f"Energy: {energy_val:.1f}kcal")
                break

        lines.append("    " + " | ".join(parts))
        lines.append("")

    lines.append("===")
    return "\n".join(lines)


def build_rag_doc_to_text(meal_description: str, contexts: list[FoodContext]) -> str:
    """Build the full RAG-augmented user message for NutriBench.

    Combines the USDA reference block with the original CoT query format.
    """
    block = format_nutrient_block(contexts)

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
