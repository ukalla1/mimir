"""Format retrieved USDA nutrient context for NutriBench prompts.

The reference block is injected before the Query line in the user message,
keeping the original CoT few-shot examples and system prompt intact.
"""

from __future__ import annotations

from nutri_rag.bench.retriever import FoodContext


# Friendly display names for nutrient keys
_DISPLAY = {
    "Carbohydrate, by difference": "Carbohydrate",
    "Protein": "Protein",
    "Total lipid (fat)": "Fat",
}

# Energy keys in preference order (DB stores under different names)
_ENERGY_KEYS = ["Energy", "Energy (Atwater General Factors)", "Energy (Atwater Specific Factors)"]


def format_nutrient_block(contexts: list[FoodContext]) -> str:
    """Build the USDA reference block from retrieved food contexts.

    Example output::

        === USDA Nutritional Reference Data (per 100g) ===
        Use these values for your calculations: carbs = (weight_g / 100) * carbs_per_100g

        [1] "Corn flour, whole-grain, yellow" (USDA #170285)
            Carbohydrate: 76.9g | Protein: 6.9g | Fat: 3.9g | Energy: 361 kcal

        [2] "Sugars, granulated" (USDA #169655)
            Carbohydrate: 99.8g | Protein: 0.0g | Fat: 0.0g | Energy: 387 kcal
        ===
    """
    # Only include references that have carbohydrate data
    matched = [
        c for c in contexts
        if c.matched and c.nutrients.get("Carbohydrate, by difference") is not None
    ]
    if not matched:
        return ""

    lines = [
        "=== USDA Nutritional Reference Data (per 100g) ===",
        "These are approximate reference values. Use them if they match your food items.",
        "For foods NOT listed here, use your own nutritional knowledge.",
        "Formula: carbs = (weight_g / 100) * carbs_per_100g",
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
