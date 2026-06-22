"""LLM-driven nutritional gap analysis (LLM Call 1).

Takes what the user has eaten so far (with nutrient profiles) and asks
the LLM to identify what's missing and produce structured JSON targets
for the next meal.
"""

from __future__ import annotations

from nutri_rag.llm import chat_completion_json

# Default balanced targets when JSON parsing fails
DEFAULT_TARGETS = {
    "protein_g": 25,
    "fat_g": 20,
    "carb_g": 50,
    "energy_kcal": 500,
}

_SYSTEM_PROMPT = (
    "You are a nutrition analyst. Analyze the user's meal history and "
    "determine what macronutrients are missing for their next meal. "
    "If a chronic health condition is provided, make the macro target "
    "clinically appropriate for it (e.g. lower carbohydrate for diabetes, "
    "lower sodium-heavy foods for hypertension). "
    "Return your analysis as JSON in this exact format:\n"
    '{"reasoning": "<your analysis>", '
    '"targets": {"protein_g": <number>, "fat_g": <number>, '
    '"carb_g": <number>, "energy_kcal": <number>}}'
)


def _format_meal_history(
    meal_items: list[dict],
) -> str:
    """Format eaten foods into a readable summary for the LLM.

    Each item in meal_items should have:
        - description: str (USDA food name)
        - nutrients: dict[str, float] (per-100g values)
        - quantity: float | None (grams if known)
        - meal_type: str (breakfast/lunch/dinner/snack)
    """
    lines = []
    current_meal = None

    for item in meal_items:
        meal_type = item.get("meal_type", "Meal")
        if meal_type != current_meal:
            current_meal = meal_type
            lines.append(f"\n{meal_type.capitalize()}:")

        desc = item.get("description", "Unknown food")
        nutrients = item.get("nutrients", {})

        carb = nutrients.get("Carbohydrate, by difference", 0)
        protein = nutrients.get("Protein", 0)
        fat = nutrients.get("Total lipid (fat)", 0)
        energy = nutrients.get("Energy", 0)

        line = f"- {desc}: {carb:.1f}g carb | {protein:.1f}g protein | {fat:.1f}g fat | {energy:.0f} kcal per 100g"

        qty = item.get("quantity")
        if qty:
            line += f" (serving: {qty:.0f}g)"

        lines.append(line)

    return "\n".join(lines)


def analyze_gap(
    meal_items: list[dict],
    next_meal: str = "lunch",
    health_condition: str | None = None,
) -> dict:
    """Analyze nutritional gaps and return target macros for the next meal.

    Args:
        meal_items: List of dicts with description, nutrients, quantity, meal_type.
        next_meal: What meal to recommend for (breakfast/lunch/dinner).
        health_condition: Optional chronic condition (e.g. "Diabetes",
            "Hypertension") that should shape the macro target direction
            (diabetes -> lower carbohydrate, etc.). None = condition-agnostic.

    Returns:
        dict with "reasoning" (str) and "targets" (dict with protein_g, fat_g,
        carb_g, energy_kcal).
    """
    history_text = _format_meal_history(meal_items)

    condition_text = ""
    if health_condition and health_condition.strip().lower() not in ("none", ""):
        condition_text = (
            f"\nThe user has the following chronic condition: {health_condition}. "
            f"Tailor the macro target to be clinically appropriate for it "
            f"(e.g. reduce carbohydrate for diabetes, reduce sodium-driving "
            f"foods for hypertension, moderate calories for obesity).\n"
        )

    user_msg = (
        f"The user has eaten the following today:\n"
        f"{history_text}\n"
        f"{condition_text}\n"
        f"Analyze the nutritional balance and recommend target macros "
        f"for their next meal ({next_meal}).\n\n"
        f"Return your analysis as JSON:\n"
        f'{{"reasoning": "...", "targets": {{"protein_g": N, "fat_g": N, '
        f'"carb_g": N, "energy_kcal": N}}}}'
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        result = chat_completion_json(messages, max_tokens=1024)
    except (ValueError, Exception) as e:
        print(f"Warning: Gap analysis LLM call failed ({e}), using defaults")
        return {"reasoning": "Fallback: using default balanced targets", "targets": DEFAULT_TARGETS}

    # Validate the response structure
    if "targets" not in result:
        return {"reasoning": result.get("reasoning", ""), "targets": DEFAULT_TARGETS}

    targets = result["targets"]
    for key in DEFAULT_TARGETS:
        if key not in targets or not isinstance(targets[key], (int, float)):
            targets[key] = DEFAULT_TARGETS[key]

    return result
