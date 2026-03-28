"""Build RAG prompts for PFoodReq recipe recommendation."""

from __future__ import annotations


def _format_nutrients(recipe: dict) -> str:
    """Format recipe nutrients as a compact string."""
    parts = []
    for key, label in [
        ("calories", "cal"),
        ("protein", "protein"),
        ("carbohydrates", "carbs"),
        ("saturated_fat", "sat_fat"),
    ]:
        val = recipe.get(key)
        if val is not None:
            parts.append(f"{label}: {val:.1f}")
    return ", ".join(parts) if parts else "nutrients: N/A"


def _format_ingredients(recipe: dict, max_items: int = 10) -> str:
    """Format recipe ingredients as a compact string."""
    ingredients = recipe.get("ingredients", [])
    if not ingredients:
        return "ingredients: N/A"
    if len(ingredients) > max_items:
        shown = ingredients[:max_items]
        return ", ".join(shown) + f" (+{len(ingredients) - max_items} more)"
    return ", ".join(ingredients)


def build_prompt(
    recipes: list[dict],
    parsed_query: dict,
) -> list[dict]:
    """Build chat messages for the PFoodReq RAG prompt.

    Args:
        recipes: list of recipe dicts from retriever (with context).
        parsed_query: output of query_parser.parse_example().

    Returns:
        list of message dicts for chat_completion().
    """
    # Format retrieved recipes block
    recipe_lines = []
    for i, r in enumerate(recipes, 1):
        name = r.get("recipe_name", "Unknown")
        nutrients = _format_nutrients(r)
        ingredients = _format_ingredients(r)
        recipe_lines.append(
            f"{i}. \"{name}\"\n"
            f"   Ingredients: {ingredients}\n"
            f"   Nutrients (per serving): {nutrients}"
        )

    recipes_block = "\n".join(recipe_lines)

    # Format user constraints
    constraint_parts = []

    query_text = parsed_query.get("original_query", parsed_query.get("query_text", ""))
    constraint_parts.append(f"Query: {query_text}")

    pos = parsed_query.get("positive_ingredients", [])
    if pos:
        constraint_parts.append(f"Must include: {', '.join(pos)}")

    neg = parsed_query.get("negative_ingredients", [])
    if neg:
        constraint_parts.append(f"Must NOT include: {', '.join(neg)}")

    for nc in parsed_query.get("nutrient_constraints", []):
        nutrient = nc.get("nutrient", "")
        level = nc.get("level", "")
        rng = nc.get("range", [])
        if rng and len(rng) == 2:
            constraint_parts.append(f"Nutrient: {level} {nutrient} ({rng[0]}-{rng[1]})")
        elif level:
            constraint_parts.append(f"Nutrient: {level} {nutrient}")

    for nutrient_name, guideline in parsed_query.get("guidelines", {}).items():
        meal = guideline.get("meal", {})
        unit = guideline.get("unit", "")
        if isinstance(meal, dict) and "lower" in meal and "upper" in meal:
            constraint_parts.append(
                f"Guideline: {nutrient_name} {meal['lower']}-{meal['upper']} {unit} per meal"
            )

    constraints_block = "\n".join(constraint_parts)

    system_msg = (
        "You are a food recommendation assistant. Given a list of candidate recipes "
        "with their ingredients and nutritional information, select the recipes that "
        "satisfy ALL of the user's requirements.\n\n"
        "Rules:\n"
        "- A recipe MUST contain all 'must include' ingredients\n"
        "- A recipe must NOT contain any 'must NOT include' ingredients\n"
        "- A recipe must satisfy all nutrient constraints\n"
        "- Return ONLY the matching recipe names as a JSON list\n"
        "- If no recipes match, return an empty list: []\n"
        "- Do NOT add recipes that are not in the candidate list"
    )

    user_msg = (
        f"=== Candidate Recipes ===\n"
        f"{recipes_block}\n"
        f"===\n\n"
        f"{constraints_block}\n\n"
        f"Which recipes from the list above satisfy ALL requirements? "
        f"Return a JSON list of recipe names."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
