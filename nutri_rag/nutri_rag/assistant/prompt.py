"""Prompt formatting for the general nutrition assistant (LLM Call 2).

Formats gap analysis results, ranked food options, and user preference
history into a prompt for generating meal recommendations.
"""

from __future__ import annotations

from nutri_rag.assistant.food_recommender import FoodOption


def format_recommendation_prompt(
    gap_reasoning: str,
    targets: dict[str, float],
    options: list[FoodOption],
    next_meal: str = "lunch",
    preference_summary: dict | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for LLM Call 2 (recommendation generation).

    Returns a list of message dicts ready for the chat completion API.
    """
    # Group options: seeds vs GAT neighbors
    seeds = [o for o in options if o.is_seed]
    neighbors = [o for o in options if not o.is_seed]

    # Build the context sections
    sections = []

    # Gap analysis section
    sections.append("=== Nutritional Gap Analysis ===")
    sections.append(gap_reasoning)
    sections.append(
        f"Target for {next_meal}: "
        f"~{targets.get('protein_g', 0):.0f}g protein, "
        f"~{targets.get('fat_g', 0):.0f}g fat, "
        f"~{targets.get('carb_g', 0):.0f}g carb, "
        f"~{targets.get('energy_kcal', 0):.0f} kcal"
    )
    sections.append("")

    # Recommended foods section
    sections.append("=== Recommended Foods (from USDA database) ===")

    for i, opt in enumerate(seeds[:5], 1):
        carb = opt.nutrients.get("Carbohydrate, by difference", 0)
        protein = opt.nutrients.get("Protein", 0)
        fat = opt.nutrients.get("Total lipid (fat)", 0)
        energy = opt.nutrients.get("Energy", 0)

        label = "(user favorite)" if opt.preference_score >= 0.7 else ""
        sections.append(
            f'  [{i}] "{opt.description}" — '
            f"{protein:.1f}g protein | {fat:.1f}g fat | "
            f"{carb:.1f}g carb | {energy:.0f} kcal per 100g "
            f"{label}".strip()
        )

    if neighbors:
        sections.append("")
        sections.append("Similar alternatives (via GAT embedding neighbors):")
        for opt in neighbors[:8]:
            carb = opt.nutrients.get("Carbohydrate, by difference", 0)
            protein = opt.nutrients.get("Protein", 0)
            fat = opt.nutrients.get("Total lipid (fat)", 0)

            sections.append(
                f'  - "{opt.description}" — '
                f"{protein:.1f}g protein | {fat:.1f}g fat | "
                f"{carb:.1f}g carb (similarity: {opt.gat_similarity:.2f})"
            )

    sections.append("")

    # User preference section
    if preference_summary:
        sections.append("=== User Preference History ===")
        favs = preference_summary.get("favorites", [])
        disliked = preference_summary.get("disliked", [])
        if favs:
            sections.append(f"User tends to choose: {', '.join(favs[:5])}")
        if disliked:
            sections.append(f"User tends to skip: {', '.join(disliked[:5])}")
        sections.append("")

    context = "\n".join(sections)

    system_msg = (
        "You are a nutrition assistant. Based on the analysis below, "
        f"recommend a balanced meal for the user's next meal ({next_meal}). "
        "Provide a specific, practical suggestion with approximate portions. "
        "Consider the user's preferences when available."
    )

    user_msg = (
        f"{context}\n"
        f'Provide a specific, practical {next_meal} suggestion with approximate portions.'
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
