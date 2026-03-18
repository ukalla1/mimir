#!/usr/bin/env python3
"""Interactive CLI for the general nutrition assistant (Mode 2).

Demonstrates the full pipeline: parse eaten foods -> gap analysis ->
GAT expansion -> preference re-ranking -> LLM recommendation.

Requires llama-server running at localhost:8080.
"""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nutri_rag.assistant.pipeline import NutriAssistant


def _parse_input(text: str) -> tuple[str, str, str]:
    """Parse user input into (eaten_description, eaten_meal_type, next_meal).

    Handles inputs like:
      "apple and milk for breakfast"
      "I ate an apple and milk for breakfast, what should I eat for lunch?"
      "apple and milk"  (defaults to breakfast -> lunch)
    """
    lower = text.lower()

    # Split off the question part (e.g., "what should I eat for lunch?")
    # Common patterns: "what should I eat/have for ...", "what to eat for ..."
    question_re = re.compile(
        r"[,;.!?\s]*\s*what\s+(?:should\s+I|to|can\s+I|do\s+you\s+recommend)\s.*$",
        re.IGNORECASE,
    )
    eaten_part = question_re.sub("", text).strip()
    if not eaten_part:
        eaten_part = text  # fallback: use full text

    # Detect what meal was eaten (from the eaten part)
    eaten_lower = eaten_part.lower()
    if "lunch" in eaten_lower:
        eaten_meal = "lunch"
    elif "dinner" in eaten_lower or "supper" in eaten_lower:
        eaten_meal = "dinner"
    elif "snack" in eaten_lower:
        eaten_meal = "snack"
    else:
        eaten_meal = "breakfast"

    # Detect what meal to recommend (from the question part, if any)
    next_meal_map = {"breakfast": "lunch", "lunch": "dinner", "dinner": "snack", "snack": "breakfast"}
    next_meal = next_meal_map[eaten_meal]  # default: next logical meal

    # Check if the user explicitly asked for a specific meal
    for meal_word in ("breakfast", "lunch", "dinner", "supper", "snack"):
        if meal_word in lower and meal_word not in eaten_lower:
            next_meal = "dinner" if meal_word == "supper" else meal_word
            break

    return eaten_part, eaten_meal, next_meal


def main():
    print("Nutrition Assistant Demo")
    print("=" * 60)
    print("This assistant analyzes what you've eaten and recommends")
    print("your next meal using GAT-based food embeddings.")
    print()
    print("Commands:")
    print("  Type what you ate to get a recommendation")
    print("  'quit' or 'exit' to stop")
    print("=" * 60)

    assistant = NutriAssistant()

    while True:
        try:
            print()
            eaten = input("What did you eat? (e.g., 'apple and milk for breakfast'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not eaten or eaten.lower() in ("quit", "exit", "q"):
            break

        eaten_part, meal_type, next_meal = _parse_input(eaten)

        print(f"\nAnalyzing your {meal_type}...")
        print("-" * 40)

        try:
            recommendation = assistant.recommend(
                user_message=f"What should I eat for {next_meal}?",
                eaten_description=eaten_part,
                eaten_meal_type=meal_type,
                next_meal=next_meal,
            )
            print(f"\nRecommendation for {next_meal}:")
            print("-" * 40)
            print(recommendation)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    assistant.close()
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
