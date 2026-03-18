#!/usr/bin/env python3
"""Interactive demo: test NutriBench RAG retrieval on sample meals.

Type a meal description and see what USDA foods are retrieved and
what the augmented prompt looks like.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nutri_rag.bench.retriever import BenchRetriever
from nutri_rag.bench.prompt import format_nutrient_block, build_rag_doc_to_text


def main():
    retriever = BenchRetriever()

    print("NutriBench RAG Retrieval Demo")
    print("Type a meal description (or 'quit' to exit)")
    print("-" * 60)

    while True:
        try:
            meal = input("\nMeal: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not meal or meal.lower() in ("quit", "exit", "q"):
            break

        contexts = retriever.retrieve(meal)

        print(f"\n--- Parsed {len(contexts)} food item(s) ---")
        for ctx in contexts:
            p = ctx.parsed
            status = "MATCHED" if ctx.matched else "NO MATCH"
            print(f"  [{status}] '{p.food_term}'"
                  f" (qty={p.quantity}, unit={p.unit})")
            if ctx.matched:
                print(f"    -> {ctx.description} (USDA #{ctx.fdc_id})")
                for name, val in ctx.nutrients.items():
                    print(f"       {name}: {val:.1f}")

        print(f"\n--- RAG Prompt ---")
        print(build_rag_doc_to_text(meal, contexts))


if __name__ == "__main__":
    main()
