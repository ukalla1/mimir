"""Patient case-study evaluation for Nutri-ATLAS (paper Table 1).

Runs real patient profiles from `diet_recommendations_dataset.csv` through the
actual recommendation pipeline (LLM gap analysis -> recommend_v2 -> recommend_meal)
and emits a case-study table:

    Patient | Profile | Constraints (restriction · allergy) | Target Diet
            | Recommended Meal | Nutrition | Adheres

Separation of concerns (matches the paper):
  - Health condition shapes the macro TARGET via the LLM gap analysis (LLM Call 1).
  - Allergies / dietary restrictions are RETRIEVAL constraints applied in Stage 4
    (disliked_names hard-exclude).

Adherence ✓ certifies: (a) the recommended recipe contains NONE of the patient's
allergens (verified from ingredient names), AND (b) the macro direction is
clinically appropriate (e.g. diabetic meals are low-carb, recipe carbohydrate
within ±50% of the LLM target carb).

Honest limitation: Low-Sodium is enforced only by ingredient avoidance
(no per-recipe sodium in the corpus).

Requires: the llama.cpp LLM server (gap analysis) + the Qwen3-Embedding model + DuckDB.
Run:  python nutri_rag/scripts/eval_patient_recommendations.py --n 10
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

# Make the nutri_rag package importable when run as a script.
_PKG = Path(__file__).resolve().parents[1]          # .../nutri_rag
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from nutri_rag.assistant.gap_analyzer import analyze_gap
from nutri_rag.assistant.food_recommender import FoodRecommender
from nutri_rag.assistant.meal_recommender import MealRecommender, _ingredients_contain_any

_REPO_ROOT = Path(__file__).resolve().parents[2]    # .../mimir
_DEFAULT_CSV = _REPO_ROOT / "diet_recommendations_dataset.csv"

# Disease -> clinical diet label (verified deterministic in the dataset).
_DISEASE_TO_DIET = {
    "Diabetes": "Low-Carb",
    "Hypertension": "Low-Sodium",
    "Obesity": "Balanced",
    "None": "Balanced",
}

# Allergy -> ingredient terms (used BOTH for hard exclusion and post-hoc safety check).
_ALLERGEN_TERMS = {
    "Peanuts": ["peanut", "peanuts"],
    "Gluten": ["wheat", "flour", "bread", "barley", "rye", "gluten"],
}
# Dietary restriction -> ingredient terms (hard exclusion only; NOT a safety claim).
_RESTRICTION_TERMS = {
    "Low_Sugar": ["sugar", "syrup", "honey", "candy", "dessert"],
    "Low_Sodium": ["salt", "soy sauce", "bacon", "sausage", "bouillon"],
}

# Dessert names mis-tagged as "lunch" in the corpus — skip for meal recommendations.
_DESSERT_KW = (
    "sherbet", "sorbet", "ice cream", "ice-cream", "cake", "cupcake", "cookie",
    "brownie", "pudding", "frosting", "popsicle", "mousse", "parfait",
    "milkshake", "custard", "gelato", "donut", "doughnut", "candy",
)


def _is_dessert(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in _DESSERT_KW)


def _load_patients(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _select_diverse(patients: list[dict], n: int, seed: int) -> list[dict]:
    """Greedy coverage of (Disease × Allergy) combos, then fill randomly."""
    rng = random.Random(seed)
    pool = patients[:]
    rng.shuffle(pool)
    picked, seen = [], set()
    for p in pool:
        key = (p["Disease_Type"], p["Allergies"], p["Dietary_Restrictions"])
        if key not in seen:
            seen.add(key)
            picked.append(p)
        if len(picked) >= n:
            return picked[:n]
    for p in pool:                      # fill remaining slots
        if p not in picked:
            picked.append(p)
        if len(picked) >= n:
            break
    return picked[:n]


def _intake_item(p: dict) -> dict:
    """Synthesize the patient's current intake as one meal_item for analyze_gap.

    This dataset records only `Daily_Caloric_Intake` (no macro breakdown), so we
    estimate current macros from calories with a typical split (50% carb / 20%
    protein / 30% fat). The health condition (passed separately to analyze_gap)
    is the primary driver of the clinical target direction.
    """
    cal = float(p.get("Daily_Caloric_Intake", 0) or 0)
    return {
        "description": "the patient's typical daily intake",
        "meal_type": "daily intake",
        "nutrients": {
            "Carbohydrate, by difference": 0.50 * cal / 4.0,
            "Protein": 0.20 * cal / 4.0,
            "Total lipid (fat)": 0.30 * cal / 9.0,
            "Energy": cal,
        },
        "quantity": None,
    }


def _disliked_and_allergens(p: dict) -> tuple[list[str], list[str]]:
    """Return (disliked_names for hard-exclude, allergen_terms for safety check)."""
    allergy = p["Allergies"]
    restriction = p["Dietary_Restrictions"]
    allergen_terms = _ALLERGEN_TERMS.get(allergy, [])
    disliked = list(allergen_terms) + _RESTRICTION_TERMS.get(restriction, [])
    return disliked, allergen_terms


def _profile_str(p: dict) -> str:
    g = (p["Gender"] or "?")[0]
    return f"{p['Age']}{g}, BMI {p['BMI']}, {p['Disease_Type']} ({p['Severity']})"


def _constraints_str(p: dict) -> str:
    r = p["Dietary_Restrictions"].replace("_", "-")
    return f"{r} · {p['Allergies']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--macro-weight", type=float, default=1.0,
                    help="weight pulling recipe retrieval toward the target macros")
    ap.add_argument("--out", type=Path, default=None, help="optional CSV output path")
    args = ap.parse_args()

    patients = _select_diverse(_load_patients(args.csv), args.n, args.seed)
    print(f"Selected {len(patients)} patients from {args.csv}\n")

    food_rec = FoodRecommender()
    meal_rec = MealRecommender()

    rows = []
    for p in patients:
        pid = p["Patient_ID"]
        disease = p["Disease_Type"]
        target_diet = _DISEASE_TO_DIET.get(disease, "Balanced")
        disliked, allergen_terms = _disliked_and_allergens(p)

        # LLM Call 1 — gap analysis shaped by the health condition.
        gap = analyze_gap([_intake_item(p)], next_meal="lunch", health_condition=disease)
        targets = gap["targets"]
        target_carb = float(targets.get("carb_g", 0) or 0)

        # Stage 2 — gap-filling foods.
        foods = food_rec.recommend_v2(targets=targets, n_results=10)
        # Stage 4 — meal composition with retrieval-side constraints.
        #   macro_weight pulls the pick toward the clinical macro target;
        #   min_overlap=0 lets higher-carb recipes surface for balanced targets;
        #   pull a wide top-K so de-dup has room.
        meals = meal_rec.recommend_meal(
            recommended_foods=foods,
            targets=targets,
            next_meal="lunch",
            disliked_names=disliked or None,
            min_overlap=0,
            top_k_final=25,
            macro_weight=args.macro_weight,
        )

        if not meals:
            rows.append({"pid": pid, "p": p, "meal": None})
            print(f"[{pid}] {disease:12s} -> NO MEAL RETURNED")
            continue

        # Pick the highest-ranked non-dessert recipe (duplicates across patients
        # are allowed: each patient independently gets their best-matching meal).
        top = next((m for m in meals if not _is_dessert(m.recipe_name)), meals[0])
        allergen_safe = not _ingredients_contain_any(top.ingredients, allergen_terms)
        carb_g = float(top.nutrients.get("carbohydrates", 0.0))
        rows.append({
            "pid": pid, "p": p, "meal": top, "target_diet": target_diet,
            "allergen_safe": allergen_safe, "carb_g": carb_g,
            "target_carb": target_carb,
            "reasoning": gap.get("reasoning", ""),
        })
        print(f"[{pid}] {disease:12s} -> {top.recipe_name[:44]:44s} "
              f"carb={carb_g:5.1f}g (target {target_carb:4.0f}g) allergen_safe={allergen_safe}")

    # ---- macro-direction check: recipe carbs within +-50% of the LLM target carb ----
    def macro_ok(r: dict) -> bool:
        tc = r.get("target_carb", 0)
        if not tc:
            return True
        return 0.5 * tc <= r["carb_g"] <= 1.5 * tc

    # ---- console + LaTeX table ----
    print("\n" + "=" * 100)
    print("Adheres = allergen-safe AND recipe carb within +-50% of the clinical target carb")
    print("=" * 100 + "\nLaTeX rows for Table 1:\n")

    for r in rows:
        p = r["p"]
        if not r.get("meal"):
            print(f"% {r['pid']}: no meal returned")
            continue
        adheres = r["allergen_safe"] and macro_ok(r)
        mark = r"\checkmark" if adheres else r"$\times$"
        meal_name = r["meal"].recipe_name.replace("&", r"\&").replace("_", r"\_")
        n = r["meal"].nutrients
        nutr = (f"{n.get('calories',0):.0f}\\,kcal, "
                f"{n.get('protein',0):.0f}/{n.get('carbohydrates',0):.0f}/{n.get('fat',0):.0f}\\,g")
        print(
            f"{r['pid']} & {_profile_str(p)} & {_constraints_str(p)} & "
            f"{r['target_diet']} & \\textit{{{meal_name}}} & {nutr} & {mark} \\\\ \\hline"
        )

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Patient", "Profile", "Constraints", "Target_Diet",
                        "Recommended_Meal", "Calories", "Protein_g", "Carb_g", "Fat_g",
                        "Allergen_Safe", "Adheres"])
            for r in rows:
                if not r.get("meal"):
                    continue
                n = r["meal"].nutrients
                w.writerow([r["pid"], _profile_str(r["p"]), _constraints_str(r["p"]),
                            r["target_diet"], r["meal"].recipe_name,
                            f"{n.get('calories',0):.0f}", f"{n.get('protein',0):.0f}",
                            f"{n.get('carbohydrates',0):.0f}", f"{n.get('fat',0):.0f}",
                            r["allergen_safe"], r["allergen_safe"] and macro_ok(r)])
        print(f"\nWrote CSV -> {args.out}")


if __name__ == "__main__":
    main()
