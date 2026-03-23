"""Per-nutrient system prompts and few-shot examples for NutriBench.

Each nutrient has its own CoT prompt template with appropriate examples.
The active nutrient is selected via the NUTRI_TARGET environment variable
(default: "carb").
"""

import os

# ── Nutrient configuration ────────────────────────────────────────────


def _get_target():
    """Read NUTRI_TARGET at call time, not import time."""
    return os.environ.get("NUTRI_TARGET", "carb")

NUTRIENT_CONFIG = {
    "carb": {
        "full_name": "carbohydrates",
        "json_key": "total_carbohydrates",
        "unit": "grams",
        "gt_column": "carb",
        "formula": "carbs = (weight_g / 100) * carbs_per_100g",
        "acc_threshold": 7.5,
    },
    "protein": {
        "full_name": "protein",
        "json_key": "total_protein",
        "unit": "grams",
        "gt_column": "protein",
        "formula": "protein = (weight_g / 100) * protein_per_100g",
        "acc_threshold": 7.5,
    },
    "fat": {
        "full_name": "fat",
        "json_key": "total_fat",
        "unit": "grams",
        "gt_column": "fat",
        "formula": "fat = (weight_g / 100) * fat_per_100g",
        "acc_threshold": 7.5,
    },
    "energy": {
        "full_name": "energy",
        "json_key": "total_energy",
        "unit": "kcal",
        "gt_column": "energy",
        "formula": "energy = (weight_g / 100) * energy_per_100g",
        "acc_threshold": 50.0,
    },
}

# ── Few-shot examples per nutrient ────────────────────────────────────

_EXAMPLES = {
    "carb": [
        {
            "query": "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
            "reasoning": (
                "The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n"
                "1 cup of oatmeal has 27g carbs.\n"
                "1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.\n"
                "1 glass of orange juice has 26g carbs.\n"
                "So the total grams of carbs in the meal = (27 + 13.5 + 26) = 66.5"
            ),
            "output_key": "total_carbohydrates",
            "output_val": 66.5,
        },
        {
            "query": "I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
            "reasoning": (
                "The meal consists of scrambled eggs made with 2 eggs and 1 toast.\n"
                "Scrambled eggs made with 2 eggs has 2g carbs.\n"
                "1 toast has 13g carbs.\n"
                "So the total grams of carbs in the meal = (2 + 13) = 15"
            ),
            "output_key": "total_carbohydrates",
            "output_val": 15,
        },
        {
            "query": "Half a peanut butter and jelly sandwich.",
            "reasoning": (
                "The meal consists of 1/2 a peanut butter and jelly sandwich.\n"
                "1 peanut butter and jelly sandwich has 50.6g carbs so half a peanut butter and jelly sandwich has (50.6*(1/2)) = 25.3g carbs.\n"
                "So the total grams of carbs in the meal = 25.3"
            ),
            "output_key": "total_carbohydrates",
            "output_val": 25.3,
        },
    ],
    "protein": [
        {
            "query": "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
            "reasoning": (
                "The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n"
                "1 cup of oatmeal has 5g protein.\n"
                "1 banana has 1.3g protein so half a banana has (1.3*(1/2)) = 0.65g protein.\n"
                "1 glass of orange juice has 1.7g protein.\n"
                "So the total grams of protein in the meal = (5 + 0.65 + 1.7) = 7.35"
            ),
            "output_key": "total_protein",
            "output_val": 7.35,
        },
        {
            "query": "I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
            "reasoning": (
                "The meal consists of scrambled eggs made with 2 eggs and 1 toast.\n"
                "Scrambled eggs made with 2 eggs has 13g protein.\n"
                "1 toast has 3g protein.\n"
                "So the total grams of protein in the meal = (13 + 3) = 16"
            ),
            "output_key": "total_protein",
            "output_val": 16,
        },
        {
            "query": "Half a peanut butter and jelly sandwich.",
            "reasoning": (
                "The meal consists of 1/2 a peanut butter and jelly sandwich.\n"
                "1 peanut butter and jelly sandwich has 13.4g protein so half has (13.4*(1/2)) = 6.7g protein.\n"
                "So the total grams of protein in the meal = 6.7"
            ),
            "output_key": "total_protein",
            "output_val": 6.7,
        },
    ],
    "fat": [
        {
            "query": "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
            "reasoning": (
                "The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n"
                "1 cup of oatmeal has 3.5g fat.\n"
                "1 banana has 0.4g fat so half a banana has (0.4*(1/2)) = 0.2g fat.\n"
                "1 glass of orange juice has 0.5g fat.\n"
                "So the total grams of fat in the meal = (3.5 + 0.2 + 0.5) = 4.2"
            ),
            "output_key": "total_fat",
            "output_val": 4.2,
        },
        {
            "query": "I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
            "reasoning": (
                "The meal consists of scrambled eggs made with 2 eggs and 1 toast.\n"
                "Scrambled eggs made with 2 eggs has 11g fat.\n"
                "1 toast has 1g fat.\n"
                "So the total grams of fat in the meal = (11 + 1) = 12"
            ),
            "output_key": "total_fat",
            "output_val": 12,
        },
        {
            "query": "Half a peanut butter and jelly sandwich.",
            "reasoning": (
                "The meal consists of 1/2 a peanut butter and jelly sandwich.\n"
                "1 peanut butter and jelly sandwich has 17.2g fat so half has (17.2*(1/2)) = 8.6g fat.\n"
                "So the total grams of fat in the meal = 8.6"
            ),
            "output_key": "total_fat",
            "output_val": 8.6,
        },
    ],
    "energy": [
        {
            "query": "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice.",
            "reasoning": (
                "The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n"
                "1 cup of oatmeal has 154 kcal.\n"
                "1 banana has 105 kcal so half a banana has (105*(1/2)) = 52.5 kcal.\n"
                "1 glass of orange juice has 112 kcal.\n"
                "So the total energy in the meal = (154 + 52.5 + 112) = 318.5"
            ),
            "output_key": "total_energy",
            "output_val": 318.5,
        },
        {
            "query": "I ate scrambled eggs made with 2 eggs and a toast for breakfast.",
            "reasoning": (
                "The meal consists of scrambled eggs made with 2 eggs and 1 toast.\n"
                "Scrambled eggs made with 2 eggs has 182 kcal.\n"
                "1 toast has 79 kcal.\n"
                "So the total energy in the meal = (182 + 79) = 261"
            ),
            "output_key": "total_energy",
            "output_val": 261,
        },
        {
            "query": "Half a peanut butter and jelly sandwich.",
            "reasoning": (
                "The meal consists of 1/2 a peanut butter and jelly sandwich.\n"
                "1 peanut butter and jelly sandwich has 376 kcal so half has (376*(1/2)) = 188 kcal.\n"
                "So the total energy in the meal = 188"
            ),
            "output_key": "total_energy",
            "output_val": 188,
        },
    ],
}


def get_nutrient_config(nutrient: str | None = None) -> dict:
    """Return the config dict for the given nutrient (default: NUTRI_TARGET env)."""
    nutrient = nutrient or _get_target()
    if nutrient not in NUTRIENT_CONFIG:
        raise ValueError(f"Unknown nutrient: {nutrient!r}. Valid: {list(NUTRIENT_CONFIG)}")
    return NUTRIENT_CONFIG[nutrient]


def build_system_prompt(nutrient: str | None = None) -> str:
    """Build the CoT system prompt for the given nutrient."""
    nutrient = nutrient or _get_target()
    cfg = get_nutrient_config(nutrient)
    examples = _EXAMPLES[nutrient]

    full_name = cfg["full_name"]
    json_key = cfg["json_key"]
    unit = cfg["unit"]

    lines = [
        f"For the given query including a meal description, think step by step as follows:",
        f"1. Parse the meal description into discrete food or beverage items along with their serving size. "
        f"If the serving size of any item in the meal is not specified, assume it is a single standard serving "
        f"based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't "
        f"relate to the item name and serving size.",
        f"2. For each food or beverage item in the meal, calculate the amount of {full_name} in {unit} for the specific serving size.",
        f"3. Respond with a dictionary object containing the total {full_name} in {unit} as follows:",
        f'{{\"{json_key}\": total {unit} of {full_name} for the serving}}',
        f"For the total {full_name}, respond with just the numeric amount without extra text. "
        f"If you don't know the answer, set the value of \"{json_key}\" to -1.",
        "",
        "Follow the format of the following examples when answering",
    ]

    for ex in examples:
        lines.append("")
        lines.append(f'Query: "{ex["query"]}"')
        lines.append("Answer: Let's think step by step.")
        lines.append(ex["reasoning"])
        lines.append(f'Output: {{"{ex["output_key"]}": {ex["output_val"]}}}')

    return "\n".join(lines)
