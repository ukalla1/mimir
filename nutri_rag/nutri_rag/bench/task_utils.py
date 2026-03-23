"""Shared evaluation utilities for all NutriBench task variants.

process_results() and clean_output() are nutrient-aware via NUTRI_TARGET env var.
Each task's utils.py imports these and adds its own doc_to_text function.
"""

import os
import re

from nutri_rag.bench.nutrient_prompts import get_nutrient_config


def _get_target():
    """Read NUTRI_TARGET at call time, not import time."""
    return os.environ.get("NUTRI_TARGET", "carb")


def process_results(doc, results):
    target = _get_target()
    cfg = get_nutrient_config(target)
    candidates = results[0]
    pred = clean_output(candidates, doc["meal_description"], "cot", target)
    gt = doc[cfg["gt_column"]]
    mae = abs(pred - gt)

    return {
        "acc": mae < cfg["acc_threshold"],
        "mae": mae,
    }


def agg_mae(items):
    return sum(items) / len(items)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def clean_output(raw_output, query, method_name, nutrition_name):
    if isinstance(raw_output, list):
        raw_output = raw_output[0] if raw_output else ""
    if not isinstance(raw_output, str):
        raw_output = str(raw_output)

    if "cot" in method_name:
        splits = raw_output.split("Output:")
        if len(splits) > 1:
            raw_output = splits[1]

    raw_output = raw_output.strip()

    if nutrition_name == 'fat':
        pattern = r'["\']\s*total_fat["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'protein':
        pattern = r'["\']\s*total_protein["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'energy':
        pattern = r'["\']\s*total_energy["\']: (-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)'
    elif nutrition_name == 'carb':
        pattern = r'["\']\s*total_carbohydrates["\']:\s*(?:["\']?(-?[0-9]+(?:\.[0-9]*)?(?:-[0-9]+(?:\.[0-9]*)?)?)["\']?|\[(-?[0-9]+(?:\.[0-9]*)?(?:,\s*-?[0-9]+(?:\.[0-9]*)?)*)\])'
    else:
        raise NotImplementedError

    match = re.search(pattern, raw_output)
    if match:
        if match.group(1):
            pred_val = match.group(1)
            if is_number(pred_val):
                return float(pred_val)
            else:
                pred_list = pred_val.split('-')
                if len(pred_list) == 2 and is_number(pred_list[0]) and is_number(pred_list[1]):
                    return (float(pred_list[0]) + float(pred_list[1])) / 2.0
                else:
                    print(f"EXCEPTION AFTER MATCHING")
                    print(f"Matched output: {raw_output}")
                    print(f"Query: {query}")
                    return -1
        elif match.group(2):
            try:
                pred_list = match.group(2).split(',')
                return (float(pred_list[0]) + float(pred_list[1])) / 2.0
            except Exception:
                print(f"EXCEPTION AFTER MATCHING")
                print(f"Matched output: {raw_output}")
                print(f"Query: {query}")
                return -1
    else:
        if is_number(raw_output):
            return float(raw_output)
        else:
            print(f"EXCEPTION")
            print(f"Matched output: {raw_output}")
            print(f"Query: {query}")
            return -1
