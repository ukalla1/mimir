"""Evaluation metrics for PFoodReq benchmark: MAP, MAR, F1.

Follows the same definitions as the PFoodReq paper (WSDM 2021):
- MAP (Mean Average Precision)
- MAR (Mean Average Recall)
- F1 (macro, averaged over all questions)
"""

from __future__ import annotations


def _normalize_name(name: str) -> str:
    """Normalize a recipe name for comparison.

    Handles backslash escapes and case differences.
    """
    name = name.replace("\\'", "'").replace('\\"', '"')
    name = name.strip().lower()
    return name


def compute_metrics(
    predicted: list[str],
    ground_truth: list[str],
) -> dict[str, float]:
    """Compute precision, recall, F1 for a single query.

    Args:
        predicted: list of predicted recipe names.
        ground_truth: list of ground truth recipe names.

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    if not ground_truth:
        return {"precision": 1.0 if not predicted else 0.0, "recall": 1.0, "f1": 1.0 if not predicted else 0.0}

    gt_set = {_normalize_name(n) for n in ground_truth}
    pred_set = {_normalize_name(n) for n in predicted}

    tp = len(gt_set & pred_set)

    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_average_precision(
    predicted_ranked: list[str],
    ground_truth: list[str],
) -> float:
    """Compute Average Precision for a single query.

    AP = (1/|relevant|) * sum_{k=1}^{n} P(k) * rel(k)
    where P(k) is precision at rank k and rel(k) is 1 if item at rank k is relevant.
    """
    if not ground_truth:
        return 1.0 if not predicted_ranked else 0.0

    gt_set = {_normalize_name(n) for n in ground_truth}

    if not predicted_ranked:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for k, pred_name in enumerate(predicted_ranked, 1):
        if _normalize_name(pred_name) in gt_set:
            hits += 1
            sum_precision += hits / k

    return sum_precision / len(gt_set) if gt_set else 0.0


def compute_average_recall(
    predicted_ranked: list[str],
    ground_truth: list[str],
) -> float:
    """Compute Average Recall for a single query.

    AR = (1/min(|relevant|, |predicted|)) * sum_{k=1}^{n} R(k) * rel(k)
    """
    if not ground_truth:
        return 1.0 if not predicted_ranked else 0.0

    gt_set = {_normalize_name(n) for n in ground_truth}

    if not predicted_ranked:
        return 0.0

    hits = 0
    sum_recall = 0.0
    for k, pred_name in enumerate(predicted_ranked, 1):
        if _normalize_name(pred_name) in gt_set:
            hits += 1
            sum_recall += hits / len(gt_set)

    denom = min(len(gt_set), len(predicted_ranked))
    return sum_recall / denom if denom > 0 else 0.0


def aggregate_metrics(all_results: list[dict]) -> dict[str, float]:
    """Compute aggregate MAP, MAR, F1 over all queries.

    Args:
        all_results: list of per-query result dicts, each containing
            "predicted" (ranked list) and "ground_truth" (list).

    Returns:
        {"MAP": float, "MAR": float, "F1": float, "n_queries": int}
    """
    maps = []
    mars = []
    f1s = []

    for result in all_results:
        predicted = result.get("predicted", [])
        ground_truth = result.get("ground_truth", [])

        ap = compute_average_precision(predicted, ground_truth)
        ar = compute_average_recall(predicted, ground_truth)
        metrics = compute_metrics(predicted, ground_truth)

        maps.append(ap)
        mars.append(ar)
        f1s.append(metrics["f1"])

    n = len(all_results)
    return {
        "MAP": sum(maps) / n * 100 if n else 0.0,
        "MAR": sum(mars) / n * 100 if n else 0.0,
        "F1": sum(f1s) / n * 100 if n else 0.0,
        "n_queries": n,
    }
