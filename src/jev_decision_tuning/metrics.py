"""Serializable sentiment metrics with an explicit, fixed class ordering."""

from collections.abc import Iterable, Mapping
from numbers import Integral
from typing import Any

from .prompts import LABEL_NAMES


def compute_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Report all three classes, including classes absent from predictions.

    Confusion matrix rows are true classes and columns are predicted classes.
    Undefined precision or recall is reported as zero. Macro scores always
    average negative, neutral, and positive, in that order.
    """
    matrix = [[0 for _ in LABEL_NAMES] for _ in LABEL_NAMES]
    count = 0
    for record in records:
        true_label, pred_label = record["true_label"], record["pred_label"]
        for name, value in (("true_label", true_label), ("pred_label", pred_label)):
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or not 0 <= value < len(LABEL_NAMES)
            ):
                raise ValueError(f"{name} must be an integer class ID: 0, 1, or 2")
        matrix[int(true_label)][int(pred_label)] += 1
        count += 1
    if not count:
        raise ValueError("cannot compute metrics for an empty evaluation")

    per_class = {}
    for class_id, name in enumerate(LABEL_NAMES):
        correct = matrix[class_id][class_id]
        support = sum(matrix[class_id])
        predicted = sum(row[class_id] for row in matrix)
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "examples": count,
        "accuracy": sum(matrix[i][i] for i in range(len(LABEL_NAMES))) / count,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(LABEL_NAMES),
        "macro_recall": sum(item["recall"] for item in per_class.values()) / len(LABEL_NAMES),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "label_names": list(LABEL_NAMES),
    }
