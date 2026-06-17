from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label, pred_label] += 1
    return matrix


def precision_recall_f1_from_confusion(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    true_positive = np.diag(matrix)
    predicted_positive = matrix.sum(axis=0)
    actual_positive = matrix.sum(axis=1)
    precision = _safe_divide(true_positive, predicted_positive)
    recall = _safe_divide(true_positive, actual_positive)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def compute_classification_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
    num_classes: int,
    class_names: list[str],
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    subject_ids = np.asarray(subject_ids)
    if not (len(y_true) == len(y_pred) == len(subject_ids)):
        raise ValueError("y_true, y_pred, and subject_ids must have equal lengths")

    matrix = confusion_matrix(y_true, y_pred, num_classes)
    precision, recall, f1 = precision_recall_f1_from_confusion(matrix)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(f1.mean()) if len(f1) else 0.0

    subject_scores: dict[str, float] = {}
    for subject in sorted(np.unique(subject_ids).tolist()):
        mask = subject_ids == subject
        subject_matrix = confusion_matrix(y_true[mask], y_pred[mask], num_classes)
        _, _, subject_f1 = precision_recall_f1_from_confusion(subject_matrix)
        subject_scores[str(subject)] = float(subject_f1.mean())

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_wise": {
            class_names[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(matrix[index].sum()),
            }
            for index in range(num_classes)
        },
        "subject_wise_macro_f1": subject_scores,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.zeros_like(numerator, dtype=np.float64)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result
