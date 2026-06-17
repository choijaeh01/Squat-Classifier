from __future__ import annotations

from typing import Any

import numpy as np

from models.classical_features import build_classical_estimator, extract_window_features
from training.metrics import compute_classification_metrics
from training.splits import LOSOSmokeSplit
from training.supervised_trainer import CLASS_NAMES


def run_classical_fold(
    *,
    model_name: str,
    seed: int,
    fold_id: int,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    split: LOSOSmokeSplit,
    return_estimator: bool = False,
) -> dict[str, Any]:
    features = extract_window_features(X)
    estimator = build_classical_estimator(model_name, seed)
    estimator.fit(features[split.train_idx], y[split.train_idx])
    train_eval = _evaluate_estimator(estimator, features, y, subject_id, split.train_idx)
    val_eval = _evaluate_estimator(estimator, features, y, subject_id, split.val_idx)
    test_eval = _evaluate_estimator(estimator, features, y, subject_id, split.test_idx)
    result = {
        "fold_metric": _fold_metric_row(model_name, seed, fold_id, split, val_eval, test_eval),
        "classwise": _classwise_rows(model_name, seed, fold_id, split.test_subject, test_eval),
        "history": [
            {
                "model_name": model_name,
                "seed": seed,
                "fold_id": fold_id,
                "epoch": 0,
                "train_loss": "",
                "val_loss": "",
                "train_accuracy": train_eval["metrics"]["accuracy"],
                "train_macro_f1": train_eval["metrics"]["macro_f1"],
                "val_accuracy": val_eval["metrics"]["accuracy"],
                "val_macro_f1": val_eval["metrics"]["macro_f1"],
                "is_best": True,
            }
        ],
        "confusion": _confusion_rows(model_name, seed, fold_id, split.test_subject, test_eval),
        "predictions": _prediction_rows(model_name, seed, fold_id, split.test_subject, test_eval),
    }
    if return_estimator:
        result["estimator"] = estimator
    return result


def _evaluate_estimator(
    estimator: Any,
    features: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    indices = np.asarray(indices, dtype=np.int64)
    pred = estimator.predict(features[indices]).astype(np.int64)
    true = y[indices].astype(np.int64)
    subjects = subject_id[indices].astype(int)
    metrics = compute_classification_metrics(
        y_true=true,
        y_pred=pred,
        subject_ids=subjects,
        num_classes=5,
        class_names=CLASS_NAMES,
    )
    return {
        "metrics": metrics,
        "loss": "",
        "y_true": true.astype(int).tolist(),
        "y_pred": pred.astype(int).tolist(),
        "subject_id": subjects.astype(int).tolist(),
        "sample_index": indices.astype(int).tolist(),
    }


def _fold_metric_row(
    model_name: str,
    seed: int,
    fold_id: int,
    split: LOSOSmokeSplit,
    val_eval: dict[str, Any],
    test_eval: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "seed": seed,
        "fold_id": fold_id,
        "train_subjects": "|".join(str(item) for item in split.train_subjects),
        "val_subject": split.val_subject,
        "test_subject": split.test_subject,
        "n_train": len(split.train_idx),
        "n_val": len(split.val_idx),
        "n_test": len(split.test_idx),
        "best_epoch": 0,
        "best_val_macro_f1": val_eval["metrics"]["macro_f1"],
        "test_accuracy": test_eval["metrics"]["accuracy"],
        "test_macro_f1": test_eval["metrics"]["macro_f1"],
        "test_weighted_f1": test_eval["metrics"]["weighted_f1"],
        "leakage_check_passed": split.leakage_check_passed,
        "scaler_fit_subjects": "|".join(str(item) for item in split.train_subjects),
        "status": "ok",
    }


def _classwise_rows(model_name: str, seed: int, fold_id: int, test_subject: int, test_eval: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        metrics = test_eval["metrics"]["class_wise"][class_name]
        rows.append(
            {
                "model_name": model_name,
                "seed": seed,
                "fold_id": fold_id,
                "test_subject": test_subject,
                "class_id": class_id,
                "class_name": class_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "support": metrics["support"],
            }
        )
    return rows


def _confusion_rows(model_name: str, seed: int, fold_id: int, test_subject: int, test_eval: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for true_class, values in enumerate(test_eval["metrics"]["confusion_matrix"]):
        for pred_class, count in enumerate(values):
            rows.append(
                {
                    "scope": "fold",
                    "model_name": model_name,
                    "seed": seed,
                    "fold_id": fold_id,
                    "test_subject": test_subject,
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "count": int(count),
                }
            )
    return rows


def _prediction_rows(model_name: str, seed: int, fold_id: int, test_subject: int, test_eval: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model_name": model_name,
            "seed": seed,
            "fold_id": fold_id,
            "test_subject": test_subject,
            "sample_index": sample_index,
            "subject_id": subject,
            "y_true": true,
            "y_pred": pred,
        }
        for sample_index, subject, true, pred in zip(
            test_eval["sample_index"],
            test_eval["subject_id"],
            test_eval["y_true"],
            test_eval["y_pred"],
        )
    ]
