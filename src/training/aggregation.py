from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def aggregate_metrics_by_model(
    fold_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    params_by_model = {str(row["model_name"]): row for row in parameter_rows}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[(str(row["model_name"]), int(row["seed"]))].append(row)

    output: list[dict[str, Any]] = []
    for (model_name, seed), rows in sorted(grouped.items()):
        successes = [row for row in rows if str(row.get("status")) in {"ok", "forward_only"}]
        failures = [row for row in rows if str(row.get("status")) == "failed"]
        accuracies = [_float(row.get("test_accuracy")) for row in successes]
        macro_f1s = [_float(row.get("test_macro_f1")) for row in successes]
        best_epochs = [_float(row.get("best_epoch")) for row in successes]
        params = params_by_model.get(model_name, {})
        output.append(
            {
                "model_name": model_name,
                "seed": seed,
                "mean_test_accuracy": _mean(accuracies),
                "std_test_accuracy": _std(accuracies),
                "mean_test_macro_f1": _mean(macro_f1s),
                "std_test_macro_f1": _std(macro_f1s),
                "min_test_macro_f1": _min(macro_f1s),
                "max_test_macro_f1": _max(macro_f1s),
                "mean_best_epoch": _mean(best_epochs),
                "total_params": int(params.get("total_params", 0) or 0),
                "encoder_params": int(params.get("encoder_params", 0) or 0),
                "aggregation_params": int(params.get("aggregation_params", 0) or 0),
                "head_params": int(params.get("head_params", 0) or 0),
                "n_success_folds": len(successes),
                "n_failed_folds": len(failures),
            }
        )
    return output


def subjectwise_metrics_by_model(
    fold_rows: list[dict[str, Any]],
    classwise_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    class_f1: dict[tuple[str, int, int], float] = {}
    for row in classwise_rows:
        key = (str(row["model_name"]), int(row["test_subject"]), int(row["class_id"]))
        class_f1[key] = _float(row.get("f1"))

    output: list[dict[str, Any]] = []
    for row in fold_rows:
        if str(row.get("status")) not in {"ok", "forward_only"}:
            continue
        model_name = str(row["model_name"])
        test_subject = int(row["test_subject"])
        payload: dict[str, Any] = {
            "model_name": model_name,
            "seed": int(row["seed"]),
            "test_subject": test_subject,
            "test_accuracy": _float(row.get("test_accuracy")),
            "test_macro_f1": _float(row.get("test_macro_f1")),
        }
        for class_id in range(5):
            payload[f"class{class_id}_f1"] = class_f1.get((model_name, test_subject, class_id), 0.0)
        output.append(payload)
    return output


def _float(value: Any) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def _mean(values: list[float]) -> float:
    clean = [value for value in values if not np.isnan(value)]
    return float(np.mean(clean)) if clean else 0.0


def _std(values: list[float]) -> float:
    clean = [value for value in values if not np.isnan(value)]
    return float(np.std(clean)) if clean else 0.0


def _min(values: list[float]) -> float:
    clean = [value for value in values if not np.isnan(value)]
    return float(np.min(clean)) if clean else 0.0


def _max(values: list[float]) -> float:
    clean = [value for value in values if not np.isnan(value)]
    return float(np.max(clean)) if clean else 0.0
