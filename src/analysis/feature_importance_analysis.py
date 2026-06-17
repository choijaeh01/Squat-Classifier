from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from models.channel_metadata import CHANNEL_ORDER


FEATURE_STATISTICS = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "energy",
    "rms",
    "ptp",
    "dominant_freq_bin",
]


def feature_definitions(input_channels: int = 18) -> list[dict[str, Any]]:
    if input_channels != len(CHANNEL_ORDER):
        channel_names = [f"imu{index:02d}" for index in range(input_channels)]
    else:
        channel_names = CHANNEL_ORDER
    rows: list[dict[str, Any]] = []
    feature_index = 0
    for statistic in FEATURE_STATISTICS:
        for channel_index, channel_name in enumerate(channel_names):
            rows.append(
                {
                    "feature_index": feature_index,
                    "feature_name": f"{channel_name}_{statistic}",
                    "channel_index": channel_index,
                    "channel_name": channel_name,
                    "statistic": statistic,
                    "source": "imu_signal",
                }
            )
            feature_index += 1
    return rows


def build_feature_audit_rows(input_channels: int = 18) -> list[dict[str, Any]]:
    rows = []
    for row in feature_definitions(input_channels=input_channels):
        payload = {
            "feature_name": row["feature_name"],
            "source": row["source"],
            "is_signal_derived": True,
            "uses_metadata": False,
            "uses_label": False,
            "uses_subject_id": False,
            "uses_window_boundary": False,
            "allowed": True,
        }
        rows.append(payload)
    return rows


def validate_feature_audit_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        allowed = (
            bool(row["is_signal_derived"])
            and not bool(row["uses_metadata"])
            and not bool(row["uses_label"])
            and not bool(row["uses_subject_id"])
            and not bool(row["uses_window_boundary"])
            and bool(row["allowed"])
        )
        if not allowed:
            raise ValueError(f"disallowed feature detected: {row}")


def random_forest_importance_rows(
    *,
    estimator: Any,
    model_name: str,
    seed: int,
    fold_id: int,
    test_subject: int,
    input_channels: int = 18,
) -> list[dict[str, Any]]:
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return []
    definitions = feature_definitions(input_channels=input_channels)
    rows = []
    for definition, importance in zip(definitions, np.asarray(importances, dtype=np.float64)):
        rows.append(
            {
                "model_name": model_name,
                "seed": seed,
                "fold_id": fold_id,
                "test_subject": test_subject,
                "feature_index": definition["feature_index"],
                "feature_name": definition["feature_name"],
                "channel_name": definition["channel_name"],
                "statistic": definition["statistic"],
                "importance": float(importance),
            }
        )
    return rows


def linear_svm_coefficient_rows(
    *,
    estimator: Any,
    model_name: str,
    seed: int,
    fold_id: int,
    test_subject: int,
    input_channels: int = 18,
) -> list[dict[str, Any]]:
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return []
    definitions = feature_definitions(input_channels=input_channels)
    rows = []
    for class_id, values in enumerate(np.asarray(coef, dtype=np.float64)):
        for definition, coefficient in zip(definitions, values):
            rows.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "fold_id": fold_id,
                    "test_subject": test_subject,
                    "class_id": class_id,
                    "feature_index": definition["feature_index"],
                    "feature_name": definition["feature_name"],
                    "channel_name": definition["channel_name"],
                    "statistic": definition["statistic"],
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                }
            )
    return rows


def aggregate_importance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["feature_name"]), str(row["channel_name"]), str(row["statistic"]))].append(float(row["importance"]))
    output = []
    for (feature_name, channel_name, statistic), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "feature_name": feature_name,
                "channel_name": channel_name,
                "statistic": statistic,
                "mean_importance": float(arr.mean()),
                "std_importance": float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
        )
    return sorted(output, key=lambda row: float(row["mean_importance"]), reverse=True)


def aggregate_coefficient_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["feature_name"]), str(row["channel_name"]), str(row["statistic"]))].append(float(row["abs_coefficient"]))
    output = []
    for (feature_name, channel_name, statistic), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "feature_name": feature_name,
                "channel_name": channel_name,
                "statistic": statistic,
                "mean_abs_coefficient": float(arr.mean()),
                "std_abs_coefficient": float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
        )
    return sorted(output, key=lambda row: float(row["mean_abs_coefficient"]), reverse=True)
