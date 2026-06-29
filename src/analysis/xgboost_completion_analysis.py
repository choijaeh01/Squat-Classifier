from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from training.supervised_trainer import write_csv


CONTROLLED_RESULT = Path("results/controlled_feature_extractor_comparison/20260629_041712_controlled_feature_extractor_comparison_v1")
MERGED_MODELS = [
    "feature_xgboost_v1",
    "feature_random_forest_v1",
    "controlled_stats_mlp",
    "controlled_shared_1d_residual",
    "controlled_shared_1d_residual_identity",
    "controlled_all_channel_1d_cnn",
    "controlled_all_channel_1d_cnn_small",
    "rescnn_bigru_attention_lite_v1",
    "lee_style_cnn_lstm_2d_v1",
]


def write_xgboost_completion_analysis(*, project_root: Path, run_dir: Path) -> None:
    controlled_dir = project_root / CONTROLLED_RESULT
    merged_dir = run_dir / "merged_with_controlled_comparison"
    merged_dir.mkdir(parents=True, exist_ok=True)

    xgb_aggregate = _read_csv(run_dir / "aggregate_metrics_by_model.csv")
    controlled_aggregate = _read_csv(controlled_dir / "aggregate_metrics_by_model.csv")
    merged = _merged_rows(xgb_aggregate, controlled_aggregate)
    write_csv(merged_dir / "merged_controlled_plus_xgboost.csv", merged)

    paired = _read_csv(run_dir / "paired_model_differences.csv")
    _write_single_comparison(merged_dir / "xgboost_vs_random_forest.csv", paired, "feature_random_forest_v1")
    _write_single_comparison(merged_dir / "xgboost_vs_stats_mlp.csv", paired, "controlled_stats_mlp")
    _write_single_comparison(merged_dir / "xgboost_vs_shared_residual.csv", paired, "controlled_shared_1d_residual")
    _write_single_comparison(merged_dir / "xgboost_vs_all_channel_cnn.csv", paired, "controlled_all_channel_1d_cnn")

    classwise = _read_csv(run_dir / "classwise_metrics_by_model.csv")
    importance = _read_csv(run_dir / "xgboost_feature_importance_aggregate.csv")
    rf_importance = _read_csv(controlled_dir / "random_forest_feature_importance_aggregate.csv")
    subjectwise = _read_csv(run_dir / "subjectwise_metrics_by_model.csv")
    (merged_dir / "xgboost_claim_memo_for_user.md").write_text(
        _claim_memo(merged, paired, classwise, importance),
        encoding="utf-8",
    )
    generate_xgboost_completion_figures(
        run_dir=run_dir,
        merged_rows=merged,
        classwise_rows=classwise,
        subjectwise_rows=subjectwise,
        xgb_importance=importance,
        rf_importance=rf_importance,
    )


def generate_xgboost_completion_figures(
    *,
    run_dir: Path,
    merged_rows: list[dict[str, str]],
    classwise_rows: list[dict[str, str]],
    subjectwise_rows: list[dict[str, str]],
    xgb_importance: list[dict[str, str]],
    rf_importance: list[dict[str, str]],
) -> None:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        (figures_dir / "figures_unavailable.txt").write_text("matplotlib is not installed.\n", encoding="utf-8")
        return
    _macro_bar(plt, merged_rows, figures_dir / "xgboost_macro_f1_vs_baselines.png")
    _top_importance(plt, xgb_importance, figures_dir / "xgboost_feature_importance_top20.png")
    _rf_xgb_importance(plt, xgb_importance, rf_importance, figures_dir / "xgboost_rf_feature_importance_comparison.png")
    _class3(plt, classwise_rows, _controlled_classwise_for_merge(run_dir), figures_dir / "xgboost_class3_f1_recall.png")
    _subjectwise(plt, subjectwise_rows, _controlled_subjectwise_for_merge(run_dir), figures_dir / "xgboost_subjectwise_macro_f1.png")


def _merged_rows(xgb_rows: list[dict[str, str]], controlled_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in xgb_rows:
        if row.get("model_name") in MERGED_MODELS:
            payload = dict(row)
            payload["result_type"] = "xgboost_only_completion_3seed_full"
            rows.append(payload)
    for row in controlled_rows:
        if row.get("model_name") in MERGED_MODELS and row.get("model_name") != "feature_xgboost_v1":
            payload = dict(row)
            payload["result_type"] = "controlled_feature_extractor_readonly"
            rows.append(payload)
    return sorted(rows, key=lambda row: float(row.get("mean_macro_f1") or 0.0), reverse=True)


def _write_single_comparison(path: Path, paired_rows: list[dict[str, str]], reference_model: str) -> None:
    rows = [
        row
        for row in paired_rows
        if row.get("reference_model") == reference_model and row.get("comparison_model") == "feature_xgboost_v1"
    ]
    write_csv(path, rows)


def _claim_memo(
    merged: list[dict[str, Any]],
    paired: list[dict[str, str]],
    classwise: list[dict[str, str]],
    importance: list[dict[str, str]],
) -> str:
    ranking = "\n".join(
        f"- {row['model_name']}: Macro F1 {row.get('mean_macro_f1')}, Accuracy {row.get('mean_accuracy')}"
        for row in merged
    )
    comparisons = "\n".join(
        f"- vs {row['reference_model']}: delta {row.get('mean_difference')}, CI [{row.get('ci_low')}, {row.get('ci_high')}], n={row.get('n_pairs')}"
        for row in paired
        if row.get("comparison_model") == "feature_xgboost_v1"
    )
    class3 = next((row for row in classwise if row.get("model_name") == "feature_xgboost_v1" and row.get("class_id") == "3"), {})
    top_features = "\n".join(
        f"- {row['feature_name']}: {row.get('mean_importance')}"
        for row in importance[:10]
    )
    return (
        "# XGBoost Completion Claim Memo For User\n\n"
        "이 메모는 수치와 관찰만 정리한다. 논문 포함 여부와 최종 해석은 사용자가 판단한다.\n\n"
        "## Merged Ranking\n\n"
        f"{ranking}\n\n"
        "## Paired Differences\n\n"
        f"{comparisons}\n\n"
        "## Class 3 Excessive Lean\n\n"
        f"- recall: {class3.get('mean_recall', '')}\n"
        f"- F1: {class3.get('mean_f1', '')}\n\n"
        "## XGBoost Top Features\n\n"
        f"{top_features}\n"
    )


def _macro_bar(plt: Any, rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    labels = [row["model_name"] for row in rows]
    values = np.asarray([float(row.get("mean_macro_f1") or 0.0) for row in rows])
    lows = np.asarray([float(row.get("macro_f1_ci_low") or value) for row, value in zip(rows, values)])
    highs = np.asarray([float(row.get("macro_f1_ci_high") or value) for row, value in zip(rows, values)])
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.85), 4))
    ax.bar(range(len(labels)), values, yerr=np.vstack([values - lows, highs - values]), capsize=3)
    ax.set_ylabel("Mean Macro F1")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _top_importance(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    top = rows[:20]
    if not top:
        return
    labels = [row["feature_name"] for row in top]
    values = [float(row.get("mean_importance") or 0.0) for row in top]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(labels)), values)
    ax.set_ylabel("Mean importance")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _rf_xgb_importance(plt: Any, xgb_rows: list[dict[str, str]], rf_rows: list[dict[str, str]], path: Path) -> None:
    top = xgb_rows[:20]
    if not top:
        return
    rf_by_feature = {row["feature_name"]: float(row.get("mean_importance") or 0.0) for row in rf_rows}
    labels = [row["feature_name"] for row in top]
    xgb_values = [float(row.get("mean_importance") or 0.0) for row in top]
    rf_values = [rf_by_feature.get(label, 0.0) for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, rf_values, width=0.4, label="RandomForest")
    ax.bar(x + 0.2, xgb_values, width=0.4, label="XGBoost")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Mean importance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _class3(plt: Any, xgb_rows: list[dict[str, str]], controlled_rows: list[dict[str, str]], path: Path) -> None:
    rows = [
        row
        for row in [*xgb_rows, *controlled_rows]
        if row.get("class_id") == "3" and row.get("model_name") in MERGED_MODELS
    ]
    rows = sorted(rows, key=lambda row: float(row.get("mean_f1") or 0.0), reverse=True)
    if not rows:
        return
    labels = [row["model_name"] for row in rows]
    f1 = [float(row.get("mean_f1") or 0.0) for row in rows]
    recall = [float(row.get("mean_recall") or 0.0) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    ax.bar(x - 0.2, recall, width=0.4, label="Recall")
    ax.bar(x + 0.2, f1, width=0.4, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _subjectwise(plt: Any, xgb_rows: list[dict[str, str]], controlled_rows: list[dict[str, str]], path: Path) -> None:
    rows = [
        row
        for row in [*xgb_rows, *controlled_rows]
        if row.get("model_name") in MERGED_MODELS
    ]
    models = sorted({row["model_name"] for row in rows})
    subjects = sorted({int(row["test_subject"]) for row in rows}) if rows else []
    if not models or not subjects:
        return
    matrix = np.zeros((len(models), len(subjects)))
    for row in rows:
        matrix[models.index(row["model_name"]), subjects.index(int(row["test_subject"]))] = float(row.get("test_macro_f1") or 0.0)
    fig, ax = plt.subplots(figsize=(8, max(4, len(models) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([str(item) for item in subjects])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _controlled_classwise_for_merge(run_dir: Path) -> list[dict[str, str]]:
    project_root = run_dir.parents[2]
    return _read_csv(project_root / CONTROLLED_RESULT / "classwise_metrics_by_model.csv")


def _controlled_subjectwise_for_merge(run_dir: Path) -> list[dict[str, str]]:
    project_root = run_dir.parents[2]
    return _read_csv(project_root / CONTROLLED_RESULT / "subjectwise_metrics_by_model.csv")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
