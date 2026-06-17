from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from analysis.full_matrix_analysis import compute_paired_model_differences
from training.supervised_trainer import write_csv


LOCKED_RUN_DIR = Path("results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1")


def merge_full_extension_with_locked_matrix(*, project_root: Path, extension_run_dir: Path) -> dict[str, Any]:
    locked_dir = (project_root / LOCKED_RUN_DIR).resolve()
    output_dir = extension_run_dir / "merged_with_locked_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)

    locked_agg = [_with_result_type(row, "locked_3seed_full") for row in _read_csv(locked_dir / "aggregate_metrics_by_model.csv")]
    extension_agg = [_with_result_type(row, "literature_extension_3seed_full") for row in _read_csv(extension_run_dir / "aggregate_metrics_by_model.csv")]
    merged = locked_agg + extension_agg
    macro_ranking = sorted(merged, key=lambda row: float(row.get("mean_macro_f1") or 0.0), reverse=True)
    accuracy_ranking = sorted(merged, key=lambda row: float(row.get("mean_accuracy") or 0.0), reverse=True)
    for index, row in enumerate(macro_ranking, start=1):
        row["rank_by_macro_f1"] = index
    for index, row in enumerate(accuracy_ranking, start=1):
        row["rank_by_accuracy"] = index

    parameter_rows = [
        {
            "model_name": row["model_name"],
            "result_type": row["result_type"],
            "total_params": row.get("total_params", ""),
            "mean_macro_f1": row.get("mean_macro_f1", ""),
        }
        for row in macro_ranking
    ]
    primary = [
        row
        for row in macro_ranking
        if row["model_name"]
        in {
            "channel_shared_posres_attention_v3",
            "all_channel_conv1d_v1",
            "all_channel_conv1d_small",
            "feature_random_forest_v1",
            "rescnn_bigru_attention_lite_v1",
            "tcn_literature_v1",
            "lee_style_cnn_lstm_2d_v1",
            "cnn_lstm_literature_v1",
        }
    ]
    locked_fold = _read_csv(locked_dir / "fold_metrics.csv")
    extension_fold = _read_csv(extension_run_dir / "fold_metrics.csv")
    paired = compute_paired_model_differences(
        locked_fold + extension_fold,
        [
            "channel_shared_posres_attention_v3",
            "all_channel_conv1d_v1",
            "all_channel_conv1d_small",
            "feature_random_forest_v1",
            "rescnn_bigru_attention_lite_v1",
            "lee_style_cnn_lstm_2d_v1",
        ],
        metric="macro_f1",
        bootstrap_n=10000,
        seed=42,
    )

    write_csv(output_dir / "merged_full_comparison.csv", merged)
    write_csv(output_dir / "merged_macro_f1_ranking.csv", macro_ranking)
    write_csv(output_dir / "merged_accuracy_ranking.csv", accuracy_ranking)
    write_csv(output_dir / "merged_parameter_vs_macro_f1.csv", parameter_rows)
    write_csv(output_dir / "merged_primary_comparisons.csv", primary)
    write_csv(output_dir / "merged_paired_differences.csv", paired)
    (output_dir / "merged_claim_warning.md").write_text(_claim_warning(), encoding="utf-8")
    return {"output_dir": str(output_dir), "n_locked_models": len(locked_agg), "n_extension_models": len(extension_agg)}


def generate_literature_extension_figures(run_dir: Path) -> None:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        (figures_dir / "figures_unavailable.txt").write_text("matplotlib is not installed.\n", encoding="utf-8")
        return

    aggregate = _read_csv(run_dir / "aggregate_metrics_by_model.csv")
    classwise = _read_csv(run_dir / "classwise_metrics_by_model.csv")
    subjectwise = _read_csv(run_dir / "subjectwise_metrics_by_model.csv")
    rf_importance = _read_csv(run_dir / "random_forest_feature_importance_aggregate.csv")
    svm_coef = _read_csv(run_dir / "linear_svm_coefficients_aggregate.csv")
    merged = _read_csv(run_dir / "merged_with_locked_matrix" / "merged_full_comparison.csv")

    _bar_ci(plt, aggregate, "mean_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", figures_dir / "literature_extension_macro_f1_bar_ci.png", "Literature extension macro F1")
    _bar_ci(plt, aggregate, "mean_accuracy", None, None, figures_dir / "literature_extension_accuracy_bar_ci.png", "Literature extension accuracy")
    _bar_ci(plt, merged, "mean_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", figures_dir / "merged_macro_f1_bar_ci.png", "Merged macro F1")
    _scatter_params(plt, merged, figures_dir / "parameter_count_vs_macro_f1_merged.png")
    _class3(plt, classwise, figures_dir / "class3_recall_f1_literature_extension.png")
    _class_recall_heatmap(plt, classwise, figures_dir / "classwise_recall_heatmap_literature_extension.png")
    _subject_heatmap(plt, subjectwise, figures_dir / "subjectwise_macro_f1_heatmap_literature_extension.png")
    _top_feature_bar(plt, rf_importance, "mean_importance", figures_dir / "random_forest_top_features.png", "RandomForest top features")
    _top_feature_bar(plt, svm_coef, "mean_abs_coefficient", figures_dir / "svm_top_coefficients.png", "LinearSVM top coefficients")
    _two_model_bar(plt, aggregate, ["cnn_lstm_literature_v1", "lee_style_cnn_lstm_2d_v1"], figures_dir / "lee_style_vs_simple_cnn_lstm.png")
    _temporal_bar(plt, aggregate, figures_dir / "temporal_baseline_comparison.png")
    (run_dir / "figure_captions.md").write_text(_figure_captions(), encoding="utf-8")


def _with_result_type(row: dict[str, str], result_type: str) -> dict[str, Any]:
    payload = dict(row)
    payload["result_type"] = result_type
    return payload


def _claim_warning() -> str:
    return (
        "# Merged Claim Warning\n\n"
        "이 병합표는 locked supervised matrix와 literature full extension을 함께 정리한다. 두 결과 모두 3 seed full protocol이지만, "
        "모델 family와 실험 목적이 다르므로 최종 학술 해석은 사용자가 판단해야 한다. 기존 locked result 파일은 수정하지 않았다.\n"
    )


def _bar_ci(plt: Any, rows: list[dict[str, str]], metric: str, low_key: str | None, high_key: str | None, path: Path, title: str) -> None:
    rows = sorted(rows, key=lambda row: float(row.get(metric) or 0.0), reverse=True)
    names = [row["model_name"] for row in rows]
    values = np.asarray([float(row.get(metric) or 0.0) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.65), 4))
    if low_key and high_key:
        lows = np.asarray([float(row.get(low_key) or value) for row, value in zip(rows, values)], dtype=float)
        highs = np.asarray([float(row.get(high_key) or value) for row, value in zip(rows, values)], dtype=float)
        yerr = np.vstack([values - lows, highs - values])
        ax.bar(range(len(names)), values, yerr=yerr, capsize=3)
    else:
        ax.bar(range(len(names)), values)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _scatter_params(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        params = row.get("total_params")
        if params in ("", None):
            continue
        ax.scatter(float(params), float(row.get("mean_macro_f1") or 0.0))
        ax.text(float(params), float(row.get("mean_macro_f1") or 0.0), row["model_name"], fontsize=7)
    ax.set_xlabel("total parameters")
    ax.set_ylabel("mean macro F1")
    ax.set_title("Parameter count vs macro F1")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _class3(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    rows = [row for row in rows if str(row.get("class_id")) == "3"]
    names = [row["model_name"] for row in rows]
    recall = [float(row.get("mean_recall") or 0.0) for row in rows]
    f1 = [float(row.get("mean_f1") or 0.0) for row in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 4))
    ax.bar(x - 0.2, recall, width=0.4, label="recall")
    ax.bar(x + 0.2, f1, width=0.4, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_title("Class 3 Excessive Lean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _class_recall_heatmap(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    models = sorted({row["model_name"] for row in rows})
    classes = sorted({int(row["class_id"]) for row in rows})
    matrix = np.zeros((len(models), len(classes)), dtype=float)
    for row in rows:
        matrix[models.index(row["model_name"]), classes.index(int(row["class_id"]))] = float(row.get("mean_recall") or 0.0)
    _heatmap(plt, matrix, models, [str(item) for item in classes], path, "Class-wise recall")


def _subject_heatmap(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    models = sorted({row["model_name"] for row in rows})
    subjects = sorted({int(row["test_subject"]) for row in rows})
    matrix = np.zeros((len(models), len(subjects)), dtype=float)
    for row in rows:
        matrix[models.index(row["model_name"]), subjects.index(int(row["test_subject"]))] = float(row.get("test_macro_f1") or 0.0)
    _heatmap(plt, matrix, models, [str(item) for item in subjects], path, "Subject-wise macro F1")


def _heatmap(plt: Any, matrix: np.ndarray, ylabels: list[str], xlabels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(xlabels)), max(4, len(ylabels) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _top_feature_bar(plt: Any, rows: list[dict[str, str]], value_key: str, path: Path, title: str) -> None:
    rows = sorted(rows, key=lambda row: float(row.get(value_key) or 0.0), reverse=True)[:20]
    names = [row["feature_name"] for row in rows]
    values = [float(row.get(value_key) or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(names)), values)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _two_model_bar(plt: Any, rows: list[dict[str, str]], names: list[str], path: Path) -> None:
    selected = [row for row in rows if row["model_name"] in names]
    _bar_ci(plt, selected, "mean_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", path, "Simple CNN-LSTM vs Lee-style CNN-LSTM")


def _temporal_bar(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    temporal = [
        row
        for row in rows
        if row["model_name"]
        in {
            "rescnn_bigru_attention_lite_v1",
            "tcn_literature_v1",
            "transformer_encoder_lite_v1",
            "cnn_lstm_literature_v1",
            "cnn_gru_literature_v1",
            "lee_style_cnn_lstm_2d_v1",
        }
    ]
    _bar_ci(plt, temporal, "mean_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", path, "Temporal baseline comparison")


def _figure_captions() -> str:
    return """# Literature Full Extension Figure Captions

- `literature_extension_macro_f1_bar_ci.png`: Literature extension macro F1 with bootstrap confidence intervals.
- `literature_extension_accuracy_bar_ci.png`: Literature extension accuracy.
- `merged_macro_f1_bar_ci.png`: Locked matrix and literature extension macro F1 summary.
- `parameter_count_vs_macro_f1_merged.png`: Parameter count and macro F1 for neural models.
- `class3_recall_f1_literature_extension.png`: Class 3 Excessive Lean recall and F1.
- `classwise_recall_heatmap_literature_extension.png`: Class-wise recall heatmap.
- `subjectwise_macro_f1_heatmap_literature_extension.png`: Subject-wise macro F1 heatmap.
- `random_forest_top_features.png`: Top RandomForest signal-derived features.
- `svm_top_coefficients.png`: Top LinearSVM signal-derived coefficients by absolute value.
- `lee_style_vs_simple_cnn_lstm.png`: Lee-style CNN-LSTM compared with simple CNN-LSTM.
- `temporal_baseline_comparison.png`: Neural temporal baseline macro F1 comparison.
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
