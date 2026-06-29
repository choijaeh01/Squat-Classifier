from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from training.supervised_trainer import write_csv


LOCKED_FULL_MATRIX = Path("results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1")
LITERATURE_FULL_EXTENSION = Path("results/literature_baseline_full_extension/20260617_183952_literature_baseline_full_extension_v1")
V3_COMPONENT_ABLATION = Path("results/v3_component_ablation/20260617_202714_v3_component_ablation_v1")


CONTROLLED_GROUPS = {
    "controlled_neural": {
        "controlled_flatten_mlp",
        "controlled_stats_mlp",
        "controlled_all_channel_1d_cnn",
        "controlled_all_channel_1d_cnn_small",
        "controlled_shared_1d",
        "controlled_shared_1d_identity",
        "controlled_shared_1d_residual",
        "controlled_shared_1d_residual_identity",
        "controlled_2d_cnn",
    },
    "classical_practical": {"feature_random_forest_v1", "feature_xgboost_v1", "feature_linear_svm_v1"},
    "literature_reference": {"rescnn_bigru_attention_lite_v1", "lee_style_cnn_lstm_2d_v1"},
}


def write_controlled_analysis_outputs(*, project_root: Path, run_dir: Path) -> None:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _read_csv(run_dir / "aggregate_metrics_by_model.csv")
    paired = _read_csv(run_dir / "paired_model_differences.csv")
    classwise = _read_csv(run_dir / "classwise_metrics_by_model.csv")
    subjectwise = _read_csv(run_dir / "subjectwise_metrics_by_model.csv")
    params = _read_csv(run_dir / "model_parameter_counts.csv")

    write_csv(analysis_dir / "controlled_architecture_comparison.csv", _filter_group(aggregate, "controlled_neural"))
    write_csv(analysis_dir / "practical_baseline_comparison.csv", _filter_group(aggregate, "classical_practical"))
    write_csv(analysis_dir / "literature_reference_comparison.csv", _filter_group(aggregate, "literature_reference"))
    write_csv(analysis_dir / "component_contribution_summary.csv", _component_summary(aggregate, paired))
    write_csv(analysis_dir / "common_head_verification.csv", _common_head_rows(params))
    write_csv(analysis_dir / "merged_readonly_reference_comparison.csv", _merged_reference_rows(project_root, aggregate))
    (analysis_dir / "claim_implication_memo_for_user.md").write_text(_claim_memo(aggregate, classwise, subjectwise), encoding="utf-8")
    generate_controlled_figures(run_dir)


def generate_controlled_figures(run_dir: Path) -> None:
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
    params = _read_csv(run_dir / "model_parameter_counts.csv")
    rf_importance = _read_csv(run_dir / "random_forest_feature_importance_aggregate.csv")
    xgb_importance = _read_csv(run_dir / "xgboost_feature_importance_aggregate.csv")
    component = _read_csv(run_dir / "analysis" / "component_contribution_summary.csv")

    _bar_ci(plt, _filter_group(aggregate, "controlled_neural"), "mean_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", figures_dir / "controlled_neural_macro_f1_bar_ci.png", "Controlled neural macro F1")
    _bar_ci(plt, _filter_group(aggregate, "controlled_neural"), "mean_accuracy", None, None, figures_dir / "controlled_neural_accuracy_bar_ci.png", "Controlled neural accuracy")
    _component_delta(plt, component, figures_dir / "component_delta_macro_f1.png")
    _common_head_bar(plt, params, figures_dir / "common_head_parameter_comparison.png")
    _bar_ci(plt, _filter_group(aggregate, "classical_practical"), "mean_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", figures_dir / "practical_baseline_macro_f1_bar_ci.png", "Practical baseline macro F1")
    _bar_ci(plt, _filter_group(aggregate, "literature_reference"), "mean_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", figures_dir / "literature_reference_macro_f1_bar_ci.png", "Literature reference macro F1")
    if xgb_importance:
        _dual_feature_importance(plt, rf_importance, xgb_importance, figures_dir / "xgboost_rf_feature_importance.png")
    _subject_heatmap(plt, subjectwise, figures_dir / "subjectwise_macro_f1_heatmap_controlled.png")
    _class_recall_heatmap(plt, classwise, figures_dir / "classwise_recall_heatmap_controlled.png")
    _class3(plt, classwise, figures_dir / "class3_f1_recall_controlled.png")
    _param_scatter(plt, aggregate, figures_dir / "parameter_count_vs_macro_f1_controlled.png")
    (run_dir / "figure_captions.md").write_text(_figure_captions(), encoding="utf-8")


def _filter_group(rows: list[dict[str, str]], group: str) -> list[dict[str, str]]:
    names = CONTROLLED_GROUPS[group]
    return [dict(row, model_group=group) for row in rows if row.get("model_name") in names]


def _component_summary(aggregate: list[dict[str, str]], paired: list[dict[str, str]]) -> list[dict[str, Any]]:
    comparisons = [
        ("controlled_shared_1d", "controlled_shared_1d_residual"),
        ("controlled_shared_1d_residual", "controlled_shared_1d_residual_identity"),
        ("controlled_shared_1d_identity", "controlled_shared_1d_residual_identity"),
        ("controlled_all_channel_1d_cnn", "controlled_shared_1d_residual_identity"),
        ("controlled_all_channel_1d_cnn_small", "controlled_shared_1d_residual_identity"),
        ("controlled_stats_mlp", "controlled_shared_1d_residual_identity"),
        ("controlled_2d_cnn", "controlled_all_channel_1d_cnn"),
    ]
    by_model = {row["model_name"]: row for row in aggregate}
    output = []
    for reference, comparison in comparisons:
        ref = by_model.get(reference, {})
        cmp = by_model.get(comparison, {})
        paired_row = _paired_lookup(paired, reference, comparison)
        output.append(
            {
                "reference_model": reference,
                "comparison_model": comparison,
                "reference_macro_f1": ref.get("mean_macro_f1", ""),
                "comparison_macro_f1": cmp.get("mean_macro_f1", ""),
                "delta_comparison_minus_reference": _delta(cmp.get("mean_macro_f1"), ref.get("mean_macro_f1")),
                "paired_delta": paired_row.get("mean_difference", ""),
                "paired_ci_low": paired_row.get("ci_low", ""),
                "paired_ci_high": paired_row.get("ci_high", ""),
                "n_pairs": paired_row.get("n_pairs", ""),
            }
        )
    return output


def _paired_lookup(rows: list[dict[str, str]], reference: str, comparison: str) -> dict[str, str]:
    for row in rows:
        if row.get("reference_model") == reference and row.get("comparison_model") == comparison:
            return row
    return {}


def _common_head_rows(params: list[dict[str, str]]) -> list[dict[str, Any]]:
    controlled = [row for row in params if row.get("model_group") == "controlled_neural"]
    head_counts = {row.get("common_head_params") for row in controlled}
    signatures = {row.get("common_head_signature") for row in controlled}
    return [
        {
            "model_name": row.get("model_name"),
            "representation_dim": row.get("representation_dim"),
            "common_head_params": row.get("common_head_params"),
            "common_head_signature": row.get("common_head_signature"),
            "common_head_param_count_shared": len(head_counts) == 1,
            "common_head_signature_shared": len(signatures) == 1,
        }
        for row in controlled
    ]


def _merged_reference_rows(project_root: Path, controlled_aggregate: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in controlled_aggregate:
        payload = dict(row)
        payload["result_type"] = "controlled_feature_extractor_3seed_full"
        rows.append(payload)
    for result_type, rel_dir in [
        ("locked_full_supervised_matrix_readonly", LOCKED_FULL_MATRIX),
        ("literature_full_extension_readonly", LITERATURE_FULL_EXTENSION),
        ("v3_component_ablation_readonly", V3_COMPONENT_ABLATION),
    ]:
        for row in _read_csv(project_root / rel_dir / "aggregate_metrics_by_model.csv"):
            payload = dict(row)
            payload["result_type"] = result_type
            rows.append(payload)
    return sorted(rows, key=lambda row: float(row.get("mean_macro_f1") or 0.0), reverse=True)


def _claim_memo(aggregate: list[dict[str, str]], classwise: list[dict[str, str]], subjectwise: list[dict[str, str]]) -> str:
    ranking = sorted(aggregate, key=lambda row: float(row.get("mean_macro_f1") or 0.0), reverse=True)
    top_lines = [f"- {row['model_name']}: Macro F1 {row.get('mean_macro_f1')}, Accuracy {row.get('mean_accuracy')}" for row in ranking[:10]]
    class3 = [row for row in classwise if str(row.get("class_id")) == "3"]
    class3_lines = [f"- {row['model_name']}: recall {row.get('mean_recall')}, F1 {row.get('mean_f1')}" for row in sorted(class3, key=lambda row: float(row.get("mean_f1") or 0.0), reverse=True)[:10]]
    return (
        "# Controlled Feature Extractor Claim Implication Memo\n\n"
        "이 메모는 수치 중심 관찰만 정리한다. 최종 학술 해석과 main model 판단은 사용자가 수행한다.\n\n"
        "## Macro F1 상위 모델\n\n"
        + "\n".join(top_lines)
        + "\n\n## Class 3 Excessive Lean 상위 모델\n\n"
        + "\n".join(class3_lines)
        + f"\n\n## subjectwise row count\n\n- rows: {len(subjectwise)}\n"
    )


def _delta(a: Any, b: Any) -> float | str:
    if a in ("", None) or b in ("", None):
        return ""
    return float(a) - float(b)


def _bar_ci(plt: Any, rows: list[dict[str, str]], metric: str, low_key: str | None, high_key: str | None, path: Path, title: str) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row.get(metric) or 0.0), reverse=True)
    labels = [row["model_name"] for row in rows]
    values = np.asarray([float(row.get(metric) or 0.0) for row in rows])
    yerr = None
    if low_key and high_key:
        lows = np.asarray([float(row.get(low_key) or value) for row, value in zip(rows, values)])
        highs = np.asarray([float(row.get(high_key) or value) for row, value in zip(rows, values)])
        yerr = np.vstack([values - lows, highs - values])
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), 4))
    ax.bar(range(len(labels)), values, yerr=yerr, capsize=3)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _component_delta(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    labels = [f"{row['comparison_model']} - {row['reference_model']}" for row in rows]
    values = [float(row.get("delta_comparison_minus_reference") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    ax.bar(range(len(labels)), values)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Component delta macro F1")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _common_head_bar(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    rows = [row for row in rows if row.get("model_group") == "controlled_neural"]
    if not rows:
        return
    labels = [row["model_name"] for row in rows]
    values = [float(row.get("common_head_params") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4))
    ax.bar(range(len(labels)), values)
    ax.set_title("Common head parameter count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _dual_feature_importance(plt: Any, rf_rows: list[dict[str, str]], xgb_rows: list[dict[str, str]], path: Path) -> None:
    top = sorted(xgb_rows, key=lambda row: float(row.get("mean_importance") or 0.0), reverse=True)[:15]
    labels = [row["feature_name"] for row in top]
    xgb_values = [float(row.get("mean_importance") or 0.0) for row in top]
    rf_by_name = {row["feature_name"]: float(row.get("mean_importance") or 0.0) for row in rf_rows}
    rf_values = [rf_by_name.get(label, 0.0) for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, rf_values, width=0.4, label="RF")
    ax.bar(x + 0.2, xgb_values, width=0.4, label="XGBoost")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _subject_heatmap(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    models = sorted({row["model_name"] for row in rows})
    subjects = sorted({int(row["test_subject"]) for row in rows}) if rows else []
    matrix = np.zeros((len(models), len(subjects)))
    for row in rows:
        matrix[models.index(row["model_name"]), subjects.index(int(row["test_subject"]))] = float(row.get("test_macro_f1") or 0.0)
    _heatmap(plt, matrix, models, [str(item) for item in subjects], path, "Subject-wise macro F1")


def _class_recall_heatmap(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    models = sorted({row["model_name"] for row in rows})
    classes = sorted({int(row["class_id"]) for row in rows}) if rows else []
    matrix = np.zeros((len(models), len(classes)))
    for row in rows:
        matrix[models.index(row["model_name"]), classes.index(int(row["class_id"]))] = float(row.get("mean_recall") or 0.0)
    _heatmap(plt, matrix, models, [str(item) for item in classes], path, "Class-wise recall")


def _class3(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    rows = [row for row in rows if str(row.get("class_id")) == "3"]
    _bar_ci(plt, rows, "mean_f1", None, None, path, "Class 3 F1")


def _param_scatter(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        params = row.get("total_params")
        if params in ("", None):
            continue
        ax.scatter(float(params), float(row.get("mean_macro_f1") or 0.0))
        ax.text(float(params), float(row.get("mean_macro_f1") or 0.0), row["model_name"], fontsize=7)
    ax.set_xlabel("total parameters")
    ax.set_ylabel("mean macro F1")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _heatmap(plt: Any, matrix: np.ndarray, ylabels: list[str], xlabels: list[str], path: Path, title: str) -> None:
    if matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(xlabels)), max(4, len(ylabels) * 0.35)))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _figure_captions() -> str:
    return """# Controlled Feature Extractor Figure Captions

- `controlled_neural_macro_f1_bar_ci.png`: Controlled neural feature extractor macro F1 with bootstrap CI.
- `controlled_neural_accuracy_bar_ci.png`: Controlled neural feature extractor accuracy.
- `component_delta_macro_f1.png`: Macro F1 delta for requested component comparisons.
- `common_head_parameter_comparison.png`: Verification that common classifier head parameter count is fixed.
- `practical_baseline_macro_f1_bar_ci.png`: RF, XGBoost, SVM macro F1.
- `literature_reference_macro_f1_bar_ci.png`: Literature reference macro F1.
- `xgboost_rf_feature_importance.png`: XGBoost and RF top signal-derived feature importance, if XGBoost ran.
- `subjectwise_macro_f1_heatmap_controlled.png`: Subject-wise macro F1 heatmap.
- `classwise_recall_heatmap_controlled.png`: Class-wise recall heatmap.
- `class3_f1_recall_controlled.png`: Class 3 Excessive Lean F1 summary.
- `parameter_count_vs_macro_f1_controlled.png`: Parameter count versus macro F1.
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
