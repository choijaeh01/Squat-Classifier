from __future__ import annotations

import csv
import json
import math
import textwrap
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


CONTROLLED_DIR = Path("results/controlled_feature_extractor_comparison/20260629_041712_controlled_feature_extractor_comparison_v1")
XGBOOST_DIR = Path("results/xgboost_only_completion/20260629_053235_xgboost_only_completion_v1")
LITERATURE_DIR = Path("results/literature_baseline_full_extension/20260617_183952_literature_baseline_full_extension_v1")
V3_ABLATION_DIR = Path("results/v3_component_ablation/20260617_202714_v3_component_ablation_v1")
FULL_MATRIX_DIR = Path("results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1")

CLASS_NAMES = {
    0: "Correct",
    1: "Knee Valgus",
    2: "Butt Wink",
    3: "Excessive Lean",
    4: "Partial Squat",
}

CONTROLLED_ORDER = [
    "controlled_stats_mlp",
    "controlled_shared_1d_residual",
    "controlled_all_channel_1d_cnn",
    "controlled_shared_1d_residual_identity",
    "feature_random_forest_v1",
    "rescnn_bigru_attention_lite_v1",
    "controlled_all_channel_1d_cnn_small",
    "feature_linear_svm_v1",
    "controlled_flatten_mlp",
    "lee_style_cnn_lstm_2d_v1",
    "controlled_shared_1d_identity",
    "controlled_2d_cnn",
    "controlled_shared_1d",
]

FLOW_MODELS = [
    "controlled_shared_1d",
    "controlled_shared_1d_identity",
    "controlled_shared_1d_residual",
    "controlled_shared_1d_residual_identity",
]

MAIN_CONFUSION_MODELS = [
    "controlled_stats_mlp",
    "controlled_shared_1d_residual",
    "controlled_all_channel_1d_cnn",
    "feature_xgboost_v1",
    "feature_random_forest_v1",
    "controlled_shared_1d",
]


@dataclass(frozen=True)
class ProfessorReportV2Paths:
    project_root: Path
    output_docs: Path
    output_obsidian: Path | None
    mapping_path: Path

    @property
    def tables(self) -> Path:
        return self.output_docs / "tables"

    @property
    def figures(self) -> Path:
        return self.output_docs / "figures"

    @property
    def diagrams(self) -> Path:
        return self.output_docs / "diagrams"

    @property
    def assets(self) -> Path:
        return self.output_docs / "assets"


def check_professor_report_v2_inputs(project_root: Path) -> dict[str, Any]:
    required = [
        CONTROLLED_DIR / "aggregate_metrics_by_model.csv",
        CONTROLLED_DIR / "model_parameter_counts.csv",
        CONTROLLED_DIR / "common_head_verification.csv",
        CONTROLLED_DIR / "classwise_metrics_by_model.csv",
        CONTROLLED_DIR / "confusion_matrices.csv",
        CONTROLLED_DIR / "random_forest_feature_importance_aggregate.csv",
        CONTROLLED_DIR / "feature_definitions.csv",
        CONTROLLED_DIR / "feature_audit.csv",
        CONTROLLED_DIR / "scaler_fit_audit.csv",
        XGBOOST_DIR / "aggregate_metrics_by_model.csv",
        XGBOOST_DIR / "confusion_matrices.csv",
        XGBOOST_DIR / "xgboost_feature_importance_aggregate.csv",
        XGBOOST_DIR / "scaler_fit_audit.csv",
        Path("docs/professor_report_v2/display_name_mapping.yaml"),
        Path("docs/professor_report_v2/tables/table_architecture_audit.csv"),
        Path("docs/professor_report_v2/tables/table_normalization_protocol.csv"),
        Path("docs/professor_report_v2/tables/table_feature_extraction_order.csv"),
    ]
    status = {str(path): (project_root / path).exists() for path in required}
    return {"all_present": all(status.values()), "files": status}


def build_professor_report_v2_assets(paths: ProfessorReportV2Paths) -> dict[str, Any]:
    for folder in (paths.output_docs, paths.tables, paths.figures, paths.diagrams, paths.assets):
        folder.mkdir(parents=True, exist_ok=True)
    mapping = load_display_mapping(paths.mapping_path)
    sources = load_sources(paths.project_root)

    tables = build_tables(paths, sources, mapping)
    figures = build_figures(paths, sources, mapping)
    diagrams = build_detailed_diagrams(paths, sources, mapping)
    captions = write_captions(paths)
    report = write_main_report(paths, sources, mapping)
    index = write_figure_table_index(paths)
    readme = write_readme(paths)
    validation = validate_report_v2(paths)
    validation_path = paths.assets / "professor_report_v2_validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_docs": str(paths.output_docs),
        "output_obsidian": str(paths.output_obsidian) if paths.output_obsidian else "",
        "tables": [str(item) for item in tables],
        "figures": [str(item) for item in figures],
        "diagrams": [str(item) for item in diagrams],
        "captions": str(captions),
        "main_report": str(report),
        "figure_table_index": str(index),
        "readme": str(readme),
        "validation_path": str(validation_path),
        "validation": validation,
    }


def load_sources(project_root: Path) -> dict[str, list[dict[str, str]]]:
    def read(rel: Path) -> list[dict[str, str]]:
        return read_csv(project_root / rel)

    return {
        "controlled_aggregate": read(CONTROLLED_DIR / "aggregate_metrics_by_model.csv"),
        "controlled_params": read(CONTROLLED_DIR / "model_parameter_counts.csv"),
        "controlled_common_head": read(CONTROLLED_DIR / "common_head_verification.csv"),
        "controlled_classwise": read(CONTROLLED_DIR / "classwise_metrics_by_model.csv"),
        "controlled_confusion": read(CONTROLLED_DIR / "confusion_matrices.csv"),
        "rf_importance": read(CONTROLLED_DIR / "random_forest_feature_importance_aggregate.csv"),
        "feature_definitions": read(CONTROLLED_DIR / "feature_definitions.csv"),
        "feature_audit": read(CONTROLLED_DIR / "feature_audit.csv"),
        "xgb_aggregate": read(XGBOOST_DIR / "aggregate_metrics_by_model.csv"),
        "xgb_classwise": read(XGBOOST_DIR / "classwise_metrics_by_model.csv"),
        "xgb_confusion": read(XGBOOST_DIR / "confusion_matrices.csv"),
        "xgb_importance": read(XGBOOST_DIR / "xgboost_feature_importance_aggregate.csv"),
        "architecture": read(Path("docs/professor_report_v2/tables/table_architecture_audit.csv")),
        "arch_params": read(Path("docs/professor_report_v2/tables/table_parameter_count_clean.csv")),
        "arch_common_head": read(Path("docs/professor_report_v2/tables/table_common_head_verification.csv")),
        "residual_features": read(Path("docs/professor_report_v2/tables/table_residual_feature_list.csv")),
        "normalization": read(Path("docs/professor_report_v2/tables/table_normalization_protocol.csv")),
        "scaler_summary": read(Path("docs/professor_report_v2/tables/table_scaler_leakage_summary.csv")),
        "feature_order": read(Path("docs/professor_report_v2/tables/table_feature_extraction_order.csv")),
    }


def build_tables(paths: ProfessorReportV2Paths, sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[Path]:
    outputs: list[Path] = []
    outputs.append(write_csv(paths.tables / "table_01_dataset_summary.csv", dataset_summary_rows()))
    outputs.append(write_csv(paths.tables / "table_02_professor_feedback_status.csv", professor_feedback_rows()))
    controlled_rows = controlled_table_rows(sources["controlled_aggregate"], mapping)
    outputs.append(write_csv(paths.tables / "table_03_feature_extractor_clean_names.csv", controlled_rows))
    outputs.append(write_csv(paths.tables / "table_04_xgboost_completion_clean_names.csv", xgb_table_rows(sources, mapping)))
    outputs.append(write_csv(paths.tables / "table_05_component_flow.csv", component_flow_rows(sources["controlled_aggregate"], mapping)))
    outputs.append(write_csv(paths.tables / "table_06_class3_summary.csv", class3_rows(sources, mapping)))
    outputs.append(write_csv(paths.tables / "table_07_feature_importance_top10.csv", feature_importance_top10(sources)))
    outputs.append(write_csv(paths.tables / "table_08_claim_audit_summary.csv", claim_audit_rows()))
    outputs.append(write_csv(paths.tables / "table_09_final_recommended_scope.csv", recommended_scope_rows()))
    outputs.append(write_csv(paths.tables / "table_parameter_count_main.csv", parameter_count_main_rows(sources, mapping)))
    outputs.append(write_csv(paths.tables / "table_confusion_matrix_summary.csv", confusion_summary_rows(sources, mapping)))
    outputs.append(write_csv(paths.tables / "table_internal_name_mapping.csv", internal_mapping_rows(mapping)))
    return outputs


def dataset_summary_rows() -> list[dict[str, str]]:
    return [
        {"item": "Sensors", "value": "3 MPU-6050 IMUs", "note": "lower back/waist, right thigh, right calf"},
        {"item": "Channels", "value": "18", "note": "3-axis accelerometer + 3-axis gyroscope per IMU"},
        {"item": "Window length", "value": "512", "note": "phase-normalized linear interpolation during conversion"},
        {"item": "Subjects", "value": "6", "note": "subject-independent LOSO"},
        {"item": "Classes", "value": "5", "note": "Correct, Knee Valgus, Butt Wink, Excessive Lean, Partial Squat"},
        {"item": "Windows", "value": "600", "note": "20 windows per subject-class combination"},
        {"item": "Validation policy", "value": "within-train stratified LOSO", "note": "train 400 / val 100 / test 100 per fold"},
        {"item": "Scaler", "value": "train-only StandardScaler", "note": "fit on train indices only, then transform train/val/test"},
    ]


def professor_feedback_rows() -> list[dict[str, str]]:
    return [
        {"feedback_item": "XGBoost comparison", "status": "done", "evidence": "XGBoost completion results", "note": "No tuning or rerun in v2."},
        {"feedback_item": "RF/XGBoost feature importance", "status": "done", "evidence": "top-10 feature importance table", "note": "Signal-derived features only."},
        {"feedback_item": "normalization/scaler clarification", "status": "done", "evidence": "normalization protocol audit", "note": "Feature extraction happens after train-only scaler transform."},
        {"feedback_item": "same classifier head comparison", "status": "done", "evidence": "common head verification table", "note": "64-dim representation and 4,485-param head."},
        {"feedback_item": "1D / residual / 2D / MLP comparison", "status": "done", "evidence": "feature extractor comparison table", "note": "Display names used for report."},
        {"feedback_item": "confusion matrix analysis", "status": "done", "evidence": "row-normalized and count confusion figures", "note": "Interpretation limited to visible patterns."},
    ]


def controlled_table_rows(rows: list[dict[str, str]], mapping: dict[str, str]) -> list[dict[str, str]]:
    by_name = {row["model_name"]: row for row in rows}
    output = []
    for model_name in CONTROLLED_ORDER:
        row = by_name.get(model_name)
        if row is None:
            continue
        output.append(
            {
                "model_display_name": display(model_name, mapping),
                "group": model_group(model_name),
                "accuracy": fmt(row.get("mean_accuracy")),
                "macro_f1": fmt(row.get("mean_macro_f1")),
                "weighted_f1": fmt(row.get("mean_weighted_f1")),
                "macro_f1_ci": f"[{fmt(row.get('macro_f1_ci_low'))}, {fmt(row.get('macro_f1_ci_high'))}]",
                "role": model_role(model_name),
                "interpretation": model_interpretation(model_name),
            }
        )
    return output


def xgb_table_rows(sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[dict[str, str]]:
    controlled = {row["model_name"]: row for row in sources["controlled_aggregate"]}
    xgb = {row["model_name"]: row for row in sources["xgb_aggregate"]}
    order = ["controlled_stats_mlp", "controlled_shared_1d_residual", "controlled_all_channel_1d_cnn", "feature_xgboost_v1", "feature_random_forest_v1"]
    rows = []
    for model_name in order:
        row = xgb.get(model_name) or controlled.get(model_name)
        if not row:
            continue
        rows.append(
            {
                "model_display_name": display(model_name, mapping),
                "accuracy": fmt(row.get("mean_accuracy")),
                "macro_f1": fmt(row.get("mean_macro_f1")),
                "macro_f1_ci": f"[{fmt(row.get('macro_f1_ci_low'))}, {fmt(row.get('macro_f1_ci_high'))}]",
                "role": model_role(model_name),
            }
        )
    return rows


def component_flow_rows(rows: list[dict[str, str]], mapping: dict[str, str]) -> list[dict[str, str]]:
    by_name = {row["model_name"]: row for row in rows}
    baseline = fnum(by_name["controlled_shared_1d"]["mean_macro_f1"])
    meaning = {
        "controlled_shared_1d": "공유 encoder만 사용하면 채널 위치와 통계 단서가 약해진다.",
        "controlled_shared_1d_identity": "identity embedding만으로는 병목 완화가 제한적이다.",
        "controlled_shared_1d_residual": "signal-derived residual branch가 병목을 크게 완화한다.",
        "controlled_shared_1d_residual_identity": "residual 위에 identity를 추가해도 추가 이득은 작았다.",
    }
    return [
        {
            "model_display_name": display(model_name, mapping),
            "macro_f1": fmt(by_name[model_name]["mean_macro_f1"]),
            "delta_from_shared_1d": signed(fnum(by_name[model_name]["mean_macro_f1"]) - baseline),
            "meaning": meaning[model_name],
        }
        for model_name in FLOW_MODELS
    ]


def class3_rows(sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[dict[str, str]]:
    rows = [*sources["controlled_classwise"], *sources["xgb_classwise"]]
    keep = ["controlled_stats_mlp", "controlled_shared_1d_residual", "controlled_all_channel_1d_cnn", "feature_xgboost_v1", "feature_random_forest_v1", "controlled_shared_1d"]
    output = []
    seen = set()
    for model_name in keep:
        for row in rows:
            if row.get("model_name") == model_name and str(row.get("class_id")) == "3" and model_name not in seen:
                seen.add(model_name)
                output.append(
                    {
                        "model_display_name": display(model_name, mapping),
                        "class3_recall": fmt(row.get("mean_recall") or row.get("recall")),
                        "class3_f1": fmt(row.get("mean_f1") or row.get("f1")),
                        "interpretation": class3_interpretation(model_name),
                    }
                )
                break
    return output


def feature_importance_top10(sources: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rf = sorted(sources["rf_importance"], key=lambda row: fnum(row.get("mean_importance")), reverse=True)[:10]
    xgb = sorted(sources["xgb_importance"], key=lambda row: fnum(row.get("mean_importance")), reverse=True)[:10]
    output = []
    for idx in range(max(len(rf), len(xgb))):
        rf_row = rf[idx] if idx < len(rf) else {}
        xgb_row = xgb[idx] if idx < len(xgb) else {}
        note = "s1_ax/s1_gx 관련 feature가 반복 관찰되지만 인과 단정은 금지" if idx == 0 else "model-specific importance"
        output.append(
            {
                "rank": str(idx + 1),
                "random_forest_feature": rf_row.get("feature_name", ""),
                "rf_importance": fmt(rf_row.get("mean_importance")),
                "xgboost_feature": xgb_row.get("feature_name", ""),
                "xgb_importance": fmt(xgb_row.get("mean_importance")),
                "note": note,
            }
        )
    return output


def parameter_count_main_rows(sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[dict[str, str]]:
    metrics = {row["model_name"]: row for row in sources["controlled_aggregate"]}
    metrics.update({row["model_name"]: row for row in sources["xgb_aggregate"]})
    params = {row["model_internal_name"]: row for row in sources["arch_params"]}
    order = [
        "controlled_stats_mlp",
        "controlled_shared_1d_residual",
        "controlled_all_channel_1d_cnn",
        "controlled_shared_1d_residual_identity",
        "controlled_all_channel_1d_cnn_small",
        "controlled_flatten_mlp",
        "controlled_shared_1d_identity",
        "controlled_2d_cnn",
        "controlled_shared_1d",
        "feature_random_forest_v1",
        "feature_xgboost_v1",
        "feature_linear_svm_v1",
    ]
    rows = []
    for model_name in order:
        metric = metrics.get(model_name, {})
        param = params.get(model_name, {})
        if model_name.startswith("feature_"):
            total = extractor = head = residual = identity = "not_applicable"
            representation_dim = "not_applicable"
            interpretation = "classical/tree model; neural parameter count와 직접 비교하지 않음."
        else:
            total = param.get("total_params", "")
            extractor = param.get("extractor_params", "")
            head = param.get("common_head_params", "")
            residual = param.get("residual_branch_params", "")
            identity = param.get("identity_embedding_params", "")
            representation_dim = param.get("representation_dim", "")
            interpretation = parameter_interpretation(model_name)
        rows.append(
            {
                "model_display_name": display(model_name, mapping),
                "group": model_group(model_name),
                "total_params": total,
                "extractor_params": extractor,
                "common_head_params": head,
                "residual_branch_params": residual,
                "identity_embedding_params": identity,
                "representation_dim": representation_dim,
                "macro_f1": fmt(metric.get("mean_macro_f1")),
                "interpretation": interpretation,
            }
        )
    return rows


def confusion_summary_rows(sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[dict[str, str]]:
    matrices = build_confusion_matrices(sources)
    rows = []
    for model_name, matrix in matrices.items():
        for class_id in range(5):
            true_count = sum(matrix[class_id])
            correct = matrix[class_id][class_id]
            recall = correct / true_count if true_count else 0.0
            confusions = sorted(
                [(pred, matrix[class_id][pred]) for pred in range(5) if pred != class_id],
                key=lambda item: item[1],
                reverse=True,
            )
            rows.append(
                {
                    "model_display_name": display(model_name, mapping),
                    "class_id": str(class_id),
                    "class_name": CLASS_NAMES[class_id],
                    "true_count": str(true_count),
                    "correct_count": str(correct),
                    "recall": f"{recall:.4f}",
                    "major_confusion_1": confusion_label(confusions[0]) if confusions else "",
                    "major_confusion_2": confusion_label(confusions[1]) if len(confusions) > 1 else "",
                    "note": confusion_note(model_name, class_id),
                }
            )
    return rows


def build_figures(paths: ProfessorReportV2Paths, sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[Path]:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    plt.rcParams["axes.unicode_minus"] = False
    outputs = [
        draw_macro_f1_bar(plt, read_csv(paths.tables / "table_03_feature_extractor_clean_names.csv"), paths.figures / "fig_01_feature_extractor_macro_f1.png"),
        draw_component_flow(plt, read_csv(paths.tables / "table_05_component_flow.csv"), paths.figures / "fig_02_residual_effect_flow.png"),
        draw_practical_bar(plt, read_csv(paths.tables / "table_04_xgboost_completion_clean_names.csv"), paths.figures / "fig_03_xgboost_practical_baselines.png"),
        draw_class3(plt, read_csv(paths.tables / "table_06_class3_summary.csv"), paths.figures / "fig_04_class3_excessive_lean.png"),
        draw_importance(plt, read_csv(paths.tables / "table_07_feature_importance_top10.csv"), paths.figures / "fig_05_rf_xgboost_feature_importance.png"),
        draw_parameter_scatter(plt, read_csv(paths.tables / "table_parameter_count_main.csv"), paths.figures / "fig_parameter_count_vs_macro_f1.png"),
    ]
    matrices = build_confusion_matrices(sources)
    confusion_files = {
        "controlled_shared_1d_residual": "fig_confusion_matrix_residual_shared.png",
        "controlled_stats_mlp": "fig_confusion_matrix_stats_mlp.png",
        "controlled_all_channel_1d_cnn": "fig_confusion_matrix_all_channel_1d.png",
        "feature_xgboost_v1": "fig_confusion_matrix_xgboost.png",
        "feature_random_forest_v1": "fig_confusion_matrix_random_forest.png",
        "controlled_shared_1d": "fig_confusion_matrix_shared_1d_failure.png",
    }
    for model_name, filename in confusion_files.items():
        outputs.append(draw_confusion_single(plt, matrices[model_name], display(model_name, mapping), paths.figures / filename, normalize=False))
    outputs.append(draw_confusion_grid(plt, matrices, mapping, paths.figures / "fig_confusion_matrix_grid_main_models.png", normalize=False))
    outputs.append(draw_confusion_grid(plt, matrices, mapping, paths.figures / "fig_confusion_matrix_row_normalized_grid.png", normalize=True))
    return outputs


def build_detailed_diagrams(paths: ProfessorReportV2Paths, sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[Path]:
    params = {row["model_display_name"]: row for row in read_csv(paths.tables / "table_parameter_count_main.csv")}
    specs = [
        (
            "diagram_common_head_detailed",
            "Common Classifier Head",
            ["Feature Extractor Output: 64-dim", "Linear 64->64", "ReLU", "Dropout p=0.1", "Linear 64->5", "5-class logits"],
            "All controlled neural models use the same classifier head. Common head params: 4,485.",
        ),
        (
            "diagram_all_channel_1d_cnn_detailed",
            "All-Channel 1D CNN",
            ["Input 512x18", "Conv1D jointly over 18 channels", "Temporal pooling/projection", "64-dim representation", "Common head"],
            param_note(params, "All-Channel 1D CNN"),
        ),
        (
            "diagram_shared_1d_encoder_detailed",
            "Shared 1D Encoder",
            ["Input 512x18", "Split into 18 single-channel streams", "Same Temporal Encoder f_theta reused", "18 tokens", "Mean pooling", "64-dim representation", "Common head"],
            param_note(params, "Shared 1D Encoder") + " No residual branch.",
        ),
        (
            "diagram_residual_channel_shared_encoder_detailed",
            "Residual Channel-Shared Encoder",
            ["Input 512x18", "Branch A: shared single-channel encoder", "Branch B: mean/std/min/max statistics", "Fusion projection", "64-dim representation", "Common head"],
            param_note(params, "Residual Channel-Shared Encoder") + " Residual branch preserves channel-specific summary cues.",
        ),
        (
            "diagram_statistical_summary_mlp_detailed",
            "Statistical Summary MLP",
            ["Input 512x18", "Signal-derived mean/std/min/max", "72 summary features", "Projection to 64-dim", "Common head"],
            param_note(params, "Statistical Summary MLP"),
        ),
        (
            "diagram_2d_cnn_detailed",
            "2D CNN",
            ["Input time x channel matrix 512x18", "2D convolution", "Pooling/projection", "64-dim representation", "Common head"],
            param_note(params, "2D CNN"),
        ),
        (
            "diagram_tree_baselines_detailed",
            "Practical Tree Baselines",
            ["Input 512x18", "Train-scaled IMU signal", "Signal-derived summary features", "Feature audit: no metadata/label/subject/boundary", "RF / XGBoost / Linear SVM"],
            "Classical model parameters are not comparable to neural parameter counts.",
        ),
        (
            "diagram_evaluation_protocol_detailed",
            "Evaluation Protocol",
            ["6 subjects", "LOSO held-out test subject", "Remaining 5 subjects", "Per subject-class: 16 train / 4 val", "Train 400 / val 100 / test 100", "StandardScaler fit train only"],
            "Test subject is never used in training, validation, or scaler fitting.",
        ),
    ]
    outputs = []
    for name, title, nodes, note in specs:
        mmd = paths.diagrams / f"{name}.mmd"
        png = paths.diagrams / f"{name}.png"
        mmd.write_text(mermaid(title, nodes, note), encoding="utf-8")
        draw_box_diagram(png, title, nodes, note)
        outputs.extend([mmd, png])
    return outputs


def write_captions(paths: ProfessorReportV2Paths) -> Path:
    path = paths.figures / "captions.md"
    path.write_text(
        """# Figure Captions v2

## fig_01_feature_extractor_macro_f1.png
- Korean caption: 공통 classifier head 조건에서 feature extractor별 Macro F1과 CI를 비교한 그림.
- English caption: Macro F1 with confidence intervals across feature extractors under a shared classifier head.
- 핵심 메시지: Residual Channel-Shared Encoder는 All-Channel 1D CNN, XGBoost, RF와 경쟁 가능한 수준이며 Statistical Summary MLP가 가장 높다.

## fig_normalization_pipeline.png
- Korean caption: raw window conversion 이후 LOSO split, train-only scaler fit, transform, model/feature extraction까지의 처리 순서.
- English caption: Processing order from raw-window conversion to LOSO split, train-only scaler fitting, transformation, and model or feature extraction.
- 핵심 메시지: validation/test subject는 scaler fitting에 사용되지 않았고 per-window z-score 및 augmentation은 사용하지 않았다.

## fig_parameter_count_vs_macro_f1.png
- Korean caption: controlled neural model의 parameter count와 Macro F1 관계.
- English caption: Parameter count versus Macro F1 for controlled neural models.
- 핵심 메시지: common head parameter는 동일하며, parameter 수와 성능은 단순 비례하지 않는다.

## fig_confusion_matrix_row_normalized_grid.png
- Korean caption: 주요 모델의 row-normalized confusion matrix. 행은 true class, 열은 predicted class다.
- English caption: Row-normalized confusion matrices for key models. Rows are true classes and columns are predicted classes.
- 핵심 메시지: Shared 1D Encoder의 failure pattern과 Class 3 Excessive Lean의 혼동 패턴을 확인하기 위한 진단용 그림이다.
""",
        encoding="utf-8",
    )
    return path


def write_main_report(paths: ProfessorReportV2Paths, sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> Path:
    table = lambda name: markdown_table(read_csv(paths.tables / name))
    report = f"""---
title: "Professor Report v2 - Residual Channel-Shared Encoder"
project: "Squat IMU"
date: "{date.today().isoformat()}"
status: "Architecture/protocol refinement"
tags:
  - research
  - squat
  - imu
  - kiee
---

# Professor Report v2 - Architecture and Protocol

## 0. 이번 v2 보강의 목적

기존 교수님 보고서는 결과 요약 중심이었다. 이번 v2는 **실제 code 기반 architecture audit, normalization/scaler 순서, parameter count, confusion matrix, residual feature audit**을 보강했다. 새 학습, CAU training, model 재학습, hyperparameter 변경은 수행하지 않았다.

## 1. 핵심 결론

같은 classifier head 조건에서 단순 Shared 1D Encoder는 낮았지만, residual branch를 추가한 Residual Channel-Shared Encoder는 All-Channel 1D CNN, XGBoost, Random Forest와 경쟁 가능한 수준까지 회복했다. 다만 Statistical Summary MLP와 tree models도 강하므로, 논문 claim은 **압도적 우월성**이 아니라 **shared encoder bottleneck 완화**로 잡는 것이 안전하다.

## 2. 교수님 피드백 반영 상태

{table("table_02_professor_feedback_status.csv")}

## 3. Dataset and Evaluation Protocol

{table("table_01_dataset_summary.csv")}

![Evaluation Protocol](diagrams/diagram_evaluation_protocol_detailed.png)

![Normalization Pipeline](figures/fig_normalization_pipeline.png)

## 4. Normalization and Feature Extraction Order

{table("table_normalization_protocol.csv")}

{table("table_feature_extraction_order.csv")}

핵심은 feature extraction도 fold 내부 scaler 정책 뒤에 위치한다는 점이다. Statistical Summary MLP와 residual branch의 mean/std/min/max는 train-only StandardScaler로 transform된 tensor에서 계산된다. RF, XGBoost, SVM도 같은 scaled signal에서 signal-derived feature를 계산한 뒤 train feature만으로 estimator를 fit한다.

## 5. Common Classifier Head

![Common Head](diagrams/diagram_common_head_detailed.png)

{table("table_common_head_verification.csv")}

모든 controlled neural model은 64차원 representation을 만들고 동일한 MLP head를 사용한다. 공통 head parameter count는 4,485다. 따라서 이번 비교는 classifier 차이가 아니라 feature extractor 차이에 더 초점을 맞춘다.

## 6. Architecture Details

### 6.1 Statistical Summary MLP

![Statistical Summary MLP](diagrams/diagram_statistical_summary_mlp_detailed.png)

Statistical Summary MLP는 scaled IMU tensor에서 mean/std/min/max 72개 feature를 계산하고 64차원 representation으로 projection한다. code audit 기준으로 energy/RMS는 이 residual/statistical MLP branch에는 포함되지 않는다.

### 6.2 All-Channel 1D CNN

![All-Channel 1D CNN](diagrams/diagram_all_channel_1d_cnn_detailed.png)

All-Channel 1D CNN은 18채널을 첫 convolution부터 함께 처리한다. 채널 간 상호작용을 직접 학습할 수 있는 baseline이다.

### 6.3 Shared 1D Encoder

![Shared 1D Encoder](diagrams/diagram_shared_1d_encoder_detailed.png)

Shared 1D Encoder는 18개 단일 채널 stream에 같은 Temporal Encoder를 재사용한다. parameter sharing은 되지만 residual branch가 없어서 channel-specific statistical cue가 약해질 수 있다.

### 6.4 Residual Channel-Shared Encoder

![Residual Channel-Shared Encoder](diagrams/diagram_residual_channel_shared_encoder_detailed.png)

Residual Channel-Shared Encoder는 shared temporal branch와 signal-derived residual branch를 결합한다. residual branch는 mean/std/min/max를 사용해 shared encoder가 잃을 수 있는 channel-specific summary signal을 보완한다.

### 6.5 2D CNN

![2D CNN](diagrams/diagram_2d_cnn_detailed.png)

2D CNN은 time x channel matrix를 대상으로 convolution을 적용하는 baseline이다.

### 6.6 RF/XGBoost/SVM

![Tree Baselines](diagrams/diagram_tree_baselines_detailed.png)

Tree baselines는 scaled IMU signal에서 signal-derived summary feature를 계산한다. feature audit 기준 metadata, label, subject ID, window boundary는 포함되지 않았다.

## 7. Parameter Count and Capacity

![Parameter Count vs Macro F1](figures/fig_parameter_count_vs_macro_f1.png)

{table("table_parameter_count_main.csv")}

Parameter count와 성능은 단순 비례하지 않았다. Raw Flatten MLP는 매우 크지만 최고 성능은 아니며, common head parameter는 controlled neural models에서 동일하다. Classical models는 neural parameter count와 직접 비교하지 않는다.

## 8. Main Controlled Results

![Feature Extractor Macro F1](figures/fig_01_feature_extractor_macro_f1.png)

{table("table_03_feature_extractor_clean_names.csv")}

핵심 수치는 Shared 1D Encoder 0.2806, Shared 1D + Identity 0.5182, Residual Channel-Shared Encoder 0.8004, Residual Channel-Shared + Identity 0.7973, All-Channel 1D CNN 0.7994, Statistical Summary MLP 0.8174, XGBoost 0.7961이다.

## 9. Residual Branch Effect

![Residual Branch Effect](figures/fig_02_residual_effect_flow.png)

{table("table_05_component_flow.csv")}

Identity만으로는 충분하지 않았고 residual branch가 가장 큰 변화를 만들었다. 국내 논문 claim은 residual branch 중심으로 잡는 것이 안전하다.

## 10. Practical Baselines: Statistical MLP, RF, XGBoost

![Practical Baselines](figures/fig_03_xgboost_practical_baselines.png)

{table("table_04_xgboost_completion_clean_names.csv")}

![Feature Importance](figures/fig_05_rf_xgboost_feature_importance.png)

{table("table_07_feature_importance_top10.csv")}

s1_ax, s1_gx 관련 feature가 반복적으로 관찰되지만, feature importance는 모델 내부 중요도일 뿐 생체역학적 인과 증거가 아니다.

## 11. Confusion Matrix Analysis

![Row-normalized Confusion Matrix Grid](figures/fig_confusion_matrix_row_normalized_grid.png)

{table("table_confusion_matrix_summary.csv")}

Confusion matrix는 행이 true class, 열이 predicted class다. 본문에서는 row-normalized grid를 우선 사용해 class별 recall과 주요 혼동 방향을 본다. 해석은 Class 3 Excessive Lean과 Shared 1D Encoder failure pattern 중심으로 제한한다.

## 12. Class 3 Excessive Lean

![Class 3 Excessive Lean](figures/fig_04_class3_excessive_lean.png)

{table("table_06_class3_summary.csv")}

Residual Channel-Shared Encoder는 Class 3에서 비교적 높은 F1을 보였지만, 이 class를 완전히 해결했다고 과장하지 않는다.

## 13. What This Means for the Paper

{table("table_08_claim_audit_summary.csv")}

## 14. Recommended KIEE Scope

{table("table_09_final_recommended_scope.csv")}

## 15. Questions for Professor

1. 논문 제목 또는 제안 구조명을 `Residual Channel-Shared Feature Extractor`로 잡아도 되는가?
2. Statistical Summary MLP가 가장 높은 점을 practical baseline으로 분리해서 배치할지?
3. position identity와 attention은 본문에서 낮추고 future work로 둘지?
4. confusion matrix와 parameter count를 본문에 어느 정도 넣을지?
5. KIEE 범위를 supervised IMU-only classification으로 확정해도 되는지?

## Appendix A. Internal Name Mapping

Internal model names are kept in `tables/table_internal_name_mapping.csv`. 이 appendix file에서만 internal names를 허용한다.

## Appendix B. Execution and Reproducibility

- 새 학습 실행: no
- CAU training 실행: no
- optimizer step/backward 실행: no
- architecture audit: synthetic input forward-only
- normalization audit: config, result CSV, runner code 기반
- local unit test: `python -m unittest discover -s tests -v`
- raw result directories: read-only input으로만 사용
"""
    path = paths.output_docs / "Professor Report v2 - Architecture and Protocol.md"
    path.write_text(report, encoding="utf-8")
    return path


def write_figure_table_index(paths: ProfessorReportV2Paths) -> Path:
    rows = [
        {"artifact": "fig_01_feature_extractor_macro_f1.png", "source": "table_03_feature_extractor_clean_names.csv", "script": "scripts/build_professor_report_assets.py", "report_section": "8", "paper_main_candidate": "yes", "appendix_candidate": "no"},
        {"artifact": "fig_normalization_pipeline.png", "source": "table_normalization_protocol.csv", "script": "scripts/audit_normalization_protocol.py", "report_section": "3-4", "paper_main_candidate": "yes", "appendix_candidate": "no"},
        {"artifact": "fig_parameter_count_vs_macro_f1.png", "source": "table_parameter_count_main.csv", "script": "scripts/build_professor_report_assets.py", "report_section": "7", "paper_main_candidate": "maybe", "appendix_candidate": "yes"},
        {"artifact": "fig_confusion_matrix_row_normalized_grid.png", "source": "confusion_matrices.csv", "script": "scripts/build_professor_report_assets.py", "report_section": "11", "paper_main_candidate": "maybe", "appendix_candidate": "yes"},
        {"artifact": "diagram_residual_channel_shared_encoder_detailed.png", "source": "table_architecture_audit.csv", "script": "scripts/build_professor_report_assets.py", "report_section": "6.4", "paper_main_candidate": "yes", "appendix_candidate": "no"},
        {"artifact": "table_parameter_count_main.csv", "source": "table_architecture_audit.csv + aggregate metrics", "script": "scripts/build_professor_report_assets.py", "report_section": "7", "paper_main_candidate": "maybe", "appendix_candidate": "yes"},
        {"artifact": "table_confusion_matrix_summary.csv", "source": "confusion_matrices.csv", "script": "scripts/build_professor_report_assets.py", "report_section": "11", "paper_main_candidate": "maybe", "appendix_candidate": "yes"},
    ]
    path = paths.output_docs / "Figure and Table Index v2.md"
    path.write_text("# Figure and Table Index v2\n\n" + markdown_table(rows) + "\n", encoding="utf-8")
    return path


def write_readme(paths: ProfessorReportV2Paths) -> Path:
    path = paths.output_docs / "README.md"
    path.write_text(
        """# Professor Report v2

이 폴더는 교수님 보고용 architecture/protocol refinement 산출물이다.

- 새 학습 없이 read-only result CSV와 code audit으로 생성했다.
- 핵심 산출물: architecture diagrams, normalization pipeline, parameter count table, confusion matrix figures.
- full raw results, processed arrays, checkpoints는 포함하지 않는다.
- Obsidian vault에는 동일 폴더 구조로 mirror된다.
""",
        encoding="utf-8",
    )
    return path


def build_confusion_matrices(sources: dict[str, list[dict[str, str]]]) -> dict[str, list[list[int]]]:
    rows = [row for row in sources["controlled_confusion"] if row.get("scope") == "fold"]
    rows.extend(row for row in sources["xgb_confusion"] if row.get("scope") == "fold")
    matrices = {model: [[0 for _ in range(5)] for _ in range(5)] for model in MAIN_CONFUSION_MODELS}
    for row in rows:
        model = row.get("model_name", "")
        if model not in matrices:
            continue
        true_class = int(row["true_class"])
        pred_class = int(row["pred_class"])
        matrices[model][true_class][pred_class] += int(float(row["count"]))
    return matrices


def draw_macro_f1_bar(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    labels = [row["model_display_name"] for row in rows]
    values = [fnum(row["macro_f1"]) for row in rows]
    lows = []
    highs = []
    for row, value in zip(rows, values):
        lo, hi = parse_ci(row["macro_f1_ci"])
        lows.append(value - lo)
        highs.append(hi - value)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(range(len(labels)), values, yerr=[lows, highs], capsize=3)
    ax.set_title("Feature Extractor Comparison")
    ax.set_ylabel("Mean Macro F1")
    ax.set_ylim(0, 0.95)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(label, 16) for label in labels], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_component_flow(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    labels = [row["model_display_name"] for row in rows]
    values = [fnum(row["macro_f1"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(range(len(labels)), values, marker="o", linewidth=2)
    ax.bar(range(len(labels)), values, alpha=0.3)
    ax.set_title("Residual Branch Effect")
    ax.set_ylabel("Mean Macro F1")
    ax.set_ylim(0, 0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(label, 18) for label in labels], rotation=25, ha="right")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_practical_bar(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    labels = [row["model_display_name"] for row in rows]
    values = [fnum(row["macro_f1"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(range(len(labels)), values)
    ax.set_title("Practical Baselines")
    ax.set_ylabel("Mean Macro F1")
    ax.set_ylim(0.72, 0.84)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(label, 16) for label in labels], rotation=35, ha="right")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.003, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_class3(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    labels = [row["model_display_name"] for row in rows]
    recall = [fnum(row["class3_recall"]) for row in rows]
    f1 = [fnum(row["class3_f1"]) for row in rows]
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar([x - 0.18 for x in xs], recall, width=0.36, label="Recall")
    ax.bar([x + 0.18 for x in xs], f1, width=0.36, label="F1")
    ax.set_title("Class 3: Excessive Lean")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([wrap_label(label, 16) for label in labels], rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_importance(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    labels = [row["rank"] for row in rows]
    rf = [fnum(row["rf_importance"]) for row in rows]
    xgb = [fnum(row["xgb_importance"]) for row in rows]
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar([x - 0.18 for x in xs], rf, width=0.36, label="Random Forest")
    ax.bar([x + 0.18 for x in xs], xgb, width=0.36, label="XGBoost")
    ax.set_title("RF/XGBoost Top Feature Importance")
    ax.set_ylabel("Mean importance")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{row['rank']}\nRF:{short_feature(row['random_forest_feature'])}\nXGB:{short_feature(row['xgboost_feature'])}" for row in rows], fontsize=7)
    ax.text(0.01, 0.96, "Importance is model-specific, not causal evidence.", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_parameter_scatter(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    neural = [row for row in rows if row["total_params"] not in {"not_applicable", ""}]
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    label_offsets = {
        "Statistical Summary MLP": (6, 6),
        "Compact All-Channel 1D CNN": (6, 10),
        "All-Channel 1D CNN": (-45, 28),
        "Shared 1D Encoder": (8, -20),
        "Shared 1D + Identity": (8, 8),
        "Residual Channel-Shared Encoder": (-48, 18),
        "Residual Channel-Shared + Identity": (8, -24),
        "2D CNN": (8, 6),
    }
    for row in neural:
        x = fnum(row["total_params"])
        y = fnum(row["macro_f1"])
        ax.scatter([x], [y], s=50)
        name = row["model_display_name"]
        offset = label_offsets.get(name, (5, 5))
        ax.annotate(
            wrap_label(name, 17),
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.5} if abs(offset[0]) > 20 or abs(offset[1]) > 16 else None,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total parameters (log scale)")
    ax.set_ylabel("Mean Macro F1")
    ax.set_title("Parameter Count vs Macro F1")
    ax.text(
        0.58,
        0.05,
        "Common head params = 4,485 for controlled neural models.\nClassical models are not shown because neural parameter count is not comparable.",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.18, y=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_confusion_single(plt: Any, matrix: list[list[int]], title: str, path: Path, *, normalize: bool) -> Path:
    values = normalize_matrix(matrix) if normalize else [[float(value) for value in row] for row in matrix]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    image = ax.imshow(values, vmin=0, vmax=1 if normalize else None)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([CLASS_NAMES[i] for i in range(5)], rotation=35, ha="right", fontsize=7)
    ax.set_yticklabels([CLASS_NAMES[i] for i in range(5)], fontsize=7)
    for i in range(5):
        for j in range(5):
            text = f"{values[i][j]:.2f}" if normalize else str(int(values[i][j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if values[i][j] > (0.5 if normalize else max(max(row) for row in values) * 0.5) else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_confusion_grid(plt: Any, matrices: dict[str, list[list[int]]], mapping: dict[str, str], path: Path, *, normalize: bool) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, model_name in zip(axes.flatten(), MAIN_CONFUSION_MODELS):
        matrix = matrices[model_name]
        values = normalize_matrix(matrix) if normalize else [[float(value) for value in row] for row in matrix]
        image = ax.imshow(values, vmin=0, vmax=1 if normalize else None)
        ax.set_title(wrap_label(display(model_name, mapping), 22), fontsize=10)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([str(i) for i in range(5)], fontsize=8)
        ax.set_yticklabels([str(i) for i in range(5)], fontsize=8)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{values[i][j]:.2f}" if normalize else str(int(values[i][j])), ha="center", va="center", fontsize=7)
    fig.suptitle("Row-normalized confusion matrices: rows=true, columns=predicted" if normalize else "Confusion matrices: rows=true, columns=predicted")
    fig.subplots_adjust(left=0.05, right=0.88, top=0.88, bottom=0.08, wspace=0.28, hspace=0.38)
    colorbar_axis = fig.add_axes([0.91, 0.2, 0.015, 0.62])
    fig.colorbar(image, cax=colorbar_axis)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_box_diagram(path: Path, title: str, nodes: list[str], note: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, max(4, len(nodes) * 0.72)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    y_positions = list(reversed([idx for idx in range(len(nodes))]))
    for idx, (node, y) in enumerate(zip(nodes, y_positions)):
        box = FancyBboxPatch((0.2, y), 0.6, 0.45, boxstyle="round,pad=0.02", linewidth=1, facecolor="#f3f6fb", edgecolor="#4b5563")
        ax.add_patch(box)
        ax.text(0.5, y + 0.225, node, ha="center", va="center", fontsize=9)
        if idx < len(nodes) - 1:
            ax.annotate("", xy=(0.5, y - 0.43), xytext=(0.5, y - 0.08), arrowprops={"arrowstyle": "->", "linewidth": 1})
    ax.text(0.5, -0.65, note, ha="center", va="top", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.1, len(nodes))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validate_report_v2(paths: ProfessorReportV2Paths) -> dict[str, Any]:
    md_files = list(paths.output_docs.rglob("*.md"))
    png_files = list(paths.output_docs.rglob("*.png"))
    csv_files = list(paths.output_docs.rglob("*.csv"))
    violations = []
    allowed = {
        "display_name_mapping.yaml",
        "table_internal_name_mapping.csv",
        "table_architecture_audit.csv",
        "table_parameter_count_clean.csv",
        "table_layer_shape_summary.csv",
    }
    for file in [*md_files, *csv_files]:
        if file.name in allowed:
            continue
        text = file.read_text(encoding="utf-8")
        if "controlled_" in text:
            violations.append(str(file))
    return {
        "markdown_files": len(md_files),
        "png_files": len(png_files),
        "csv_files": len(csv_files),
        "controlled_name_violations": violations,
        "controlled_name_violation_count": len(violations),
        "table_pipe_errors": markdown_table_pipe_errors(md_files),
        "missing_image_links": missing_image_links(paths.output_docs),
        "passed": not violations and not markdown_table_pipe_errors(md_files) and not missing_image_links(paths.output_docs),
    }


def load_display_mapping(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data["mapping"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def display(model_name: str, mapping: dict[str, str]) -> str:
    return mapping.get(model_name, model_name)


def fnum(value: Any) -> float:
    if value in (None, "", "not_applicable"):
        return 0.0
    return float(value)


def fmt(value: Any, digits: int = 4) -> str:
    return f"{fnum(value):.{digits}f}"


def signed(value: float) -> str:
    return f"{value:+.4f}"


def parse_ci(value: str) -> tuple[float, float]:
    clean = value.strip().strip("[]")
    lo, hi = clean.split(",")
    return float(lo), float(hi)


def model_group(model_name: str) -> str:
    if model_name in {"controlled_stats_mlp", "feature_random_forest_v1", "feature_xgboost_v1", "feature_linear_svm_v1"}:
        return "Practical Baseline"
    if model_name in {"rescnn_bigru_attention_lite_v1", "lee_style_cnn_lstm_2d_v1"}:
        return "Literature Reference"
    if model_name == "controlled_shared_1d_residual":
        return "Proposed Core"
    return "Neural Baseline"


def model_role(model_name: str) -> str:
    roles = {
        "controlled_stats_mlp": "summary-statistics practical baseline",
        "controlled_shared_1d_residual": "proposed core extractor",
        "controlled_all_channel_1d_cnn": "all-channel neural baseline",
        "feature_random_forest_v1": "tree-based practical baseline",
        "feature_xgboost_v1": "boosted tree practical baseline",
        "feature_linear_svm_v1": "linear feature baseline",
    }
    return roles.get(model_name, "comparison model")


def model_interpretation(model_name: str) -> str:
    interpretations = {
        "controlled_stats_mlp": "현재 데이터에서 signal summary가 매우 강함.",
        "controlled_shared_1d_residual": "residual branch가 shared encoder 병목을 완화.",
        "controlled_all_channel_1d_cnn": "채널 정보를 직접 학습하는 강한 neural baseline.",
        "controlled_shared_1d": "단독 shared pooling은 underfit.",
        "controlled_shared_1d_identity": "identity만으로는 병목 해결이 부족.",
        "controlled_shared_1d_residual_identity": "residual-only와 비슷해 identity 추가 이득은 제한적.",
        "controlled_2d_cnn": "time-channel 2D baseline은 낮게 관찰.",
        "controlled_flatten_mlp": "parameter가 크지만 최고 성능은 아님.",
    }
    return interpretations.get(model_name, "비교용 모델.")


def class3_interpretation(model_name: str) -> str:
    if model_name == "controlled_shared_1d_residual":
        return "Class 3에서 높은 F1을 보인 proposed core."
    if model_name == "controlled_shared_1d":
        return "Shared-only failure pattern 확인용."
    if model_name == "feature_xgboost_v1":
        return "tree baseline reference."
    return "class-wise reference."


def parameter_interpretation(model_name: str) -> str:
    if model_name == "controlled_flatten_mlp":
        return "매우 큰 parameter count에도 최고 성능은 아님."
    if model_name == "controlled_shared_1d_residual":
        return "shared encoder에 작은 residual branch를 결합."
    if model_name == "controlled_stats_mlp":
        return "작은 neural summary baseline."
    return "controlled neural parameter count."


def confusion_label(item: tuple[int, int]) -> str:
    pred, count = item
    return f"{CLASS_NAMES[pred]} ({count})"


def confusion_note(model_name: str, class_id: int) -> str:
    if model_name == "controlled_shared_1d" and class_id in {0, 3}:
        return "Shared-only failure pattern 확인 필요."
    if class_id == 3:
        return "Excessive Lean 주요 관심 class."
    return ""


def claim_audit_rows() -> list[dict[str, str]]:
    return [
        {"claim": "Residual branch가 shared encoder bottleneck을 완화했다.", "status": "safe", "rationale": "Shared 1D 0.2806 -> Residual Channel-Shared 0.8004.", "safe_wording": "잔차 branch는 naive shared encoder의 성능 저하를 크게 완화했다."},
        {"claim": "제안 모델이 모든 baseline보다 통계적으로 유의하게 우수하다.", "status": "avoid", "rationale": "Statistical Summary MLP가 가장 높고 CI overlap이 존재한다.", "safe_wording": "강한 neural/practical baseline과 경쟁 가능한 성능을 보였다."},
        {"claim": "feature importance가 생체역학적 원인을 증명한다.", "status": "avoid", "rationale": "importance는 모델 내부 기준이다.", "safe_wording": "s1_ax/s1_gx 관련 feature가 반복적으로 중요하게 관찰되었다."},
        {"claim": "transfer learning이 검증되었다.", "status": "avoid", "rationale": "이번 범위에는 external/SSL 실험이 없다.", "safe_wording": "후속 연구로 남긴다."},
    ]


def recommended_scope_rows() -> list[dict[str, str]]:
    return [
        {"item": "Supervised IMU-only target dataset", "include_in_kiee": "yes", "reason": "논문 핵심 범위", "future_work": ""},
        {"item": "Clean-room conversion and LOSO protocol", "include_in_kiee": "yes", "reason": "재현성과 누수 통제 근거", "future_work": ""},
        {"item": "Controlled feature extractor comparison", "include_in_kiee": "yes", "reason": "교수님 피드백 핵심", "future_work": ""},
        {"item": "RF/XGBoost/Stats MLP baselines", "include_in_kiee": "yes", "reason": "강한 practical baselines 투명 보고", "future_work": ""},
        {"item": "SSL/external transfer", "include_in_kiee": "no", "reason": "이번 supervised paper 범위 밖", "future_work": "후속 논문"},
    ]


def internal_mapping_rows(mapping: dict[str, str]) -> list[dict[str, str]]:
    return [{"internal_model_name": key, "display_name": value} for key, value in sorted(mapping.items())]


def normalize_matrix(matrix: list[list[int]]) -> list[list[float]]:
    out = []
    for row in matrix:
        total = sum(row)
        out.append([value / total if total else 0.0 for value in row])
    return out


def param_note(params: dict[str, dict[str, str]], display_name: str) -> str:
    row = params.get(display_name, {})
    if not row:
        return ""
    return f"Params: total={row.get('total_params')}, extractor={row.get('extractor_params')}, head={row.get('common_head_params')}."


def mermaid(title: str, nodes: list[str], note: str) -> str:
    lines = ["flowchart TD"]
    for idx, node in enumerate(nodes):
        lines.append(f'  N{idx}["{node}"]')
        if idx > 0:
            lines.append(f"  N{idx - 1} --> N{idx}")
    lines.append(f'  NOTE["{note}"]')
    lines.append(f"  N{len(nodes) - 1} -.-> NOTE")
    return "\n".join(lines) + "\n"


def markdown_table(rows: list[dict[str, str]], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "&#124;").replace("\n", " ") for col in columns) + " |")
    return "\n".join(lines)


def wrap_label(label: str, width: int) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def short_feature(feature: str) -> str:
    return feature.replace("_", " ")


def markdown_table_pipe_errors(files: list[Path]) -> list[str]:
    errors = []
    for path in files:
        expected = None
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not (line.startswith("|") and line.endswith("|")):
                expected = None
                continue
            count = line.count("|")
            if expected is None:
                expected = count
            elif count != expected:
                errors.append(f"{path}:{line_no}: expected {expected}, found {count}")
    return errors


def missing_image_links(root: Path) -> list[str]:
    missing = []
    for path in root.rglob("*.md"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("![") and "](" in stripped and stripped.endswith(")"):
                target = stripped.split("](", 1)[1].rsplit(")", 1)[0]
                if not target.startswith("http") and not (path.parent / target).exists():
                    missing.append(f"{path}:{line_no}:{target}")
    return missing
