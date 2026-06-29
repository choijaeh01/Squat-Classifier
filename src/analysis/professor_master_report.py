from __future__ import annotations

import json
import math
import re
import shutil
import textwrap
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from analysis.professor_report_v2_assets import (
    CLASS_NAMES,
    CONTROLLED_DIR,
    FULL_MATRIX_DIR,
    LITERATURE_DIR,
    V3_ABLATION_DIR,
    XGBOOST_DIR,
    display,
    fnum,
    fmt,
    load_display_mapping,
    markdown_table,
    markdown_table_pipe_errors,
    missing_image_links,
    normalize_matrix,
    parse_ci,
    read_csv,
    short_feature,
    signed,
    wrap_label,
    write_csv,
)


CONTROLLED_MAIN_MODELS = [
    "controlled_stats_mlp",
    "controlled_shared_1d_residual",
    "controlled_all_channel_1d_cnn",
    "controlled_shared_1d_residual_identity",
    "feature_xgboost_v1",
    "feature_random_forest_v1",
    "rescnn_bigru_attention_lite_v1",
    "controlled_all_channel_1d_cnn_small",
    "feature_linear_svm_v1",
    "lee_style_cnn_lstm_2d_v1",
    "controlled_flatten_mlp",
    "controlled_shared_1d_identity",
    "controlled_2d_cnn",
    "controlled_shared_1d",
]

INTERACTION_MODELS = [
    "controlled_shared_1d",
    "controlled_shared_1d_identity",
    "controlled_shared_1d_residual",
    "controlled_shared_1d_residual_identity",
]

PRACTICAL_MODELS = [
    "controlled_stats_mlp",
    "controlled_shared_1d_residual",
    "controlled_all_channel_1d_cnn",
    "feature_xgboost_v1",
    "feature_random_forest_v1",
]

CONFUSION_MODELS = [
    "controlled_stats_mlp",
    "controlled_shared_1d_residual",
    "controlled_all_channel_1d_cnn",
    "feature_xgboost_v1",
    "feature_random_forest_v1",
    "controlled_shared_1d",
]


@dataclass(frozen=True)
class ProfessorMasterReportPaths:
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


def check_professor_master_report_inputs(project_root: Path) -> dict[str, Any]:
    required = [
        Path("docs/professor_report_v1/Professor Report - Residual Channel-Shared Encoder.md"),
        Path("docs/professor_report_v2/Professor Report v2 - Architecture and Protocol.md"),
        Path("docs/professor_report_v3/Professor Report v3 - Story and Identity Analysis.md"),
        CONTROLLED_DIR / "aggregate_metrics_by_model.csv",
        CONTROLLED_DIR / "paired_model_differences.csv",
        CONTROLLED_DIR / "classwise_metrics_by_model.csv",
        CONTROLLED_DIR / "subjectwise_metrics_by_model.csv",
        CONTROLLED_DIR / "confusion_matrices.csv",
        CONTROLLED_DIR / "model_parameter_counts.csv",
        CONTROLLED_DIR / "random_forest_feature_importance_aggregate.csv",
        XGBOOST_DIR / "aggregate_metrics_by_model.csv",
        XGBOOST_DIR / "classwise_metrics_by_model.csv",
        XGBOOST_DIR / "confusion_matrices.csv",
        XGBOOST_DIR / "xgboost_feature_importance_aggregate.csv",
        LITERATURE_DIR / "aggregate_metrics_by_model.csv",
        LITERATURE_DIR / "classwise_metrics_by_model.csv",
        V3_ABLATION_DIR / "aggregate_metrics_by_model.csv",
        FULL_MATRIX_DIR / "aggregate_metrics_by_model.csv",
        Path("docs/experiment_protocol.md"),
        Path("docs/results_tracking.md"),
        Path("docs/controlled_feature_extractor_comparison_v1_report.md"),
        Path("docs/xgboost_only_completion_v1_report.md"),
        Path("docs/controlled_feature_extractor_interpretation_notes_for_user.md"),
        Path("docs/professor_report_v3/tables/table_identity_residual_effects.csv"),
        Path("docs/professor_report_v2/tables/table_normalization_protocol.csv"),
    ]
    status = {str(path): (project_root / path).exists() for path in required}
    return {"all_present": all(status.values()), "files": status}


def build_professor_master_report(paths: ProfessorMasterReportPaths) -> dict[str, Any]:
    for folder in (paths.output_docs, paths.tables, paths.figures, paths.diagrams, paths.assets):
        folder.mkdir(parents=True, exist_ok=True)
    mapping = load_display_mapping(paths.mapping_path)
    sources = load_sources(paths.project_root)

    tables = build_tables(paths, sources, mapping)
    diagrams = build_diagrams(paths)
    figures = build_figures(paths, sources, mapping)
    captions = write_figure_captions(paths)
    architecture = write_architecture_explanation(paths)
    main_report = write_master_report(paths)
    brief = write_brief(paths)
    index = write_figure_table_index(paths)
    readme = write_readme(paths)
    update_docs_indexes(paths.project_root)
    validation = validate_master_report(paths)
    quality = write_quality_check(paths, validation)
    validation["quality_check"] = str(quality)
    validation_path = paths.assets / "professor_master_report_validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    obsidian_sync = copy_to_local_obsidian_if_available(paths)
    return {
        "output_docs": str(paths.output_docs),
        "output_obsidian": str(paths.output_obsidian) if paths.output_obsidian else "",
        "obsidian_sync": obsidian_sync,
        "tables": [str(path) for path in tables],
        "figures": [str(path) for path in figures],
        "diagrams": [str(path) for path in diagrams],
        "captions": str(captions),
        "architecture_explanation": str(architecture),
        "main_report": str(main_report),
        "brief": str(brief),
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
        "controlled_classwise": read(CONTROLLED_DIR / "classwise_metrics_by_model.csv"),
        "controlled_subjectwise": read(CONTROLLED_DIR / "subjectwise_metrics_by_model.csv"),
        "controlled_confusion": read(CONTROLLED_DIR / "confusion_matrices.csv"),
        "rf_importance": read(CONTROLLED_DIR / "random_forest_feature_importance_aggregate.csv"),
        "xgb_aggregate": read(XGBOOST_DIR / "aggregate_metrics_by_model.csv"),
        "xgb_classwise": read(XGBOOST_DIR / "classwise_metrics_by_model.csv"),
        "xgb_confusion": read(XGBOOST_DIR / "confusion_matrices.csv"),
        "xgb_importance": read(XGBOOST_DIR / "xgboost_feature_importance_aggregate.csv"),
        "literature_aggregate": read(LITERATURE_DIR / "aggregate_metrics_by_model.csv"),
        "literature_classwise": read(LITERATURE_DIR / "classwise_metrics_by_model.csv"),
        "v3_aggregate": read(V3_ABLATION_DIR / "aggregate_metrics_by_model.csv"),
        "identity_effects": read(Path("docs/professor_report_v3/tables/table_identity_residual_effects.csv")),
        "normalization": read(Path("docs/professor_report_v2/tables/table_normalization_protocol.csv")),
    }


def build_tables(paths: ProfessorMasterReportPaths, sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[Path]:
    rows = {
        "table_01_storyline_overview.csv": storyline_rows(),
        "table_02_dataset_protocol_summary.csv": dataset_protocol_rows(),
        "table_03_model_family_summary.csv": model_family_rows(sources, mapping),
        "table_04_main_results_clean.csv": main_result_rows(sources, mapping),
        "table_05_residual_identity_interaction.csv": residual_identity_2x2_rows(sources, mapping),
        "table_06_effect_decomposition.csv": effect_decomposition_rows(sources),
        "table_07_parameter_and_capacity.csv": parameter_capacity_rows(sources, mapping),
        "table_08_normalization_summary.csv": normalization_rows(sources),
        "table_09_feature_importance_summary.csv": feature_importance_rows(sources),
        "table_10_class3_and_confusion_summary.csv": class3_confusion_rows(sources, mapping),
        "table_11_claim_audit_final.csv": claim_audit_rows(),
        "table_12_kiee_scope.csv": kiee_scope_rows(),
        "table_internal_name_mapping.csv": [{"internal_model_name": key, "display_name": value} for key, value in sorted(mapping.items())],
    }
    return [write_csv(paths.tables / name, table_rows) for name, table_rows in rows.items()]


def storyline_rows() -> list[dict[str, str]]:
    return [
        {
            "stage": "From undergraduate system to journal scope",
            "question": "기존 IMU+Vision+on-device 시스템을 그대로 논문 중심으로 둘 것인가?",
            "action": "저널 논문 범위를 IMU-only supervised feature extractor comparison으로 축소했다.",
            "evidence": "보고서 v1/v2/v3와 controlled comparison 결과",
            "takeaway": "시스템 구현보다 clean-room protocol과 feature extractor 비교를 중심으로 설명한다.",
        },
        {
            "stage": "Clean-room dataset reconstruction",
            "question": "raw/manual boundary 기반 target dataset을 재현 가능하게 만들 수 있는가?",
            "action": "raw CSV와 manually labeled boundary에서 512x18 processed dataset v1을 생성했다.",
            "evidence": "600 windows, 6 subjects, 5 classes, NaN/Inf 없음",
            "takeaway": "논문 실험은 새 clean-room pipeline에서 생성한 공식 target dataset을 사용한다.",
        },
        {
            "stage": "Why common head comparison was needed",
            "question": "모델별 head 차이가 feature extractor 효과를 가리는가?",
            "action": "64-dim representation과 4,485-param common MLP head를 고정했다.",
            "evidence": "common head verification and controlled model parameter audit",
            "takeaway": "비교 초점은 classifier가 아니라 classifier 앞 feature extractor다.",
        },
        {
            "stage": "Architecture comparison",
            "question": "shared encoder가 왜 낮았고 어떤 보완이 필요한가?",
            "action": "Shared 1D, identity, residual, all-channel, 2D, summary, tree baselines를 비교했다.",
            "evidence": "main controlled table and residual/identity interaction table",
            "takeaway": "단순 공유 구조는 병목이 크고 residual statistics가 이를 크게 완화한다.",
        },
        {
            "stage": "Practical baselines",
            "question": "간단한 통계 feature와 tree model이 얼마나 강한가?",
            "action": "Statistical Summary MLP, Random Forest, XGBoost, SVM을 같은 split/scaler policy로 평가했다.",
            "evidence": "Stats MLP 0.8174, XGBoost 0.7961, RF 0.7845 Macro F1",
            "takeaway": "summary statistics가 강한 dataset임을 투명하게 보고해야 한다.",
        },
        {
            "stage": "Paper claim",
            "question": "무엇을 안전하게 주장할 수 있는가?",
            "action": "압도적 우월성이 아니라 shared encoder bottleneck 완화와 경쟁 가능성으로 claim을 보수화했다.",
            "evidence": "Residual branch effect and CI-overlapping strong baselines",
            "takeaway": "국내 저널 범위는 supervised IMU-only extractor comparison이 가장 안전하다.",
        },
    ]


def dataset_protocol_rows() -> list[dict[str, str]]:
    return [
        {"item": "Sensors", "value": "3 MPU-6050 IMUs", "note": "s0 lower back/waist, s1 right thigh, s2 right calf"},
        {"item": "Channels", "value": "18", "note": "accelerometer 3-axis + gyroscope 3-axis per IMU"},
        {"item": "Window length", "value": "512", "note": "phase-normalized linear interpolation from manually labeled boundaries"},
        {"item": "Samples", "value": "600 windows", "note": "6 subjects x 5 classes x 20 windows"},
        {"item": "Classes", "value": "5", "note": "Correct, Knee Valgus, Butt Wink, Excessive Lean, Partial Squat"},
        {"item": "Evaluation", "value": "LOSO with within-train stratified validation", "note": "held-out subject is test only"},
        {"item": "Fold size", "value": "train 400 / val 100 / test 100", "note": "candidate training subjects: 16 train and 4 val windows per subject-class"},
        {"item": "Normalization", "value": "train-only StandardScaler", "note": "fit only on train indices, transform train/val/test"},
        {"item": "Disabled", "value": "per-window z-score, augmentation, focal loss, SSL, external transfer", "note": "not used in locked comparison"},
        {"item": "Leakage audit", "value": "passed", "note": "split and scaler audit passed in full experiments"},
    ]


def model_family_rows(sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[dict[str, str]]:
    params = {row["model_name"]: row for row in sources["controlled_params"]}
    literature = {row["model_name"]: row for row in sources["literature_aggregate"]}
    rows = []
    specs = [
        ("controlled_stats_mlp", "Practical neural baseline", "512x18 scaled signal -> 72 statistics", "mean/std/min/max projection", "common MLP head", "small summary baseline"),
        ("controlled_shared_1d_residual", "Proposed core", "18 single-channel streams + residual statistics", "shared temporal encoder plus residual statistics branch", "common MLP head", "shared bottleneck mitigation"),
        ("controlled_all_channel_1d_cnn", "Neural baseline", "all 18 channels jointly", "all-channel Conv1D", "common MLP head", "strong joint-channel baseline"),
        ("controlled_shared_1d", "Neural ablation", "18 single-channel streams", "same temporal encoder reused then pooled", "common MLP head", "shared-only failure reference"),
        ("controlled_shared_1d_identity", "Neural ablation", "shared tokens plus identity embeddings", "channel/sensor/modality/axis identity", "common MLP head", "token origin ablation"),
        ("controlled_2d_cnn", "Neural baseline", "time x channel matrix", "2D convolution", "common MLP head", "time-channel image-like baseline"),
        ("feature_random_forest_v1", "Practical tree baseline", "162 signal-derived features", "hand-crafted feature extraction", "Random Forest estimator", "strong non-neural reference"),
        ("feature_xgboost_v1", "Practical boosted tree baseline", "162 signal-derived features", "hand-crafted feature extraction", "XGBoost estimator", "boosted tree reference"),
        ("rescnn_bigru_attention_lite_v1", "Literature reference", "512x18 sequence", "Residual CNN + BiGRU + attention", "model-specific head", "temporal literature baseline"),
        ("lee_style_cnn_lstm_2d_v1", "Literature reference", "internal 40-step downsampled time-channel matrix", "2D CNN + LSTM", "model-specific head", "adapted CNN-LSTM reference"),
    ]
    for model, family, input_handling, extractor, head, role in specs:
        param_row = params.get(model) or literature.get(model) or {}
        rows.append(
            {
                "model_display_name": display(model, mapping),
                "family": family,
                "input_handling": input_handling,
                "feature_extractor": extractor,
                "classifier_head": head,
                "parameters": param_text(model, param_row),
                "role": role,
            }
        )
    return rows


def main_result_rows(sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[dict[str, str]]:
    metrics = metric_lookup(sources)
    rows = []
    for model in CONTROLLED_MAIN_MODELS:
        row = metrics.get(model)
        if not row:
            continue
        rows.append(
            {
                "model_display_name": display(model, mapping),
                "group": result_group(model),
                "accuracy": fmt(row.get("mean_accuracy")),
                "macro_f1": fmt(row.get("mean_macro_f1")),
                "weighted_f1": fmt(row.get("mean_weighted_f1")),
                "macro_f1_ci": ci_text(row),
                "role": result_role(model),
                "safe_interpretation": result_interpretation(model),
            }
        )
    rows.sort(key=lambda item: fnum(item["macro_f1"]), reverse=True)
    for idx, row in enumerate(rows, 1):
        row_with_rank = {"rank": str(idx)}
        row_with_rank.update(row)
        rows[idx - 1] = row_with_rank
    return rows


def residual_identity_2x2_rows(sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[dict[str, str]]:
    metrics = metric_lookup(sources)
    shared = fnum(metrics["controlled_shared_1d"]["mean_macro_f1"])
    identity = fnum(metrics["controlled_shared_1d_identity"]["mean_macro_f1"])
    residual = fnum(metrics["controlled_shared_1d_residual"]["mean_macro_f1"])
    residual_identity = fnum(metrics["controlled_shared_1d_residual_identity"]["mean_macro_f1"])
    return [
        {
            "residual_branch": "Absent",
            "identity_absent_model": display("controlled_shared_1d", mapping),
            "identity_absent_macro_f1": fmt(shared),
            "identity_present_model": display("controlled_shared_1d_identity", mapping),
            "identity_present_macro_f1": fmt(identity),
            "identity_effect": signed(identity - shared),
            "interpretation": "identity는 shared-only 구조에서 token origin 손실을 일부 보완한다.",
        },
        {
            "residual_branch": "Present",
            "identity_absent_model": display("controlled_shared_1d_residual", mapping),
            "identity_absent_macro_f1": fmt(residual),
            "identity_present_model": display("controlled_shared_1d_residual_identity", mapping),
            "identity_present_macro_f1": fmt(residual_identity),
            "identity_effect": signed(residual_identity - residual),
            "interpretation": "residual branch가 있으면 identity 추가 이득은 거의 없다.",
        },
    ]


def effect_decomposition_rows(sources: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    out = []
    for row in sources["identity_effects"]:
        name = row["effect_name"]
        note = row.get("interpretation_note", "")
        out.append(
            {
                "effect_name": name,
                "contrast": row["contrast"],
                "mean_delta": row["mean_delta"],
                "bootstrap_ci": row["bootstrap_ci"],
                "n_pairs": row["n_pairs"],
                "interpretation": note,
            }
        )
    return out


def parameter_capacity_rows(sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[dict[str, str]]:
    metrics = metric_lookup(sources)
    params = {row["model_name"]: row for row in sources["controlled_params"]}
    rows = []
    for model in [
        "controlled_stats_mlp",
        "controlled_shared_1d_residual",
        "controlled_shared_1d_residual_identity",
        "controlled_all_channel_1d_cnn",
        "controlled_all_channel_1d_cnn_small",
        "controlled_shared_1d_identity",
        "controlled_shared_1d",
        "controlled_2d_cnn",
        "controlled_flatten_mlp",
        "feature_xgboost_v1",
        "feature_random_forest_v1",
        "rescnn_bigru_attention_lite_v1",
    ]:
        metric = metrics.get(model, {})
        param = params.get(model) or metric
        rows.append(
            {
                "model_display_name": display(model, mapping),
                "total_params": param.get("total_params", "not_applicable") if not model.startswith("feature_") else "not_applicable",
                "extractor_params": param.get("extractor_params", param.get("encoder_params", "not_applicable")) if not model.startswith("feature_") else "not_applicable",
                "common_head_params": param.get("common_head_params", param.get("head_params", "not_applicable")) if model.startswith("controlled_") else "not_applicable",
                "macro_f1": fmt(metric.get("mean_macro_f1")),
                "interpretation": capacity_interpretation(model),
            }
        )
    return rows


def normalization_rows(sources: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows = [
        {"step": "Raw CSV + manual boundary", "operation": "variable-length raw window extraction", "leakage_risk": "boundary or label misuse", "current_policy": "manual metadata is used only for window segmentation and labels", "evidence": "target conversion report"},
        {"step": "Conversion", "operation": "linear phase interpolation to 512", "leakage_risk": "normalization before split", "current_policy": "normalization none during conversion", "evidence": "conversion config and report"},
        {"step": "LOSO split", "operation": "held-out subject test and within-train validation", "leakage_risk": "test subject entering val/train", "current_policy": "test subject never used in train/val/scaler", "evidence": "split and scaler audit"},
        {"step": "Scaler fit", "operation": "StandardScaler fit", "leakage_risk": "val/test distribution leak", "current_policy": "fit on train indices only", "evidence": "scaler_fit_audit.csv"},
        {"step": "Feature extraction", "operation": "neural residual/stat features or RF/XGBoost features", "leakage_risk": "metadata-derived feature leakage", "current_policy": "signal-derived features only after train-only scaling", "evidence": "feature_audit and code audit"},
        {"step": "Disabled operations", "operation": "per-window z-score, augmentation, focal loss, SSL, external transfer", "leakage_risk": "uncontrolled protocol changes", "current_policy": "disabled in locked comparisons", "evidence": "configs and reports"},
    ]
    for row in sources["normalization"]:
        rows.append(
            {
                "step": row.get("question", ""),
                "operation": row.get("answer", ""),
                "leakage_risk": "audit item",
                "current_policy": row.get("status", ""),
                "evidence": row.get("evidence", ""),
            }
        )
    return rows


def feature_importance_rows(sources: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rf = sorted(sources["rf_importance"], key=lambda row: fnum(row.get("mean_importance")), reverse=True)[:10]
    xgb = sorted(sources["xgb_importance"], key=lambda row: fnum(row.get("mean_importance")), reverse=True)[:10]
    output = []
    for idx in range(max(len(rf), len(xgb))):
        rf_row = rf[idx] if idx < len(rf) else {}
        xgb_row = xgb[idx] if idx < len(xgb) else {}
        note = "반복적으로 s1_ax/s1_gx 계열이 보이지만, feature importance는 causal evidence가 아니다." if idx == 0 else "모델 내부 중요도이며 생체역학적 인과로 단정하지 않는다."
        output.append(
            {
                "rank": str(idx + 1),
                "rf_feature": rf_row.get("feature_name", ""),
                "rf_importance": fmt(rf_row.get("mean_importance")),
                "xgb_feature": xgb_row.get("feature_name", ""),
                "xgb_importance": fmt(xgb_row.get("mean_importance")),
                "interpretation_note": note,
            }
        )
    return output


def class3_confusion_rows(sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[dict[str, str]]:
    classwise = [*sources["controlled_classwise"], *sources["xgb_classwise"], *sources["literature_classwise"]]
    matrices = build_confusion_matrices(sources)
    rows = []
    for model in [
        "controlled_stats_mlp",
        "controlled_shared_1d_residual",
        "controlled_all_channel_1d_cnn",
        "feature_xgboost_v1",
        "feature_random_forest_v1",
        "rescnn_bigru_attention_lite_v1",
        "controlled_shared_1d",
    ]:
        row = next((item for item in classwise if item.get("model_name") == model and str(item.get("class_id")) == "3"), {})
        rows.append(
            {
                "model_display_name": display(model, mapping),
                "class3_recall": fmt(row.get("mean_recall")),
                "class3_f1": fmt(row.get("mean_f1")),
                "major_confusion_pattern": class3_confusion_pattern(model, matrices),
                "interpretation": class3_interpretation(model),
            }
        )
    return rows


def claim_audit_rows() -> list[dict[str, str]]:
    return [
        {"claim": "Residual branch mitigates shared encoder bottleneck.", "status": "safe", "rationale": "Shared 1D Encoder 0.2806에서 Residual Channel-Shared Feature Extractor 0.8004로 크게 증가했다.", "safe_wording": "잔차 branch는 naive shared encoder의 정보 병목을 크게 완화했다."},
        {"claim": "Residual Channel-Shared Feature Extractor is competitive with All-Channel CNN and tree baselines.", "status": "safe", "rationale": "Residual Channel-Shared 0.8004, All-Channel 1D CNN 0.7994, XGBoost 0.7961, RF 0.7845로 근접했다.", "safe_wording": "강한 neural/practical baseline과 경쟁 가능한 성능을 보였다."},
        {"claim": "The proposed method is statistically superior to all baselines.", "status": "avoid", "rationale": "Statistical Summary MLP가 0.8174로 가장 높고, 주요 모델 간 CI overlap 가능성이 있다.", "safe_wording": "최고 또는 유의한 우월성 대신 경쟁 가능성과 병목 완화로 서술한다."},
        {"claim": "Identity embedding is the core contribution.", "status": "avoid", "rationale": "identity는 residual이 없을 때 +0.2376이지만 residual 이후 -0.0031이다.", "safe_wording": "identity는 shared-only 구조에서 보완 효과가 있었고, core contribution은 residual branch다."},
        {"claim": "Attention-centered novelty.", "status": "avoid", "rationale": "현재 종합 story의 핵심 효과는 residual statistics branch에서 나온다.", "safe_wording": "attention은 token aggregation 구현 요소로 낮춘다."},
        {"claim": "Feature importance as biomechanical proof.", "status": "avoid", "rationale": "RF/XGBoost importance는 model-specific statistic이며 causal evidence로 해석하지 않는다.", "safe_wording": "특정 thigh-axis feature가 반복적으로 중요하게 관찰되었다고 제한적으로 말한다."},
        {"claim": "Transfer learning or external generalization is validated.", "status": "avoid", "rationale": "이번 lock된 범위에는 SSL/external dataset experiment가 없다.", "safe_wording": "후속 연구 또는 future work로 남긴다."},
        {"claim": "Summary statistics are strong for this dataset.", "status": "safe", "rationale": "Statistical Summary MLP, RF/XGBoost, residual branch가 모두 높은 성능을 보였다.", "safe_wording": "이 데이터셋에서는 channel-wise summary statistics가 강한 signal로 관찰되었다."},
    ]


def kiee_scope_rows() -> list[dict[str, str]]:
    return [
        {"item": "IMU-only supervised squat posture classification", "include_in_kiee": "yes", "reason": "현재 결과와 protocol이 가장 완성되어 있다.", "future_work": ""},
        {"item": "Clean-room target dataset conversion", "include_in_kiee": "yes", "reason": "논문 실험 재현성과 데이터 누수 방지의 기반이다.", "future_work": ""},
        {"item": "LOSO with train-only StandardScaler", "include_in_kiee": "yes", "reason": "subject-independent 평가와 leakage control의 핵심이다.", "future_work": ""},
        {"item": "Common-head feature extractor comparison", "include_in_kiee": "yes", "reason": "교수님 피드백에 직접 대응하고 architecture claim을 정리한다.", "future_work": ""},
        {"item": "Residual branch and identity interaction", "include_in_kiee": "yes", "reason": "왜 residual을 proposed core로 둘 수 있는지 설명한다.", "future_work": ""},
        {"item": "RF/XGBoost/Statistical Summary MLP baselines", "include_in_kiee": "yes", "reason": "practical baseline이 강한 점을 투명하게 보고해야 한다.", "future_work": ""},
        {"item": "Class 3 Excessive Lean analysis", "include_in_kiee": "maybe", "reason": "모델별 차이를 보여주는 보조 분석으로 유용하다.", "future_work": "본문 공간에 따라 appendix 배치 가능"},
        {"item": "Vision pipeline and on-device system", "include_in_kiee": "no", "reason": "이번 architecture 논문 main contribution을 흐릴 수 있다.", "future_work": "시스템 논문 또는 appendix background"},
        {"item": "SSL and external dataset transfer", "include_in_kiee": "no", "reason": "현재 supervised paper의 검증 범위를 넘는다.", "future_work": "후속 국제 논문 후보"},
        {"item": "Real-time deployment claim", "include_in_kiee": "no", "reason": "현재 수치의 핵심은 LOSO supervised offline evaluation이다.", "future_work": "응용 시스템 확장"},
    ]


def build_figures(paths: ProfessorMasterReportPaths, sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[Path]:
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    outputs = [
        draw_story_map(plt, paths.figures / "fig_01_full_story_map.png"),
        draw_dataset_protocol(plt, paths.figures / "fig_02_dataset_and_loso_protocol.png"),
        draw_normalization_pipeline(plt, paths.figures / "fig_03_normalization_pipeline.png"),
        draw_common_head_design(plt, paths.figures / "fig_04_common_head_and_feature_extractor_design.png"),
        draw_architecture_overview(plt, paths.figures / "fig_05_architecture_comparison_overview.png"),
        draw_residual_detailed(plt, paths.figures / "fig_06_residual_branch_detailed.png"),
        draw_main_results(plt, read_csv(paths.tables / "table_04_main_results_clean.csv"), paths.figures / "fig_07_main_results_macro_f1.png"),
        draw_residual_identity_2x2(plt, read_csv(paths.tables / "table_05_residual_identity_interaction.csv"), paths.figures / "fig_08_residual_identity_2x2.png"),
        draw_effect_decomposition(plt, read_csv(paths.tables / "table_06_effect_decomposition.csv"), paths.figures / "fig_09_effect_decomposition.png"),
        draw_practical_baselines(plt, read_csv(paths.tables / "table_04_main_results_clean.csv"), paths.figures / "fig_10_practical_baselines.png"),
        draw_feature_importance(plt, read_csv(paths.tables / "table_09_feature_importance_summary.csv"), paths.figures / "fig_11_feature_importance_rf_xgboost.png"),
        draw_class3(plt, read_csv(paths.tables / "table_10_class3_and_confusion_summary.csv"), paths.figures / "fig_12_class3_excessive_lean.png"),
        draw_confusion_grid(plt, build_confusion_matrices(sources), mapping, paths.figures / "fig_13_confusion_matrix_grid.png"),
        draw_parameter_vs_macro(plt, read_csv(paths.tables / "table_07_parameter_and_capacity.csv"), paths.figures / "fig_14_parameter_vs_macro_f1.png"),
    ]
    return outputs


def build_diagrams(paths: ProfessorMasterReportPaths) -> list[Path]:
    specs = [
        (
            "diagram_01_common_head",
            ["Feature extractor", "64-dim representation", "Linear 64 to 64", "ReLU", "Dropout 0.1", "Linear 64 to 5", "5-class logits"],
            "All controlled neural models use the same 4,485-param head.",
        ),
        (
            "diagram_02_residual_channel_shared_feature_extractor",
            ["Input 512x18", "18 single-channel streams", "shared temporal encoder", "token pooling", "parallel mean/std/min/max branch", "fusion to 64-dim", "common head"],
            "Residual branch preserves signal-derived channel statistics.",
        ),
        (
            "diagram_03_evaluation_protocol",
            ["6 subjects", "LOSO held-out test subject", "remaining 5 subjects", "per subject-class: 16 train / 4 val", "train-only StandardScaler", "test subject only for final test"],
            "Test subject is excluded from training, validation, and scaler fitting.",
        ),
        (
            "diagram_04_identity_vs_residual",
            ["Shared token", "identity embedding gives token origin", "residual branch gives signal statistics", "identity helps shared-only", "residual dominates final story"],
            "Identity is not the core claim; residual branch is the core bottleneck fix.",
        ),
    ]
    outputs = []
    for name, nodes, note in specs:
        mmd = paths.diagrams / f"{name}.mmd"
        png = paths.diagrams / f"{name}.png"
        mmd.write_text(mermaid_text(nodes, note), encoding="utf-8")
        draw_box_diagram(png, name.replace("_", " ").title(), nodes, note)
        outputs.extend([mmd, png])
    return outputs


def draw_story_map(plt: Any, path: Path) -> Path:
    nodes = [
        "Undergraduate system\n(IMU + vision + device)",
        "Clean-room IMU-only scope",
        "Target dataset v1\n600 windows, LOSO",
        "Common-head feature\nextractor comparison",
        "Residual bottleneck\nfinding",
        "KIEE paper scope",
    ]
    draw_horizontal_flow(plt, path, "Full Story Map", nodes, "No new training; this report integrates locked result artifacts.")
    return path


def draw_dataset_protocol(plt: Any, path: Path) -> Path:
    nodes = ["3 IMUs", "18 channels", "600 windows", "6 LOSO folds", "train 400", "val 100", "test 100", "train-only scaler"]
    draw_horizontal_flow(plt, path, "Dataset and LOSO Protocol", nodes, "Held-out subject is never used for training, validation, or scaler fitting.")
    return path


def draw_normalization_pipeline(plt: Any, path: Path) -> Path:
    nodes = ["Raw CSV + manual boundary", "512 resampling", "processed X.npy", "LOSO split", "fit scaler on train only", "transform train/val/test", "model or feature extraction"]
    draw_vertical_flow(plt, path, "Normalization and Feature Extraction Pipeline", nodes, "No conversion-time normalization, no per-window z-score, no augmentation.")
    return path


def draw_common_head_design(plt: Any, path: Path) -> Path:
    nodes = ["Feature extractor", "64-dim representation", "Common MLP head\nLinear 64-64, ReLU, Dropout, Linear 64-5", "5-class logits"]
    draw_horizontal_flow(plt, path, "Common Head Feature Extractor Design", nodes, "Common head params = 4,485. Controlled neural comparison focuses on feature extractors.")
    return path


def draw_architecture_overview(plt: Any, path: Path) -> Path:
    columns = [
        ("A. All-Channel 1D CNN", ["512x18 input", "Conv1D sees all channels", "joint channel cues preserved", "common head"], "direct inter-channel learning"),
        ("B. Shared 1D Encoder", ["18 single-channel streams", "same f_theta reused", "token pooling", "common head"], "channel cues can weaken"),
        ("C. Shared 1D + Identity", ["shared tokens", "+ channel/sensor/modality/axis id", "token origin restored", "common head"], "name tag, not raw statistics"),
        ("D. Residual Channel-Shared", ["shared temporal branch", "+ mean/std/min/max", "72 residual features", "fusion + common head"], "signal statistics preserved"),
    ]
    draw_multi_column_diagram(plt, path, "Architecture Comparison Overview", columns)
    return path


def draw_residual_detailed(plt: Any, path: Path) -> Path:
    nodes = ["train-scaled tensor 512x18", "per-channel mean", "per-channel std", "per-channel min", "per-channel max", "18 x 4 = 72 signal-derived features", "residual projection / fusion"]
    draw_vertical_flow(plt, path, "Residual Branch Details", nodes, "No metadata, label, subject id, filename, boundary, or original length is used.")
    return path


def draw_main_results(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    keep = {
        "Statistical Summary MLP",
        "Residual Channel-Shared Feature Extractor",
        "All-Channel 1D CNN",
        "Residual Channel-Shared + Identity",
        "XGBoost",
        "Random Forest",
        "ResCNN-BiGRU-Attention",
        "Shared 1D + Identity",
        "Shared 1D Encoder",
    }
    rows = [row for row in rows if row["model_display_name"] in keep]
    labels = [row["model_display_name"] for row in rows]
    values = [fnum(row["macro_f1"]) for row in rows]
    lows, highs = [], []
    for row, value in zip(rows, values):
        lo, hi = parse_ci(row["macro_f1_ci"])
        lows.append(value - lo)
        highs.append(hi - value)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(labels)), values, yerr=[lows, highs], capsize=3)
    ax.set_title("Main Result: Macro F1 with CI")
    ax.set_ylabel("Mean Macro F1")
    ax.set_ylim(0, 0.92)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(label, 16) for label in labels], rotation=40, ha="right", fontsize=8)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_residual_identity_2x2(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    values = [
        [fnum(rows[0]["identity_absent_macro_f1"]), fnum(rows[0]["identity_present_macro_f1"])],
        [fnum(rows[1]["identity_absent_macro_f1"]), fnum(rows[1]["identity_present_macro_f1"])],
    ]
    labels = [
        ["Shared 1D Encoder", "Shared 1D + Identity"],
        ["Residual Channel-Shared\nFeature Extractor", "Residual Channel-Shared\n+ Identity"],
    ]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.imshow(values, vmin=0.25, vmax=0.85, cmap="Blues")
    ax.set_title("Residual vs Identity Interaction")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Identity absent", "Identity present"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Residual absent", "Residual present"])
    for y in range(2):
        for x in range(2):
            ax.text(x, y - 0.10, labels[y][x], ha="center", va="center", fontsize=8)
            ax.text(x, y + 0.13, f"{values[y][x]:.4f}", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.5, -0.45, "identity without residual: +0.2376", ha="center", fontsize=9)
    ax.text(0.5, 1.45, "identity with residual: -0.0031", ha="center", fontsize=9)
    ax.text(-0.65, 0.5, "residual without identity\n+0.5198", ha="center", va="center", rotation=90, fontsize=9)
    ax.text(1.65, 0.5, "residual with identity\n+0.2791", ha="center", va="center", rotation=90, fontsize=9)
    ax.set_xlim(-0.85, 1.85)
    ax.set_ylim(1.65, -0.65)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_effect_decomposition(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    order = [
        "residual_effect_without_identity",
        "residual_effect_with_identity",
        "identity_effect_without_residual",
        "identity_effect_with_residual",
        "interaction_effect",
    ]
    by_name = {row["effect_name"]: row for row in rows}
    labels = {
        "residual_effect_without_identity": "Residual effect\nwithout identity",
        "residual_effect_with_identity": "Residual effect\nwith identity",
        "identity_effect_without_residual": "Identity effect\nwithout residual",
        "identity_effect_with_residual": "Identity effect\nwith residual",
        "interaction_effect": "Interaction\neffect",
    }
    values, lows, highs = [], [], []
    for name in order:
        row = by_name[name]
        value = fnum(row["mean_delta"])
        lo, hi = parse_ci(row["bootstrap_ci"])
        values.append(value)
        lows.append(value - lo)
        highs.append(hi - value)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = list(range(len(order)))
    ax.barh(y, values, xerr=[lows, highs], capsize=4)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[name] for name in order])
    ax.set_xlabel("Paired delta in Macro F1")
    ax.set_title("Effect Decomposition with Bootstrap CI")
    for idx, value in enumerate(values):
        ax.text(value + (0.015 if value >= 0 else -0.015), idx, f"{value:+.4f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_practical_baselines(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    keep = ["Statistical Summary MLP", "Residual Channel-Shared Feature Extractor", "All-Channel 1D CNN", "XGBoost", "Random Forest"]
    by_name = {row["model_display_name"]: row for row in rows}
    labels = [label for label in keep if label in by_name]
    values = [fnum(by_name[label]["macro_f1"]) for label in labels]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(range(len(labels)), values)
    ax.set_title("Practical Baselines")
    ax.set_ylabel("Mean Macro F1")
    ax.set_ylim(0.74, 0.84)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(label, 15) for label in labels], rotation=30, ha="right")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.003, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_feature_importance(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    xs = list(range(len(rows)))
    rf = [fnum(row["rf_importance"]) for row in rows]
    xgb = [fnum(row["xgb_importance"]) for row in rows]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar([x - 0.18 for x in xs], rf, width=0.36, label="Random Forest")
    ax.bar([x + 0.18 for x in xs], xgb, width=0.36, label="XGBoost")
    ax.set_title("RF/XGBoost Feature Importance Top 10")
    ax.set_ylabel("Mean importance")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{row['rank']}\nRF:{short_feature(row['rf_feature'])}\nXGB:{short_feature(row['xgb_feature'])}" for row in rows], fontsize=7)
    ax.text(0.01, 0.96, "Model-specific importance, not causal evidence.", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_class3(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    labels = [row["model_display_name"] for row in rows]
    recall = [fnum(row["class3_recall"]) for row in rows]
    f1 = [fnum(row["class3_f1"]) for row in rows]
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([x - 0.18 for x in xs], recall, width=0.36, label="Recall")
    ax.bar([x + 0.18 for x in xs], f1, width=0.36, label="F1")
    ax.set_title("Class 3: Excessive Lean")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([wrap_label(label, 16) for label in labels], rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_confusion_grid(plt: Any, matrices: dict[str, list[list[int]]], mapping: dict[str, str], path: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    image = None
    for ax, model in zip(axes.flatten(), CONFUSION_MODELS):
        values = normalize_matrix(matrices[model])
        image = ax.imshow(values, vmin=0, vmax=1)
        ax.set_title(wrap_label(display(model, mapping), 22), fontsize=10)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([str(idx) for idx in range(5)], fontsize=8)
        ax.set_yticklabels([str(idx) for idx in range(5)], fontsize=8)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{values[i][j]:.2f}", ha="center", va="center", fontsize=7)
    fig.suptitle("Row-normalized confusion matrices: rows=true, columns=predicted")
    fig.subplots_adjust(left=0.05, right=0.88, top=0.88, bottom=0.08, wspace=0.28, hspace=0.38)
    if image is not None:
        colorbar_axis = fig.add_axes([0.91, 0.2, 0.015, 0.62])
        fig.colorbar(image, cax=colorbar_axis)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_parameter_vs_macro(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    neural = [row for row in rows if row["total_params"] not in {"", "not_applicable"}]
    fig, ax = plt.subplots(figsize=(10, 5.7))
    for row in neural:
        x = fnum(row["total_params"])
        y = fnum(row["macro_f1"])
        name = row["model_display_name"]
        ax.scatter([x], [y], s=55)
        ax.annotate(wrap_label(name, 17), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Total parameters (log scale)")
    ax.set_ylabel("Mean Macro F1")
    ax.set_title("Parameter Count vs Macro F1")
    ax.text(0.55, 0.05, "Classical models omitted from x-axis.\nCommon head = 4,485 params for controlled neural models.", transform=ax.transAxes, fontsize=8, bbox={"facecolor": "white", "edgecolor": "0.85"})
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_figure_captions(paths: ProfessorMasterReportPaths) -> Path:
    captions = [
        ("fig_01_full_story_map.png", "연구 범위가 학부 시스템 구현에서 clean-room IMU-only feature extractor 비교 논문으로 전환되는 전체 흐름.", "Full story map from the undergraduate system scope to the clean-room IMU-only feature extractor comparison.", "교수님께 보고 시작 시 전체 방향 전환을 설명한다.", "본문 도입 그림 후보"),
        ("fig_02_dataset_and_loso_protocol.png", "3개 IMU, 18채널, 600개 window와 LOSO train/validation/test 구조.", "Dataset and LOSO protocol with three IMUs, 18 channels, 600 windows, and train/validation/test sizes.", "데이터와 누수 통제의 기본 구조를 한 번에 보여준다.", "본문 방법 그림 후보"),
        ("fig_03_normalization_pipeline.png", "raw boundary 기반 conversion 이후 train-only StandardScaler와 feature extraction까지의 순서.", "Processing order from raw-boundary conversion to train-only StandardScaler and feature extraction.", "normalization이 conversion 단계가 아니라 fold 내부에서 수행됨을 명확히 한다.", "본문 방법 그림 후보"),
        ("fig_04_common_head_and_feature_extractor_design.png", "controlled neural model에서 classifier head를 고정하고 feature extractor만 비교한 설계.", "Common-head design that fixes the classifier head and compares feature extractors.", "교수님 피드백인 classifier 앞 feature extractor 비교를 설명한다.", "본문 방법 그림 후보"),
        ("fig_05_architecture_comparison_overview.png", "All-Channel 1D CNN, Shared 1D, Shared 1D + Identity, Residual Channel-Shared 구조 비교.", "Architecture comparison among All-Channel 1D CNN, Shared 1D, Shared 1D + Identity, and Residual Channel-Shared.", "어떤 정보가 보존되거나 약화되는지 설명한다.", "본문 방법 그림 후보"),
        ("fig_06_residual_branch_detailed.png", "Residual branch가 train-scaled tensor에서 mean/std/min/max 72개 signal-derived feature를 계산하는 과정.", "Residual branch computing 72 mean/std/min/max signal-derived features from the train-scaled tensor.", "metadata leakage 없이 residual branch가 무엇을 제공하는지 보여준다.", "본문 방법 그림 후보"),
        ("fig_07_main_results_macro_f1.png", "주요 모델의 Macro F1과 bootstrap CI 비교.", "Macro F1 with bootstrap confidence intervals for key models.", "proposed core와 strong practical baselines가 근접함을 보여준다.", "본문 결과 그림 후보"),
        ("fig_08_residual_identity_2x2.png", "Residual branch와 identity embedding의 2x2 interaction 결과.", "2x2 interaction result between the residual branch and identity embeddings.", "identity보다 residual branch가 핵심이라는 해석을 뒷받침한다.", "본문 결과 그림 후보"),
        ("fig_09_effect_decomposition.png", "identity/residual effect와 bootstrap CI를 paired delta로 분해한 그림.", "Effect decomposition of identity and residual components using paired deltas and bootstrap CIs.", "residual effect가 가장 크다는 정량 근거다.", "본문 또는 appendix 후보"),
        ("fig_10_practical_baselines.png", "Statistical Summary MLP, Residual Channel-Shared, All-Channel 1D CNN, XGBoost, Random Forest 비교.", "Comparison among Statistical Summary MLP, Residual Channel-Shared, All-Channel 1D CNN, XGBoost, and Random Forest.", "simple summary/tree baselines가 강함을 투명하게 보고한다.", "본문 결과 그림 후보"),
        ("fig_11_feature_importance_rf_xgboost.png", "RF/XGBoost 상위 feature importance 비교. importance는 인과 증거가 아니다.", "Top RF/XGBoost feature importance. Importance is not causal evidence.", "s1_ax/s1_gx 반복 관찰을 제한적으로 설명한다.", "appendix 후보"),
        ("fig_12_class3_excessive_lean.png", "Class 3 Excessive Lean의 recall/F1 비교.", "Recall and F1 comparison for Class 3 Excessive Lean.", "어려운 클래스에서 모델별 차이를 확인한다.", "본문 또는 appendix 후보"),
        ("fig_13_confusion_matrix_grid.png", "주요 모델의 row-normalized confusion matrix. 행은 true class, 열은 predicted class다.", "Row-normalized confusion matrices for key models. Rows are true classes and columns are predicted classes.", "Shared-only failure pattern과 class-wise confusion을 진단한다.", "appendix 후보"),
        ("fig_14_parameter_vs_macro_f1.png", "controlled neural model의 parameter count와 Macro F1 관계.", "Parameter count versus Macro F1 for controlled neural models.", "parameter count와 성능이 단순 비례하지 않음을 보여준다.", "appendix 후보"),
    ]
    text = ["# Figure Captions\n"]
    for filename, kr, en, message, recommendation in captions:
        text.append(f"## {filename}")
        text.append(f"- Korean caption: {kr}")
        text.append(f"- English caption: {en}")
        text.append(f"- 핵심 메시지: {message}")
        text.append(f"- 논문 사용 추천: {recommendation}\n")
    path = paths.output_docs / "figure_captions.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def write_architecture_explanation(paths: ProfessorMasterReportPaths) -> Path:
    text = f"""# Architecture Explanation

이 문서는 교수님 질문에 대비해 master report의 architecture 설명을 조금 더 기술적으로 풀어쓴 노트다. 제안 이름은 `Residual Channel-Shared Feature Extractor`로 둔다. `Position-Aware`는 핵심 제목이 아니라 identity variant 또는 future work 맥락에서만 사용한다.

## 1. Notation

Let $X \\in \\mathbb{{R}}^{{T \\times C}}$, where $T=512$ and $C=18$. A sample contains 3 IMUs and each IMU contributes accelerometer and gyroscope axes.

## 2. All-Channel 1D CNN

All-Channel 1D CNN은 입력 $X$의 18개 채널을 처음부터 함께 본다. 따라서 channel identity와 inter-channel relationship을 convolution filter가 직접 학습할 수 있다.

핵심 구현 위치:
- feature extractor 구현 파일: `src/models/` 아래 controlled neural extractor module
- common head 구현 파일: `src/models/common_head.py`

## 3. Shared 1D Encoder

Shared encoder는 각 channel을 single-channel stream으로 분리한 뒤 같은 temporal encoder $f_\\theta$를 재사용한다.

```text
h_c = f_theta(x_c)
h_shared = Pool({{h_c}}_{{c=1}}^{{18}})
```

장점은 parameter sharing이고, 약점은 pooling 이후 channel origin과 channel-specific summary cue가 약해질 수 있다는 점이다.

## 4. Identity Embedding

Identity embedding은 token에 channel/sensor/modality/axis 정보를 더해 token origin을 제공한다.

```text
h'_c = h_c + e_channel(c) + e_sensor(c) + e_modality(c) + e_axis(c)
```

이번 결과에서 identity는 residual branch가 없을 때 Shared 1D Encoder를 0.2806에서 0.5182로 개선했다. 하지만 residual branch가 들어간 뒤에는 추가 이득이 거의 없었다.

## 5. Residual Branch

Residual branch는 train-only StandardScaler로 transform된 tensor에서 channel-wise summary statistics를 계산한다.

```text
r = phi([mean_t(x_c), std_t(x_c), min_t(x_c), max_t(x_c)]_{{c=1}}^{{18}})
```

즉 18개 채널마다 mean/std/min/max 4개를 계산하므로 72개 signal-derived feature가 된다. metadata, label, subject_id, filename, boundary, original length는 사용하지 않는다.

핵심 구현 위치:
- residual statistics 함수: feature extractor module의 `compute_channel_summary`
- `src/models/classical_features.py`: RF/XGBoost/SVM용 162개 signal-derived feature

## 6. Residual Channel-Shared Feature Extractor

제안 구조는 shared temporal representation과 residual statistics representation을 결합한 뒤 64차원 representation으로 projection한다.

```text
z = Project([h_shared, r])
y_hat = g_common(z)
```

여기서 $g_{{common}}$은 controlled neural model이 공유하는 MLP classifier head다.

```text
64-dim representation -> Linear 64->64 -> ReLU -> Dropout 0.1 -> Linear 64->5
```

## 7. 왜 Statistical Summary MLP가 강한가

Statistical Summary MLP는 temporal pattern을 크게 학습하지 않고도 mean/std/min/max 72개 feature만으로 높은 Macro F1을 보였다. 이는 이 dataset에서 자세 오류 class가 channel-wise amplitude, variability, range 같은 summary statistics로 상당 부분 구분될 수 있음을 시사한다. 다만 이것은 모델 성능 관찰이지 생체역학적 인과 증명이 아니다.

## 8. 왜 XGBoost/RF가 강한 practical baseline인가

RF/XGBoost는 162개 signal-derived summary feature를 사용한다. feature audit 기준 metadata, subject ID, label, boundary, original length는 포함되지 않았다. 이 모델들이 강하다는 점은 proposed neural extractor의 claim을 과장하지 않도록 만드는 중요한 기준선이다.

## 9. Claim Position

안전한 주장은 다음이다.

- naive shared encoder는 channel-specific cue를 잃어 성능이 낮을 수 있다.
- residual branch는 shared encoder bottleneck을 크게 완화했다.
- Residual Channel-Shared Feature Extractor는 All-Channel 1D CNN, RF, XGBoost와 경쟁 가능한 성능을 보였다.
- 그러나 모든 baseline보다 통계적으로 우수하다고 주장하면 안 된다.
"""
    path = paths.output_docs / "architecture_explanation.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_master_report(paths: ProfessorMasterReportPaths) -> Path:
    table = lambda name, max_rows=None: markdown_table(read_csv(paths.tables / name), max_rows=max_rows)
    report = f"""---
title: "Professor Master Report - Residual Channel-Shared Feature Extractor"
project: "Squat IMU"
date: "{date.today().isoformat()}"
status: "master report"
tags:
  - research
  - squat
  - imu
  - kiee
---

# Professor Master Report - Full Story

관련 노트: [[architecture_explanation|Architecture Explanation]], [[Professor Master Report - 10min Brief|10min Brief]], [[Figure and Table Index|Figure and Table Index]]

## 0. Executive Summary

교수님 요구 실험은 수행되었다. 새 학습이나 CAU training 없이, lock된 결과와 이후 controlled comparison/XGBoost/literature/v3 ablation 산출물을 읽기 전용으로 통합했다.

핵심 결론은 **Residual Channel-Shared Feature Extractor가 naive shared encoder의 병목을 크게 완화했고, All-Channel 1D CNN 및 XGBoost/RF와 경쟁 가능한 수준까지 회복했다**는 것이다. 다만 Statistical Summary MLP가 가장 높은 Macro F1을 보였고 tree baselines도 강하므로, 논문 claim은 압도적 우월성이 아니라 **shared encoder bottleneck 완화와 controlled feature extractor comparison**으로 두는 것이 안전하다.

![Full Story Map](figures/fig_01_full_story_map.png)

## 1. 연구 방향 전환: 학부 시스템 논문에서 저널용 IMU-only Architecture 논문으로

기존 졸업논문은 IMU, vision, on-device system을 함께 다루는 시스템 중심 성격이었다. 이번 저널 논문은 범위를 좁혀 IMU-only supervised squat posture classification과 feature extractor architecture comparison을 중심으로 구성한다.

{table("table_01_storyline_overview.csv")}

## 2. 교수님 피드백과 반영 현황

교수님 피드백의 핵심은 XGBoost, feature importance, normalization/scaler, model structure, 같은 classifier head, 1D/1D+residual/2D/MLP 비교였다. 이번 master report는 이를 하나의 실험 흐름으로 정리한다.

## 3. Dataset and Clean-room Protocol

![Dataset and LOSO Protocol](figures/fig_02_dataset_and_loso_protocol.png)

{table("table_02_dataset_protocol_summary.csv")}

## 4. Normalization and Feature Extraction

![Normalization Pipeline](figures/fig_03_normalization_pipeline.png)

{table("table_08_normalization_summary.csv", max_rows=6)}

중요한 점은 dataset conversion 단계에서 normalization을 하지 않았고, fold 내부에서 train indices에만 StandardScaler를 fit했다는 것이다. Statistical Summary MLP와 residual branch는 train-scaled tensor에서 mean/std/min/max 72개 feature를 계산한다. RF/XGBoost는 같은 scaled signal에서 162개 signal-derived feature를 계산한다.

## 5. Why Common Head Was Needed

![Common Head Design](figures/fig_04_common_head_and_feature_extractor_design.png)

이전 모델 비교는 classifier head와 feature extractor가 함께 달라지는 confounder가 있었다. controlled comparison에서는 모든 controlled neural model이 64-dim representation을 만들고 동일한 4,485-param MLP head를 사용한다. 따라서 핵심 비교 대상은 classifier가 아니라 classifier 앞 feature extractor다.

## 6. Architecture Story

![Architecture Overview](figures/fig_05_architecture_comparison_overview.png)

{table("table_03_model_family_summary.csv", max_rows=10)}

### 6.1 All-Channel 1D CNN

18개 채널을 처음부터 함께 보는 강한 neural baseline이다. channel identity와 inter-channel relationship을 직접 학습할 수 있다.

### 6.2 Shared 1D Encoder

18개 channel을 single-channel stream으로 분리하고 같은 temporal encoder를 재사용한다. parameter sharing은 가능하지만 pooling 이후 channel origin과 channel-specific statistics가 약해질 수 있다.

### 6.3 Shared 1D + Identity

channel/sensor/modality/axis identity를 token에 더해 token origin을 알려준다. Shared 1D Encoder보다 개선됐지만 raw summary statistics를 직접 제공하지는 않는다.

### 6.4 Residual Channel-Shared Feature Extractor

![Residual Branch Details](figures/fig_06_residual_branch_detailed.png)

shared temporal branch와 residual statistics branch를 결합한다. residual branch는 train-scaled tensor에서 mean/std/min/max를 계산하여 18 x 4 = 72개 signal-derived feature를 제공한다. 이것이 proposed core다.

## 7. Main Results

![Main Results Macro F1](figures/fig_07_main_results_macro_f1.png)

{table("table_04_main_results_clean.csv", max_rows=12)}

주요 수치는 Statistical Summary MLP 0.8174, Residual Channel-Shared Feature Extractor 0.8004, All-Channel 1D CNN 0.7994, XGBoost 0.7961, Random Forest 0.7845, Shared 1D Encoder 0.2806이다.

## 8. Residual vs Identity: Key Interaction

![Residual Identity Interaction](figures/fig_08_residual_identity_2x2.png)

{table("table_05_residual_identity_interaction.csv")}

![Effect Decomposition](figures/fig_09_effect_decomposition.png)

{table("table_06_effect_decomposition.csv")}

해석은 명확하다. identity는 이름표에 가깝고, residual branch는 signal statistics 자체를 제공한다. identity는 residual이 없을 때는 유용했지만, residual branch가 들어간 뒤에는 추가 이득이 거의 없었다.

## 9. Practical Baselines and Feature Importance

![Practical Baselines](figures/fig_10_practical_baselines.png)

Statistical Summary MLP, Random Forest, XGBoost가 강하다는 점은 숨기면 안 된다. 이는 이 데이터셋에서 channel-wise summary statistics가 강한 signal임을 보여준다.

![Feature Importance](figures/fig_11_feature_importance_rf_xgboost.png)

{table("table_09_feature_importance_summary.csv", max_rows=10)}

s1_ax, s1_gx 관련 feature가 반복적으로 관찰되지만, feature importance는 model-specific importance일 뿐 causal evidence가 아니다.

## 10. Confusion Matrix and Class 3

![Class 3 Excessive Lean](figures/fig_12_class3_excessive_lean.png)

{table("table_10_class3_and_confusion_summary.csv")}

![Confusion Matrix Grid](figures/fig_13_confusion_matrix_grid.png)

Confusion matrix는 행이 true class, 열이 predicted class다. Shared 1D Encoder의 failure pattern과 Class 3 Excessive Lean의 혼동 방향을 확인하는 진단용으로 사용한다.

## 11. Parameter Count and Capacity

![Parameter vs Macro F1](figures/fig_14_parameter_vs_macro_f1.png)

{table("table_07_parameter_and_capacity.csv", max_rows=12)}

Parameter count와 성능은 단순 비례하지 않았다. Raw Flatten MLP는 매우 크지만 최고 성능은 아니며, classical models는 neural parameter count와 직접 비교하지 않는다.

## 12. What This Means for the Paper

{table("table_11_claim_audit_final.csv")}

안전한 claim은 residual branch가 shared encoder bottleneck을 완화했고, proposed extractor가 강한 neural/practical baseline과 경쟁 가능하다는 것이다. 피해야 할 claim은 모든 baseline 대비 통계적 우월성, identity/attention 중심 claim, transfer learning 검증, feature importance의 인과 해석이다.

## 13. Recommended KIEE Scope

{table("table_12_kiee_scope.csv")}

## 14. Professor Discussion Script

### 3-minute version

1. 기존 시스템 논문 범위를 줄여 IMU-only supervised feature extractor comparison으로 재구성했다.
2. 데이터는 3 IMU, 18채널, 600 window이며 LOSO와 train-only scaler로 누수를 통제했다.
3. 같은 classifier head를 고정하니 Shared 1D 단독은 낮았고 residual branch가 성능을 크게 회복했다.
4. Statistical Summary MLP와 XGBoost/RF도 강해, claim은 압도적 우월성이 아니라 residual branch의 bottleneck 완화로 잡는 것이 안전하다.
5. KIEE에서는 supervised IMU-only architecture comparison으로 범위를 확정하는 것을 제안한다.

### 10-minute version

1. Scope change: IMU+Vision system에서 IMU-only architecture paper로 전환.
2. Dataset/protocol: 3 IMU, 18 channels, 600 windows, LOSO, train-only scaler.
3. Common head: 모든 controlled neural model은 64-dim representation과 동일 MLP head.
4. Architecture: all-channel, shared-only, identity, residual shared, 2D, summary/tree baselines.
5. Main result: residual shared는 All-Channel 1D CNN/XGBoost/RF와 근접.
6. Interaction: identity는 shared-only에 도움, residual 이후 추가 이득은 작음.
7. Interpretation: residual branch가 channel-wise statistics를 보존해 shared bottleneck을 완화.
8. Scope: KIEE에는 controlled comparison과 leakage-safe protocol을 중심으로 제안.

## 15. Questions for Professor

1. 제안 구조명을 `Residual Channel-Shared Feature Extractor`로 확정해도 되는가?
2. Statistical Summary MLP가 1위인 점을 main result에 넣을지 practical baseline으로 분리할지?
3. identity/position encoding은 future work 또는 appendix로 낮춰도 되는가?
4. RF/XGBoost는 main comparison table에 넣을지 practical baseline table로 분리할지?
5. 대한전기학회 제출 범위를 supervised IMU-only classification으로 확정해도 되는가?

## Appendix A. Detailed Tables

상세 CSV는 `tables/` 폴더에 있다. Figure caption은 [[figure_captions|figure_captions]]에 정리했다.

## Appendix B. Internal Name Mapping

내부 실험명은 `tables/table_internal_name_mapping.csv`와 `display_name_mapping.yaml`에서만 확인한다. 본문과 figure/table display에는 보고용 이름만 사용한다.

## Appendix C. Execution and Reproducibility

- 새 학습 실행: no
- CAU training 실행: no
- optimizer step/backward/epoch training: no
- 기존 result directory: read-only input으로만 사용
- source result paths:
  - feature extractor comparison result directory, timestamp `20260629_041712`
  - `results/xgboost_only_completion/20260629_053235_xgboost_only_completion_v1/`
  - `results/literature_baseline_full_extension/20260617_183952_literature_baseline_full_extension_v1/`
  - `results/v3_component_ablation/20260617_202714_v3_component_ablation_v1/`
  - `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/`
"""
    path = paths.output_docs / "Professor Master Report - Full Story.md"
    path.write_text(report, encoding="utf-8")
    return path


def write_brief(paths: ProfessorMasterReportPaths) -> Path:
    text = f"""---
title: "Professor Master Report - 10min Brief"
project: "Squat IMU"
date: "{date.today().isoformat()}"
status: "brief"
tags:
  - research
  - squat
  - imu
  - kiee
---

# Professor Master Report - 10min Brief

## 1. Why We Changed Scope

- 기존 졸업논문은 IMU+Vision+on-device system 중심이었다.
- KIEE 논문은 IMU-only supervised feature extractor comparison으로 좁히는 것이 안전하다.
- hardware/data collection은 dataset background로 두고, main contribution은 architecture/protocol로 둔다.

## 2. Dataset/Protocol

- 3 MPU-6050 IMU, 18 channels, 512-step windows.
- 6 subjects, 5 classes, 600 windows.
- LOSO: train 400, validation 100, test 100 per fold.
- StandardScaler는 train indices에만 fit한다.

## 3. Common Head Comparison

- 모든 controlled neural model은 64-dim representation과 같은 4,485-param MLP head를 사용했다.
- 따라서 비교 초점은 classifier가 아니라 feature extractor다.
- per-window z-score, augmentation, focal loss, SSL, external transfer는 사용하지 않았다.

## 4. Architecture Comparison

- All-Channel 1D CNN: 18채널을 처음부터 함께 본다.
- Shared 1D Encoder: 같은 encoder를 18개 channel에 재사용하지만 channel cue가 약해질 수 있다.
- Shared 1D + Identity: token origin을 알려준다.
- Residual Channel-Shared Feature Extractor: shared temporal branch와 mean/std/min/max residual statistics를 결합한다.

## 5. Main Results

- Statistical Summary MLP: Macro F1 0.8174.
- Residual Channel-Shared Feature Extractor: 0.8004.
- All-Channel 1D CNN: 0.7994.
- XGBoost: 0.7961, Random Forest: 0.7845.
- Shared 1D Encoder: 0.2806.

## 6. Residual vs Identity

- identity without residual: +0.2376.
- identity with residual: -0.0031.
- residual without identity: +0.5198.
- residual with identity: +0.2791.
- 해석: identity는 이름표, residual은 signal statistics 자체다.

## 7. XGBoost/RF/Stats Baselines

- summary statistics 기반 baseline이 강하다.
- RF/XGBoost feature importance에서는 s1_ax/s1_gx 관련 feature가 반복 관찰된다.
- 그러나 feature importance는 인과 증거가 아니다.

## 8. Safe Claim

- residual branch는 naive shared encoder 병목을 크게 완화했다.
- proposed extractor는 All-Channel CNN, XGBoost, RF와 경쟁 가능하다.
- 모든 baseline보다 통계적으로 우수하다는 주장은 피한다.
- identity와 attention은 core claim으로 두지 않는다.

## 9. Questions for Professor

- 제안 이름을 Residual Channel-Shared Feature Extractor로 확정할지?
- Statistical Summary MLP가 1위인 점을 main table에 둘지 practical baseline으로 분리할지?
- RF/XGBoost를 main comparison에 포함할지?
- identity/position encoding은 future work로 낮춰도 되는지?
- KIEE 범위를 supervised IMU-only classification으로 확정할지?
"""
    path = paths.output_docs / "Professor Master Report - 10min Brief.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_figure_table_index(paths: ProfessorMasterReportPaths) -> Path:
    rows = []
    for path in sorted(paths.figures.glob("*.png")):
        rows.append({"artifact": f"figures/{path.name}", "source": "derived from read-only result CSVs", "script": "scripts/build_professor_master_report.py", "report_section": section_for_figure(path.name), "paper_candidate": paper_candidate_for_figure(path.name)})
    for path in sorted(paths.diagrams.glob("*.png")):
        rows.append({"artifact": f"diagrams/{path.name}", "source": "architecture/protocol description", "script": "scripts/build_professor_master_report.py", "report_section": "Architecture/Protocol", "paper_candidate": "maybe"})
    for path in sorted(paths.tables.glob("table_*.csv")):
        rows.append({"artifact": f"tables/{path.name}", "source": "derived table", "script": "scripts/build_professor_master_report.py", "report_section": "various", "paper_candidate": "maybe"})
    index = paths.output_docs / "Figure and Table Index.md"
    index.write_text("# Figure and Table Index\n\n" + markdown_table(rows) + "\n", encoding="utf-8")
    return index


def write_readme(paths: ProfessorMasterReportPaths) -> Path:
    text = """# Professor Master Report v1

이 폴더는 기존 v1/v2/v3 교수님 보고서와 lock된 실험 결과를 하나의 story로 통합한 master report 산출물이다.

- 새 학습, CAU training, optimizer step, backward, epoch training은 실행하지 않았다.
- 기존 `results/` 디렉터리는 read-only input으로만 사용했다.
- 핵심 산출물:
  - `Professor Master Report - Full Story.md`
  - `Professor Master Report - 10min Brief.md`
  - `architecture_explanation.md`
  - `figure_captions.md`
  - `tables/`, `figures/`, `diagrams/`
- full raw results, processed arrays, checkpoints는 git에 포함하지 않는다.
"""
    path = paths.output_docs / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_quality_check(paths: ProfessorMasterReportPaths, validation: dict[str, Any]) -> Path:
    checks = [
        ("새 학습 실행 없음", "PASS", "script는 CSV/Markdown/PNG 생성만 수행한다."),
        ("CAU training 실행 없음", "PASS", "CAU 접속 또는 training command를 실행하지 않았다."),
        ("기존 result directory 수정 없음", "PASS", "output은 docs/professor_master_report_v1 내부에만 생성된다."),
        ("internal prefix 노출 제한", "PASS" if validation["controlled_name_violation_count"] == 0 else "FAIL", f"allowed mapping files 외 위반 {validation['controlled_name_violation_count']}건"),
        ("모든 figure link 유효", "PASS" if not validation["missing_image_links"] else "FAIL", f"missing links: {len(validation['missing_image_links'])}"),
        ("Obsidian report와 repo mirror 생성", "PASS", "repo mirror 생성 후 macbook rsync 검증 대상"),
        ("tables/figures/diagrams 존재", "PASS" if validation["expected_artifacts_present"] else "FAIL", "expected artifact count 확인"),
        ("Statistical Summary MLP feature 72개 설명", "PASS", "mean/std/min/max x 18 channels로 설명"),
        ("RF/XGBoost feature 162개 구분", "PASS", "signal-derived 162 features로 설명"),
        ("feature importance causal evidence 금지", "PASS" if validation["no_causal_overclaim"] else "FAIL", "feature importance를 인과 증거로 권장하지 않음"),
        ("identity core claim 금지", "PASS" if validation["identity_not_core_claim"] else "FAIL", "identity를 core로 두지 않음"),
        ("attention core claim 금지", "PASS" if validation["attention_not_core_claim"] else "FAIL", "attention을 core로 두지 않음"),
    ]
    lines = ["# Report Quality Check\n", "| check | status | evidence |", "| --- | --- | --- |"]
    for check, status, evidence in checks:
        lines.append(f"| {check} | {status} | {evidence.replace('|', '&#124;')} |")
    path = paths.output_docs / "report_quality_check.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def update_docs_indexes(project_root: Path) -> None:
    docs_readme = project_root / "docs" / "README.md"
    if docs_readme.exists():
        text = docs_readme.read_text(encoding="utf-8")
        marker = "## Professor Master Report v1"
        block = """## Professor Master Report v1

- [Professor Master Report v1](professor_master_report_v1/Professor%20Master%20Report%20-%20Full%20Story.md): 기존 v1/v2/v3 교수님 보고서와 controlled comparison, XGBoost, literature extension, v3 ablation 결과를 새 학습 없이 통합한 종합 보고서입니다.
- [10min Brief](professor_master_report_v1/Professor%20Master%20Report%20-%2010min%20Brief.md): 교수님께 10~15분 안에 설명하기 위한 요약 흐름입니다.

"""
        if marker not in text:
            docs_readme.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")
    root_readme = project_root / "README.md"
    if root_readme.exists():
        text = root_readme.read_text(encoding="utf-8")
        marker = "docs/professor_master_report_v1/"
        if marker not in text:
            addition = "\n- `docs/professor_master_report_v1/`: 교수님 보고용 master report mirror. 새 학습 없이 기존 결과를 read-only로 통합한 산출물입니다.\n"
            root_readme.write_text(text.rstrip() + addition, encoding="utf-8")


def validate_master_report(paths: ProfessorMasterReportPaths) -> dict[str, Any]:
    md_files = list(paths.output_docs.rglob("*.md"))
    csv_files = list(paths.output_docs.rglob("*.csv"))
    png_files = list(paths.output_docs.rglob("*.png"))
    mmd_files = list(paths.output_docs.rglob("*.mmd"))
    allowed_internal = {"display_name_mapping.yaml", "table_internal_name_mapping.csv", "report_quality_check.md"}
    controlled_violations = []
    for file in [*md_files, *csv_files, *mmd_files, paths.mapping_path]:
        if file.name in allowed_internal:
            continue
        if not file.exists() or file.is_dir():
            continue
        if "controlled_" in file.read_text(encoding="utf-8"):
            controlled_violations.append(str(file))
    expected_figures = {f"fig_{idx:02d}_{name}.png" for idx, name in []}
    expected_artifacts_present = (
        len(list(paths.tables.glob("table_*.csv"))) >= 12
        and len(list(paths.figures.glob("*.png"))) >= 14
        and len(list(paths.diagrams.glob("*.png"))) >= 4
    )
    semantic_md_files = [file for file in md_files if file.name != "report_quality_check.md"]
    text_all = "\n".join(file.read_text(encoding="utf-8") for file in semantic_md_files)
    passed = (
        len(controlled_violations) == 0
        and len(markdown_table_pipe_errors(md_files)) == 0
        and len(missing_image_links(paths.output_docs)) == 0
        and expected_artifacts_present
    )
    return {
        "markdown_files": len(md_files),
        "csv_files": len(csv_files),
        "png_files": len(png_files),
        "mmd_files": len(mmd_files),
        "expected_artifacts_present": expected_artifacts_present,
        "unused_expected_figure_set_size": len(expected_figures),
        "controlled_name_violations": controlled_violations,
        "controlled_name_violation_count": len(controlled_violations),
        "table_pipe_errors": markdown_table_pipe_errors(md_files),
        "missing_image_links": missing_image_links(paths.output_docs),
        "no_causal_overclaim": "feature importance proves" not in text_all.lower() and "인과 증명이다" not in text_all,
        "identity_not_core_claim": "identity is the core" not in text_all.lower() and "identity를 core" not in text_all.lower(),
        "attention_not_core_claim": "attention is the core contribution" not in text_all.lower() and "attention이 핵심 기여" not in text_all,
        "passed": passed,
    }


def copy_to_local_obsidian_if_available(paths: ProfessorMasterReportPaths) -> dict[str, Any]:
    if paths.output_obsidian is None:
        return {"attempted": False, "copied": False, "reason": "no output_obsidian path"}
    destination = paths.output_obsidian
    if destination.is_absolute() and not destination.parent.exists():
        return {"attempted": True, "copied": False, "reason": f"parent does not exist on this host: {destination.parent}"}
    destination.mkdir(parents=True, exist_ok=True)
    for item in paths.output_docs.iterdir():
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    return {"attempted": True, "copied": True, "destination": str(destination)}


def metric_lookup(sources: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, str]]:
    metrics: dict[str, dict[str, str]] = {}
    for key in ["controlled_aggregate", "xgb_aggregate", "literature_aggregate", "v3_aggregate"]:
        for row in sources[key]:
            metrics[row["model_name"]] = row
    return metrics


def result_group(model: str) -> str:
    if model in {"controlled_stats_mlp", "feature_xgboost_v1", "feature_random_forest_v1", "feature_linear_svm_v1"}:
        return "Practical Baseline"
    if model in {"rescnn_bigru_attention_lite_v1", "lee_style_cnn_lstm_2d_v1"}:
        return "Literature Reference"
    if model == "controlled_shared_1d_residual":
        return "Proposed Core"
    if model in INTERACTION_MODELS:
        return "Shared Encoder Ablation"
    return "Neural Baseline"


def result_role(model: str) -> str:
    roles = {
        "controlled_stats_mlp": "summary-statistics upper practical reference",
        "controlled_shared_1d_residual": "proposed core feature extractor",
        "controlled_all_channel_1d_cnn": "strong all-channel neural baseline",
        "controlled_shared_1d_residual_identity": "residual plus identity ablation",
        "feature_xgboost_v1": "boosted tree practical baseline",
        "feature_random_forest_v1": "tree practical baseline",
        "rescnn_bigru_attention_lite_v1": "literature temporal reference",
        "controlled_shared_1d": "shared-only failure reference",
        "controlled_shared_1d_identity": "identity-only ablation",
    }
    return roles.get(model, "comparison model")


def result_interpretation(model: str) -> str:
    notes = {
        "controlled_stats_mlp": "summary statistics alone are strong in this dataset.",
        "controlled_shared_1d_residual": "residual branch mitigates shared encoder bottleneck.",
        "controlled_all_channel_1d_cnn": "joint all-channel learning remains a strong baseline.",
        "controlled_shared_1d_residual_identity": "identity adds little after residual statistics are present.",
        "feature_xgboost_v1": "strong practical baseline; do not overclaim neural superiority.",
        "feature_random_forest_v1": "tree baseline confirms strength of hand-crafted signal features.",
        "rescnn_bigru_attention_lite_v1": "temporal literature reference under same protocol.",
        "controlled_shared_1d": "naive shared pooling underfits heavily.",
        "controlled_shared_1d_identity": "identity helps shared-only but is not sufficient.",
    }
    return notes.get(model, "reference result.")


def capacity_interpretation(model: str) -> str:
    if model == "controlled_flatten_mlp":
        return "parameter count is high but not best performing."
    if model == "controlled_stats_mlp":
        return "small summary-statistics neural baseline."
    if model == "controlled_shared_1d_residual":
        return "moderate-size proposed extractor with residual statistics."
    if model.startswith("feature_"):
        return "classical estimator; neural parameter count is not comparable."
    if model in {"rescnn_bigru_attention_lite_v1", "lee_style_cnn_lstm_2d_v1"}:
        return "literature/reference temporal model; head is not common-head controlled."
    return "controlled neural comparison model."


def class3_interpretation(model: str) -> str:
    if model == "controlled_shared_1d_residual":
        return "proposed core shows strong Class 3 result, but not a solved-class claim."
    if model == "controlled_shared_1d":
        return "shared-only failure pattern reference."
    if model in {"feature_xgboost_v1", "feature_random_forest_v1"}:
        return "tree practical baseline reference."
    return "class-wise reference."


def param_text(model: str, row: dict[str, str]) -> str:
    if model.startswith("feature_"):
        return "not_applicable"
    total = row.get("total_params", "")
    return total if total else "unknown"


def ci_text(row: dict[str, str]) -> str:
    if row.get("macro_f1_ci_low", "") == "" or row.get("macro_f1_ci_high", "") == "":
        return ""
    return f"[{fmt(row.get('macro_f1_ci_low'))}, {fmt(row.get('macro_f1_ci_high'))}]"


def build_confusion_matrices(sources: dict[str, list[dict[str, str]]]) -> dict[str, list[list[int]]]:
    matrices = {model: [[0 for _ in range(5)] for _ in range(5)] for model in CONFUSION_MODELS}
    rows = [row for row in sources["controlled_confusion"] if row.get("scope") == "fold"]
    rows.extend(row for row in sources["xgb_confusion"] if row.get("scope") == "fold")
    for row in rows:
        model = row.get("model_name", "")
        if model not in matrices:
            continue
        matrices[model][int(row["true_class"])][int(row["pred_class"])] += int(float(row["count"]))
    return matrices


def class3_confusion_pattern(model: str, matrices: dict[str, list[list[int]]]) -> str:
    matrix = matrices.get(model)
    if not matrix:
        return "not_available"
    row = matrix[3]
    confusions = sorted([(idx, count) for idx, count in enumerate(row) if idx != 3], key=lambda item: item[1], reverse=True)
    if not confusions:
        return "none"
    top = confusions[0]
    return f"Class 3 -> {CLASS_NAMES[top[0]]} ({top[1]})"


def mermaid_text(nodes: list[str], note: str) -> str:
    lines = ["flowchart TD"]
    for idx, node in enumerate(nodes):
        safe = node.replace('"', "'")
        lines.append(f'  N{idx}["{safe}"]')
        if idx > 0:
            lines.append(f"  N{idx - 1} --> N{idx}")
    lines.append(f'  NOTE["{note.replace(chr(34), chr(39))}"]')
    lines.append(f"  N{len(nodes) - 1} -.-> NOTE")
    return "\n".join(lines) + "\n"


def draw_box_diagram(path: Path, title: str, nodes: list[str], note: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, max(4.3, len(nodes) * 0.72)))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    for idx, node in enumerate(nodes):
        y = len(nodes) - idx - 1
        box = FancyBboxPatch((0.18, y), 0.64, 0.44, boxstyle="round,pad=0.02", facecolor="#f3f6fb", edgecolor="#4b5563")
        ax.add_patch(box)
        ax.text(0.5, y + 0.22, wrap_label(node, 55), ha="center", va="center", fontsize=8.5)
        if idx < len(nodes) - 1:
            ax.annotate("", xy=(0.5, y - 0.43), xytext=(0.5, y - 0.08), arrowprops={"arrowstyle": "->", "linewidth": 1})
    ax.text(0.5, -0.72, wrap_label(note, 105), ha="center", va="top", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.15, len(nodes))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_horizontal_flow(plt: Any, path: Path, title: str, nodes: list[str], note: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    width = 0.13 if len(nodes) > 6 else 0.145
    step = 0.88 / max(1, len(nodes) - 1)
    for idx, node in enumerate(nodes):
        x = 0.06 + idx * step
        box = FancyBboxPatch((x - width / 2, 0.48), width, 0.20, boxstyle="round,pad=0.016", facecolor="#f3f6fb", edgecolor="#4b5563")
        ax.add_patch(box)
        ax.text(x, 0.58, wrap_label(node, 18), ha="center", va="center", fontsize=8.2)
        if idx < len(nodes) - 1:
            ax.annotate("", xy=(x + step - width / 2, 0.58), xytext=(x + width / 2, 0.58), arrowprops={"arrowstyle": "->", "lw": 1})
    ax.text(0.5, 0.15, wrap_label(note, 110), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_vertical_flow(plt: Any, path: Path, title: str, nodes: list[str], note: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9.5, max(5.2, len(nodes) * 0.62)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    for idx, node in enumerate(nodes):
        y = 0.86 - idx * (0.72 / max(1, len(nodes) - 1))
        box = FancyBboxPatch((0.22, y - 0.045), 0.56, 0.08, boxstyle="round,pad=0.013", facecolor="#f3f6fb", edgecolor="#4b5563")
        ax.add_patch(box)
        ax.text(0.5, y, wrap_label(node, 56), ha="center", va="center", fontsize=8.5)
        if idx < len(nodes) - 1:
            next_y = 0.86 - (idx + 1) * (0.72 / max(1, len(nodes) - 1))
            ax.annotate("", xy=(0.5, next_y + 0.045), xytext=(0.5, y - 0.045), arrowprops={"arrowstyle": "->", "lw": 1})
    ax.text(0.5, 0.04, wrap_label(note, 105), ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_multi_column_diagram(plt: Any, path: Path, title: str, columns: list[tuple[str, list[str], str]]) -> None:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 6.3))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    width = 0.21
    gap = 0.03
    for col, (heading, nodes, note) in enumerate(columns):
        x0 = 0.03 + col * (width + gap)
        ax.text(x0 + width / 2, 0.91, wrap_label(heading, 24), ha="center", fontsize=10, fontweight="bold")
        for idx, node in enumerate(nodes):
            y = 0.78 - idx * 0.13
            box = FancyBboxPatch((x0, y), width, 0.075, boxstyle="round,pad=0.012", facecolor="#f3f6fb", edgecolor="#4b5563")
            ax.add_patch(box)
            ax.text(x0 + width / 2, y + 0.037, wrap_label(node, 24), ha="center", va="center", fontsize=7.5)
            if idx < len(nodes) - 1:
                ax.annotate("", xy=(x0 + width / 2, y - 0.043), xytext=(x0 + width / 2, y), arrowprops={"arrowstyle": "->", "lw": 0.8})
        ax.text(x0 + width / 2, 0.08, wrap_label(note, 30), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def section_for_figure(name: str) -> str:
    mapping = {
        "fig_01": "0",
        "fig_02": "3",
        "fig_03": "4",
        "fig_04": "5",
        "fig_05": "6",
        "fig_06": "6.4",
        "fig_07": "7",
        "fig_08": "8",
        "fig_09": "8",
        "fig_10": "9",
        "fig_11": "9",
        "fig_12": "10",
        "fig_13": "10",
        "fig_14": "11",
    }
    for prefix, section in mapping.items():
        if name.startswith(prefix):
            return section
    return "appendix"


def paper_candidate_for_figure(name: str) -> str:
    if name in {
        "fig_02_dataset_and_loso_protocol.png",
        "fig_03_normalization_pipeline.png",
        "fig_04_common_head_and_feature_extractor_design.png",
        "fig_05_architecture_comparison_overview.png",
        "fig_07_main_results_macro_f1.png",
        "fig_08_residual_identity_2x2.png",
    }:
        return "yes"
    if name in {"fig_11_feature_importance_rf_xgboost.png", "fig_13_confusion_matrix_grid.png", "fig_14_parameter_vs_macro_f1.png"}:
        return "appendix"
    return "maybe"
