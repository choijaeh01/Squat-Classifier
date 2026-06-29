from __future__ import annotations

import csv
import json
import math
import shutil
import textwrap
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


CONTROLLED_DIR = Path("results/controlled_feature_extractor_comparison/20260629_041712_controlled_feature_extractor_comparison_v1")
XGBOOST_DIR = Path("results/xgboost_only_completion/20260629_053235_xgboost_only_completion_v1")
FULL_MATRIX_DIR = Path("results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1")
LITERATURE_DIR = Path("results/literature_baseline_full_extension/20260617_183952_literature_baseline_full_extension_v1")
V3_ABLATION_DIR = Path("results/v3_component_ablation/20260617_202714_v3_component_ablation_v1")
PROCESSED_DATASET_DIR = Path("data/target/processed/v1_manual_windows_resample512")

KEY_MODELS = [
    "controlled_stats_mlp",
    "controlled_shared_1d_residual",
    "controlled_all_channel_1d_cnn",
    "feature_xgboost_v1",
    "feature_random_forest_v1",
    "rescnn_bigru_attention_lite_v1",
]

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

CLAIM_AUDIT_ROWS = [
    {
        "claim": "Residual branch가 shared encoder bottleneck을 크게 완화했다.",
        "status": "safe",
        "rationale": "Shared 1D Macro F1 0.2806에서 Residual Channel-Shared Encoder 0.8004로 상승했다.",
        "safe_wording": "잔차 통계 branch는 naive shared encoder의 성능 저하를 크게 완화했다.",
    },
    {
        "claim": "Residual Channel-Shared Encoder가 모든 baseline보다 우월하다.",
        "status": "avoid",
        "rationale": "Statistical Summary MLP가 가장 높고, XGBoost 및 all-channel CNN과의 paired CI가 0을 포함한다.",
        "safe_wording": "Residual Channel-Shared Encoder는 강한 practical/neural baseline과 경쟁 가능한 성능을 보였다.",
    },
    {
        "claim": "Attention이 핵심 성능 요인이다.",
        "status": "avoid",
        "rationale": "v3 ablation에서 no-attention meanpool이 original v3와 가까웠다.",
        "safe_wording": "attention은 해석 보조 요소이며, 핵심 claim은 residual branch와 shared bottleneck 완화에 둔다.",
    },
    {
        "claim": "RF/XGBoost feature importance는 원인 설명이다.",
        "status": "avoid",
        "rationale": "feature importance는 모델 내부 기준이며 생체역학적 인과를 직접 증명하지 않는다.",
        "safe_wording": "s1_ax 및 s1_gx 관련 요약 통계가 모델에서 반복적으로 중요하게 관찰되었다.",
    },
    {
        "claim": "Transfer learning 효과를 검증했다.",
        "status": "avoid",
        "rationale": "이번 보고 범위에는 SSL/external dataset adapter/transfer learning 실험이 없다.",
        "safe_wording": "Transfer learning은 후속 연구 범위로 남긴다.",
    },
]


@dataclass(frozen=True)
class ProfessorReportPaths:
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


def check_professor_report_inputs(project_root: Path) -> dict[str, Any]:
    required = [
        CONTROLLED_DIR / "aggregate_metrics_by_model.csv",
        CONTROLLED_DIR / "paired_model_differences.csv",
        CONTROLLED_DIR / "classwise_metrics_by_model.csv",
        CONTROLLED_DIR / "subjectwise_metrics_by_model.csv",
        CONTROLLED_DIR / "random_forest_feature_importance_aggregate.csv",
        XGBOOST_DIR / "aggregate_metrics_by_model.csv",
        XGBOOST_DIR / "xgboost_feature_importance_aggregate.csv",
        XGBOOST_DIR / "merged_with_controlled_comparison/merged_controlled_plus_xgboost.csv",
        FULL_MATRIX_DIR / "aggregate_metrics_by_model.csv",
        LITERATURE_DIR / "aggregate_metrics_by_model.csv",
        V3_ABLATION_DIR / "aggregate_metrics_by_model.csv",
        Path("docs/controlled_feature_extractor_comparison_v1_report.md"),
        Path("docs/xgboost_only_completion_v1_report.md"),
        Path("docs/experiment_protocol.md"),
        Path("docs/results_tracking.md"),
    ]
    status = {str(path): (project_root / path).exists() for path in required}
    return {"all_present": all(status.values()), "files": status}


def build_professor_report_assets(paths: ProfessorReportPaths) -> dict[str, Any]:
    paths.output_docs.mkdir(parents=True, exist_ok=True)
    for folder in (paths.tables, paths.figures, paths.diagrams, paths.assets):
        folder.mkdir(parents=True, exist_ok=True)

    mapping = load_display_mapping(paths.mapping_path)
    sources = load_sources(paths.project_root)

    tables = build_tables(paths, sources, mapping)
    figures = build_figures(paths, tables)
    diagrams = build_diagrams(paths)
    captions = write_captions(paths)
    main_report = write_main_report(paths, tables, figures, diagrams)
    figure_index = write_figure_table_index(paths)
    validation = validate_professor_report(paths)
    validation_path = paths.assets / "professor_report_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "output_docs": str(paths.output_docs),
        "output_obsidian": str(paths.output_obsidian) if paths.output_obsidian else "",
        "tables": [str(p) for p in tables],
        "figures": [str(p) for p in figures],
        "diagrams": [str(p) for p in diagrams],
        "captions": str(captions),
        "main_report": str(main_report),
        "figure_table_index": str(figure_index),
        "validation_path": str(validation_path),
        "validation": validation,
    }


def load_display_mapping(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data["mapping"])


def load_sources(project_root: Path) -> dict[str, list[dict[str, str]]]:
    def read(rel: Path) -> list[dict[str, str]]:
        return read_csv(project_root / rel)

    return {
        "controlled_aggregate": read(CONTROLLED_DIR / "aggregate_metrics_by_model.csv"),
        "controlled_classwise": read(CONTROLLED_DIR / "classwise_metrics_by_model.csv"),
        "controlled_subjectwise": read(CONTROLLED_DIR / "subjectwise_metrics_by_model.csv"),
        "rf_importance": read(CONTROLLED_DIR / "random_forest_feature_importance_aggregate.csv"),
        "xgb_aggregate": read(XGBOOST_DIR / "aggregate_metrics_by_model.csv"),
        "xgb_classwise": read(XGBOOST_DIR / "classwise_metrics_by_model.csv"),
        "xgb_subjectwise": read(XGBOOST_DIR / "subjectwise_metrics_by_model.csv"),
        "xgb_importance": read(XGBOOST_DIR / "xgboost_feature_importance_aggregate.csv"),
        "xgb_merged": read(XGBOOST_DIR / "merged_with_controlled_comparison/merged_controlled_plus_xgboost.csv"),
        "xgb_paired": read(XGBOOST_DIR / "paired_model_differences.csv"),
        "full_aggregate": read(FULL_MATRIX_DIR / "aggregate_metrics_by_model.csv"),
        "literature_aggregate": read(LITERATURE_DIR / "aggregate_metrics_by_model.csv"),
        "v3_ablation_aggregate": read(V3_ABLATION_DIR / "aggregate_metrics_by_model.csv"),
    }


def build_tables(paths: ProfessorReportPaths, sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[Path]:
    outputs: list[Path] = []
    outputs.append(write_table(paths.tables / "table_01_dataset_summary.csv", dataset_summary_rows()))
    outputs.append(write_table(paths.tables / "table_02_experiment_checklist.csv", experiment_checklist_rows()))
    controlled = controlled_table_rows(sources["controlled_aggregate"], mapping)
    controlled_table = write_table(paths.tables / "table_03_controlled_comparison_clean_names.csv", controlled)
    outputs.append(controlled_table)
    professor_facing_feature_table = paths.tables / "table_03_feature_extractor_clean_names.csv"
    shutil.copyfile(controlled_table, professor_facing_feature_table)
    outputs.append(professor_facing_feature_table)
    xgb_rows = xgboost_table_rows(sources["xgb_merged"], mapping)
    outputs.append(write_table(paths.tables / "table_04_xgboost_completion_clean_names.csv", xgb_rows))
    flow = component_flow_rows(sources["controlled_aggregate"], mapping)
    outputs.append(write_table(paths.tables / "table_05_component_flow.csv", flow))
    class3 = class3_rows(sources, mapping)
    outputs.append(write_table(paths.tables / "table_06_class3_summary.csv", class3))
    importance = feature_importance_rows(sources["rf_importance"], sources["xgb_importance"])
    outputs.append(write_table(paths.tables / "table_07_feature_importance_top10.csv", importance))
    outputs.append(write_table(paths.tables / "table_08_claim_audit_summary.csv", CLAIM_AUDIT_ROWS))
    outputs.append(write_table(paths.tables / "table_09_final_recommended_scope.csv", final_scope_rows()))
    return outputs


def dataset_summary_rows() -> list[dict[str, str]]:
    return [
        {"item": "Sensors", "value": "3 MPU-6050 IMUs", "note": "허리, 오른쪽 허벅지, 오른쪽 종아리"},
        {"item": "Channels", "value": "18", "note": "각 IMU acc 3축 + gyro 3축"},
        {"item": "Window length", "value": "512", "note": "phase-normalized linear interpolation"},
        {"item": "Subjects", "value": "6", "note": "subject-independent LOSO"},
        {"item": "Classes", "value": "5", "note": "Correct, Knee Valgus, Butt Wink, Excessive Lean, Partial Squat"},
        {"item": "Windows", "value": "600", "note": "subject-class 조합별 20 windows"},
        {"item": "Validation policy", "value": "LOSO with within-train stratified validation", "note": "train 400 / val 100 / test 100"},
        {"item": "Scaler", "value": "Train-only StandardScaler", "note": "held-out test subject는 fit에 미사용"},
    ]


def experiment_checklist_rows() -> list[dict[str, str]]:
    return [
        {"professor_request": "XGBoost 비교", "executed": "yes", "evidence": "xgboost_only_completion_v1", "note": "동일 feature set, 18 runs 완료"},
        {"professor_request": "RandomForest feature importance 확인", "executed": "yes", "evidence": "random_forest_feature_importance_aggregate.csv", "note": "s1_ax 관련 feature 반복 관찰"},
        {"professor_request": "normalization/scaler 확인", "executed": "yes", "evidence": "scaler_fit_audit.csv", "note": "train indices only"},
        {"professor_request": "같은 classifier head 비교", "executed": "yes", "evidence": "common_head_verification.csv", "note": "64-dim representation + 동일 MLP head"},
        {"professor_request": "1D / residual / 2D / MLP 비교", "executed": "yes", "evidence": "controlled comparison", "note": "display name table로 정리"},
        {"professor_request": "국내 저널 범위 축소", "executed": "yes", "evidence": "table_09_final_recommended_scope.csv", "note": "SSL/transfer는 후속 연구로 분리"},
    ]


def controlled_table_rows(rows: list[dict[str, str]], mapping: dict[str, str]) -> list[dict[str, str]]:
    by_model = {row["model_name"]: row for row in rows}
    output = []
    for model in CONTROLLED_ORDER:
        row = by_model.get(model)
        if not row:
            continue
        output.append(
            {
                "model_display_name": display(model, mapping),
                "group": model_group(model),
                "accuracy": fmt(row.get("mean_accuracy")),
                "macro_f1": fmt(row.get("mean_macro_f1")),
                "weighted_f1": fmt(row.get("mean_weighted_f1")),
                "macro_f1_ci": ci(row),
                "role": model_role(model),
                "interpretation": interpretation_for_model(model),
            }
        )
    return output


def xgboost_table_rows(rows: list[dict[str, str]], mapping: dict[str, str]) -> list[dict[str, str]]:
    keep = ["controlled_stats_mlp", "controlled_shared_1d_residual", "controlled_all_channel_1d_cnn", "feature_xgboost_v1", "feature_random_forest_v1"]
    by_model = {row["model_name"]: row for row in rows}
    output = []
    for model in keep:
        row = by_model.get(model)
        if not row:
            continue
        output.append(
            {
                "model_display_name": display(model, mapping),
                "accuracy": fmt(row.get("mean_accuracy")),
                "macro_f1": fmt(row.get("mean_macro_f1")),
                "macro_f1_ci": ci(row),
                "role": model_role(model),
            }
        )
    return output


def component_flow_rows(rows: list[dict[str, str]], mapping: dict[str, str]) -> list[dict[str, str]]:
    by_model = {row["model_name"]: row for row in rows}
    baseline = fnum(by_model["controlled_shared_1d"]["mean_macro_f1"])
    meanings = {
        "controlled_shared_1d": "공유 encoder만 사용하면 채널 위치 정보가 약해진다.",
        "controlled_shared_1d_identity": "identity만 추가하면 일부 개선되지만 충분하지 않다.",
        "controlled_shared_1d_residual": "raw summary residual branch가 병목을 크게 완화한다.",
        "controlled_shared_1d_residual_identity": "residual 위에 identity를 더해도 추가 이득은 작았다.",
    }
    output = []
    for model in FLOW_MODELS:
        value = fnum(by_model[model]["mean_macro_f1"])
        output.append(
            {
                "model_display_name": display(model, mapping),
                "macro_f1": fmt(value),
                "delta_from_shared_1d": signed(value - baseline),
                "meaning": meanings[model],
            }
        )
    return output


def class3_rows(sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[dict[str, str]]:
    rows = [*sources["controlled_classwise"], *sources["xgb_classwise"]]
    keep = ["controlled_stats_mlp", "controlled_shared_1d_residual", "controlled_all_channel_1d_cnn", "feature_xgboost_v1", "feature_random_forest_v1", "rescnn_bigru_attention_lite_v1"]
    by_model = {row["model_name"]: row for row in rows if row.get("class_id") == "3"}
    output = []
    for model in keep:
        row = by_model.get(model)
        if not row:
            continue
        output.append(
            {
                "model_display_name": display(model, mapping),
                "class3_recall": fmt(row.get("mean_recall")),
                "class3_f1": fmt(row.get("mean_f1")),
                "interpretation": class3_interpretation(model),
            }
        )
    return output


def feature_importance_rows(rf_rows: list[dict[str, str]], xgb_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for idx in range(10):
        rf = rf_rows[idx]
        xgb = xgb_rows[idx]
        note = "s1/s0 중심 signal-derived feature; 인과 단정 금지" if idx == 0 else "model-specific importance"
        output.append(
            {
                "rank": str(idx + 1),
                "random_forest_feature": rf["feature_name"],
                "rf_importance": fmt(rf.get("mean_importance")),
                "xgboost_feature": xgb["feature_name"],
                "xgb_importance": fmt(xgb.get("mean_importance")),
                "note": note,
            }
        )
    return output


def final_scope_rows() -> list[dict[str, str]]:
    return [
        {"item": "Clean-room dataset conversion", "include_in_kiee": "yes", "reason": "데이터/누수 통제의 신뢰 기반", "future_work": "외부 공개 가능성 검토"},
        {"item": "Controlled feature extractor comparison", "include_in_kiee": "yes", "reason": "교수님 피드백의 핵심 비교", "future_work": "추가 dataset에서 재검증"},
        {"item": "Residual Channel-Shared Encoder", "include_in_kiee": "yes", "reason": "shared encoder 병목 완화 claim의 중심", "future_work": "위치 embedding/attention 세부 ablation 확장"},
        {"item": "RF/XGBoost/Stats MLP practical baselines", "include_in_kiee": "yes", "reason": "강한 practical baseline을 투명하게 보고", "future_work": "feature robustness 분석"},
        {"item": "SSL", "include_in_kiee": "no", "reason": "이번 supervised protocol 범위 밖", "future_work": "후속 국제 논문 후보"},
        {"item": "External transfer benchmark", "include_in_kiee": "no", "reason": "adapter 및 protocol 미확정", "future_work": "후속 연구"},
        {"item": "Real-time system", "include_in_kiee": "limited", "reason": "배경/응용 가능성으로만 언급", "future_work": "시스템 논문으로 분리"},
    ]


def build_figures(paths: ProfessorReportPaths, tables: list[Path]) -> list[Path]:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    plt.rcParams["axes.unicode_minus"] = False

    table_data = {path.name: read_csv(path) for path in tables}
    requested_controlled_figure = paths.figures / "fig_01_controlled_macro_f1.png"
    professor_facing_feature_figure = paths.figures / "fig_01_feature_extractor_macro_f1.png"
    outputs = [
        draw_bar_with_ci(
            plt,
            table_data["table_03_controlled_comparison_clean_names.csv"],
            requested_controlled_figure,
            title="Controlled Feature Extractor Comparison",
            value_key="macro_f1",
            ci_key="macro_f1_ci",
            models=[
                "Statistical Summary MLP",
                "Residual Channel-Shared Encoder",
                "All-Channel 1D CNN",
                "Residual Channel-Shared + Identity",
                "Random Forest",
                "ResCNN-BiGRU-Attention",
                "Compact All-Channel 1D CNN",
                "Linear SVM",
                "Raw Flatten MLP",
                "Lee-style CNN-LSTM",
                "Shared 1D + Identity",
                "2D CNN",
                "Shared 1D Encoder",
            ],
        ),
        draw_component_flow(plt, table_data["table_05_component_flow.csv"], paths.figures / "fig_02_residual_effect_flow.png"),
        draw_practical_baselines(plt, table_data["table_04_xgboost_completion_clean_names.csv"], paths.figures / "fig_03_xgboost_practical_baselines.png"),
        draw_class3(plt, table_data["table_06_class3_summary.csv"], paths.figures / "fig_04_class3_excessive_lean.png"),
        draw_feature_importance(plt, table_data["table_07_feature_importance_top10.csv"], paths.figures / "fig_05_rf_xgboost_feature_importance.png"),
        draw_subjectwise(paths, plt, paths.figures / "fig_06_subjectwise_macro_f1.png"),
        draw_parameter_vs_macro(paths, plt, paths.figures / "fig_07_parameter_vs_macro_f1.png"),
    ]
    shutil.copyfile(requested_controlled_figure, professor_facing_feature_figure)
    outputs.append(professor_facing_feature_figure)
    return outputs


def build_diagrams(paths: ProfessorReportPaths) -> list[Path]:
    diagram_specs = [
        (
            "diagram_01_common_head",
            "Common Head",
            ["Feature Extractor", "64-dim Representation", "Linear 64->64", "ReLU", "Dropout 0.1", "Linear 64->5", "5-class logits"],
            "모든 controlled neural model에서 classifier head는 동일하다.",
        ),
        (
            "diagram_02_all_channel_1d_cnn",
            "All-Channel 1D CNN",
            ["Input 512x18", "Conv1D over all channels", "Global pooling / projection", "64-dim representation", "Common MLP head"],
            "18채널을 처음부터 함께 보며 channel-specific information을 직접 학습한다.",
        ),
        (
            "diagram_03_shared_1d_encoder",
            "Shared 1D Encoder",
            ["18 single-channel streams", "same Temporal Encoder f_theta reused", "token pooling", "64-dim representation", "Common MLP head"],
            "parameter sharing은 하지만 channel-specific 정보가 약화될 수 있다.",
        ),
        (
            "diagram_04_residual_channel_shared_encoder",
            "Residual Channel-Shared Encoder",
            ["18 single-channel streams", "shared Temporal Encoder f_theta", "pooled shared representation", "raw IMU residual summary branch", "fusion to 64-dim", "Common MLP head"],
            "shared encoder 병목을 residual branch가 보완한다.",
        ),
        (
            "diagram_05_statistical_summary_mlp",
            "Statistical Summary MLP",
            ["Input 512x18", "signal-derived summary statistics", "projection to 64-dim", "Common MLP head"],
            "channel-wise summary statistics가 강한 practical baseline이다.",
        ),
        (
            "diagram_06_practical_tree_baselines",
            "Practical Tree Baselines",
            ["Input 512x18", "signal-derived summary features", "Random Forest / XGBoost / Linear SVM", "5-class prediction"],
            "same feature audit, no metadata/label/subject leakage.",
        ),
        (
            "diagram_07_evaluation_protocol",
            "Evaluation Protocol",
            ["6 subjects", "LOSO held-out test subject", "remaining 5 subjects", "per subject-class: 16 train / 4 val", "train 400 / val 100 / test 100", "StandardScaler fit train only"],
            "test subject는 scaler, validation, training에 사용하지 않는다.",
        ),
    ]
    outputs: list[Path] = []
    for name, title, nodes, note in diagram_specs:
        mmd_path = paths.diagrams / f"{name}.mmd"
        png_path = paths.diagrams / f"{name}.png"
        mmd_path.write_text(mermaid_diagram(title, nodes, note), encoding="utf-8")
        draw_box_diagram(png_path, title, nodes, note)
        outputs.extend([mmd_path, png_path])
    return outputs


def write_captions(paths: ProfessorReportPaths) -> Path:
    caption_path = paths.figures / "captions.md"
    caption_path.write_text(
        """# Figure Captions

## fig_01_feature_extractor_macro_f1.png

- Korean caption: 공통 classifier head 조건에서 feature extractor별 Macro F1을 비교한 그림.
- English caption: Macro F1 comparison across feature extractors under a shared classifier head.
- 핵심 메시지: Statistical Summary MLP가 가장 높고, Residual Channel-Shared Encoder는 All-Channel 1D CNN 및 XGBoost와 유사한 수준이다.
- 논문 사용 추천: main 또는 ablation/control figure 후보.

## fig_02_residual_effect_flow.png

- Korean caption: Shared 1D Encoder에서 identity와 residual branch를 추가했을 때 Macro F1 변화.
- English caption: Macro F1 progression from shared 1D encoder to residual channel-shared variants.
- 핵심 메시지: identity만으로는 부족했고 residual branch가 가장 큰 성능 회복을 만들었다.
- 논문 사용 추천: proposed structure motivation figure 후보.

## fig_03_xgboost_practical_baselines.png

- Korean caption: XGBoost completion을 포함한 practical/neural baseline Macro F1 비교.
- English caption: Macro F1 comparison including XGBoost completion and practical baselines.
- 핵심 메시지: XGBoost와 RF도 강하므로 practical baseline을 투명하게 보고해야 한다.
- 논문 사용 추천: baseline comparison 또는 appendix 후보.

## fig_04_class3_excessive_lean.png

- Korean caption: Class 3 Excessive Lean에서 주요 모델의 recall 및 F1 비교.
- English caption: Class 3 Excessive Lean recall and F1 across key models.
- 핵심 메시지: Residual Channel-Shared 계열이 어려운 class에서 비교적 높은 F1을 보였다.
- 논문 사용 추천: class-wise analysis 후보.

## fig_05_rf_xgboost_feature_importance.png

- Korean caption: RF와 XGBoost의 상위 signal-derived feature importance 비교.
- English caption: Top signal-derived feature importances from Random Forest and XGBoost.
- 핵심 메시지: s1_ax, s1_gx 관련 feature가 반복적으로 관찰되지만 인과 단정은 금지한다.
- 논문 사용 추천: discussion 또는 appendix 후보.

## fig_06_subjectwise_macro_f1.png

- Korean caption: 주요 모델의 subject-wise Macro F1 비교.
- English caption: Subject-wise Macro F1 comparison for key models.
- 핵심 메시지: 평균 성능만이 아니라 subject별 변동을 함께 봐야 한다.
- 논문 사용 추천: appendix 후보.

## fig_07_parameter_vs_macro_f1.png

- Korean caption: controlled neural model의 parameter count와 Macro F1 관계.
- English caption: Parameter count versus Macro F1 for controlled neural models.
- 핵심 메시지: parameter 수가 많다고 자동으로 좋은 성능을 보장하지 않는다.
- 논문 사용 추천: supplementary 또는 discussion 후보.
""",
        encoding="utf-8",
    )
    return caption_path


def write_main_report(paths: ProfessorReportPaths, tables: list[Path], figures: list[Path], diagrams: list[Path]) -> Path:
    today = date.today().isoformat()
    report_path = paths.output_docs / "Professor Report - Residual Channel-Shared Encoder.md"
    report_path.write_text(main_report_markdown(today), encoding="utf-8")
    return report_path


def write_figure_table_index(paths: ProfessorReportPaths) -> Path:
    path = paths.output_docs / "Figure and Table Index.md"
    path.write_text(figure_table_index_markdown(), encoding="utf-8")
    return path


def validate_professor_report(paths: ProfessorReportPaths) -> dict[str, Any]:
    md_files = list(paths.output_docs.rglob("*.md"))
    png_files = list(paths.output_docs.rglob("*.png"))
    csv_files = list(paths.output_docs.rglob("*.csv"))
    controlled_violations: list[str] = []
    allowed_internal = {"display_name_mapping.yaml", "Figure and Table Index.md"}
    for file in [*md_files, *csv_files]:
        if file.name in allowed_internal or "display_name_mapping" in file.name:
            continue
        text = file.read_text(encoding="utf-8")
        if "controlled_" in text:
            controlled_violations.append(str(file))
    table_pipe_errors = markdown_table_pipe_errors(md_files)
    missing_images = missing_image_links(paths.output_docs)
    return {
        "markdown_files": len(md_files),
        "png_files": len(png_files),
        "csv_files": len(csv_files),
        "controlled_name_violations": controlled_violations,
        "controlled_name_violation_count": len(controlled_violations),
        "table_pipe_errors": table_pipe_errors,
        "missing_image_links": missing_images,
        "passed": not controlled_violations and not table_pipe_errors and not missing_images,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_table(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def display(model_name: str, mapping: dict[str, str]) -> str:
    return mapping.get(model_name, model_name)


def fnum(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def fmt(value: Any, digits: int = 4) -> str:
    return f"{fnum(value):.{digits}f}"


def signed(value: float) -> str:
    return f"{value:+.4f}"


def ci(row: dict[str, str]) -> str:
    return f"[{fmt(row.get('macro_f1_ci_low'))}, {fmt(row.get('macro_f1_ci_high'))}]"


def model_group(model: str) -> str:
    if model in {"controlled_stats_mlp", "feature_random_forest_v1", "feature_xgboost_v1", "feature_linear_svm_v1"}:
        return "Practical Baseline"
    if model in {"rescnn_bigru_attention_lite_v1", "lee_style_cnn_lstm_2d_v1"}:
        return "Literature Reference"
    if model == "controlled_shared_1d_residual":
        return "Proposed Core"
    return "Neural Baseline"


def model_role(model: str) -> str:
    roles = {
        "controlled_stats_mlp": "summary-statistics practical baseline",
        "controlled_shared_1d_residual": "proposed core extractor",
        "controlled_all_channel_1d_cnn": "all-channel neural baseline",
        "controlled_shared_1d_residual_identity": "residual plus identity variant",
        "feature_random_forest_v1": "tree-based practical baseline",
        "feature_xgboost_v1": "boosted tree practical baseline",
        "rescnn_bigru_attention_lite_v1": "literature temporal reference",
        "lee_style_cnn_lstm_2d_v1": "adapted CNN-LSTM reference",
    }
    return roles.get(model, "comparison model")


def interpretation_for_model(model: str) -> str:
    interpretations = {
        "controlled_stats_mlp": "현재 데이터에서 signal summary가 매우 강함.",
        "controlled_shared_1d_residual": "shared encoder 병목을 residual branch가 완화.",
        "controlled_all_channel_1d_cnn": "채널별 정보를 직접 학습하는 강한 neural baseline.",
        "controlled_shared_1d_residual_identity": "residual-only와 비슷해 identity 추가 이득은 제한적.",
        "feature_random_forest_v1": "feature baseline이 강하다는 점을 확인.",
        "feature_linear_svm_v1": "linear feature baseline으로 lower bound 역할.",
        "controlled_shared_1d": "단독 shared pooling은 underfit.",
        "controlled_shared_1d_identity": "identity만으로는 병목 해결이 부족.",
        "controlled_2d_cnn": "time-channel 2D baseline은 낮게 관찰.",
        "controlled_flatten_mlp": "큰 parameter 수에도 inductive bias가 약함.",
        "rescnn_bigru_attention_lite_v1": "문헌형 temporal reference로 비교 가능.",
        "lee_style_cnn_lstm_2d_v1": "adapted CNN-LSTM은 본 protocol에서 낮음.",
    }
    return interpretations.get(model, "비교용 모델.")


def class3_interpretation(model: str) -> str:
    if model == "controlled_shared_1d_residual":
        return "어려운 Excessive Lean class에서 높은 F1."
    if model == "feature_xgboost_v1":
        return "precision은 높지만 recall은 제한적."
    if model == "controlled_stats_mlp":
        return "summary feature도 Class 3에서 강함."
    return "class-wise reference."


def draw_bar_with_ci(plt: Any, rows: list[dict[str, str]], path: Path, *, title: str, value_key: str, ci_key: str, models: list[str]) -> Path:
    by_name = {row["model_display_name"]: row for row in rows}
    selected = [by_name[name] for name in models if name in by_name]
    labels = [row["model_display_name"] for row in selected]
    values = [fnum(row[value_key]) for row in selected]
    lows: list[float] = []
    highs: list[float] = []
    for row, value in zip(selected, values):
        lo, hi = parse_ci(row[ci_key])
        lows.append(value - lo)
        highs.append(hi - value)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(range(len(labels)), values, yerr=[lows, highs], capsize=3)
    ax.set_title(title)
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
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(range(len(labels)), values, marker="o", linewidth=2)
    ax.bar(range(len(labels)), values, alpha=0.3)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_title("Residual Branch Effect")
    ax.set_ylabel("Mean Macro F1")
    ax.set_ylim(0, 0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(label, 18) for label in labels], rotation=20, ha="right")
    ax.annotate("Residual branch", xy=(2, values[2]), xytext=(1.1, 0.72), arrowprops={"arrowstyle": "->"})
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_practical_baselines(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    ordered = ["Statistical Summary MLP", "Residual Channel-Shared Encoder", "All-Channel 1D CNN", "XGBoost", "Random Forest"]
    by_name = {row["model_display_name"]: row for row in rows}
    selected = [by_name[name] for name in ordered if name in by_name]
    labels = [row["model_display_name"] for row in selected]
    values = [fnum(row["macro_f1"]) for row in selected]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(range(len(labels)), values)
    ax.set_title("Practical Baselines Including XGBoost")
    ax.set_ylabel("Mean Macro F1")
    ax.set_ylim(0.7, 0.86)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([wrap_label(label, 16) for label in labels], rotation=35, ha="right")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.004, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_class3(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    labels = [row["model_display_name"] for row in rows]
    recall = [fnum(row["class3_recall"]) for row in rows]
    f1 = [fnum(row["class3_f1"]) for row in rows]
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar([x - 0.18 for x in xs], recall, width=0.36, label="Recall")
    ax.bar([x + 0.18 for x in xs], f1, width=0.36, label="F1")
    ax.set_title("Class 3 Excessive Lean")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([wrap_label(label, 15) for label in labels], rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_feature_importance(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    labels = [row["rank"] for row in rows]
    rf = [fnum(row["rf_importance"]) for row in rows]
    xgb = [fnum(row["xgb_importance"]) for row in rows]
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([x - 0.18 for x in xs], rf, width=0.36, label="Random Forest")
    ax.bar([x + 0.18 for x in xs], xgb, width=0.36, label="XGBoost")
    ax.set_title("RF/XGBoost Top-10 Feature Importance")
    ax.set_ylabel("Mean importance")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{row['rank']}\nRF:{short_feature(row['random_forest_feature'])}\nXGB:{short_feature(row['xgboost_feature'])}" for row in rows], fontsize=7)
    ax.text(0.01, 0.97, "Feature importance is model-specific, not causal evidence.", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_subjectwise(paths: ProfessorReportPaths, plt: Any, path: Path) -> Path:
    mapping = load_display_mapping(paths.mapping_path)
    rows = [*read_csv(paths.project_root / CONTROLLED_DIR / "subjectwise_metrics_by_model.csv"), *read_csv(paths.project_root / XGBOOST_DIR / "subjectwise_metrics_by_model.csv")]
    keep = ["controlled_stats_mlp", "controlled_shared_1d_residual", "controlled_all_channel_1d_cnn", "feature_xgboost_v1", "feature_random_forest_v1", "rescnn_bigru_attention_lite_v1"]
    subjects = sorted({row["test_subject"] for row in rows}, key=lambda x: int(float(x)))
    data = []
    labels = []
    for model in keep:
        labels.append(display(model, mapping))
        model_rows = {row["test_subject"]: fnum(row["test_macro_f1"]) for row in rows if row["model_name"] == model}
        data.append([model_rows.get(subj, 0.0) for subj in subjects])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(data, vmin=0, vmax=1, aspect="auto")
    ax.set_title("Subject-wise Macro F1")
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([f"S{subj}" for subj in subjects])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([wrap_label(label, 24) for label in labels], fontsize=8)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_parameter_vs_macro(paths: ProfessorReportPaths, plt: Any, path: Path) -> Path:
    mapping = load_display_mapping(paths.mapping_path)
    rows = read_csv(paths.project_root / CONTROLLED_DIR / "aggregate_metrics_by_model.csv")
    neural = [row for row in rows if row["model_name"].startswith("controlled_") or row["model_name"] in {"rescnn_bigru_attention_lite_v1", "lee_style_cnn_lstm_2d_v1"}]
    classical = [row for row in rows if row["model_name"].startswith("feature_")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in neural:
        params = fnum(row["total_params"])
        macro = fnum(row["mean_macro_f1"])
        ax.scatter(params, macro, s=50)
        ax.text(params, macro + 0.01, wrap_label(display(row["model_name"], mapping), 14), fontsize=7)
    for idx, row in enumerate(classical):
        macro = fnum(row["mean_macro_f1"])
        ax.scatter(0, macro, marker="x", s=60)
        ax.text(1000, macro + 0.01, wrap_label(display(row["model_name"], mapping), 14), fontsize=7)
    ax.set_title("Parameter Count vs Macro F1")
    ax.set_xlabel("Trainable parameters (classical models marked near zero)")
    ax.set_ylabel("Mean Macro F1")
    ax.set_ylim(0.2, 0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_box_diagram(path: Path, title: str, nodes: list[str], note: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(nodes) * 0.75)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    y_positions = list(reversed([idx for idx in range(len(nodes))]))
    for idx, (node, y) in enumerate(zip(nodes, y_positions)):
        box = FancyBboxPatch((0.25, y), 0.5, 0.45, boxstyle="round,pad=0.02", linewidth=1, facecolor="#f3f6fb", edgecolor="#4b5563")
        ax.add_patch(box)
        ax.text(0.5, y + 0.225, node, ha="center", va="center", fontsize=10)
        if idx < len(nodes) - 1:
            ax.annotate("", xy=(0.5, y - 0.45), xytext=(0.5, y - 0.08), arrowprops={"arrowstyle": "->", "linewidth": 1})
    ax.text(0.5, -0.65, note, ha="center", va="top", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.1, len(nodes))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def mermaid_diagram(title: str, nodes: list[str], note: str) -> str:
    lines = ["flowchart TD"]
    for idx, node in enumerate(nodes):
        safe = node.replace('"', "'")
        lines.append(f'  N{idx}["{safe}"]')
        if idx > 0:
            lines.append(f"  N{idx-1} --> N{idx}")
    lines.append(f'  Note["{note.replace(chr(34), chr(39))}"]')
    lines.append(f"  N{len(nodes)-1} --> Note")
    return "\n".join(lines) + "\n"


def parse_ci(value: str) -> tuple[float, float]:
    clean = value.strip().strip("[]")
    lo, hi = clean.split(",")
    return float(lo), float(hi)


def wrap_label(label: str, width: int) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def short_feature(feature: str) -> str:
    return feature.replace("_", " ")


def markdown_table(rows: list[dict[str, str]], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = [escape_table(str(row.get(col, ""))) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def escape_table(value: str) -> str:
    return value.replace("|", "&#124;").replace("\n", " ")


def main_report_markdown(today: str) -> str:
    table03 = read_csv(Path("docs/professor_report_v1/tables/table_03_controlled_comparison_clean_names.csv"))
    table04 = read_csv(Path("docs/professor_report_v1/tables/table_04_xgboost_completion_clean_names.csv"))
    table05 = read_csv(Path("docs/professor_report_v1/tables/table_05_component_flow.csv"))
    table06 = read_csv(Path("docs/professor_report_v1/tables/table_06_class3_summary.csv"))
    table07 = read_csv(Path("docs/professor_report_v1/tables/table_07_feature_importance_top10.csv"))
    table08 = read_csv(Path("docs/professor_report_v1/tables/table_08_claim_audit_summary.csv"))
    table09 = read_csv(Path("docs/professor_report_v1/tables/table_09_final_recommended_scope.csv"))
    checklist = read_csv(Path("docs/professor_report_v1/tables/table_02_experiment_checklist.csv"))
    return f"""---
title: "Residual Channel-Shared Encoder 보고자료"
project: "Squat IMU"
date: "{today}"
status: "Professor report draft"
tags:
  - research
  - squat
  - imu
  - kiee
---

# Residual Channel-Shared Encoder 보고자료

## 0. 한 줄 결론

교수님께서 요구한 **동일 classifier head 기반 feature extractor 비교와 XGBoost 보완 비교는 수행되었다.** 결과상 residual branch는 shared encoder의 병목을 크게 완화했지만, statistical summary 및 tree 기반 baseline도 강했다. 따라서 논문 주장은 “압도적 우월성”이 아니라 **잔차 채널 공유 구조가 naive shared encoder 문제를 완화하고 강한 baseline과 경쟁 가능한 성능을 보였다는 검증**으로 잡는 것이 안전하다.

## 1. 교수님 피드백 반영 요약

{markdown_table(checklist)}

## 2. 데이터셋 및 평가 프로토콜

데이터는 3개 MPU-6050 IMU로 수집한 18채널 squat 자세 window다. 센서 위치는 허리, 오른쪽 허벅지, 오른쪽 종아리이며 각 센서는 accelerometer 3축과 gyroscope 3축을 제공한다. Raw window는 512 time steps로 phase-normalized linear interpolation을 적용해 정렬했다.

평가는 6명 subject에 대한 LOSO 방식이다. 각 fold에서 test subject 1명은 완전히 held-out이고, 나머지 5명의 각 subject-class 조합에서 16개 window를 train, 4개 window를 validation으로 사용했다. fold당 train 400, validation 100, test 100이며 StandardScaler는 train indices에만 fit했다.

![Evaluation Protocol](diagrams/diagram_07_evaluation_protocol.png)

## 3. 왜 기존 결과만으로 부족했는가

기존 full matrix는 모델마다 classifier head와 representation pathway가 달라 feature extractor 자체의 효과를 분리하기 어려웠다. 이번 controlled comparison에서는 모든 neural model이 64차원 representation을 만들고 동일한 MLP head를 사용하도록 고정했다. 이 설계로 “앞단 feature extractor가 바뀌었을 때 성능이 어떻게 변하는가”를 더 직접적으로 볼 수 있다.

![Common Head](diagrams/diagram_01_common_head.png)

## 4. 비교한 아키텍처 구조

### 4.1 Statistical Summary MLP

Statistical Summary MLP는 512x18 IMU signal에서 channel별 mean, std, energy 등 signal-derived summary statistics만 계산한 뒤 64차원 representation과 공통 MLP head를 사용한다. 현재 데이터에서는 이 요약 통계가 매우 강한 baseline으로 나타났다.

![Statistical Summary MLP](diagrams/diagram_05_statistical_summary_mlp.png)

### 4.2 All-Channel 1D CNN

All-Channel 1D CNN은 18채널을 처음부터 함께 보고 temporal convolution을 학습한다. 채널 위치와 상호작용을 직접 학습할 수 있지만, 작은 subject 수에서는 자유도가 커질 수 있다.

![All-Channel 1D CNN](diagrams/diagram_02_all_channel_1d_cnn.png)

### 4.3 Shared 1D Encoder

Shared 1D Encoder는 18개 단일 채널 stream에 같은 temporal encoder를 재사용한다. parameter sharing은 가능하지만 channel identity와 위치별 차이가 약해질 수 있다.

![Shared 1D Encoder](diagrams/diagram_03_shared_1d_encoder.png)

### 4.4 Residual Channel-Shared Encoder

Residual Channel-Shared Encoder는 shared temporal encoder를 유지하면서 raw IMU signal에서 나온 residual statistical branch를 병렬로 결합한다. 이 branch가 shared encoder가 잃을 수 있는 channel-specific summary signal을 보완한다.

![Residual Channel-Shared Encoder](diagrams/diagram_04_residual_channel_shared_encoder.png)

### 4.5 2D CNN

2D CNN은 time x channel matrix로 입력을 보고 2D convolution을 적용하는 baseline이다. 이번 controlled comparison에서는 상위 baseline보다 낮게 나타났다.

### 4.6 RF/XGBoost/SVM

Random Forest, XGBoost, Linear SVM은 동일한 signal-derived summary feature set을 사용한다. feature audit에서 metadata, label, subject ID, boundary, original length feature는 포함되지 않았다.

![Practical Tree Baselines](diagrams/diagram_06_practical_tree_baselines.png)

## 5. Controlled Feature Extractor 결과

![Feature Extractor Comparison](figures/fig_01_feature_extractor_macro_f1.png)

{markdown_table(table03)}

핵심은 세 가지다. 첫째, Statistical Summary MLP가 가장 높다. 둘째, Residual Channel-Shared Encoder는 All-Channel 1D CNN과 매우 가까운 Macro F1을 보인다. 셋째, Shared 1D Encoder 단독은 낮아서 단순 parameter sharing만으로는 충분하지 않다.

## 6. Residual branch 효과

![Residual Branch Effect](figures/fig_02_residual_effect_flow.png)

{markdown_table(table05)}

Shared 1D Encoder의 Macro F1은 0.2806이었다. Identity만 추가하면 0.5182로 개선되지만 충분하지 않았다. Residual Channel-Shared Encoder는 0.8004로 크게 상승했다. 따라서 이번 결과에서 가장 방어 가능한 구조적 메시지는 **residual branch가 shared encoder 병목을 완화한다**는 것이다.

## 7. XGBoost 및 tree baseline 결과

![XGBoost Practical Baselines](figures/fig_03_xgboost_practical_baselines.png)

{markdown_table(table04)}

XGBoost completion 결과는 Macro F1 0.7961이고 Random Forest는 0.7845였다. XGBoost와 RF, Statistical Summary MLP, Residual Channel-Shared Encoder, All-Channel 1D CNN의 paired confidence interval은 모두 0을 포함했다. 따라서 XGBoost가 명확히 우월하다고 말하지 않고, tree-based practical baseline이 강하다고 정리하는 것이 안전하다.

![RF/XGBoost Feature Importance](figures/fig_05_rf_xgboost_feature_importance.png)

{markdown_table(table07)}

RF와 XGBoost 모두 s1_ax, s1_gx 관련 feature가 중요하게 나타났다. 다만 feature importance는 모델 내부 중요도이므로 생체역학적 인과로 단정하지 않는다.

## 8. Class 3 Excessive Lean 분석

![Class 3 Excessive Lean](figures/fig_04_class3_excessive_lean.png)

{markdown_table(table06)}

Class 3 Excessive Lean은 pilot 단계부터 어려운 class였다. Residual Channel-Shared Encoder 계열은 이 class에서 비교적 높은 F1을 보였지만, 완전히 해결했다고 과장해서는 안 된다.

## 9. 현재 논문에서 주장 가능한 것 / 피해야 할 것

{markdown_table(table08)}

안전한 주장은 “Residual Channel-Shared Encoder가 naive shared encoder의 병목을 완화했고, 강한 practical/neural baseline과 경쟁 가능한 성능을 보였다”는 수준이다. 피해야 할 주장은 “모든 baseline보다 통계적으로 유의하게 우월하다”, “attention이 핵심이다”, “transfer learning을 검증했다”는 식의 과장이다.

## 10. 국내 저널용 최종 범위 제안

{markdown_table(table09)}

국내 저널에서는 supervised target dataset, clean-room conversion, LOSO protocol, controlled feature extractor comparison, RF/XGBoost/Stats MLP practical baseline, Class 3 분석까지가 적절하다. SSL, external transfer, real-time system은 후속 연구 또는 별도 논문으로 분리하는 것이 안전하다.

## 11. 교수님께 확인받을 질문

1. 제안 모델명을 `Residual Channel-Shared Feature Extractor`로 정리해도 되는가?
2. Statistical Summary MLP가 가장 높은 점을 main table에 넣을지, practical baseline table로 분리할지?
3. position identity와 attention은 국내 논문 본문에서 줄이고 appendix/후속 연구로 둘지?
4. RF/XGBoost를 main comparison에 포함할지, reviewer-facing practical baseline으로 별도 제시할지?
5. 대한전기학회 투고 범위를 supervised IMU classification으로 제한해도 되는지?

## 12. Appendix: 실행 이력 및 commit

- GitHub repo: https://github.com/choijaeh01/Squat-Classifier
- Feature extractor comparison: read-only local result directory recorded in `docs/results_tracking.md`.
- XGBoost completion: `results/xgboost_only_completion/20260629_053235_xgboost_only_completion_v1/`
- Locked full matrix: `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/`
- Literature extension: `results/literature_baseline_full_extension/20260617_183952_literature_baseline_full_extension_v1/`
- v3 component ablation: `results/v3_component_ablation/20260617_202714_v3_component_ablation_v1/`
- GitHub selected artifacts: `docs/results_artifacts/`
- Report mirror: `docs/professor_report_v1/`

검증 결과, full matrix integrity check와 unit test가 통과했다. 이 보고서 작성 과정에서 새 학습, CAU training, hyperparameter tuning, split/preprocessing/scaler/feature set 변경은 수행하지 않았다.
"""


def figure_table_index_markdown() -> str:
    rows = [
        {"artifact": "fig_01_feature_extractor_macro_f1.png", "source": "table_03_feature_extractor_clean_names.csv", "script": "scripts/build_professor_report_assets.py", "report_location": "Section 5", "paper_candidate": "yes"},
        {"artifact": "fig_02_residual_effect_flow.png", "source": "table_05_component_flow.csv", "script": "scripts/build_professor_report_assets.py", "report_location": "Section 6", "paper_candidate": "yes"},
        {"artifact": "fig_03_xgboost_practical_baselines.png", "source": "table_04_xgboost_completion_clean_names.csv", "script": "scripts/build_professor_report_assets.py", "report_location": "Section 7", "paper_candidate": "maybe"},
        {"artifact": "fig_04_class3_excessive_lean.png", "source": "table_06_class3_summary.csv", "script": "scripts/build_professor_report_assets.py", "report_location": "Section 8", "paper_candidate": "yes"},
        {"artifact": "fig_05_rf_xgboost_feature_importance.png", "source": "table_07_feature_importance_top10.csv", "script": "scripts/build_professor_report_assets.py", "report_location": "Section 7", "paper_candidate": "appendix"},
        {"artifact": "fig_06_subjectwise_macro_f1.png", "source": "subjectwise_metrics_by_model.csv", "script": "scripts/build_professor_report_assets.py", "report_location": "supporting", "paper_candidate": "appendix"},
        {"artifact": "fig_07_parameter_vs_macro_f1.png", "source": "aggregate_metrics_by_model.csv", "script": "scripts/build_professor_report_assets.py", "report_location": "supporting", "paper_candidate": "appendix"},
        {"artifact": "table_03_feature_extractor_clean_names.csv", "source": "feature extractor aggregate metrics", "script": "scripts/build_professor_report_assets.py", "report_location": "Section 5", "paper_candidate": "yes"},
        {"artifact": "table_05_component_flow.csv", "source": "feature extractor aggregate metrics", "script": "scripts/build_professor_report_assets.py", "report_location": "Section 6", "paper_candidate": "yes"},
        {"artifact": "table_08_claim_audit_summary.csv", "source": "manual claim audit from result evidence", "script": "scripts/build_professor_report_assets.py", "report_location": "Section 9", "paper_candidate": "no"},
    ]
    return "# Figure and Table Index\n\n" + markdown_table(rows) + "\n"


def markdown_table_pipe_errors(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        expected: int | None = None
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not (line.startswith("|") and line.endswith("|")):
                expected = None
                continue
            count = line.count("|")
            if expected is None:
                expected = count
            elif count != expected:
                errors.append(f"{path}:{line_no}: expected {expected} pipes, found {count}")
    return errors


def missing_image_links(root: Path) -> list[str]:
    missing: list[str] = []
    for path in root.rglob("*.md"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "![ " in line:
                continue
            marker = "]("
            if line.strip().startswith("![") and marker in line and line.rstrip().endswith(")"):
                target = line.split(marker, 1)[1].rsplit(")", 1)[0]
                if target.startswith("http"):
                    continue
                if not (path.parent / target).exists():
                    missing.append(f"{path}:{line_no}:{target}")
    return missing
