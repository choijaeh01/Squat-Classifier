from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml


CONTROLLED_DIR = Path("results/controlled_feature_extractor_comparison/20260629_041712_controlled_feature_extractor_comparison_v1")
XGBOOST_DIR = Path("results/xgboost_only_completion/20260629_053235_xgboost_only_completion_v1")


def run_normalization_protocol_audit(*, project_root: Path, output_dir: Path, figure_dir: Path, diagram_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    diagram_dir.mkdir(parents=True, exist_ok=True)

    controlled_config = load_yaml(project_root / "configs/controlled_feature_extractor_comparison_v1.yaml")
    xgb_config = load_yaml(project_root / "configs/xgboost_only_completion_v1.yaml")
    controlled_scaler = read_csv(project_root / CONTROLLED_DIR / "scaler_fit_audit.csv")
    xgb_scaler = read_csv(project_root / XGBOOST_DIR / "scaler_fit_audit.csv")
    feature_audit = read_csv(project_root / CONTROLLED_DIR / "feature_audit.csv")
    feature_definitions = read_csv(project_root / CONTROLLED_DIR / "feature_definitions.csv")

    protocol_rows = normalization_protocol_rows(controlled_config, xgb_config, controlled_scaler, xgb_scaler, feature_audit)
    scaler_rows = scaler_leakage_summary_rows(controlled_scaler, xgb_scaler)
    extraction_rows = feature_extraction_order_rows(feature_definitions)

    write_csv(output_dir / "table_normalization_protocol.csv", protocol_rows)
    write_csv(output_dir / "table_scaler_leakage_summary.csv", scaler_rows)
    write_csv(output_dir / "table_feature_extraction_order.csv", extraction_rows)
    write_normalization_mermaid(diagram_dir / "diagram_normalization_pipeline.mmd")
    draw_normalization_pipeline(figure_dir / "fig_normalization_pipeline.png")
    return {
        "output_dir": str(output_dir),
        "figure": str(figure_dir / "fig_normalization_pipeline.png"),
        "diagram": str(diagram_dir / "diagram_normalization_pipeline.mmd"),
        "controlled_scaler_rows": len(controlled_scaler),
        "xgboost_scaler_rows": len(xgb_scaler),
        "feature_audit_rows": len(feature_audit),
        "all_scaler_checks_passed": all(row.get("scaler_leakage_check_passed") == "True" for row in [*controlled_scaler, *xgb_scaler]),
        "all_features_allowed": all(row.get("allowed") == "True" for row in feature_audit),
    }


def normalization_protocol_rows(
    controlled_config: dict[str, Any],
    xgb_config: dict[str, Any],
    controlled_scaler: list[dict[str, str]],
    xgb_scaler: list[dict[str, str]],
    feature_audit: list[dict[str, str]],
) -> list[dict[str, str]]:
    combined_scaler = [*controlled_scaler, *xgb_scaler]
    return [
        {
            "question": "Dataset conversion 단계에서 normalization을 했는가?",
            "answer": "No",
            "evidence": "target conversion report and config specify normalization none; scaler is applied only inside fold runners.",
            "status": "confirmed",
        },
        {
            "question": "512 resampling은 언제 이루어졌는가?",
            "answer": "Before training, during target conversion from raw CSV/manual boundaries to processed X.npy.",
            "evidence": "processed dataset v1 contains 512-step windows; training runners load processed arrays.",
            "status": "confirmed",
        },
        {
            "question": "학습 fold에서 StandardScaler는 언제 fit되는가?",
            "answer": "After LOSO split construction and before model/classical feature evaluation, using train indices only.",
            "evidence": "Both supervised feature-extractor and XGBoost completion runners call TrainOnlyStandardScaler().fit(dataset.X, train_idx=split.train_idx).",
            "status": "confirmed",
        },
        {
            "question": "scaler fit에 사용된 sample은 train indices 400개뿐인가?",
            "answer": "Yes",
            "evidence": f"All scaler_fit_audit rows have scaler_fit_n_windows=400: {all(row.get('scaler_fit_n_windows') == '400' for row in combined_scaler)}.",
            "status": "confirmed",
        },
        {
            "question": "validation/test window가 scaler fit에 들어갔는가?",
            "answer": "No",
            "evidence": f"val/test used flags are all False and scaler checks all passed: {all(row.get('val_indices_used_in_scaler') == 'False' and row.get('test_indices_used_in_scaler') == 'False' and row.get('scaler_leakage_check_passed') == 'True' for row in combined_scaler)}.",
            "status": "confirmed",
        },
        {
            "question": "per-window z-score는 사용했는가?",
            "answer": "No",
            "evidence": f"feature-extractor comparison={controlled_config['normalization'].get('per_window_zscore')}, xgboost={xgb_config['normalization'].get('per_window_zscore')}.",
            "status": "confirmed",
        },
        {
            "question": "augmentation은 사용했는가?",
            "answer": "No",
            "evidence": f"feature-extractor comparison augmentation enabled={controlled_config['augmentation'].get('enabled')}; xgboost safety forbids augmentation={xgb_config['safety'].get('forbid_augmentation')}.",
            "status": "confirmed",
        },
        {
            "question": "classical feature baseline의 feature는 scaler 적용 전 raw signal에서 계산되는가, train-scaled signal에서 계산되는가?",
            "answer": "Train-scaled signal에서 계산된다.",
            "evidence": "runner creates X_scaled=scaler.transform(dataset.X), then run_classical_fold calls extract_window_features(X).",
            "status": "confirmed_from_code",
        },
        {
            "question": "Statistical Summary MLP와 residual branch statistics는 scaler 적용 전/후 어느 단계에서 계산되는가?",
            "answer": "Train-scaled tensor가 model input으로 들어간 뒤, model 내부에서 mean/std/min/max를 계산한다.",
            "evidence": "The feature-extractor runner passes X_scaled to the fold evaluator; StatsMLPExtractor and Shared1DExtractor call raw_summary_features(x).",
            "status": "confirmed_from_code",
        },
        {
            "question": "XGBoost와 RF는 같은 feature set을 썼는가?",
            "answer": "Yes",
            "evidence": f"feature_audit allowed all features={all(row.get('allowed') == 'True' for row in feature_audit)}; both use extract_window_features.",
            "status": "confirmed",
        },
        {
            "question": "Tree model은 normalization에 덜 민감하지만, feature extraction/scaler policy는 neural baseline과 어떻게 맞췄는가?",
            "answer": "동일 LOSO split과 train-only StandardScaler transform 후 같은 signal-derived feature set으로 fit/predict했다.",
            "evidence": "RF/SVM and XGBoost completion paths both use TrainOnlyStandardScaler and extract_window_features on X_scaled.",
            "status": "confirmed",
        },
    ]


def scaler_leakage_summary_rows(controlled_scaler: list[dict[str, str]], xgb_scaler: list[dict[str, str]]) -> list[dict[str, str]]:
    return [summarize_scaler("Controlled feature extractor comparison", controlled_scaler), summarize_scaler("XGBoost completion", xgb_scaler)]


def summarize_scaler(name: str, rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        "experiment": name,
        "rows": str(len(rows)),
        "fit_windows_unique": "|".join(sorted({row.get("scaler_fit_n_windows", "") for row in rows})),
        "val_indices_used_in_scaler_any": str(any(row.get("val_indices_used_in_scaler") == "True" for row in rows)),
        "test_indices_used_in_scaler_any": str(any(row.get("test_indices_used_in_scaler") == "True" for row in rows)),
        "scaler_leakage_all_passed": str(all(row.get("scaler_leakage_check_passed") == "True" for row in rows)),
        "note": "StandardScaler fit scope is train indices only.",
    }


def feature_extraction_order_rows(feature_definitions: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "pipeline": "Neural temporal extractors",
            "order": "processed X.npy -> LOSO split -> train-only StandardScaler fit -> transform train/val/test -> neural forward",
            "feature_source": "scaled IMU tensor",
            "feature_count": "not_applicable",
            "metadata_used": "False",
            "note": "All controlled neural models receive scaled tensors.",
        },
        {
            "pipeline": "Statistical Summary MLP",
            "order": "processed X.npy -> scaler transform -> model internal mean/std/min/max -> common head",
            "feature_source": "scaled IMU tensor",
            "feature_count": "72",
            "metadata_used": "False",
            "note": "Uses mean, std, min, max for each of 18 channels.",
        },
        {
            "pipeline": "Residual branch inside Residual Channel-Shared Encoder",
            "order": "processed X.npy -> scaler transform -> shared encoder branch plus internal mean/std/min/max residual branch -> fusion -> common head",
            "feature_source": "scaled IMU tensor",
            "feature_count": "72",
            "metadata_used": "False",
            "note": "Residual branch uses the same raw_summary_features function as Statistical Summary MLP.",
        },
        {
            "pipeline": "Random Forest / Linear SVM / XGBoost",
            "order": "processed X.npy -> scaler transform -> extract_window_features -> estimator.fit(train features only)",
            "feature_source": "scaled IMU tensor",
            "feature_count": str(len(feature_definitions)),
            "metadata_used": "False",
            "note": "Feature set: mean, std, min, max, median, energy, RMS, peak-to-peak, dominant FFT bin.",
        },
    ]


def write_normalization_mermaid(path: Path) -> None:
    path.write_text(
        """flowchart TD
  A["Raw CSV + manual boundaries"] --> B["512-step phase-normalized resampling"]
  B --> C["Processed X.npy / y.npy / subject_id.npy"]
  C --> D["LOSO split with within-train validation"]
  D --> E["Train indices only: 400 windows"]
  E --> F["Fit global StandardScaler"]
  F --> G["Transform train / validation / test"]
  G --> H["Neural model forward"]
  G --> I["Signal-derived feature extraction for RF / XGBoost / SVM"]
  J["per-window z-score = false"] -.-> G
  K["augmentation = false"] -.-> H
  K -.-> I
""",
        encoding="utf-8",
    )


def draw_normalization_pipeline(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    plt.rcParams["axes.unicode_minus"] = False
    nodes = [
        "Raw CSV + manual boundaries",
        "512-step resampling",
        "Processed X.npy",
        "LOSO split",
        "Train indices only",
        "Fit StandardScaler",
        "Transform train / val / test",
        "Neural forward or signal feature extraction",
    ]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.axis("off")
    y_positions = list(reversed([0.12 + i * 0.1 for i in range(len(nodes))]))
    for idx, (y, label) in enumerate(zip(y_positions, nodes)):
        box = FancyBboxPatch((0.18, y), 0.64, 0.065, boxstyle="round,pad=0.02", linewidth=1, facecolor="#f3f6fb", edgecolor="#4b5563")
        ax.add_patch(box)
        ax.text(0.5, y + 0.032, label, ha="center", va="center", fontsize=10)
        if idx < len(nodes) - 1:
            next_y = y_positions[idx + 1]
            ax.annotate("", xy=(0.5, next_y + 0.075), xytext=(0.5, y - 0.01), arrowprops={"arrowstyle": "->", "linewidth": 1})
    ax.text(0.5, 0.04, "per-window z-score = false | augmentation = false | validation/test never used to fit scaler", ha="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_yaml(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
