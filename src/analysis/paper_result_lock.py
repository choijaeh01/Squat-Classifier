from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_MODELS = [
    "all_channel_conv1d_v1",
    "all_channel_conv1d_small",
    "cnn2d_baseline_v1",
    "channel_shared_meanpool_v2",
    "channel_shared_attentionpool_v2",
    "channel_shared_posres_attention_v3",
    "modality_shared_sensorattn_v3",
]
EXPECTED_SEEDS = [42, 123, 2025]
EXPECTED_FOLDS = [1, 2, 3, 4, 5, 6]
PRIMARY_MODEL = "channel_shared_posres_attention_v3"


def validate_result_integrity(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_plan = _read_csv(run_dir / "run_plan.csv")
    fold_metrics = _read_csv(run_dir / "fold_metrics.csv")
    failed_runs = _read_csv(run_dir / "failed_runs.csv")
    validation = _read_csv(run_dir / "validation_policy_summary.csv")
    scaler_audit = _read_csv(run_dir / "scaler_fit_audit.csv")
    aggregate = _read_csv(run_dir / "aggregate_metrics_by_model.csv")

    expected_combos = {(model, seed, fold) for model in EXPECTED_MODELS for seed in EXPECTED_SEEDS for fold in EXPECTED_FOLDS}
    observed_combos = {
        (str(row["model_name"]), int(row["seed"]), int(row["fold_id"]))
        for row in fold_metrics
        if str(row.get("status")) == "ok"
    }

    aggregate_match = _aggregate_matches_fold_metrics(fold_metrics, aggregate)
    split_sizes_ok = all(
        int(row["n_train"]) == 400 and int(row["n_val"]) == 100 and int(row["n_test"]) == 100
        for row in validation
    )
    fold_sizes_ok = all(
        int(row["n_train"]) == 400 and int(row["n_val"]) == 100 and int(row["n_test"]) == 100
        for row in fold_metrics
        if str(row.get("status")) == "ok"
    )
    all_leakage = all(str(row.get("leakage_check_passed")) == "True" for row in fold_metrics)
    all_scaler = all(str(row.get("scaler_leakage_check_passed")) == "True" for row in fold_metrics)
    scaler_audit_ok = all(
        str(row.get("val_indices_used_in_scaler")) == "False"
        and str(row.get("test_indices_used_in_scaler")) == "False"
        and str(row.get("scaler_leakage_check_passed")) == "True"
        for row in scaler_audit
    )
    required_files = {
        "config_snapshot.yaml": (run_dir / "config_snapshot.yaml").exists(),
        "config_hash.txt": (run_dir / "config_hash.txt").exists(),
        "git_commit": (run_dir / "git_commit.txt").exists() or (run_dir / "git_commit_unavailable.txt").exists(),
        "model_parameter_counts.csv": (run_dir / "model_parameter_counts.csv").exists(),
        "aggregate_metrics_by_model.csv": (run_dir / "aggregate_metrics_by_model.csv").exists(),
    }
    checks = {
        "run_plan_total": len(run_plan),
        "run_plan_total_ok": len(run_plan) == 126,
        "fold_metric_success_runs": len([row for row in fold_metrics if str(row.get("status")) == "ok"]),
        "fold_metric_success_runs_ok": len([row for row in fold_metrics if str(row.get("status")) == "ok"]) == 126,
        "failed_run_count": len(failed_runs),
        "failed_runs_empty": len(failed_runs) == 0,
        "all_leakage_checks_passed": all_leakage,
        "all_scaler_leakage_checks_passed": all_scaler and scaler_audit_ok,
        "split_sizes_ok": split_sizes_ok and fold_sizes_ok,
        "all_model_seed_fold_combinations_present": observed_combos == expected_combos,
        "missing_combinations": sorted(f"{m}|{s}|{f}" for m, s, f in expected_combos - observed_combos),
        "required_files": required_files,
        "required_files_ok": all(required_files.values()),
        "aggregate_metrics_match_fold_metrics": aggregate_match,
    }
    checks["all_checks_passed"] = all(
        bool(checks[key])
        for key in [
            "run_plan_total_ok",
            "fold_metric_success_runs_ok",
            "failed_runs_empty",
            "all_leakage_checks_passed",
            "all_scaler_leakage_checks_passed",
            "split_sizes_ok",
            "all_model_seed_fold_combinations_present",
            "required_files_ok",
            "aggregate_metrics_match_fold_metrics",
        ]
    )
    return checks


def lock_paper_results(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    output_dir = run_dir / "paper_lock"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    integrity = validate_result_integrity(run_dir)
    _write_json(output_dir / "result_integrity_check.json", integrity)

    aggregate = _read_csv(run_dir / "aggregate_metrics_by_model.csv")
    params = _read_csv(run_dir / "model_parameter_counts.csv")
    classwise = _read_csv(run_dir / "classwise_metrics_by_model.csv")
    subjectwise = _read_csv(run_dir / "subjectwise_metrics_by_model.csv")
    paired = _read_csv(run_dir / "paired_model_differences.csv")
    fold_metrics = _read_csv(run_dir / "fold_metrics.csv")

    locked_summary = _locked_summary_rows(run_dir, integrity, aggregate)
    ranking = _ranking_rows(aggregate)
    primary = _primary_comparisons_rows(paired)
    _write_csv(output_dir / "locked_result_summary.csv", locked_summary)
    _write_csv(output_dir / "locked_model_ranking.csv", ranking)
    _write_csv(output_dir / "locked_primary_comparisons.csv", primary)
    (output_dir / "locked_claim_audit.md").write_text(_claim_audit_markdown(), encoding="utf-8")

    _write_csv(tables_dir / "table1_dataset_summary.csv", _dataset_summary_rows())
    _write_csv(tables_dir / "table2_model_parameter_counts.csv", params)
    _write_csv(tables_dir / "table3_main_model_comparison.csv", _main_model_comparison_rows(aggregate, params))
    _write_csv(tables_dir / "table4_classwise_f1_recall.csv", classwise)
    _write_csv(tables_dir / "table5_subjectwise_macro_f1.csv", subjectwise)
    _write_csv(tables_dir / "table6_paired_differences.csv", paired)
    _write_csv(tables_dir / "table7_ablation_summary.csv", _ablation_summary_rows(aggregate, paired))

    _create_figures(run_dir, figures_dir)
    (output_dir / "figure_captions.md").write_text(_figure_captions_markdown(), encoding="utf-8")
    _write_csv(output_dir / "locked_seed_fold_metrics.csv", fold_metrics)
    return output_dir


def _aggregate_matches_fold_metrics(fold_metrics: list[dict[str, str]], aggregate: list[dict[str, str]]) -> bool:
    aggregate_by_model = {row["model_name"]: row for row in aggregate}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fold_metrics:
        if str(row.get("status")) == "ok":
            grouped[row["model_name"]].append(row)
    for model, rows in grouped.items():
        agg = aggregate_by_model.get(model)
        if agg is None:
            return False
        checks = {
            "mean_accuracy": np.mean([float(row["test_accuracy"]) for row in rows]),
            "mean_macro_f1": np.mean([float(row["test_macro_f1"]) for row in rows]),
            "mean_weighted_f1": np.mean([float(row["test_weighted_f1"]) for row in rows]),
        }
        for key, value in checks.items():
            if abs(float(agg[key]) - float(value)) > 1e-9:
                return False
    return True


def _locked_summary_rows(run_dir: Path, integrity: dict[str, Any], aggregate: list[dict[str, str]]) -> list[dict[str, Any]]:
    best = max(aggregate, key=lambda row: float(row["mean_macro_f1"]))
    return [
        {"field": "run_dir", "value": str(run_dir)},
        {"field": "total_runs", "value": integrity["run_plan_total"]},
        {"field": "success_runs", "value": integrity["fold_metric_success_runs"]},
        {"field": "failed_runs", "value": integrity["failed_run_count"]},
        {"field": "all_checks_passed", "value": integrity["all_checks_passed"]},
        {"field": "best_model_by_macro_f1", "value": best["model_name"]},
        {"field": "best_macro_f1", "value": best["mean_macro_f1"]},
    ]


def _ranking_rows(aggregate: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for rank, row in enumerate(sorted(aggregate, key=lambda r: float(r["mean_macro_f1"]), reverse=True), start=1):
        payload = dict(row)
        payload["rank_by_macro_f1"] = rank
        rows.append(payload)
    return rows


def _primary_comparisons_rows(paired: list[dict[str, str]]) -> list[dict[str, Any]]:
    wanted = {
        ("all_channel_conv1d_v1", PRIMARY_MODEL),
        ("all_channel_conv1d_small", PRIMARY_MODEL),
    }
    rows = []
    for row in paired:
        key = (row["reference_model"], row["comparison_model"])
        if key in wanted and row["metric"] == "macro_f1":
            rows.append(dict(row))
    return rows


def _dataset_summary_rows() -> list[dict[str, Any]]:
    return [
        {"item": "samples", "value": 600, "notes": "processed windows"},
        {"item": "window_length", "value": 512, "notes": "linear phase interpolation"},
        {"item": "channels", "value": 18, "notes": "3 IMUs x acc/gyro x xyz"},
        {"item": "subjects", "value": 6, "notes": "LOSO evaluation"},
        {"item": "classes", "value": 5, "notes": "squat posture classes"},
        {"item": "windows_per_subject_class", "value": 20, "notes": "balanced"},
        {"item": "train_per_fold", "value": 400, "notes": "16 windows per candidate subject-class"},
        {"item": "validation_per_fold", "value": 100, "notes": "4 windows per candidate subject-class"},
        {"item": "test_per_fold", "value": 100, "notes": "held-out subject only"},
    ]


def _main_model_comparison_rows(aggregate: list[dict[str, str]], params: list[dict[str, str]]) -> list[dict[str, Any]]:
    params_by_model = {row["model_name"]: row for row in params}
    rows = []
    for rank, row in enumerate(sorted(aggregate, key=lambda r: float(r["mean_macro_f1"]), reverse=True), start=1):
        model = row["model_name"]
        rows.append(
            {
                "model_name": model,
                "total_params": params_by_model[model]["total_params"],
                "accuracy_mean": row["mean_accuracy"],
                "accuracy_std": row["std_accuracy"],
                "macro_f1_mean": row["mean_macro_f1"],
                "macro_f1_std": row["std_macro_f1"],
                "macro_f1_ci_low": row["macro_f1_ci_low"],
                "macro_f1_ci_high": row["macro_f1_ci_high"],
                "weighted_f1_mean": row["mean_weighted_f1"],
                "rank_by_macro_f1": rank,
                "safe_interpretation": _safe_interpretation(model, rank),
            }
        )
    return rows


def _safe_interpretation(model: str, rank: int) -> str:
    if model == PRIMARY_MODEL:
        return "Best mean Macro F1; competitive with all-channel baselines; paired CI vs baselines includes zero."
    if model in {"all_channel_conv1d_v1", "all_channel_conv1d_small"}:
        return "Strong all-channel baseline."
    if "meanpool_v2" in model or "attentionpool_v2" in model:
        return "Naive shared encoder underfits and loses channel identity."
    return "Secondary baseline or ablation context."


def _ablation_summary_rows(aggregate: list[dict[str, str]], paired: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_model = {row["model_name"]: row for row in aggregate}
    comparisons = [
        ("channel_shared_meanpool_v2", PRIMARY_MODEL, "naive shared meanpool v2 vs v3"),
        ("channel_shared_attentionpool_v2", PRIMARY_MODEL, "shared attentionpool v2 vs v3"),
        ("all_channel_conv1d_v1", PRIMARY_MODEL, "all-channel v1 vs v3"),
        ("all_channel_conv1d_small", PRIMARY_MODEL, "small all-channel vs v3"),
        ("modality_shared_sensorattn_v3", PRIMARY_MODEL, "modality shared sensor attention vs v3"),
        ("cnn2d_baseline_v1", PRIMARY_MODEL, "2D CNN baseline vs v3"),
    ]
    paired_lookup = {
        (row["reference_model"], row["comparison_model"], row["metric"]): row
        for row in paired
    }
    rows = []
    for base, v3, label in comparisons:
        base_macro = float(by_model[base]["mean_macro_f1"])
        v3_macro = float(by_model[v3]["mean_macro_f1"])
        paired_row = paired_lookup.get((base, v3, "macro_f1"))
        rows.append(
            {
                "comparison": label,
                "baseline_model": base,
                "proposed_model": v3,
                "baseline_macro_f1": base_macro,
                "proposed_macro_f1": v3_macro,
                "aggregate_delta_macro_f1": v3_macro - base_macro,
                "paired_delta_macro_f1": "" if paired_row is None else paired_row["mean_difference"],
                "paired_ci": "" if paired_row is None else f"[{paired_row['ci_low']}, {paired_row['ci_high']}]",
                "safe_interpretation": _ablation_interpretation(base),
            }
        )
    return rows


def _ablation_interpretation(model: str) -> str:
    if model.startswith("channel_shared_"):
        return "Supports channel identity and residual branch necessity."
    if model.startswith("all_channel_"):
        return "V3 is competitive; do not claim statistically significant superiority."
    if model.startswith("modality_shared_"):
        return "Modality-shared design improves over naive shared but remains below v3."
    return "Baseline comparison."


def _create_figures(run_dir: Path, figures_dir: Path) -> None:
    source = run_dir / "figures"
    copy_map = {
        "model_macro_f1_bar_ci.png": "fig1_model_macro_f1_ci.png",
        "model_parameter_count_vs_macro_f1.png": "fig2_parameter_count_vs_macro_f1.png",
        "classwise_recall_heatmap.png": "fig3_classwise_recall_heatmap.png",
        "class3_recall_f1_by_model.png": "fig4_class3_recall_f1_by_model.png",
        "subjectwise_macro_f1_heatmap.png": "fig5_subjectwise_macro_f1_heatmap.png",
    }
    for src_name, dst_name in copy_map.items():
        src = source / src_name
        if src.exists():
            shutil.copy2(src, figures_dir / dst_name)
    _plot_v3_paired_delta(run_dir, figures_dir / "fig6_v3_vs_baseline_paired_delta.png")
    _plot_single_confusion(run_dir, PRIMARY_MODEL, figures_dir / "fig7_aggregate_confusion_matrix_v3.png")
    _plot_single_confusion(run_dir, "all_channel_conv1d_v1", figures_dir / "fig8_aggregate_confusion_matrix_all_channel_v1.png")


def _plot_v3_paired_delta(run_dir: Path, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        fallback = run_dir / "figures" / "paired_delta_macro_f1_vs_all_channel_v1.png"
        if fallback.exists():
            shutil.copy2(fallback, path)
        return

    rows = _read_csv(run_dir / "paired_model_differences.csv")
    selected = [
        row for row in rows
        if row["metric"] == "macro_f1"
        and row["comparison_model"] == PRIMARY_MODEL
        and row["reference_model"] in {"all_channel_conv1d_v1", "all_channel_conv1d_small"}
    ]
    labels = [row["reference_model"] for row in selected]
    values = np.asarray([float(row["mean_difference"]) for row in selected])
    lows = np.asarray([float(row["ci_low"]) for row in selected])
    highs = np.asarray([float(row["ci_high"]) for row in selected])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(range(len(labels)), values, yerr=np.vstack([values - lows, highs - values]), capsize=3)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylabel("Paired Macro F1 delta")
    ax.set_title("Proposed v3 vs all-channel baselines")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_single_confusion(run_dir: Path, model_name: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        fallback = run_dir / "figures" / "aggregate_confusion_matrix_per_model.png"
        if fallback.exists():
            shutil.copy2(fallback, path)
        return

    rows = [
        row for row in _read_csv(run_dir / "confusion_matrices.csv")
        if row.get("scope") == "aggregate" and row.get("model_name") == model_name
    ]
    matrix = np.zeros((5, 5), dtype=float)
    for row in rows:
        matrix[int(row["true_class"]), int(row["pred_class"])] += float(row["count"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_title(model_name)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(5), [str(i) for i in range(5)])
    ax.set_yticks(range(5), [str(i) for i in range(5)])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _figure_captions_markdown() -> str:
    return """# Figure Captions for Paper Result Lock v1

## fig1_model_macro_f1_ci.png
- Korean: 모델별 평균 Macro F1과 bootstrap confidence interval을 비교한 그림.
- English: Mean Macro F1 with bootstrap confidence intervals across supervised LOSO runs.
- Key message: position-aware shared v3가 가장 높은 평균 Macro F1을 보인다.
- Suggested placement: Results, main model comparison.

## fig2_parameter_count_vs_macro_f1.png
- Korean: 모델 parameter count와 평균 Macro F1의 관계.
- English: Parameter count versus mean Macro F1 for each model.
- Key message: v3는 all-channel v1과 유사한 parameter budget에서 더 높은 평균 Macro F1을 보인다.
- Suggested placement: Results or Discussion.

## fig3_classwise_recall_heatmap.png
- Korean: 모델별 class-wise recall heatmap.
- English: Class-wise recall heatmap across models.
- Key message: naive shared v2는 특히 class 3에서 낮고, v3는 class-wise 균형이 개선된다.
- Suggested placement: Class-wise analysis.

## fig4_class3_recall_f1_by_model.png
- Korean: Class 3 Excessive Lean의 recall과 F1 비교.
- English: Recall and F1 for Class 3 Excessive Lean.
- Key message: v3가 naive shared v2 대비 class 3 성능을 크게 개선한다.
- Suggested placement: Class-wise analysis.

## fig5_subjectwise_macro_f1_heatmap.png
- Korean: subject-wise Macro F1 heatmap.
- English: Subject-wise Macro F1 heatmap.
- Key message: subject 5가 여러 모델에서 어려운 held-out subject로 나타난다.
- Suggested placement: Robustness analysis.

## fig6_v3_vs_baseline_paired_delta.png
- Korean: v3와 all-channel baselines의 paired Macro F1 차이.
- English: Paired Macro F1 differences between the proposed v3 model and all-channel baselines.
- Key message: v3 평균 차이는 양수지만 CI가 0을 포함하므로 우월성 주장은 조심해야 한다.
- Suggested placement: Main results or Discussion.

## fig7_aggregate_confusion_matrix_v3.png
- Korean: v3 모델의 aggregate confusion matrix.
- English: Aggregate confusion matrix for the proposed v3 model.
- Key message: v3의 주요 class confusion 구조를 보여준다.
- Suggested placement: Error analysis.

## fig8_aggregate_confusion_matrix_all_channel_v1.png
- Korean: all-channel Conv1D v1의 aggregate confusion matrix.
- English: Aggregate confusion matrix for the all-channel Conv1D v1 baseline.
- Key message: v3와 baseline의 error pattern 비교 기준을 제공한다.
- Suggested placement: Error analysis.
"""


def _claim_audit_markdown() -> str:
    return """# Locked Claim Audit

## Safe claims

- The full supervised matrix completed 126/126 runs without failed runs.
- All split and scaler leakage checks passed.
- The proposed position-aware shared encoder v3 achieved the highest mean Macro F1 in this matrix.
- Naive shared v2 models performed substantially worse than v3.
- The paired differences between v3 and all-channel baselines were positive on average, but the confidence intervals included zero.

## Unsafe claims

- Do not claim statistically significant superiority over all-channel Conv1D.
- Do not claim transfer learning effectiveness.
- Do not claim SSL, augmentation, focal loss, or balanced sampling benefits.
- Do not claim external dataset generalization.
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
