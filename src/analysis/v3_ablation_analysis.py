from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from analysis.full_matrix_analysis import compute_paired_model_differences
from models.channel_metadata import AXIS_NAMES, CHANNEL_ORDER, MODALITY_NAMES, SENSOR_NAMES, TARGET_CHANNELS
from training.supervised_trainer import write_csv


ABLATION_MODELS = {
    "channel_shared_posres_attention_v3_no_residual",
    "channel_shared_posres_attention_v3_residual_only_mlp",
    "channel_shared_posres_attention_v3_no_identity",
    "channel_shared_posres_meanpool_v3_no_attention",
}


def merge_v3_ablation_with_existing_results(*, project_root: Path, run_dir: Path, config: dict[str, Any]) -> None:
    refs = config["reference_models_read_only"]
    locked_dir = (project_root / refs["locked_full_matrix_dir"]).resolve()
    literature_dir = (project_root / refs["literature_full_extension_dir"]).resolve()
    output_dir = run_dir / "merged_with_existing_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    locked_agg = [_with_result_type(row, "locked_3seed_full") for row in _read_csv(locked_dir / "aggregate_metrics_by_model.csv")]
    literature_agg = [_with_result_type(row, "literature_extension_3seed_full") for row in _read_csv(literature_dir / "aggregate_metrics_by_model.csv")]
    ablation_agg = [_with_result_type(row, "v3_component_ablation_3seed_full") for row in _read_csv(run_dir / "aggregate_metrics_by_model.csv")]
    reference_models = set(str(item) for item in refs["reference_models"])
    merged = [
        row
        for row in locked_agg + literature_agg + ablation_agg
        if row["model_name"] in reference_models or row["model_name"] in ABLATION_MODELS
    ]
    merged = sorted(merged, key=lambda row: float(row.get("mean_macro_f1") or 0.0), reverse=True)
    for rank, row in enumerate(merged, start=1):
        row["rank_by_macro_f1"] = rank
    write_csv(output_dir / "merged_ablation_comparison.csv", merged)

    locked_fold = _read_csv(locked_dir / "fold_metrics.csv")
    literature_fold = _read_csv(literature_dir / "fold_metrics.csv")
    ablation_fold = _read_csv(run_dir / "fold_metrics.csv")
    combined_fold = locked_fold + literature_fold + ablation_fold
    write_csv(output_dir / "paired_differences_vs_v3.csv", compute_paired_model_differences(combined_fold, ["channel_shared_posres_attention_v3"], metric="macro_f1", bootstrap_n=10000, seed=42))
    write_csv(output_dir / "paired_differences_vs_rf.csv", compute_paired_model_differences(combined_fold, ["feature_random_forest_v1"], metric="macro_f1", bootstrap_n=10000, seed=42))
    write_csv(output_dir / "paired_differences_vs_rescnn_bigru.csv", compute_paired_model_differences(combined_fold, ["rescnn_bigru_attention_lite_v1"], metric="macro_f1", bootstrap_n=10000, seed=42))

    component_rows = _component_contribution_rows(merged, combined_fold)
    write_csv(output_dir / "component_contribution_summary.csv", component_rows)
    write_csv(output_dir / "class3_component_ablation_summary.csv", _class3_summary(locked_dir, literature_dir, run_dir))
    write_csv(output_dir / "subjectwise_component_ablation_summary.csv", _subjectwise_summary(locked_dir, literature_dir, run_dir))
    (output_dir / "claim_implication_memo_for_user.md").write_text(_claim_memo(component_rows), encoding="utf-8")
    analyze_attention_rf_alignment(project_root=project_root, output_dir=output_dir)
    generate_v3_ablation_figures(run_dir)


def analyze_attention_rf_alignment(*, project_root: Path, output_dir: Path) -> None:
    attention_dir = project_root / "results" / "review_bundles" / "v3_attention_token_importance_inference_only"
    literature_dir = project_root / "results" / "literature_baseline_full_extension" / "20260617_183952_literature_baseline_full_extension_v1"
    attention_path = attention_dir / "v3_attention_weight_summary_by_channel.csv"
    attention_class_path = attention_dir / "v3_attention_weight_summary_by_class_channel.csv"
    rf_path = literature_dir / "random_forest_feature_importance_aggregate.csv"
    rf_fold_path = literature_dir / "random_forest_feature_importance_by_fold.csv"
    if not attention_path.exists() or not rf_path.exists():
        (output_dir / "attention_rf_alignment_summary.md").write_text(
            "# Attention-RF Alignment\n\n분석 skipped: v3 attention summary 또는 RandomForest feature importance 파일이 없다.\n",
            encoding="utf-8",
        )
        write_csv(output_dir / "attention_rf_alignment_channel.csv", [])
        write_csv(output_dir / "attention_rf_alignment_group.csv", [])
        write_csv(output_dir / "attention_rf_alignment_top_attention_by_class.csv", [])
        return

    attention_rows = _read_csv(attention_path)
    rf_rows = _read_csv(rf_path)
    rf_channel = _rf_channel_importance(rf_rows)
    attention_by_channel = {row["channel_name"]: float(row["mean_attention_weight"]) for row in attention_rows}
    channel_rows = []
    for channel_name in CHANNEL_ORDER:
        channel_rows.append(
            {
                "channel_name": channel_name,
                "v3_mean_attention_weight": attention_by_channel.get(channel_name, 0.0),
                "rf_channel_importance": rf_channel.get(channel_name, 0.0),
            }
        )
    pearson = _pearson([row["v3_mean_attention_weight"] for row in channel_rows], [row["rf_channel_importance"] for row in channel_rows])
    spearman = _spearman([row["v3_mean_attention_weight"] for row in channel_rows], [row["rf_channel_importance"] for row in channel_rows])
    for row in channel_rows:
        row["pearson_all_channels"] = pearson
        row["spearman_all_channels"] = spearman
    write_csv(output_dir / "attention_rf_alignment_channel.csv", channel_rows)

    group_rows = _group_alignment_rows(channel_rows)
    write_csv(output_dir / "attention_rf_alignment_group.csv", group_rows)
    class_rows = _read_csv(attention_class_path) if attention_class_path.exists() else []
    top_by_class = _top_attention_by_class(class_rows)
    write_csv(output_dir / "attention_rf_alignment_top_attention_by_class.csv", top_by_class)
    rf_top = sorted(rf_rows, key=lambda row: float(row.get("mean_importance") or 0.0), reverse=True)[:10]
    fold_note = "available" if rf_fold_path.exists() else "missing"
    (output_dir / "attention_rf_alignment_summary.md").write_text(
        _alignment_summary(pearson, spearman, group_rows, top_by_class, rf_top, fold_note),
        encoding="utf-8",
    )


def generate_v3_ablation_figures(run_dir: Path) -> None:
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
    merged_dir = run_dir / "merged_with_existing_results"
    component = _read_csv(merged_dir / "component_contribution_summary.csv")
    alignment = _read_csv(merged_dir / "attention_rf_alignment_channel.csv")
    _bar_ci(plt, aggregate, "mean_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", figures_dir / "ablation_macro_f1_bar_ci.png", "v3 component ablation macro F1")
    _bar_ci(plt, aggregate, "mean_accuracy", None, None, figures_dir / "ablation_accuracy_bar_ci.png", "v3 component ablation accuracy")
    _component_delta(plt, component, figures_dir / "component_contribution_delta_macro_f1.png")
    _class3(plt, classwise, figures_dir / "class3_ablation_f1_recall.png")
    _subject_heatmap(plt, subjectwise, figures_dir / "subjectwise_ablation_heatmap.png")
    _residual_vs_rf(plt, run_dir, figures_dir / "residual_only_vs_rf_macro_f1.png")
    _attention_rf_scatter(plt, alignment, figures_dir / "v3_attention_vs_rf_importance_channel.png")
    _attention_top_by_class(plt, run_dir, figures_dir / "v3_attention_top_channels_by_class.png")


def _component_contribution_rows(merged_rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = [
        ("v3 original vs no_residual", "channel_shared_posres_attention_v3", "channel_shared_posres_attention_v3_no_residual"),
        ("v3 original vs residual_only_mlp", "channel_shared_posres_attention_v3", "channel_shared_posres_attention_v3_residual_only_mlp"),
        ("v3 original vs no_identity", "channel_shared_posres_attention_v3", "channel_shared_posres_attention_v3_no_identity"),
        ("v3 original vs no_attention", "channel_shared_posres_attention_v3", "channel_shared_posres_meanpool_v3_no_attention"),
        ("residual_only_mlp vs feature_random_forest_v1", "feature_random_forest_v1", "channel_shared_posres_attention_v3_residual_only_mlp"),
        ("no_residual vs all_channel_conv1d_small", "all_channel_conv1d_small", "channel_shared_posres_attention_v3_no_residual"),
        ("no_identity vs channel_shared_attentionpool_v2", "channel_shared_attentionpool_v2", "channel_shared_posres_attention_v3_no_identity"),
        ("no_identity vs channel_shared_meanpool_v2", "channel_shared_meanpool_v2", "channel_shared_posres_attention_v3_no_identity"),
    ]
    by_model = {row["model_name"]: row for row in merged_rows}
    diff_rows = compute_paired_model_differences(fold_rows, sorted({reference for _, reference, _ in pairs}), metric="macro_f1", bootstrap_n=10000, seed=42)
    diff_by_pair = {(row["reference_model"], row["comparison_model"]): row for row in diff_rows}
    rows = []
    for comparison_name, reference, comparison in pairs:
        ref_metric = float(by_model.get(reference, {}).get("mean_macro_f1") or 0.0)
        cmp_metric = float(by_model.get(comparison, {}).get("mean_macro_f1") or 0.0)
        paired = diff_by_pair.get((reference, comparison), {})
        rows.append(
            {
                "comparison": comparison_name,
                "reference_model": reference,
                "comparison_model": comparison,
                "reference_macro_f1": ref_metric,
                "comparison_macro_f1": cmp_metric,
                "mean_delta_comparison_minus_reference": paired.get("mean_difference", cmp_metric - ref_metric),
                "ci_low": paired.get("ci_low", ""),
                "ci_high": paired.get("ci_high", ""),
                "n_pairs": paired.get("n_pairs", ""),
            }
        )
    return rows


def _class3_summary(*dirs: Path) -> list[dict[str, Any]]:
    rows = []
    for directory in dirs:
        for row in _read_csv(directory / "classwise_metrics_by_model.csv"):
            if str(row.get("class_id")) == "3":
                payload = dict(row)
                payload["source_dir"] = str(directory)
                rows.append(payload)
    keep = ABLATION_MODELS | {"channel_shared_posres_attention_v3", "feature_random_forest_v1", "rescnn_bigru_attention_lite_v1", "all_channel_conv1d_small", "all_channel_conv1d_v1"}
    return [row for row in rows if row["model_name"] in keep]


def _subjectwise_summary(*dirs: Path) -> list[dict[str, Any]]:
    rows = []
    keep = ABLATION_MODELS | {"channel_shared_posres_attention_v3", "feature_random_forest_v1", "rescnn_bigru_attention_lite_v1", "all_channel_conv1d_small", "all_channel_conv1d_v1"}
    for directory in dirs:
        for row in _read_csv(directory / "subjectwise_metrics_by_model.csv"):
            if row["model_name"] in keep:
                payload = dict(row)
                payload["source_dir"] = str(directory)
                rows.append(payload)
    return rows


def _rf_channel_importance(rows: list[dict[str, str]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for row in rows:
        channel_name = str(row["channel_name"])
        output[channel_name] = output.get(channel_name, 0.0) + float(row.get("mean_importance") or 0.0)
    return output


def _group_alignment_rows(channel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    spec_by_name = {spec.name: spec for spec in TARGET_CHANNELS}
    for row in channel_rows:
        spec = spec_by_name[row["channel_name"]]
        keys = [
            ("sensor", SENSOR_NAMES[spec.sensor_id]),
            ("modality", MODALITY_NAMES[spec.modality_id]),
            ("axis", AXIS_NAMES[spec.axis_id]),
        ]
        for group_type, group_name in keys:
            payload = groups.setdefault((group_type, group_name), {"attention": [], "rf": []})
            payload["attention"].append(float(row["v3_mean_attention_weight"]))
            payload["rf"].append(float(row["rf_channel_importance"]))
    output = []
    for (group_type, group_name), values in sorted(groups.items()):
        output.append(
            {
                "group_type": group_type,
                "group_name": group_name,
                "v3_mean_attention_weight": float(np.mean(values["attention"])),
                "rf_mean_channel_importance": float(np.mean(values["rf"])),
                "rf_sum_channel_importance": float(np.sum(values["rf"])),
            }
        )
    return output


def _top_attention_by_class(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["class_id"]), []).append(row)
    output = []
    for class_id, values in sorted(grouped.items(), key=lambda item: int(item[0])):
        top = sorted(values, key=lambda row: float(row.get("mean_attention_weight") or 0.0), reverse=True)[:5]
        for rank, row in enumerate(top, start=1):
            payload = dict(row)
            payload["rank_within_class"] = rank
            output.append(payload)
    return output


def _alignment_summary(pearson: float, spearman: float, group_rows: list[dict[str, Any]], top_by_class: list[dict[str, str]], rf_top: list[dict[str, str]], fold_note: str) -> str:
    lines = [
        "# Attention-RF Alignment Summary",
        "",
        "이 분석은 기존 v3 attention weight와 RandomForest feature importance를 사용한 inference-only/post-hoc 비교다. Attention weight와 feature importance는 같은 의미가 아니므로 정성적 alignment로만 해석한다.",
        "",
        f"- channel-level Pearson correlation: {pearson:.4f}",
        f"- channel-level Spearman correlation: {spearman:.4f}",
        f"- RF by-fold importance file: {fold_note}",
        "",
        "## Group Aggregate",
        "",
        "| group type | group name | v3 attention | RF sum importance |",
        "|---|---|---:|---:|",
    ]
    for row in group_rows:
        lines.append(f"| {row['group_type']} | {row['group_name']} | {float(row['v3_mean_attention_weight']):.4f} | {float(row['rf_sum_channel_importance']):.4f} |")
    lines.extend(["", "## Class-wise Top Attention Channels", "", "| class | rank | channel | attention |", "|---:|---:|---|---:|"])
    for row in top_by_class:
        lines.append(f"| {row['class_id']} | {row['rank_within_class']} | {row['channel_name']} | {float(row['mean_attention_weight']):.4f} |")
    lines.extend(["", "## RF Top Features", "", "| rank | feature | channel | importance |", "|---:|---|---|---:|"])
    for rank, row in enumerate(rf_top, start=1):
        lines.append(f"| {rank} | {row['feature_name']} | {row['channel_name']} | {float(row['mean_importance']):.4f} |")
    return "\n".join(lines) + "\n"


def _claim_memo(component_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Claim Implication Memo For User",
        "",
        "이 메모는 가능한 해석 시나리오만 정리한다. 최종 학술적 판단은 사용자가 한다.",
        "",
        "| comparison | delta comparison-reference | CI |",
        "|---|---:|---|",
    ]
    for row in component_rows:
        lines.append(
            f"| {row['comparison']} | {float(row['mean_delta_comparison_minus_reference']):.4f} | "
            f"[{row['ci_low']}, {row['ci_high']}] |"
        )
    lines.extend(
        [
            "",
            "## 주의",
            "",
            "- paired CI가 0을 포함하면 우월성/열등성 표현을 피한다.",
            "- `no_identity` 결과는 residual branch가 남아 있어 token identity embedding 효과만 분리한 것이다.",
            "- `residual_only_mlp`가 높게 나오면 v3의 일부 성능이 summary statistics로 설명될 수 있음을 뜻하지만 RandomForest와 같은 모델은 아니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _bar_ci(plt: Any, rows: list[dict[str, str]], metric: str, low_key: str | None, high_key: str | None, path: Path, title: str) -> None:
    rows = sorted(rows, key=lambda row: float(row.get(metric) or 0.0), reverse=True)
    names = [row["model_name"] for row in rows]
    values = np.asarray([float(row.get(metric) or 0.0) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.8), 4))
    if low_key and high_key:
        lows = np.asarray([float(row.get(low_key) or value) for row, value in zip(rows, values)], dtype=float)
        highs = np.asarray([float(row.get(high_key) or value) for row, value in zip(rows, values)], dtype=float)
        ax.bar(range(len(names)), values, yerr=np.vstack([values - lows, highs - values]), capsize=3)
    else:
        ax.bar(range(len(names)), values)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _component_delta(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    names = [row["comparison"] for row in rows]
    values = [float(row.get("mean_delta_comparison_minus_reference") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.75), 4))
    ax.bar(range(len(names)), values)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Component contribution delta macro F1")
    ax.set_ylabel("comparison - reference")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _class3(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    rows = [row for row in rows if str(row.get("class_id")) == "3"]
    names = [row["model_name"] for row in rows]
    recall = [float(row.get("mean_recall") or 0.0) for row in rows]
    f1 = [float(row.get("mean_f1") or 0.0) for row in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.8), 4))
    ax.bar(x - 0.2, recall, width=0.4, label="recall")
    ax.bar(x + 0.2, f1, width=0.4, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_title("Class 3 Excessive Lean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _subject_heatmap(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    models = sorted({row["model_name"] for row in rows})
    subjects = sorted({int(row["test_subject"]) for row in rows})
    matrix = np.zeros((len(models), len(subjects)), dtype=float)
    for row in rows:
        matrix[models.index(row["model_name"]), subjects.index(int(row["test_subject"]))] = float(row.get("test_macro_f1") or 0.0)
    fig, ax = plt.subplots(figsize=(7, max(4, len(models) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([str(item) for item in subjects])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title("Subject-wise ablation macro F1")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _residual_vs_rf(plt: Any, run_dir: Path, path: Path) -> None:
    merged = _read_csv(run_dir / "merged_with_existing_results" / "merged_ablation_comparison.csv")
    selected = [row for row in merged if row["model_name"] in {"channel_shared_posres_attention_v3_residual_only_mlp", "feature_random_forest_v1"}]
    _bar_ci(plt, selected, "mean_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", path, "Residual-only MLP vs RandomForest")


def _attention_rf_scatter(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for row in rows:
        ax.scatter(float(row.get("v3_mean_attention_weight") or 0.0), float(row.get("rf_channel_importance") or 0.0))
        ax.text(float(row.get("v3_mean_attention_weight") or 0.0), float(row.get("rf_channel_importance") or 0.0), row["channel_name"], fontsize=7)
    ax.set_xlabel("v3 mean attention weight")
    ax.set_ylabel("RF channel importance")
    ax.set_title("v3 attention vs RF importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _attention_top_by_class(plt: Any, run_dir: Path, path: Path) -> None:
    source = run_dir / "merged_with_existing_results" / "attention_rf_alignment_top_attention_by_class.csv"
    rows = _read_csv(source)
    if not rows:
        return
    labels = [f"c{row['class_id']}:{row['channel_name']}" for row in rows]
    values = [float(row.get("mean_attention_weight") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 4))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_title("Top v3 attention channels by class")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _with_result_type(row: dict[str, str], result_type: str) -> dict[str, Any]:
    payload = dict(row)
    payload["result_type"] = result_type
    return payload


def _pearson(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2 or np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        j = index
        while j + 1 < len(order) and math.isclose(values[order[j + 1]], values[order[index]]):
            j += 1
        rank = (index + j) / 2.0 + 1.0
        for k in range(index, j + 1):
            ranks[order[k]] = rank
        index = j + 1
    return ranks


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
