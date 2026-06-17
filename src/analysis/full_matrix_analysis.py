from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def aggregate_full_matrix_by_model_seed(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[(str(row["model_name"]), int(row["seed"]))].append(row)

    output: list[dict[str, Any]] = []
    for (model_name, seed), rows in sorted(grouped.items()):
        success = [row for row in rows if str(row.get("status")) in {"ok", "forward_only"}]
        failed = [row for row in rows if str(row.get("status")) == "failed"]
        output.append(
            {
                "model_name": model_name,
                "seed": seed,
                "mean_test_accuracy": _mean(_values(success, "test_accuracy")),
                "std_test_accuracy": _std(_values(success, "test_accuracy")),
                "mean_test_macro_f1": _mean(_values(success, "test_macro_f1")),
                "std_test_macro_f1": _std(_values(success, "test_macro_f1")),
                "mean_test_weighted_f1": _mean(_values(success, "test_weighted_f1")),
                "std_test_weighted_f1": _std(_values(success, "test_weighted_f1")),
                "min_test_macro_f1": _min(_values(success, "test_macro_f1")),
                "max_test_macro_f1": _max(_values(success, "test_macro_f1")),
                "mean_best_epoch": _mean(_values(success, "best_epoch")),
                "n_success_folds": len(success),
                "n_failed_folds": len(failed),
            }
        )
    return output


def aggregate_full_matrix_by_model(
    fold_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    *,
    bootstrap_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    params_by_model = {str(row["model_name"]): row for row in parameter_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[str(row["model_name"])].append(row)

    output: list[dict[str, Any]] = []
    for model_name, rows in sorted(grouped.items()):
        success = [row for row in rows if str(row.get("status")) in {"ok", "forward_only"}]
        failed = [row for row in rows if str(row.get("status")) == "failed"]
        macro_values = _values(success, "test_macro_f1")
        ci_low, ci_high = bootstrap_mean_ci(macro_values, n_bootstrap=bootstrap_n, seed=seed)
        params = params_by_model.get(model_name, {})
        output.append(
            {
                "model_name": model_name,
                "total_params": int(params.get("total_params", 0) or 0),
                "encoder_params": int(params.get("encoder_params", 0) or 0),
                "aggregation_params": int(params.get("aggregation_params", 0) or 0),
                "head_params": int(params.get("head_params", 0) or 0),
                "mean_accuracy": _mean(_values(success, "test_accuracy")),
                "std_accuracy": _std(_values(success, "test_accuracy")),
                "mean_macro_f1": _mean(macro_values),
                "std_macro_f1": _std(macro_values),
                "mean_weighted_f1": _mean(_values(success, "test_weighted_f1")),
                "std_weighted_f1": _std(_values(success, "test_weighted_f1")),
                "macro_f1_ci_low": ci_low,
                "macro_f1_ci_high": ci_high,
                "mean_best_epoch": _mean(_values(success, "best_epoch")),
                "n_success_runs": len(success),
                "n_failed_runs": len(failed),
            }
        )
    return output


def aggregate_classwise_by_model(classwise_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in classwise_rows:
        grouped[(str(row["model_name"]), int(row["class_id"]), str(row["class_name"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model_name, class_id, class_name), rows in sorted(grouped.items()):
        output.append(
            {
                "model_name": model_name,
                "class_id": class_id,
                "class_name": class_name,
                "mean_precision": _mean(_values(rows, "precision")),
                "std_precision": _std(_values(rows, "precision")),
                "mean_recall": _mean(_values(rows, "recall")),
                "std_recall": _std(_values(rows, "recall")),
                "mean_f1": _mean(_values(rows, "f1")),
                "std_f1": _std(_values(rows, "f1")),
                "total_support": int(sum(float(row.get("support", 0) or 0) for row in rows)),
            }
        )
    return output


def aggregate_subjectwise_by_model(
    fold_rows: list[dict[str, Any]],
    classwise_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fold_grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        if str(row.get("status")) not in {"ok", "forward_only"}:
            continue
        fold_grouped[(str(row["model_name"]), int(row["test_subject"]))].append(row)

    class_grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in classwise_rows:
        class_grouped[(str(row["model_name"]), int(row["test_subject"]), int(row["class_id"]))].append(row)

    output: list[dict[str, Any]] = []
    for (model_name, test_subject), rows in sorted(fold_grouped.items()):
        payload: dict[str, Any] = {
            "model_name": model_name,
            "test_subject": test_subject,
            "test_accuracy": _mean(_values(rows, "test_accuracy")),
            "test_macro_f1": _mean(_values(rows, "test_macro_f1")),
        }
        for class_id in range(5):
            payload[f"class{class_id}_f1"] = _mean(_values(class_grouped.get((model_name, test_subject, class_id), []), "f1"))
        output.append(payload)
    return output


def compute_bootstrap_confidence_intervals(
    fold_rows: list[dict[str, Any]],
    *,
    bootstrap_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[str(row["model_name"])].append(row)
    output: list[dict[str, Any]] = []
    metric_map = {
        "accuracy": "test_accuracy",
        "macro_f1": "test_macro_f1",
        "weighted_f1": "test_weighted_f1",
    }
    for model_name, rows in sorted(grouped.items()):
        success = [row for row in rows if str(row.get("status")) in {"ok", "forward_only"}]
        for metric_name, column in metric_map.items():
            values = _values(success, column)
            low, high = bootstrap_mean_ci(values, n_bootstrap=bootstrap_n, seed=seed)
            output.append(
                {
                    "model_name": model_name,
                    "metric": metric_name,
                    "mean": _mean(values),
                    "ci_low": low,
                    "ci_high": high,
                    "n": len(values),
                }
            )
    return output


def compute_paired_model_differences(
    fold_rows: list[dict[str, Any]],
    reference_models: list[str],
    *,
    metric: str,
    bootstrap_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    metric_column = f"test_{metric}" if not metric.startswith("test_") else metric
    rows_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    models: set[str] = set()
    for row in fold_rows:
        if str(row.get("status")) not in {"ok", "forward_only"}:
            continue
        model_name = str(row["model_name"])
        models.add(model_name)
        rows_by_key[(model_name, int(row["seed"]), int(row["fold_id"]))] = row

    output: list[dict[str, Any]] = []
    for reference in reference_models:
        for comparison in sorted(models):
            if comparison == reference:
                continue
            differences: list[float] = []
            for (model_name, seed_value, fold_id), ref_row in rows_by_key.items():
                if model_name != reference:
                    continue
                cmp_row = rows_by_key.get((comparison, seed_value, fold_id))
                if cmp_row is None:
                    continue
                differences.append(_float(cmp_row[metric_column]) - _float(ref_row[metric_column]))
            low, high = bootstrap_mean_ci(differences, n_bootstrap=bootstrap_n, seed=seed)
            output.append(
                {
                    "reference_model": reference,
                    "comparison_model": comparison,
                    "metric": metric,
                    "mean_difference": _mean(differences),
                    "std_difference": _std(differences),
                    "ci_low": low,
                    "ci_high": high,
                    "n_pairs": len(differences),
                }
            )
    return output


def bootstrap_mean_ci(values: list[float], *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    clean = np.asarray([value for value in values if not np.isnan(value)], dtype=np.float64)
    if clean.size == 0:
        return 0.0, 0.0
    if clean.size == 1 or n_bootstrap <= 1:
        value = float(clean.mean())
        return value, value
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(clean, size=(int(n_bootstrap), clean.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def generate_full_matrix_figures(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
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
    confusion = _read_csv(run_dir / "confusion_matrices.csv")
    history = _read_csv(run_dir / "training_history.csv")
    paired = _read_csv(run_dir / "paired_model_differences.csv")
    _plot_bar_ci(plt, aggregate, "mean_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", figures_dir / "model_macro_f1_bar_ci.png", "Mean macro F1")
    _plot_bar_ci(plt, aggregate, "mean_accuracy", None, None, figures_dir / "model_accuracy_bar_ci.png", "Mean accuracy")
    _plot_param_vs_metric(plt, aggregate, figures_dir / "model_parameter_count_vs_macro_f1.png")
    _plot_subject_heatmap(plt, subjectwise, figures_dir / "subjectwise_macro_f1_heatmap.png")
    _plot_class_recall_heatmap(plt, classwise, figures_dir / "classwise_recall_heatmap.png")
    _plot_class3(plt, classwise, figures_dir / "class3_recall_f1_by_model.png")
    _plot_confusion(plt, confusion, figures_dir / "aggregate_confusion_matrix_per_model.png")
    _plot_history(plt, history, figures_dir / "training_curves_summary.png")
    _plot_paired_delta(plt, paired, "all_channel_conv1d_v1", figures_dir / "paired_delta_macro_f1_vs_all_channel_v1.png")
    _plot_paired_delta(plt, paired, "all_channel_conv1d_small", figures_dir / "paired_delta_macro_f1_vs_all_channel_small.png")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [_float(row.get(key)) for row in rows if row.get(key) not in ("", None)]


def _float(value: Any) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def _mean(values: list[float]) -> float:
    clean = [value for value in values if not np.isnan(value)]
    return float(np.mean(clean)) if clean else 0.0


def _std(values: list[float]) -> float:
    clean = [value for value in values if not np.isnan(value)]
    return float(np.std(clean)) if clean else 0.0


def _min(values: list[float]) -> float:
    clean = [value for value in values if not np.isnan(value)]
    return float(np.min(clean)) if clean else 0.0


def _max(values: list[float]) -> float:
    clean = [value for value in values if not np.isnan(value)]
    return float(np.max(clean)) if clean else 0.0


def _plot_bar_ci(plt: Any, rows: list[dict[str, str]], value_key: str, low_key: str | None, high_key: str | None, path: Path, title: str) -> None:
    if not rows:
        return
    labels = [row["model_name"] for row in rows]
    values = np.asarray([float(row[value_key]) for row in rows])
    yerr = None
    if low_key and high_key:
        low = np.asarray([float(row[low_key]) for row in rows])
        high = np.asarray([float(row[high_key]) for row in rows])
        yerr = np.vstack([values - low, high - values])
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
    ax.bar(range(len(labels)), values, yerr=yerr, capsize=3)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_param_vs_metric(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in rows:
        ax.scatter(float(row["total_params"]), float(row["mean_macro_f1"]))
        ax.annotate(row["model_name"], (float(row["total_params"]), float(row["mean_macro_f1"])), fontsize=7)
    ax.set_xlabel("Total parameters")
    ax.set_ylabel("Mean macro F1")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_subject_heatmap(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    models = sorted({row["model_name"] for row in rows})
    subjects = sorted({int(row["test_subject"]) for row in rows})
    matrix = np.zeros((len(models), len(subjects)))
    for i, model in enumerate(models):
        for j, subject in enumerate(subjects):
            vals = [float(row["test_macro_f1"]) for row in rows if row["model_name"] == model and int(row["test_subject"]) == subject]
            matrix[i, j] = float(np.mean(vals)) if vals else 0.0
    _imshow(plt, matrix, models, [str(s) for s in subjects], path, "Subject-wise macro F1")


def _plot_class_recall_heatmap(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    models = sorted({row["model_name"] for row in rows})
    classes = sorted({int(row["class_id"]) for row in rows})
    matrix = np.zeros((len(models), len(classes)))
    for i, model in enumerate(models):
        for j, class_id in enumerate(classes):
            vals = [float(row["mean_recall"]) for row in rows if row["model_name"] == model and int(row["class_id"]) == class_id]
            matrix[i, j] = vals[0] if vals else 0.0
    _imshow(plt, matrix, models, [str(c) for c in classes], path, "Class-wise recall")


def _plot_class3(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    class3 = [row for row in rows if str(row.get("class_id")) == "3"]
    if not class3:
        return
    labels = [row["model_name"] for row in class3]
    recall = [float(row["mean_recall"]) for row in class3]
    f1 = [float(row["mean_f1"]) for row in class3]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
    ax.bar(x - 0.2, recall, width=0.4, label="recall")
    ax.bar(x + 0.2, f1, width=0.4, label="F1")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_title("Class 3 Excessive Lean")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confusion(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    aggregate = [row for row in rows if row.get("scope") == "aggregate"]
    if not aggregate:
        return
    models = sorted({row["model_name"] for row in aggregate})
    fig, axes = plt.subplots(1, len(models), figsize=(max(4 * len(models), 6), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        matrix = np.zeros((5, 5), dtype=float)
        for row in aggregate:
            if row["model_name"] == model:
                matrix[int(row["true_class"]), int(row["pred_class"])] += float(row["count"])
        ax.imshow(matrix, aspect="auto")
        ax.set_title(model, fontsize=8)
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_history(plt: Any, rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        return
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["model_name"]].append((int(row["epoch"]), float(row["val_macro_f1"])))
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, pairs in sorted(grouped.items()):
        by_epoch: dict[int, list[float]] = defaultdict(list)
        for epoch, value in pairs:
            by_epoch[epoch].append(value)
        xs = sorted(by_epoch)
        ys = [float(np.mean(by_epoch[x])) for x in xs]
        ax.plot(xs, ys, label=model)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean val macro F1")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_paired_delta(plt: Any, rows: list[dict[str, str]], reference: str, path: Path) -> None:
    data = [row for row in rows if row.get("reference_model") == reference and row.get("metric") == "macro_f1"]
    if not data:
        return
    labels = [row["comparison_model"] for row in data]
    values = np.asarray([float(row["mean_difference"]) for row in data])
    low = np.asarray([float(row["ci_low"]) for row in data])
    high = np.asarray([float(row["ci_high"]) for row in data])
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(range(len(labels)), values, yerr=np.vstack([values - low, high - values]), capsize=3)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel(f"Macro F1 delta vs {reference}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _imshow(plt: Any, matrix: np.ndarray, ylabels: list[str], xlabels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(xlabels)), max(4, len(ylabels) * 0.45)))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
