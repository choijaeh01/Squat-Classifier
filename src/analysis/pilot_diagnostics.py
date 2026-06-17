from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES = ["Correct", "Knee Valgus", "Butt Wink", "Excessive Lean", "Partial Squat"]


def analyze_pilot_loso_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    diagnostics_dir = run_path / "diagnostics"
    figures_dir = diagnostics_dir / "figures"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = _read_csv(run_path / "fold_metrics.csv")
    history_rows = _read_csv(run_path / "training_history.csv")
    prediction_rows = _read_csv(run_path / "predictions.csv")
    classwise_rows = _read_csv(run_path / "classwise_metrics.csv")
    subjectwise_rows = _read_csv(run_path / "subjectwise_metrics_by_model.csv")

    ranking_rows = recompute_model_ranking(fold_rows)
    fold_diagnostic_rows = compute_fold_metric_diagnostics(fold_rows, history_rows)
    prediction_distribution_rows = compute_prediction_distribution(prediction_rows)
    class3_rows = compute_class3_error_analysis(prediction_rows)
    gap_rows = compute_train_val_test_gap_by_model(fold_diagnostic_rows)
    best_epoch_rows = compute_best_epoch_distribution(fold_rows)
    collapse_rows = diagnose_prediction_collapse(prediction_rows)
    classwise_recomputed = recompute_classwise_metrics(prediction_rows)
    classwise_for_figures = classwise_rows or classwise_recomputed
    subjectwise_for_figures = subjectwise_rows or recompute_subjectwise_metrics(prediction_rows)
    summary = diagnostic_summary(ranking_rows, collapse_rows, class3_rows, gap_rows)

    _write_csv(diagnostics_dir / "model_ranking_recomputed.csv", ranking_rows)
    _write_csv(diagnostics_dir / "fold_metric_diagnostics.csv", fold_diagnostic_rows)
    _write_csv(diagnostics_dir / "prediction_distribution_by_model.csv", prediction_distribution_rows)
    _write_csv(diagnostics_dir / "class3_error_analysis.csv", class3_rows)
    _write_csv(diagnostics_dir / "train_val_test_gap_by_model.csv", gap_rows)
    _write_csv(diagnostics_dir / "best_epoch_distribution.csv", best_epoch_rows)
    _write_csv(diagnostics_dir / "collapse_diagnosis.csv", collapse_rows)
    (diagnostics_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generate_diagnostic_figures(
        figures_dir=figures_dir,
        prediction_distribution_rows=prediction_distribution_rows,
        gap_rows=gap_rows,
        class3_rows=class3_rows,
        best_epoch_rows=best_epoch_rows,
        classwise_rows=classwise_for_figures,
        subjectwise_rows=subjectwise_for_figures,
    )
    return {
        "run_dir": str(run_path),
        "diagnostics_dir": str(diagnostics_dir),
        "summary": summary,
    }


def recompute_model_ranking(fold_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fold_rows:
        if row.get("status") == "ok":
            grouped[row["model_name"]].append(row)
    output = []
    for model_name, rows in grouped.items():
        acc = [_float(row["test_accuracy"]) for row in rows]
        f1 = [_float(row["test_macro_f1"]) for row in rows]
        output.append(
            {
                "model_name": model_name,
                "n_folds": len(rows),
                "mean_test_accuracy": float(np.mean(acc)) if acc else 0.0,
                "std_test_accuracy": float(np.std(acc)) if acc else 0.0,
                "mean_test_macro_f1": float(np.mean(f1)) if f1 else 0.0,
                "std_test_macro_f1": float(np.std(f1)) if f1 else 0.0,
                "min_test_macro_f1": float(np.min(f1)) if f1 else 0.0,
                "max_test_macro_f1": float(np.max(f1)) if f1 else 0.0,
            }
        )
    output.sort(key=lambda row: row["mean_test_macro_f1"], reverse=True)
    for rank, row in enumerate(output, start=1):
        row["rank_by_macro_f1"] = rank
    return output


def compute_fold_metric_diagnostics(
    fold_rows: list[dict[str, str]],
    history_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    history_by_key = {
        (row["model_name"], int(row["fold_id"]), int(row["epoch"])): row
        for row in history_rows
        if row.get("epoch")
    }
    output = []
    for row in fold_rows:
        if row.get("status") != "ok":
            continue
        key = (row["model_name"], int(row["fold_id"]), int(float(row["best_epoch"])))
        hist = history_by_key.get(key, {})
        train_macro_f1 = _float(hist.get("train_macro_f1", 0.0))
        val_macro_f1 = _float(hist.get("val_macro_f1", row.get("best_val_macro_f1", 0.0)))
        test_macro_f1 = _float(row["test_macro_f1"])
        train_accuracy = _float(hist.get("train_accuracy", 0.0))
        val_accuracy = _float(hist.get("val_accuracy", 0.0))
        test_accuracy = _float(row["test_accuracy"])
        output.append(
            {
                "model_name": row["model_name"],
                "seed": int(row["seed"]),
                "fold_id": int(row["fold_id"]),
                "test_subject": int(row["test_subject"]),
                "best_epoch": int(float(row["best_epoch"])),
                "train_accuracy_at_best": train_accuracy,
                "val_accuracy_at_best": val_accuracy,
                "test_accuracy": test_accuracy,
                "train_macro_f1_at_best": train_macro_f1,
                "val_macro_f1_at_best": val_macro_f1,
                "test_macro_f1": test_macro_f1,
                "train_val_macro_f1_gap": train_macro_f1 - val_macro_f1,
                "val_test_macro_f1_gap": val_macro_f1 - test_macro_f1,
                "train_test_macro_f1_gap": train_macro_f1 - test_macro_f1,
                "status": row["status"],
            }
        )
    return output


def compute_prediction_distribution(prediction_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in prediction_rows:
        grouped[row["model_name"]].append(int(row["y_pred"]))
    output = []
    for model_name, preds in sorted(grouped.items()):
        total = len(preds)
        for class_id in range(5):
            count = int(np.sum(np.asarray(preds) == class_id))
            output.append(
                {
                    "model_name": model_name,
                    "pred_class": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "count": count,
                    "pct": count / total if total else 0.0,
                    "n_predictions": total,
                }
            )
    return output


def compute_class3_error_analysis(prediction_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in prediction_rows:
        if int(row["y_true"]) == 3:
            grouped[row["model_name"]].append(int(row["y_pred"]))
    output = []
    for model_name, preds in sorted(grouped.items()):
        total = len(preds)
        for class_id in range(5):
            count = int(np.sum(np.asarray(preds) == class_id))
            output.append(
                {
                    "model_name": model_name,
                    "true_class": 3,
                    "true_class_name": CLASS_NAMES[3],
                    "pred_class": class_id,
                    "pred_class_name": CLASS_NAMES[class_id],
                    "count": count,
                    "pct_within_class3": count / total if total else 0.0,
                    "is_correct": class_id == 3,
                    "class3_support": total,
                }
            )
    return output


def compute_train_val_test_gap_by_model(fold_diagnostic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fold_diagnostic_rows:
        grouped[row["model_name"]].append(row)
    output = []
    for model_name, rows in sorted(grouped.items()):
        output.append(
            {
                "model_name": model_name,
                "mean_train_accuracy_at_best": _mean([row["train_accuracy_at_best"] for row in rows]),
                "mean_val_accuracy_at_best": _mean([row["val_accuracy_at_best"] for row in rows]),
                "mean_test_accuracy": _mean([row["test_accuracy"] for row in rows]),
                "mean_train_macro_f1_at_best": _mean([row["train_macro_f1_at_best"] for row in rows]),
                "mean_val_macro_f1_at_best": _mean([row["val_macro_f1_at_best"] for row in rows]),
                "mean_test_macro_f1": _mean([row["test_macro_f1"] for row in rows]),
                "mean_train_val_macro_f1_gap": _mean([row["train_val_macro_f1_gap"] for row in rows]),
                "mean_val_test_macro_f1_gap": _mean([row["val_test_macro_f1_gap"] for row in rows]),
                "mean_train_test_macro_f1_gap": _mean([row["train_test_macro_f1_gap"] for row in rows]),
            }
        )
    return output


def compute_best_epoch_distribution(fold_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in fold_rows:
        if row.get("status") == "ok":
            grouped[row["model_name"]].append(int(float(row["best_epoch"])))
    output = []
    for model_name, epochs in sorted(grouped.items()):
        values = np.asarray(epochs, dtype=float)
        output.append(
            {
                "model_name": model_name,
                "n_folds": len(epochs),
                "min_best_epoch": int(values.min()) if len(values) else 0,
                "mean_best_epoch": float(values.mean()) if len(values) else 0.0,
                "median_best_epoch": float(np.median(values)) if len(values) else 0.0,
                "max_best_epoch": int(values.max()) if len(values) else 0,
                "folds_best_epoch_ge_25": int(np.sum(values >= 25)) if len(values) else 0,
            }
        )
    return output


def diagnose_prediction_collapse(prediction_rows: list[dict[str, str]], dominance_threshold: float = 0.70) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in prediction_rows:
        grouped[row["model_name"]].append(int(row["y_pred"]))
    output = []
    for model_name, preds in sorted(grouped.items()):
        total = len(preds)
        counts = np.asarray([int(np.sum(np.asarray(preds) == class_id)) for class_id in range(5)], dtype=float)
        dominant = int(np.argmax(counts)) if total else -1
        dominant_pct = float(counts[dominant] / total) if total else 0.0
        probs = counts / total if total else np.zeros(5)
        nonzero = probs[probs > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum()) if len(nonzero) else 0.0
        unique_pred_classes = int(np.sum(counts > 0))
        collapse_flag = bool(dominant_pct >= dominance_threshold or unique_pred_classes <= 2)
        output.append(
            {
                "model_name": model_name,
                "n_predictions": total,
                "unique_pred_classes": unique_pred_classes,
                "dominant_pred_class": dominant,
                "dominant_pred_class_name": CLASS_NAMES[dominant] if dominant >= 0 else "",
                "dominant_pred_count": int(counts[dominant]) if dominant >= 0 else 0,
                "dominant_pred_pct": dominant_pct,
                "prediction_entropy": entropy,
                "collapse_flag": collapse_flag,
            }
        )
    return output


def recompute_classwise_metrics(prediction_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[row["model_name"]].append((int(row["y_true"]), int(row["y_pred"])))
    output = []
    for model_name, pairs in sorted(grouped.items()):
        true = np.asarray([item[0] for item in pairs], dtype=int)
        pred = np.asarray([item[1] for item in pairs], dtype=int)
        for class_id in range(5):
            tp = int(np.sum((true == class_id) & (pred == class_id)))
            fp = int(np.sum((true != class_id) & (pred == class_id)))
            fn = int(np.sum((true == class_id) & (pred != class_id)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            output.append(
                {
                    "model_name": model_name,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": int(np.sum(true == class_id)),
                }
            )
    return output


def recompute_subjectwise_metrics(prediction_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(row["model_name"], int(row["test_subject"]))].append((int(row["y_true"]), int(row["y_pred"])))
    output = []
    for (model_name, subject), pairs in sorted(grouped.items()):
        true = np.asarray([item[0] for item in pairs], dtype=int)
        pred = np.asarray([item[1] for item in pairs], dtype=int)
        class_f1 = []
        for class_id in range(5):
            tp = int(np.sum((true == class_id) & (pred == class_id)))
            fp = int(np.sum((true != class_id) & (pred == class_id)))
            fn = int(np.sum((true == class_id) & (pred != class_id)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            class_f1.append(f1)
        output.append(
            {
                "model_name": model_name,
                "test_subject": subject,
                "test_macro_f1": float(np.mean(class_f1)),
                "test_accuracy": float(np.mean(true == pred)) if len(true) else 0.0,
            }
        )
    return output


def diagnostic_summary(
    ranking_rows: list[dict[str, Any]],
    collapse_rows: list[dict[str, Any]],
    class3_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    class3_correct = [row for row in class3_rows if int(row["pred_class"]) == 3]
    class3_recall_by_model = {
        row["model_name"]: float(row["pct_within_class3"])
        for row in class3_correct
    }
    return {
        "top_model_by_macro_f1": ranking_rows[0]["model_name"] if ranking_rows else "",
        "lowest_class3_recall_model": min(class3_recall_by_model, key=class3_recall_by_model.get) if class3_recall_by_model else "",
        "collapsed_models": [row["model_name"] for row in collapse_rows if _bool(row["collapse_flag"])],
        "largest_train_test_gap_model": max(gap_rows, key=lambda row: row["mean_train_test_macro_f1_gap"])["model_name"] if gap_rows else "",
        "note": "1 seed pilot diagnostics; not final performance evidence.",
    }


def generate_diagnostic_figures(
    *,
    figures_dir: Path,
    prediction_distribution_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    class3_rows: list[dict[str, Any]],
    best_epoch_rows: list[dict[str, Any]],
    classwise_rows: list[dict[str, Any]],
    subjectwise_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        (figures_dir / "figures_unavailable.txt").write_text(
            "matplotlib is not installed; diagnostic figures were not generated.\n",
            encoding="utf-8",
        )
        return
    _plot_prediction_distribution(prediction_distribution_rows, figures_dir / "prediction_distribution_by_model.png", plt)
    _plot_train_val_test_gap(gap_rows, figures_dir / "train_val_test_gap_by_model.png", plt)
    _plot_class3_confusion(class3_rows, figures_dir / "class3_confusion_by_model.png", plt)
    _plot_best_epoch(best_epoch_rows, figures_dir / "best_epoch_distribution.png", plt)
    _plot_model_class_recall_heatmap(classwise_rows, figures_dir / "model_vs_class_recall_heatmap.png", plt)
    _plot_subject_model_heatmap(subjectwise_rows, figures_dir / "subject_vs_model_macro_f1_heatmap.png", plt)


def _plot_prediction_distribution(rows: list[dict[str, Any]], output: Path, plt: Any) -> None:
    models = sorted({row["model_name"] for row in rows})
    matrix = np.zeros((len(models), 5), dtype=float)
    for row in rows:
        matrix[models.index(row["model_name"]), int(row["pred_class"])] = float(row["pct"])
    _heatmap(matrix, models, [str(i) for i in range(5)], "Predicted class", "Model", "Prediction distribution", output, plt)


def _plot_train_val_test_gap(rows: list[dict[str, Any]], output: Path, plt: Any) -> None:
    models = [row["model_name"] for row in rows]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.25, [float(row["mean_train_macro_f1_at_best"]) for row in rows], width=0.25, label="train")
    ax.bar(x, [float(row["mean_val_macro_f1_at_best"]) for row in rows], width=0.25, label="val")
    ax.bar(x + 0.25, [float(row["mean_test_macro_f1"]) for row in rows], width=0.25, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_class3_confusion(rows: list[dict[str, Any]], output: Path, plt: Any) -> None:
    models = sorted({row["model_name"] for row in rows})
    matrix = np.zeros((len(models), 5), dtype=float)
    for row in rows:
        matrix[models.index(row["model_name"]), int(row["pred_class"])] = float(row["pct_within_class3"])
    _heatmap(matrix, models, [str(i) for i in range(5)], "Predicted class for true class 3", "Model", "Class 3 error pattern", output, plt)


def _plot_best_epoch(rows: list[dict[str, Any]], output: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    models = [row["model_name"] for row in rows]
    ax.bar(range(len(models)), [float(row["mean_best_epoch"]) for row in rows])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean best epoch")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_model_class_recall_heatmap(rows: list[dict[str, Any]], output: Path, plt: Any) -> None:
    if not rows:
        return
    models = sorted({row["model_name"] for row in rows})
    matrix = np.zeros((len(models), 5), dtype=float)
    for model in models:
        for class_id in range(5):
            values = [
                _float(row["recall"])
                for row in rows
                if row["model_name"] == model and int(row["class_id"]) == class_id
            ]
            matrix[models.index(model), class_id] = float(np.mean(values)) if values else 0.0
    _heatmap(matrix, models, [str(i) for i in range(5)], "Class", "Model", "Class recall", output, plt)


def _plot_subject_model_heatmap(rows: list[dict[str, Any]], output: Path, plt: Any) -> None:
    if not rows:
        return
    models = sorted({row["model_name"] for row in rows})
    subjects = sorted({int(row["test_subject"]) for row in rows})
    matrix = np.zeros((len(subjects), len(models)), dtype=float)
    for row in rows:
        matrix[subjects.index(int(row["test_subject"])), models.index(row["model_name"])] = _float(row["test_macro_f1"])
    _heatmap(matrix, [str(s) for s in subjects], models, "Model", "Subject", "Subject vs model macro F1", output, plt)


def _heatmap(matrix: np.ndarray, y_labels: list[str], x_labels: list[str], xlabel: str, ylabel: str, title: str, output: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


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
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"
