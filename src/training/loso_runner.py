from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from data.processed_loader import ProcessedTargetDataset
from models.registry import build_model, count_parameter_groups
from training.aggregation import aggregate_metrics_by_model, subjectwise_metrics_by_model
from training.checkpointing import load_model_state, save_checkpoint
from training.metrics import compute_classification_metrics
from training.scalers import TrainOnlyStandardScaler
from training.splits import LOSOSmokeSplit, class_counts, make_cyclic_loso_splits
from training.supervised_trainer import (
    CLASS_NAMES,
    append_run_log,
    create_run_dir,
    git_commit,
    git_status,
    manifest_checksums,
    resolve_path,
    write_csv,
)
from utils.reproducibility import set_global_seed


def load_pilot_loso_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def validate_pilot_loso_safety(config: dict[str, Any]) -> None:
    split = config["split"]
    training = config["training"]
    normalization = config["normalization"]
    augmentation = config["augmentation"]
    safety = config.get("safety", {})

    if split.get("type") != "loso_with_cyclic_subject_validation":
        raise ValueError("pilot LOSO requires split.type=loso_with_cyclic_subject_validation")
    if split.get("validation_policy") != "next_subject_cyclic":
        raise ValueError("pilot LOSO requires validation_policy=next_subject_cyclic")
    subjects = [int(item) for item in split.get("subjects", [])]
    if len(subjects) != 6 or len(set(subjects)) != 6:
        raise ValueError("pilot LOSO requires exactly six unique subjects")
    if split.get("strict_subject_isolation") is not True:
        raise ValueError("strict subject isolation must be enabled")

    if training.get("pilot_mode") is not True:
        raise ValueError("run_pilot_loso requires training.pilot_mode=true")
    if training.get("allow_full_training") is not False:
        raise ValueError("pilot config must keep allow_full_training=false")
    max_epochs = int(training.get("max_epochs", 0))
    max_allowed_epochs = int(safety.get("max_allowed_epochs_for_pilot", 30))
    if max_epochs < 1 or max_epochs > max_allowed_epochs or max_epochs > 30:
        raise ValueError("pilot max_epochs must be in [1, 30]")
    if isinstance(config.get("seed"), list):
        raise ValueError("pilot LOSO allows exactly one scalar seed")
    if len(config.get("seeds", [config.get("seed")])) > int(safety.get("max_allowed_seeds_for_pilot", 1)):
        raise ValueError("pilot LOSO allows one seed only")
    if training.get("loss") != "cross_entropy":
        raise ValueError("pilot LOSO forbids focal loss and only supports cross_entropy")
    if training.get("optimizer") != "adam":
        raise ValueError("pilot LOSO only supports adam for this fixed pilot")
    if training.get("mixed_precision") is not False:
        raise ValueError("mixed precision is disabled for pilot reproducibility")
    if augmentation.get("enabled") is not False or safety.get("forbid_augmentation") is not True:
        raise ValueError("augmentation must remain disabled")
    if safety.get("forbid_focal_loss") is not True:
        raise ValueError("safety.forbid_focal_loss must be true")
    if safety.get("forbid_ssl") is not True:
        raise ValueError("safety.forbid_ssl must be true")
    if safety.get("forbid_external_dataset") is not True:
        raise ValueError("safety.forbid_external_dataset must be true")
    if normalization.get("global_standard_scaler") is not True:
        raise ValueError("pilot LOSO expects global_standard_scaler=true")
    if normalization.get("fit_scaler_on") != "train_subjects_only":
        raise ValueError("scaler must be fit on train subjects only")
    if normalization.get("per_window_zscore") is not False:
        raise ValueError("per-window z-score must remain disabled")


def run_pilot_loso(
    config: dict[str, Any],
    *,
    project_root: Path,
    confirm_pilot: bool = False,
    dry_run: bool = False,
    device_name: str | None = None,
) -> dict[str, Any]:
    validate_pilot_loso_safety(config)
    if not dry_run and not confirm_pilot:
        raise ValueError("refusing to run pilot training without --confirm-pilot or --dry-run")

    seed = int(config["seed"])
    set_global_seed(seed)
    _configure_deterministic(bool(config.get("reproducibility", {}).get("deterministic", True)))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_dir = resolve_path(project_root, config["dataset_dir"])
    dataset = ProcessedTargetDataset(dataset_dir)
    splits = make_cyclic_loso_splits(
        subject_id=dataset.subject_id,
        y=dataset.y,
        subjects=[int(item) for item in config["split"]["subjects"]],
        strict_subject_isolation=bool(config["split"]["strict_subject_isolation"]),
    )
    run_dir = create_run_dir(project_root, config)
    _write_prelude(run_dir, project_root, config, dataset_dir, device, dry_run)

    split_plan_rows = [_split_plan_row(index + 1, split, dataset.y) for index, split in enumerate(splits)]
    write_csv(run_dir / "split_plan.csv", split_plan_rows)

    scaled_by_fold: dict[int, np.ndarray] = {}
    scaler_rows: list[dict[str, Any]] = []
    for fold_id, split in enumerate(splits, start=1):
        scaler = TrainOnlyStandardScaler().fit(dataset.X, train_idx=split.train_idx)
        scaled_by_fold[fold_id] = scaler.transform(dataset.X)
        scaler_rows.append(_scaler_row(fold_id, split, scaler))
    write_csv(run_dir / "scaler_stats_by_fold.csv", scaler_rows)

    parameter_rows = _parameter_rows(config, device)
    write_csv(run_dir / "model_parameter_counts.csv", parameter_rows)

    fold_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for model_name in config["models"]:
        for fold_id, split in enumerate(splits, start=1):
            try:
                set_global_seed(seed)
                model = build_model(str(model_name), input_length=512, input_channels=18, num_classes=5).to(device)
                result = _run_model_fold(
                    model_name=str(model_name),
                    fold_id=fold_id,
                    model=model,
                    X=scaled_by_fold[fold_id],
                    y=dataset.y,
                    subject_id=dataset.subject_id,
                    split=split,
                    config=config,
                    run_dir=run_dir,
                    device=device,
                    dry_run=dry_run,
                )
            except Exception as exc:
                failed = _failed_row(str(model_name), seed, fold_id, split, exc)
                failed_rows.append(failed)
                fold_rows.append(_failed_fold_metric_row(failed))
                write_csv(run_dir / "failed_runs.csv", failed_rows)
                if config["training"].get("failure_policy", "abort") == "abort":
                    raise
                continue
            fold_rows.append(result["fold_metric"])
            classwise_rows.extend(result["classwise"])
            history_rows.extend(result["history"])
            confusion_rows.extend(result["confusion"])
            prediction_rows.extend(result["predictions"])

    aggregate_rows = aggregate_metrics_by_model(fold_rows, parameter_rows)
    subjectwise_rows = subjectwise_metrics_by_model(fold_rows, classwise_rows)
    confusion_rows.extend(_aggregate_confusion_rows(confusion_rows))

    write_csv(run_dir / "fold_metrics.csv", fold_rows)
    write_csv(run_dir / "classwise_metrics.csv", classwise_rows)
    write_csv(run_dir / "training_history.csv", history_rows)
    write_csv(run_dir / "confusion_matrices.csv", confusion_rows)
    write_csv(run_dir / "predictions.csv", prediction_rows)
    write_csv(run_dir / "aggregate_metrics_by_model.csv", aggregate_rows)
    write_csv(run_dir / "subjectwise_metrics_by_model.csv", subjectwise_rows)
    write_csv(run_dir / "failed_runs.csv", failed_rows)
    generate_figures(run_dir)
    append_run_log(run_dir / "run_log.txt", [f"errors={len(failed_rows)}", "status=finished"])
    return {
        "run_dir": str(run_dir),
        "device": str(device),
        "dry_run": bool(dry_run),
        "models": [str(item) for item in config["models"]],
        "n_folds": len(splits),
        "n_failed": len(failed_rows),
    }


def _run_model_fold(
    *,
    model_name: str,
    fold_id: int,
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    split: LOSOSmokeSplit,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    dry_run: bool,
) -> dict[str, Any]:
    training = config["training"]
    batch_size = int(training["batch_size"])
    criterion = torch.nn.CrossEntropyLoss()
    history_rows: list[dict[str, Any]] = []

    if dry_run:
        train_eval = evaluate_full(model, X, y, subject_id, split.train_idx, batch_size, criterion, device)
        val_eval = evaluate_full(model, X, y, subject_id, split.val_idx, batch_size, criterion, device)
        test_eval = evaluate_full(model, X, y, subject_id, split.test_idx, batch_size, criterion, device)
        best_epoch = 0
        best_val_macro_f1 = float(val_eval["metrics"]["macro_f1"])
        status = "forward_only"
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        fit_result = fit_with_early_stopping(
            model_name=model_name,
            fold_id=fold_id,
            model=model,
            X=X,
            y=y,
            subject_id=subject_id,
            split=split,
            config=config,
            run_dir=run_dir,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            history_rows=history_rows,
        )
        best_epoch = fit_result["best_epoch"]
        best_val_macro_f1 = fit_result["best_val_macro_f1"]
        train_eval = fit_result["best_train_eval"]
        val_eval = fit_result["best_val_eval"]
        test_eval = evaluate_full(model, X, y, subject_id, split.test_idx, batch_size, criterion, device)
        status = "ok"

    return {
        "fold_metric": _fold_metric_row(
            model_name=model_name,
            seed=int(config["seed"]),
            fold_id=fold_id,
            split=split,
            best_epoch=best_epoch,
            best_val_macro_f1=best_val_macro_f1,
            test_eval=test_eval,
            status=status,
        ),
        "classwise": _classwise_rows(model_name, int(config["seed"]), fold_id, split.test_subject, test_eval),
        "history": history_rows,
        "confusion": _confusion_rows(model_name, int(config["seed"]), fold_id, split.test_subject, test_eval),
        "predictions": _prediction_rows(model_name, int(config["seed"]), fold_id, split.test_subject, test_eval),
    }


def fit_with_early_stopping(
    *,
    model_name: str,
    fold_id: int,
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    split: LOSOSmokeSplit,
    config: dict[str, Any],
    run_dir: Path,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    history_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    training = config["training"]
    early = training["early_stopping"]
    batch_size = int(training["batch_size"])
    max_epochs = int(training["max_epochs"])
    patience = int(early["patience"])
    min_delta = float(early["min_delta"])
    best_score = -float("inf")
    best_epoch = 0
    best_train_eval: dict[str, Any] | None = None
    best_val_eval: dict[str, Any] | None = None
    stale_epochs = 0
    checkpoint_path = _checkpoint_path(run_dir, model_name, fold_id)

    for epoch in range(1, max_epochs + 1):
        train_eval = train_one_epoch(
            model,
            X,
            y,
            subject_id,
            split.train_idx,
            batch_size,
            criterion,
            optimizer,
            device,
            int(config["seed"]) + epoch,
            training.get("gradient_clip_norm"),
        )
        val_eval = evaluate_full(model, X, y, subject_id, split.val_idx, batch_size, criterion, device)
        score = float(val_eval["metrics"]["macro_f1"])
        improved = score > best_score + min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            best_train_eval = train_eval
            best_val_eval = val_eval
            stale_epochs = 0
            if config["checkpointing"].get("save_best_checkpoint") is True:
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    model_name=model_name,
                    fold_id=fold_id,
                    epoch=epoch,
                    val_macro_f1=score,
                    config=config,
                )
        else:
            stale_epochs += 1
        history_rows.append(
            {
                "model_name": model_name,
                "seed": int(config["seed"]),
                "fold_id": fold_id,
                "epoch": epoch,
                "train_loss": train_eval["loss"],
                "val_loss": val_eval["loss"],
                "train_accuracy": train_eval["metrics"]["accuracy"],
                "train_macro_f1": train_eval["metrics"]["macro_f1"],
                "val_accuracy": val_eval["metrics"]["accuracy"],
                "val_macro_f1": val_eval["metrics"]["macro_f1"],
                "is_best": improved,
            }
        )
        if early.get("enabled") is True and stale_epochs >= patience:
            break

    if config["checkpointing"].get("save_best_checkpoint") is True and checkpoint_path.exists():
        load_model_state(checkpoint_path, model, device)
    if best_train_eval is None or best_val_eval is None:
        raise RuntimeError("no validation checkpoint was selected")
    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "best_train_eval": best_train_eval,
        "best_val_eval": best_val_eval,
    }


def train_one_epoch(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    shuffle_seed: int,
    gradient_clip_norm: Any,
) -> dict[str, Any]:
    model.train()
    rng = np.random.default_rng(shuffle_seed)
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    true: list[int] = []
    pred: list[int] = []
    subjects: list[int] = []
    losses: list[float] = []
    sample_indices: list[int] = []
    for batch_idx in _batch_indices(shuffled, batch_size):
        xb, yb = _tensors_for_batch(X, y, batch_idx, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
        optimizer.step()
        preds = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
        true.extend(y[batch_idx].astype(int).tolist())
        pred.extend(preds.tolist())
        subjects.extend(subject_id[batch_idx].astype(int).tolist())
        sample_indices.extend(batch_idx.astype(int).tolist())
        losses.append(float(loss.detach().cpu().item()))
    return _eval_payload(true, pred, subjects, sample_indices, losses)


def evaluate_full(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    true: list[int] = []
    pred: list[int] = []
    subjects: list[int] = []
    losses: list[float] = []
    sample_indices: list[int] = []
    with torch.no_grad():
        for batch_idx in _batch_indices(indices, batch_size):
            xb, yb = _tensors_for_batch(X, y, batch_idx, device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
            true.extend(y[batch_idx].astype(int).tolist())
            pred.extend(preds.tolist())
            subjects.extend(subject_id[batch_idx].astype(int).tolist())
            sample_indices.extend(batch_idx.astype(int).tolist())
            losses.append(float(loss.detach().cpu().item()))
    return _eval_payload(true, pred, subjects, sample_indices, losses)


def generate_figures(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        (figures_dir / "figures_unavailable.txt").write_text(
            "matplotlib is not installed; pilot LOSO figures were not generated.\n",
            encoding="utf-8",
        )
        append_run_log(run_dir / "run_log.txt", ["figures=skipped_missing_matplotlib"])
        return
    aggregate = _read_csv(run_dir / "aggregate_metrics_by_model.csv")
    subjectwise = _read_csv(run_dir / "subjectwise_metrics_by_model.csv")
    classwise = _read_csv(run_dir / "classwise_metrics.csv")
    confusion = _read_csv(run_dir / "confusion_matrices.csv")
    history = _read_csv(run_dir / "training_history.csv")
    _plot_bar(aggregate, "mean_test_macro_f1", figures_dir / "model_macro_f1_bar.png", "Mean test macro F1")
    _plot_bar(aggregate, "mean_test_accuracy", figures_dir / "model_accuracy_bar.png", "Mean test accuracy")
    _plot_subject_heatmap(subjectwise, figures_dir / "subjectwise_macro_f1_heatmap.png")
    _plot_class_recall_heatmap(classwise, figures_dir / "classwise_recall_heatmap.png")
    _plot_confusion_by_model(confusion, figures_dir / "aggregated_confusion_matrix_per_model.png")
    _plot_param_vs_macro_f1(aggregate, figures_dir / "parameter_count_vs_macro_f1.png")
    _plot_training_curves(history, figures_dir / "training_curves_per_model_fold.png")


def _eval_payload(
    true: list[int],
    pred: list[int],
    subjects: list[int],
    sample_indices: list[int],
    losses: list[float],
) -> dict[str, Any]:
    metrics = compute_classification_metrics(
        y_true=np.asarray(true, dtype=np.int64),
        y_pred=np.asarray(pred, dtype=np.int64),
        subject_ids=np.asarray(subjects, dtype=np.int64),
        num_classes=5,
        class_names=CLASS_NAMES,
    )
    return {
        "metrics": metrics,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "y_true": true,
        "y_pred": pred,
        "subject_id": subjects,
        "sample_index": sample_indices,
    }


def _batch_indices(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _tensors_for_batch(X: np.ndarray, y: np.ndarray, batch_idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(X[batch_idx], dtype=torch.float32, device=device),
        torch.as_tensor(y[batch_idx], dtype=torch.long, device=device),
    )


def _fold_metric_row(
    *,
    model_name: str,
    seed: int,
    fold_id: int,
    split: LOSOSmokeSplit,
    best_epoch: int,
    best_val_macro_f1: float,
    test_eval: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "seed": seed,
        "fold_id": fold_id,
        "train_subjects": "|".join(str(item) for item in split.train_subjects),
        "val_subject": split.val_subject,
        "test_subject": split.test_subject,
        "n_train": len(split.train_idx),
        "n_val": len(split.val_idx),
        "n_test": len(split.test_idx),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test_accuracy": test_eval["metrics"]["accuracy"],
        "test_macro_f1": test_eval["metrics"]["macro_f1"],
        "leakage_check_passed": split.leakage_check_passed,
        "scaler_fit_subjects": "|".join(str(item) for item in split.train_subjects),
        "status": status,
    }


def _classwise_rows(
    model_name: str,
    seed: int,
    fold_id: int,
    test_subject: int,
    test_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        metrics = test_eval["metrics"]["class_wise"][class_name]
        rows.append(
            {
                "model_name": model_name,
                "seed": seed,
                "fold_id": fold_id,
                "test_subject": test_subject,
                "class_id": class_id,
                "class_name": class_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "support": metrics["support"],
            }
        )
    return rows


def _confusion_rows(
    model_name: str,
    seed: int,
    fold_id: int,
    test_subject: int,
    test_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    matrix = test_eval["metrics"]["confusion_matrix"]
    for true_class, values in enumerate(matrix):
        for pred_class, count in enumerate(values):
            rows.append(
                {
                    "scope": "fold",
                    "model_name": model_name,
                    "seed": seed,
                    "fold_id": fold_id,
                    "test_subject": test_subject,
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "count": int(count),
                }
            )
    return rows


def _aggregate_confusion_rows(confusion_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, int, int, int], int] = {}
    for row in confusion_rows:
        if row.get("scope") != "fold":
            continue
        key = (str(row["model_name"]), int(row["seed"]), int(row["true_class"]), int(row["pred_class"]))
        totals[key] = totals.get(key, 0) + int(row["count"])
    rows = []
    for (model_name, seed, true_class, pred_class), count in sorted(totals.items()):
        rows.append(
            {
                "scope": "aggregate",
                "model_name": model_name,
                "seed": seed,
                "fold_id": "all",
                "test_subject": "all",
                "true_class": true_class,
                "pred_class": pred_class,
                "count": count,
            }
        )
    return rows


def _prediction_rows(
    model_name: str,
    seed: int,
    fold_id: int,
    test_subject: int,
    test_eval: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "model_name": model_name,
            "seed": seed,
            "fold_id": fold_id,
            "test_subject": test_subject,
            "sample_index": test_eval["sample_index"][index],
            "subject_id": test_eval["subject_id"][index],
            "y_true": test_eval["y_true"][index],
            "y_pred": test_eval["y_pred"][index],
        }
        for index in range(len(test_eval["y_true"]))
    ]


def _parameter_rows(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows = []
    x = torch.zeros(2, 512, 18, dtype=torch.float32, device=device)
    for model_name in config["models"]:
        model = build_model(str(model_name), input_length=512, input_channels=18, num_classes=5).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        groups = count_parameter_groups(model)
        rows.append(
            {
                "model_name": str(model_name),
                "total_params": groups["total_params"],
                "encoder_params": groups["encoder_params"],
                "aggregation_params": groups["aggregation_params"],
                "head_params": groups["head_params"],
                "other_params": groups["other_params"],
                "input_shape": "(batch, 512, 18)",
                "output_shape": str(tuple(logits.shape)),
            }
        )
    return rows


def _split_plan_row(fold_id: int, split: LOSOSmokeSplit, y: np.ndarray) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        "train_subjects": "|".join(str(item) for item in split.train_subjects),
        "val_subject": split.val_subject,
        "test_subject": split.test_subject,
        "n_train": len(split.train_idx),
        "n_val": len(split.val_idx),
        "n_test": len(split.test_idx),
        "train_class_counts": json.dumps(class_counts(y[split.train_idx]), sort_keys=True),
        "val_class_counts": json.dumps(class_counts(y[split.val_idx]), sort_keys=True),
        "test_class_counts": json.dumps(class_counts(y[split.test_idx]), sort_keys=True),
        "leakage_check_passed": split.leakage_check_passed,
    }


def _scaler_row(fold_id: int, split: LOSOSmokeSplit, scaler: TrainOnlyStandardScaler) -> dict[str, Any]:
    payload = scaler.to_dict()
    return {
        "fold_id": fold_id,
        "fit_scope": payload["fit_scope"],
        "scaler_fit_subjects": "|".join(str(item) for item in split.train_subjects),
        "fit_sample_count": payload["fit_sample_count"],
        "fit_window_count": payload["fit_window_count"],
        "mean": json.dumps(payload["mean"]),
        "scale": json.dumps(payload["scale"]),
    }


def _failed_row(model_name: str, seed: int, fold_id: int, split: LOSOSmokeSplit, exc: Exception) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "seed": seed,
        "fold_id": fold_id,
        "train_subjects": "|".join(str(item) for item in split.train_subjects),
        "val_subject": split.val_subject,
        "test_subject": split.test_subject,
        "status": "failed",
        "error": str(exc),
    }


def _failed_fold_metric_row(failed: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": failed["model_name"],
        "seed": failed["seed"],
        "fold_id": failed["fold_id"],
        "train_subjects": failed["train_subjects"],
        "val_subject": failed["val_subject"],
        "test_subject": failed["test_subject"],
        "n_train": "",
        "n_val": "",
        "n_test": "",
        "best_epoch": "",
        "best_val_macro_f1": "",
        "test_accuracy": "",
        "test_macro_f1": "",
        "leakage_check_passed": False,
        "scaler_fit_subjects": failed["train_subjects"],
        "status": "failed",
    }


def _checkpoint_path(run_dir: Path, model_name: str, fold_id: int) -> Path:
    safe_model_name = model_name.replace("/", "_")
    return run_dir / "checkpoints" / safe_model_name / f"fold_{fold_id:02d}_best.pt"


def _write_prelude(
    run_dir: Path,
    project_root: Path,
    config: dict[str, Any],
    dataset_dir: Path,
    device: torch.device,
    dry_run: bool,
) -> None:
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "git_status.txt").write_text(git_status(project_root) + "\n", encoding="utf-8")
    commit = git_commit(project_root)
    if commit:
        (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    else:
        (run_dir / "git_commit_unavailable.txt").write_text("git commit unavailable\n", encoding="utf-8")
    (run_dir / "manifest_checksums.json").write_text(
        json.dumps(manifest_checksums(dataset_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    device_detail = "cpu"
    if device.type == "cuda" and torch.cuda.is_available():
        device_detail = torch.cuda.get_device_name(device)
    append_run_log(
        run_dir / "run_log.txt",
        [
            f"experiment={config['experiment_name']}",
            f"mode={'dry_run_forward_only' if dry_run else 'confirm_pilot'}",
            f"device={device}",
            f"device_detail={device_detail}",
            f"torch_version={torch.__version__}",
            f"dataset_dir={dataset_dir}",
            "validation_policy=next_subject_cyclic",
            "augmentation=disabled",
            "per_window_zscore=false",
            "scaler_fit=train_subjects_only",
        ],
    )


def _configure_deterministic(enabled: bool) -> None:
    if not enabled:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_bar(rows: list[dict[str, str]], metric: str, output: Path, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [row["model_name"] for row in rows]
    values = [float(row.get(metric, 0.0) or 0.0) for row in rows]
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_subject_heatmap(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({row["model_name"] for row in rows})
    subjects = sorted({int(row["test_subject"]) for row in rows})
    matrix = np.zeros((len(models), len(subjects)), dtype=float)
    for row in rows:
        matrix[models.index(row["model_name"]), subjects.index(int(row["test_subject"]))] = float(row["test_macro_f1"])
    _heatmap(matrix, models, [str(item) for item in subjects], "Test subject", "Model", "Subject-wise macro F1", output, plt)


def _plot_class_recall_heatmap(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({row["model_name"] for row in rows})
    classes = list(range(5))
    matrix = np.zeros((len(models), len(classes)), dtype=float)
    for model in models:
        for class_id in classes:
            values = [
                float(row["recall"])
                for row in rows
                if row["model_name"] == model and int(row["class_id"]) == class_id
            ]
            matrix[models.index(model), class_id] = float(np.mean(values)) if values else 0.0
    _heatmap(matrix, models, [str(item) for item in classes], "Class", "Model", "Mean class recall", output, plt)


def _plot_confusion_by_model(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = [row for row in rows if row.get("scope") == "aggregate"]
    models = sorted({row["model_name"] for row in aggregate})
    cols = 3
    fig, axes = plt.subplots(2, cols, figsize=(10, 7))
    axes_flat = axes.flatten()
    for ax, model in zip(axes_flat, models):
        matrix = np.zeros((5, 5), dtype=float)
        for row in aggregate:
            if row["model_name"] == model:
                matrix[int(row["true_class"]), int(row["pred_class"])] = float(row["count"])
        ax.imshow(matrix, aspect="auto")
        ax.set_title(model, fontsize=8)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
    for ax in axes_flat[len(models) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_param_vs_macro_f1(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        x = float(row.get("total_params", 0) or 0)
        y = float(row.get("mean_test_macro_f1", 0.0) or 0.0)
        ax.scatter([x], [y])
        ax.annotate(row["model_name"], (x, y), fontsize=7)
    ax.set_xlabel("Total parameters")
    ax.set_ylabel("Mean test macro F1")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_training_curves(rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["model_name"], row["fold_id"]), []).append(row)
    for (model, fold_id), values in grouped.items():
        values = sorted(values, key=lambda item: int(item["epoch"]))
        ax.plot(
            [int(item["epoch"]) for item in values],
            [float(item["val_macro_f1"]) for item in values],
            linewidth=0.8,
            label=f"{model}/f{fold_id}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val macro F1")
    ax.set_ylim(0.0, 1.0)
    if grouped:
        ax.legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _heatmap(matrix: np.ndarray, y_labels: list[str], x_labels: list[str], xlabel: str, ylabel: str, title: str, output: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
