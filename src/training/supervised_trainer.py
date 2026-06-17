from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from data.processed_loader import ProcessedTargetDataset
from models.registry import build_model, count_parameter_groups
from training.metrics import compute_classification_metrics
from training.scalers import TrainOnlyStandardScaler
from training.splits import LOSOSmokeSplit, make_loso_smoke_split, summarize_split
from utils.reproducibility import set_global_seed


CLASS_NAMES = ["Correct", "Knee Valgus", "Butt Wink", "Excessive Lean", "Partial Squat"]


def load_smoke_training_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return dict(payload)


def validate_smoke_safety(config: dict[str, Any]) -> None:
    training = config["training"]
    normalization = config["normalization"]
    augmentation = config["augmentation"]
    if training.get("smoke_mode") is not True:
        raise ValueError("smoke_train_real requires training.smoke_mode=true")
    if training.get("allow_full_training") is not False:
        raise ValueError("smoke_train_real requires training.allow_full_training=false")
    if int(training.get("max_epochs", 0)) > 2:
        raise ValueError("smoke safety guard: max_epochs must be <= 2")
    if int(training.get("max_train_batches", 0)) > 5:
        raise ValueError("smoke safety guard: max_train_batches must be <= 5")
    if training.get("loss") != "cross_entropy":
        raise ValueError("smoke training only supports cross_entropy")
    if training.get("optimizer") != "adam":
        raise ValueError("smoke training only supports adam")
    if training.get("mixed_precision") is not False:
        raise ValueError("mixed precision is disabled for smoke reproducibility")
    if augmentation.get("enabled") is not False:
        raise ValueError("augmentation must be disabled for smoke training")
    if normalization.get("fit_scaler_on") != "train_subjects_only":
        raise ValueError("scaler must be fit on train subjects only")
    if normalization.get("per_window_zscore") is not False:
        raise ValueError("per-window z-score must be disabled in smoke config")


def run_real_smoke_training(
    config: dict[str, Any],
    *,
    project_root: Path,
    confirm_smoke: bool = False,
    forward_only: bool = False,
    device_name: str | None = None,
) -> dict[str, Any]:
    validate_smoke_safety(config)
    if not confirm_smoke and not forward_only:
        raise ValueError("refusing to run smoke training without --confirm-smoke or --dry-run-forward-only")
    set_global_seed(int(config["seed"]))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset_dir = resolve_path(project_root, config["dataset_dir"])
    dataset = ProcessedTargetDataset(dataset_dir)
    split = make_loso_smoke_split(
        subject_id=dataset.subject_id,
        y=dataset.y,
        test_subject_id=int(config["split"]["test_subject_id"]),
        val_subject_id=int(config["split"]["val_subject_id"]),
        strict_subject_isolation=bool(config["split"]["strict_subject_isolation"]),
    )
    scaler = TrainOnlyStandardScaler().fit(dataset.X, train_idx=split.train_idx)
    X_scaled = scaler.transform(dataset.X)

    run_dir = create_run_dir(project_root, config)
    write_run_prelude(run_dir, project_root, config, dataset_dir, scaler, device)

    parameter_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for model_name in config["models"]:
        set_global_seed(int(config["seed"]))
        model = build_model(str(model_name), input_length=512, input_channels=18, num_classes=5).to(device)
        parameter_row = {"model_name": str(model_name), **count_parameter_groups(model)}
        parameter_rows.append(parameter_row)
        split_rows.append(summarize_split(split, y=dataset.y, subject_id=dataset.subject_id, model_name=str(model_name), seed=int(config["seed"])))
        try:
            result = run_one_model(
                model_name=str(model_name),
                model=model,
                X=X_scaled,
                y=dataset.y,
                subject_id=dataset.subject_id,
                split=split,
                config=config,
                device=device,
                forward_only=forward_only,
            )
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
            metric_rows.append({"model_name": str(model_name), "status": "failed", "error": str(exc)})
            continue
        metric_rows.append(result["metric_row"])
        history_rows.extend(result["history_rows"])
        confusion_rows.extend(result["confusion_rows"])
        prediction_rows.extend(result["prediction_rows"])

    write_csv(run_dir / "split_summary.csv", split_rows)
    write_csv(run_dir / "model_parameter_counts.csv", parameter_rows)
    write_csv(run_dir / "per_model_smoke_metrics.csv", metric_rows)
    write_csv(run_dir / "per_model_training_history.csv", history_rows)
    write_csv(run_dir / "per_model_confusion_matrix.csv", confusion_rows)
    write_csv(run_dir / "predictions_preview.csv", prediction_rows[:100])
    append_run_log(
        run_dir / "run_log.txt",
        [
            f"mode={'forward_only' if forward_only else 'confirm_smoke'}",
            f"device={device}",
            f"errors={len(errors)}",
            *[f"error: {error}" for error in errors],
        ],
    )
    return {
        "run_dir": str(run_dir),
        "device": str(device),
        "errors": errors,
        "models": list(config["models"]),
        "forward_only": forward_only,
    }


def run_one_model(
    *,
    model_name: str,
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    split: LOSOSmokeSplit,
    config: dict[str, Any],
    device: torch.device,
    forward_only: bool,
) -> dict[str, Any]:
    training = config["training"]
    batch_size = int(training["batch_size"])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None
    if not forward_only:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )

    history_rows: list[dict[str, Any]] = []
    if forward_only:
        train_eval = evaluate_limited(model, X, y, subject_id, split.train_idx, batch_size, 1, criterion, device)
        val_eval = evaluate_limited(model, X, y, subject_id, split.val_idx, batch_size, int(training["max_val_batches"]), criterion, device)
        test_eval = evaluate_limited(model, X, y, subject_id, split.test_idx, batch_size, int(training["max_test_batches"]), criterion, device)
    else:
        train_eval = train_limited(
            model,
            X,
            y,
            subject_id,
            split.train_idx,
            batch_size,
            int(training["max_epochs"]),
            int(training["max_train_batches"]),
            criterion,
            optimizer,
            device,
            history_rows,
            model_name,
            training.get("gradient_clip_norm"),
        )
        val_eval = evaluate_limited(model, X, y, subject_id, split.val_idx, batch_size, int(training["max_val_batches"]), criterion, device)
        test_eval = evaluate_limited(model, X, y, subject_id, split.test_idx, batch_size, int(training["max_test_batches"]), criterion, device)

    metric_row = flatten_metric_row(model_name, train_eval, val_eval, test_eval, "forward_only" if forward_only else "ok")
    confusion_rows = confusion_rows_for_model(model_name, test_eval["metrics"]["confusion_matrix"])
    prediction_rows = prediction_preview_rows(model_name, test_eval)
    return {
        "metric_row": metric_row,
        "history_rows": history_rows,
        "confusion_rows": confusion_rows,
        "prediction_rows": prediction_rows,
    }


def train_limited(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    max_epochs: int,
    max_batches: int,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    history_rows: list[dict[str, Any]],
    model_name: str,
    gradient_clip_norm: Any,
) -> dict[str, Any]:
    model.train()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_subjects: list[int] = []
    losses: list[float] = []
    batch_counter = 0
    for epoch in range(max_epochs):
        for batch_index, batch_idx in enumerate(batch_indices(indices, batch_size)):
            if batch_index >= max_batches:
                break
            xb, yb = tensors_for_batch(X, y, batch_idx, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
            preds = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
            all_true.extend(y[batch_idx].astype(int).tolist())
            all_pred.extend(preds.tolist())
            all_subjects.extend(subject_id[batch_idx].astype(int).tolist())
            losses.append(float(loss.detach().cpu().item()))
            history_rows.append(
                {
                    "model_name": model_name,
                    "phase": "train",
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "loss": float(loss.detach().cpu().item()),
                }
            )
            batch_counter += 1
    return eval_payload(all_true, all_pred, all_subjects, losses, batch_counter)


def evaluate_limited(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    max_batches: int,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_subjects: list[int] = []
    losses: list[float] = []
    batch_counter = 0
    with torch.no_grad():
        for batch_index, batch_idx in enumerate(batch_indices(indices, batch_size)):
            if batch_index >= max_batches:
                break
            xb, yb = tensors_for_batch(X, y, batch_idx, device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
            all_true.extend(y[batch_idx].astype(int).tolist())
            all_pred.extend(preds.tolist())
            all_subjects.extend(subject_id[batch_idx].astype(int).tolist())
            losses.append(float(loss.detach().cpu().item()))
            batch_counter += 1
    return eval_payload(all_true, all_pred, all_subjects, losses, batch_counter)


def eval_payload(true: list[int], pred: list[int], subjects: list[int], losses: list[float], batches: int) -> dict[str, Any]:
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
        "batches": int(batches),
        "y_true": true,
        "y_pred": pred,
        "subject_id": subjects,
    }


def batch_indices(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def tensors_for_batch(X: np.ndarray, y: np.ndarray, batch_idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xb = torch.as_tensor(X[batch_idx], dtype=torch.float32, device=device)
    yb = torch.as_tensor(y[batch_idx], dtype=torch.long, device=device)
    return xb, yb


def flatten_metric_row(model_name: str, train_eval: dict[str, Any], val_eval: dict[str, Any], test_eval: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "status": status,
        "train_batches": train_eval["batches"],
        "val_batches": val_eval["batches"],
        "test_batches": test_eval["batches"],
        "train_loss": train_eval["loss"],
        "val_loss": val_eval["loss"],
        "test_loss": test_eval["loss"],
        "train_accuracy": train_eval["metrics"]["accuracy"],
        "val_accuracy": val_eval["metrics"]["accuracy"],
        "test_accuracy": test_eval["metrics"]["accuracy"],
        "train_macro_f1": train_eval["metrics"]["macro_f1"],
        "val_macro_f1": val_eval["metrics"]["macro_f1"],
        "test_macro_f1": test_eval["metrics"]["macro_f1"],
    }


def confusion_rows_for_model(model_name: str, matrix: list[list[int]]) -> list[dict[str, Any]]:
    rows = []
    for true_label, values in enumerate(matrix):
        for pred_label, count in enumerate(values):
            rows.append({"model_name": model_name, "split": "test", "true_label": true_label, "pred_label": pred_label, "count": int(count)})
    return rows


def prediction_preview_rows(model_name: str, eval_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model_name": model_name,
            "split": "test",
            "row_index": index,
            "subject_id": eval_result["subject_id"][index],
            "y_true": eval_result["y_true"][index],
            "y_pred": eval_result["y_pred"][index],
        }
        for index in range(len(eval_result["y_true"]))
    ]


def create_run_dir(project_root: Path, config: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_path(project_root, config["output_root"]) / f"{timestamp}_{config['experiment_name']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_prelude(
    run_dir: Path,
    project_root: Path,
    config: dict[str, Any],
    dataset_dir: Path,
    scaler: TrainOnlyStandardScaler,
    device: torch.device,
) -> None:
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
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
    (run_dir / "scaler_stats.json").write_text(
        json.dumps(scaler.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_run_log(
        run_dir / "run_log.txt",
        [
            f"experiment={config['experiment_name']}",
            f"device={device}",
            f"torch_version={torch.__version__}",
            f"dataset_dir={dataset_dir}",
            "augmentation=disabled",
            "per_window_zscore=false",
            "scaler_fit=train_subjects_only",
        ],
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_run_log(path: Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line) + "\n")


def manifest_checksums(dataset_dir: Path) -> dict[str, Any]:
    checksums_path = dataset_dir / "checksums.json"
    payload: dict[str, Any] = {}
    if checksums_path.exists():
        payload["source_checksums_json"] = json.loads(checksums_path.read_text(encoding="utf-8"))
    for name in ["X.npy", "y.npy", "subject_id.npy", "metadata.csv"]:
        path = dataset_dir / name
        if path.exists():
            payload[name] = sha256_file(path)
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_status(project_root: Path) -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=project_root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return "git status unavailable"
    return result.stdout.strip() or "clean"


def git_commit(project_root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()
