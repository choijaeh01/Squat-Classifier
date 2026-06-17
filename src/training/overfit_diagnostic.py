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
from training.loso_runner import evaluate_full, train_one_epoch
from training.scalers import TrainOnlyStandardScaler
from training.supervised_trainer import append_run_log, create_run_dir, resolve_path, write_csv
from utils.reproducibility import set_global_seed


def load_overfit_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def validate_overfit_safety(config: dict[str, Any]) -> None:
    data = config["data"]
    training = config["training"]
    normalization = config["normalization"]
    augmentation = config["augmentation"]
    safety = config.get("safety", {})

    if training.get("diagnostic_mode") is not True:
        raise ValueError("overfit diagnostic requires training.diagnostic_mode=true")
    if training.get("allow_generalization_claims") is not False:
        raise ValueError("overfit diagnostic forbids generalization claims")
    if int(training.get("max_epochs", 0)) < 1 or int(training.get("max_epochs", 0)) > 150:
        raise ValueError("overfit diagnostic max_epochs must be in [1, 150]")
    if training.get("loss") != "cross_entropy":
        raise ValueError("overfit diagnostic forbids focal loss and only supports cross_entropy")
    if training.get("optimizer") != "adam":
        raise ValueError("overfit diagnostic only supports adam")
    if training.get("early_stopping", {}).get("enabled") is not False:
        raise ValueError("overfit diagnostic early stopping must remain disabled")
    if float(training.get("target_train_accuracy", 0.0)) != 0.95:
        raise ValueError("overfit diagnostic target_train_accuracy must be 0.95")
    if training.get("stop_when_target_reached") is not True:
        raise ValueError("overfit diagnostic must stop when target is reached")
    if augmentation.get("enabled") is not False:
        raise ValueError("augmentation must be disabled")
    if safety.get("forbid_ssl") is not True:
        raise ValueError("safety.forbid_ssl must be true")
    if safety.get("forbid_augmentation") is not True:
        raise ValueError("safety.forbid_augmentation must be true")
    if safety.get("forbid_external_dataset") is not True:
        raise ValueError("safety.forbid_external_dataset must be true")
    if safety.get("forbid_focal_loss") is not True:
        raise ValueError("safety.forbid_focal_loss must be true")
    if normalization.get("global_standard_scaler") is not True:
        raise ValueError("global_standard_scaler must be true")
    if normalization.get("fit_scaler_on") != "selected_subset_only":
        raise ValueError("scaler must be fit on selected subset only")
    if normalization.get("per_window_zscore") is not False:
        raise ValueError("per-window z-score must be disabled")
    if data.get("sampling_policy") != "balanced_by_class_from_train_subjects":
        raise ValueError("overfit diagnostic requires balanced_by_class_from_train_subjects")
    if data.get("scaler_fit_on") != "selected_subset_only":
        raise ValueError("data.scaler_fit_on must be selected_subset_only")
    if int(data["samples_per_class"]) * 5 != int(data["total_samples"]):
        raise ValueError("total_samples must equal samples_per_class * 5")


def select_balanced_overfit_subset(
    *,
    y: np.ndarray,
    subject_id: np.ndarray,
    subject_ids: list[int],
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    subject_id = np.asarray(subject_id)
    allowed_subjects = {int(item) for item in subject_ids}
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for class_id in range(5):
        candidates = np.where((y == class_id) & np.isin(subject_id.astype(int), list(allowed_subjects)))[0]
        if len(candidates) < samples_per_class:
            raise ValueError(f"class {class_id} has {len(candidates)} candidates, need {samples_per_class}")
        chosen = rng.choice(candidates, size=samples_per_class, replace=False)
        selected.extend(int(item) for item in chosen.tolist())
    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    return selected_array


def run_overfit_diagnostic(
    config: dict[str, Any],
    *,
    project_root: Path,
    confirm_diagnostic: bool = False,
    dry_run: bool = False,
    device_name: str | None = None,
) -> dict[str, Any]:
    validate_overfit_safety(config)
    if not dry_run and not confirm_diagnostic:
        raise ValueError("refusing to run overfit diagnostic without --confirm-diagnostic or --dry-run")

    seed = int(config["seed"])
    set_global_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_dir = resolve_path(project_root, config["dataset_dir"])
    dataset = ProcessedTargetDataset(dataset_dir)
    subset_idx = select_balanced_overfit_subset(
        y=dataset.y,
        subject_id=dataset.subject_id,
        subject_ids=[int(item) for item in config["data"]["subject_ids"]],
        samples_per_class=int(config["data"]["samples_per_class"]),
        seed=seed,
    )
    scaler = TrainOnlyStandardScaler().fit(dataset.X, train_idx=subset_idx)
    X_scaled = scaler.transform(dataset.X)
    run_dir = create_run_dir(project_root, config)
    _write_prelude(run_dir, config, scaler, device, dry_run)
    _write_selected_subset(run_dir / "selected_subset.csv", subset_idx, dataset.y, dataset.subject_id)

    parameter_rows = []
    metric_rows = []
    history_rows = []
    for model_name in config["models"]:
        set_global_seed(seed)
        model = build_model(str(model_name), input_length=512, input_channels=18, num_classes=5).to(device)
        groups = count_parameter_groups(model)
        parameter_rows.append({"model_name": str(model_name), **groups})
        metric, history = _run_one_model(
            model_name=str(model_name),
            model=model,
            X=X_scaled,
            y=dataset.y,
            subject_id=dataset.subject_id,
            subset_idx=subset_idx,
            config=config,
            device=device,
            dry_run=dry_run,
            total_params=int(groups["total_params"]),
        )
        metric_rows.append(metric)
        history_rows.extend(history)

    write_csv(run_dir / "model_parameter_counts.csv", parameter_rows)
    write_csv(run_dir / "overfit_metrics.csv", metric_rows)
    write_csv(run_dir / "overfit_training_history.csv", history_rows)
    append_run_log(run_dir / "run_log.txt", [f"status=finished", f"dry_run={dry_run}"])
    return {
        "run_dir": str(run_dir),
        "device": str(device),
        "dry_run": dry_run,
        "n_samples": int(len(subset_idx)),
        "models": [str(item) for item in config["models"]],
    }


def _run_one_model(
    *,
    model_name: str,
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    subset_idx: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    dry_run: bool,
    total_params: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = config["training"]
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = int(training["batch_size"])
    max_epochs = int(training["max_epochs"])
    target = float(training["target_train_accuracy"])
    optimizer = None
    if not dry_run:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )

    history_rows: list[dict[str, Any]] = []
    reached_epoch = ""
    final_eval = evaluate_full(model, X, y, subject_id, subset_idx, batch_size, criterion, device)
    if dry_run:
        status = "forward_only"
    else:
        status = "ok"
        for epoch in range(1, max_epochs + 1):
            train_one_epoch(
                model,
                X,
                y,
                subject_id,
                subset_idx,
                batch_size,
                criterion,
                optimizer,
                device,
                int(config["seed"]) + epoch,
                training.get("gradient_clip_norm"),
            )
            final_eval = evaluate_full(model, X, y, subject_id, subset_idx, batch_size, criterion, device)
            history_rows.append(
                {
                    "model_name": model_name,
                    "seed": int(config["seed"]),
                    "epoch": epoch,
                    "train_loss": final_eval["loss"],
                    "train_accuracy": final_eval["metrics"]["accuracy"],
                    "train_macro_f1": final_eval["metrics"]["macro_f1"],
                    "generalization_claim_allowed": False,
                }
            )
            if final_eval["metrics"]["accuracy"] >= target:
                reached_epoch = epoch
                break
    reached = reached_epoch != ""
    metric = {
        "model_name": model_name,
        "seed": int(config["seed"]),
        "n_samples": int(len(subset_idx)),
        "n_classes": 5,
        "total_params": total_params,
        "reached_target_train_accuracy": bool(reached),
        "epoch_reached_95_train_accuracy": reached_epoch,
        "final_train_accuracy": final_eval["metrics"]["accuracy"],
        "final_train_macro_f1": final_eval["metrics"]["macro_f1"],
        "final_train_loss": final_eval["loss"],
        "status": status,
    }
    return metric, history_rows


def _write_prelude(
    run_dir: Path,
    config: dict[str, Any],
    scaler: TrainOnlyStandardScaler,
    device: torch.device,
    dry_run: bool,
) -> None:
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    scaler_payload = scaler.to_dict()
    scaler_payload["fit_scope"] = "selected_subset_only"
    (run_dir / "scaler_stats.json").write_text(
        json.dumps(scaler_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    device_detail = "cpu"
    if device.type == "cuda" and torch.cuda.is_available():
        device_detail = torch.cuda.get_device_name(device)
    append_run_log(
        run_dir / "run_log.txt",
        [
            f"experiment={config['experiment_name']}",
            f"mode={'dry_run_forward_only' if dry_run else 'confirm_diagnostic'}",
            f"device={device}",
            f"device_detail={device_detail}",
            f"torch_version={torch.__version__}",
            "purpose=overfit_capacity_sanity_check",
            "generalization_claims=false",
            "augmentation=disabled",
            "per_window_zscore=false",
            "scaler_fit=selected_subset_only",
        ],
    )


def _write_selected_subset(path: Path, subset_idx: np.ndarray, y: np.ndarray, subject_id: np.ndarray) -> None:
    rows = [
        {
            "subset_order": order,
            "sample_index": int(index),
            "subject_id": int(subject_id[index]),
            "class_id": int(y[index]),
        }
        for order, index in enumerate(subset_idx.tolist())
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subset_order", "sample_index", "subject_id", "class_id"])
        writer.writeheader()
        writer.writerows(rows)
