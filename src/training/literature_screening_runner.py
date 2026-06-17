from __future__ import annotations

import csv
import json
import platform
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from analysis.full_matrix_analysis import (
    aggregate_classwise_by_model,
    aggregate_full_matrix_by_model,
    aggregate_full_matrix_by_model_seed,
    aggregate_subjectwise_by_model,
    generate_full_matrix_figures,
)
from analysis.literature_baseline_analysis import merge_screening_with_locked_matrix
from data.processed_loader import ProcessedTargetDataset
from models.classical_features import is_classical_model, sklearn_available
from models.registry import build_model, count_parameter_groups
from training.classical_trainer import run_classical_fold
from training.loso_runner import (
    _make_splits_for_config,
    _run_model_fold,
    build_scaler_fit_audit_row,
)
from training.scalers import TrainOnlyStandardScaler
from training.splits import LOSOSmokeSplit, class_counts
from training.supervised_trainer import append_run_log, create_run_dir, git_commit, git_status, manifest_checksums, resolve_path, write_csv
from utils.reproducibility import set_global_seed


def load_literature_screening_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def validate_literature_screening_safety(config: dict[str, Any]) -> None:
    training = config["training"]
    split = config["split"]
    normalization = config["normalization"]
    augmentation = config["augmentation"]
    safety = config["safety"]
    seeds = config.get("seeds", [])

    if training.get("screening_mode") is not True:
        raise ValueError("literature screening requires training.screening_mode=true")
    if training.get("allow_full_training") is not False:
        raise ValueError("literature screening keeps allow_full_training=false")
    if len(seeds) != 1 or len(seeds) > int(safety.get("max_allowed_seeds_for_screening", 1)):
        raise ValueError("literature screening allows exactly one seed")
    if split.get("type") != "loso_with_within_train_subject_stratified_validation":
        raise ValueError("screening must use the final within-train stratified LOSO policy")
    if int(split.get("train_windows_per_subject_class", 0)) != 16 or int(split.get("val_windows_per_subject_class", 0)) != 4:
        raise ValueError("screening split must use 16 train and 4 validation windows per subject-class")
    if split.get("strict_subject_isolation_for_test") is not True or split.get("strict_index_disjoint_train_val") is not True:
        raise ValueError("screening split leakage guards must be enabled")
    if training.get("loss") != "cross_entropy" or training.get("optimizer") != "adam":
        raise ValueError("screening uses fixed cross_entropy and adam settings")
    if training.get("mixed_precision") is not False:
        raise ValueError("mixed precision must remain disabled")
    if augmentation.get("enabled") is not False or safety.get("forbid_augmentation") is not True:
        raise ValueError("augmentation is forbidden")
    if safety.get("forbid_focal_loss") is not True or safety.get("forbid_balanced_sampling") is not True:
        raise ValueError("focal loss and balanced sampling are forbidden")
    if safety.get("forbid_ssl") is not True or safety.get("forbid_external_dataset") is not True:
        raise ValueError("SSL and external dataset adapters are forbidden")
    if safety.get("require_confirm_screening") is not True or safety.get("forbid_full_matrix") is not True:
        raise ValueError("screening requires explicit confirmation and must forbid full matrix")
    if normalization.get("global_standard_scaler") is not True or normalization.get("fit_scaler_on") != "train_indices_only":
        raise ValueError("scaler must be fit on train indices only")
    if normalization.get("per_window_zscore") is not False:
        raise ValueError("per-window z-score must remain disabled")


def build_literature_screening_run_plan(config: dict[str, Any], splits: list[LOSOSmokeSplit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = 0
    for seed in config["seeds"]:
        for fold_id, split in enumerate(splits, start=1):
            for model_name in config["models"]:
                run_id += 1
                rows.append(
                    {
                        "run_id": run_id,
                        "model_name": str(model_name),
                        "seed": int(seed),
                        "fold_id": fold_id,
                        "test_subject": split.test_subject,
                        "status": "pending",
                        "output_subdir": f"{model_name}/seed_{int(seed)}/fold_{fold_id:02d}",
                    }
                )
    return rows


def run_literature_baseline_screening(
    config: dict[str, Any],
    *,
    project_root: Path,
    confirm_screening: bool = False,
    dry_run: bool = False,
    device_name: str | None = None,
) -> dict[str, Any]:
    validate_literature_screening_safety(config)
    if not dry_run and not confirm_screening:
        raise ValueError("refusing literature screening without --confirm-screening")

    seed = int(config["seeds"][0])
    seed_config = dict(config)
    seed_config["seed"] = seed
    set_global_seed(seed)
    _configure_deterministic(bool(config.get("reproducibility", {}).get("deterministic", True)))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_dir = resolve_path(project_root, config["dataset_dir"])
    dataset = ProcessedTargetDataset(dataset_dir)
    splits = _make_splits_for_config(seed_config, dataset)
    run_dir = create_run_dir(project_root, config)
    _write_screening_prelude(run_dir, project_root, config, dataset_dir, device, dry_run)

    run_plan = build_literature_screening_run_plan(config, splits)
    write_csv(run_dir / "run_plan.csv", run_plan)
    write_csv(run_dir / "model_parameter_counts.csv", _parameter_rows(config, device))
    split_rows = [_split_plan_row(seed, fold_id, split, dataset.y) for fold_id, split in enumerate(splits, start=1)]
    write_csv(run_dir / "split_plan.csv", split_rows)

    scaler_rows: list[dict[str, Any]] = []
    scaler_audit_rows: list[dict[str, Any]] = []
    for fold_id, split in enumerate(splits, start=1):
        scaler = TrainOnlyStandardScaler().fit(dataset.X, train_idx=split.train_idx)
        scaler_rows.append(_scaler_row(seed, fold_id, split, scaler))
        for model_name in config["models"]:
            audit = build_scaler_fit_audit_row(
                fold_id=fold_id,
                model_name=str(model_name),
                split=split,
                scaler=scaler,
                scaler_fit_indices=split.train_idx,
            )
            audit["seed"] = seed
            scaler_audit_rows.append(audit)
    write_csv(run_dir / "scaler_stats_by_fold.csv", scaler_rows)
    write_csv(run_dir / "scaler_fit_audit.csv", scaler_audit_rows)

    if dry_run:
        _write_empty_outputs(run_dir)
        append_run_log(run_dir / "run_log.txt", [f"expected_runs={len(run_plan)}", "status=dry_run_no_training"])
        return {"run_dir": str(run_dir), "device": str(device), "dry_run": True, "expected_runs": len(run_plan)}

    fold_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    run_status: dict[tuple[str, int, int], str] = {}

    for fold_id, split in enumerate(splits, start=1):
        scaler = TrainOnlyStandardScaler().fit(dataset.X, train_idx=split.train_idx)
        X_scaled = scaler.transform(dataset.X)
        for model_name in config["models"]:
            key = (str(model_name), seed, fold_id)
            audit = build_scaler_fit_audit_row(fold_id=fold_id, model_name=str(model_name), split=split, scaler=scaler, scaler_fit_indices=split.train_idx)
            if not audit["scaler_leakage_check_passed"]:
                raise ValueError(f"scaler leakage detected for {key}")
            if is_classical_model(str(model_name)) and not sklearn_available():
                skipped = _skipped_row(str(model_name), seed, fold_id, split, "scikit-learn is not installed")
                skipped_rows.append(skipped)
                run_status[key] = "skipped"
                _update_run_plan_status(run_plan, key, "skipped")
                continue
            try:
                set_global_seed(seed)
                if is_classical_model(str(model_name)):
                    result = run_classical_fold(
                        model_name=str(model_name),
                        seed=seed,
                        fold_id=fold_id,
                        X=X_scaled,
                        y=dataset.y,
                        subject_id=dataset.subject_id,
                        split=split,
                    )
                else:
                    model = build_model(str(model_name), input_length=512, input_channels=18, num_classes=5).to(device)
                    result = _run_model_fold(
                        model_name=str(model_name),
                        fold_id=fold_id,
                        model=model,
                        X=X_scaled,
                        y=dataset.y,
                        subject_id=dataset.subject_id,
                        split=split,
                        config=seed_config,
                        run_dir=run_dir / "runs" / str(model_name) / f"seed_{seed}",
                        device=device,
                        dry_run=False,
                    )
                fold_rows.append(_screening_fold_metric_row(result["fold_metric"], split, audit))
                classwise_rows.extend(result["classwise"])
                history_rows.extend(result["history"])
                confusion_rows.extend(result["confusion"])
                prediction_rows.extend(result["predictions"])
                run_status[key] = "ok"
            except Exception as exc:  # noqa: BLE001 - preserve failed run evidence.
                failed_rows.append(_failed_row(str(model_name), seed, fold_id, split, exc))
                run_status[key] = "failed"
            _update_run_plan_status(run_plan, key, run_status[key])
            _write_intermediate(run_dir, run_plan, fold_rows, classwise_rows, history_rows, confusion_rows, prediction_rows, failed_rows, skipped_rows)

    confusion_rows = [row for row in confusion_rows if row.get("scope") == "fold"]
    confusion_rows = confusion_rows + _aggregate_confusion_rows_full(confusion_rows)
    parameter_rows = _read_csv_dicts(run_dir / "model_parameter_counts.csv")
    aggregate_seed_rows = aggregate_full_matrix_by_model_seed(fold_rows)
    aggregate_rows = aggregate_full_matrix_by_model(fold_rows, parameter_rows, bootstrap_n=1000, seed=seed)
    subjectwise_rows = aggregate_subjectwise_by_model(fold_rows, classwise_rows)
    classwise_by_model_rows = aggregate_classwise_by_model(classwise_rows)

    _write_intermediate(run_dir, run_plan, fold_rows, classwise_rows, history_rows, confusion_rows, prediction_rows, failed_rows, skipped_rows)
    write_csv(run_dir / "aggregate_metrics_by_model_seed.csv", aggregate_seed_rows)
    write_csv(run_dir / "aggregate_metrics_by_model.csv", aggregate_rows)
    write_csv(run_dir / "subjectwise_metrics_by_model.csv", subjectwise_rows)
    write_csv(run_dir / "classwise_metrics_by_model.csv", classwise_by_model_rows)
    generate_full_matrix_figures(run_dir)
    merge_screening_with_locked_matrix(project_root=project_root, screening_run_dir=run_dir)
    append_run_log(run_dir / "run_log.txt", [f"errors={len(failed_rows)}", f"skipped={len(skipped_rows)}", "status=finished"])
    return {
        "run_dir": str(run_dir),
        "device": str(device),
        "dry_run": False,
        "expected_runs": len(run_plan),
        "n_success_runs": sum(1 for status in run_status.values() if status == "ok"),
        "n_failed_runs": sum(1 for status in run_status.values() if status == "failed"),
        "n_skipped_runs": sum(1 for status in run_status.values() if status == "skipped"),
    }


def _parameter_rows(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x = torch.zeros(2, 512, 18, dtype=torch.float32, device=device)
    for model_name in config["models"]:
        if is_classical_model(str(model_name)):
            rows.append(
                {
                    "model_name": str(model_name),
                    "total_params": "",
                    "encoder_params": "",
                    "aggregation_params": "",
                    "head_params": "",
                    "other_params": "",
                    "input_shape": "(batch, 512, 18)",
                    "output_shape": "not_applicable",
                    "notes": "classical sklearn baseline; parameters not comparable",
                }
            )
            continue
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
                "notes": "neural literature temporal baseline",
            }
        )
    return rows


def _write_screening_prelude(run_dir: Path, project_root: Path, config: dict[str, Any], dataset_dir: Path, device: torch.device, dry_run: bool) -> None:
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "git_status.txt").write_text(git_status(project_root) + "\n", encoding="utf-8")
    commit = git_commit(project_root)
    if commit:
        (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    else:
        (run_dir / "git_commit_unavailable.txt").write_text("git commit unavailable\n", encoding="utf-8")
    (run_dir / "environment_info.json").write_text(json.dumps(_environment_info(device), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "manifest_checksums.json").write_text(json.dumps(manifest_checksums(dataset_dir), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_run_log(
        run_dir / "run_log.txt",
        [
            f"experiment={config['experiment_name']}",
            f"mode={'dry_run_no_training' if dry_run else 'confirm_screening'}",
            f"device={device}",
            f"dataset_dir={dataset_dir}",
            "validation_policy=loso_with_within_train_subject_stratified_validation",
            "augmentation=disabled",
            "per_window_zscore=false",
            "scaler_fit=train_indices_only",
        ],
    )


def _environment_info(device: torch.device) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else "cpu",
        "sklearn_available": sklearn_available(),
    }


def _split_plan_row(seed: int, fold_id: int, split: LOSOSmokeSplit, y: np.ndarray) -> dict[str, Any]:
    return {
        "seed": seed,
        "fold_id": fold_id,
        "train_subjects": "|".join(str(item) for item in split.train_subjects),
        "val_subjects": "|".join(str(item) for item in split.train_subjects),
        "test_subject": split.test_subject,
        "n_train": len(split.train_idx),
        "n_val": len(split.val_idx),
        "n_test": len(split.test_idx),
        "train_class_counts": json.dumps(class_counts(y[split.train_idx]), sort_keys=True),
        "val_class_counts": json.dumps(class_counts(y[split.val_idx]), sort_keys=True),
        "test_class_counts": json.dumps(class_counts(y[split.test_idx]), sort_keys=True),
        "leakage_check_passed": split.leakage_check_passed,
    }


def _scaler_row(seed: int, fold_id: int, split: LOSOSmokeSplit, scaler: TrainOnlyStandardScaler) -> dict[str, Any]:
    payload = scaler.to_dict()
    return {
        "seed": seed,
        "fold_id": fold_id,
        "fit_scope": "train_indices_only",
        "scaler_fit_subjects": "|".join(str(item) for item in split.train_subjects),
        "fit_sample_count": payload["fit_sample_count"],
        "fit_window_count": payload["fit_window_count"],
        "mean": json.dumps(payload["mean"]),
        "scale": json.dumps(payload["scale"]),
    }


def _screening_fold_metric_row(row: dict[str, Any], split: LOSOSmokeSplit, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": row["model_name"],
        "seed": row["seed"],
        "fold_id": row["fold_id"],
        "train_subjects": row["train_subjects"],
        "val_subjects": "|".join(str(item) for item in split.train_subjects),
        "test_subject": row["test_subject"],
        "n_train": row["n_train"],
        "n_val": row["n_val"],
        "n_test": row["n_test"],
        "best_epoch": row["best_epoch"],
        "best_val_macro_f1": row["best_val_macro_f1"],
        "test_accuracy": row["test_accuracy"],
        "test_macro_f1": row["test_macro_f1"],
        "test_weighted_f1": row["test_weighted_f1"],
        "leakage_check_passed": row["leakage_check_passed"],
        "scaler_leakage_check_passed": audit["scaler_leakage_check_passed"],
        "status": row["status"],
    }


def _failed_row(model_name: str, seed: int, fold_id: int, split: LOSOSmokeSplit, exc: Exception) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "seed": seed,
        "fold_id": fold_id,
        "train_subjects": "|".join(str(item) for item in split.train_subjects),
        "val_subjects": "|".join(str(item) for item in split.train_subjects),
        "test_subject": split.test_subject,
        "status": "failed",
        "error": str(exc),
    }


def _skipped_row(model_name: str, seed: int, fold_id: int, split: LOSOSmokeSplit, reason: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "seed": seed,
        "fold_id": fold_id,
        "train_subjects": "|".join(str(item) for item in split.train_subjects),
        "val_subjects": "|".join(str(item) for item in split.train_subjects),
        "test_subject": split.test_subject,
        "status": "skipped",
        "reason": reason,
    }


def _update_run_plan_status(run_plan: list[dict[str, Any]], key: tuple[str, int, int], status: str) -> None:
    model_name, seed, fold_id = key
    for row in run_plan:
        if str(row["model_name"]) == model_name and int(row["seed"]) == seed and int(row["fold_id"]) == fold_id:
            row["status"] = status
            return


def _write_intermediate(
    run_dir: Path,
    run_plan: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    classwise_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
) -> None:
    write_csv(run_dir / "run_plan.csv", run_plan)
    write_csv(run_dir / "fold_metrics.csv", fold_rows)
    write_csv(run_dir / "classwise_metrics.csv", classwise_rows)
    write_csv(run_dir / "training_history.csv", history_rows)
    write_csv(run_dir / "confusion_matrices.csv", confusion_rows)
    write_csv(run_dir / "predictions.csv", prediction_rows)
    _write_csv_with_header(run_dir / "failed_runs.csv", failed_rows, ["model_name", "seed", "fold_id", "train_subjects", "val_subjects", "test_subject", "status", "error"])
    _write_csv_with_header(run_dir / "skipped_runs.csv", skipped_rows, ["model_name", "seed", "fold_id", "train_subjects", "val_subjects", "test_subject", "status", "reason"])


def _write_empty_outputs(run_dir: Path) -> None:
    _write_csv_with_header(run_dir / "fold_metrics.csv", [], ["model_name", "seed", "fold_id", "train_subjects", "val_subjects", "test_subject", "n_train", "n_val", "n_test", "best_epoch", "best_val_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1", "leakage_check_passed", "scaler_leakage_check_passed", "status"])
    _write_csv_with_header(run_dir / "classwise_metrics.csv", [], ["model_name", "seed", "fold_id", "test_subject", "class_id", "class_name", "precision", "recall", "f1", "support"])
    _write_csv_with_header(run_dir / "training_history.csv", [], ["model_name", "seed", "fold_id", "epoch", "train_loss", "val_loss", "train_accuracy", "train_macro_f1", "val_accuracy", "val_macro_f1", "is_best"])
    _write_csv_with_header(run_dir / "confusion_matrices.csv", [], ["scope", "model_name", "seed", "fold_id", "test_subject", "true_class", "pred_class", "count"])
    _write_csv_with_header(run_dir / "predictions.csv", [], ["model_name", "seed", "fold_id", "test_subject", "sample_index", "subject_id", "y_true", "y_pred"])
    _write_csv_with_header(run_dir / "failed_runs.csv", [], ["model_name", "seed", "fold_id", "train_subjects", "val_subjects", "test_subject", "status", "error"])
    _write_csv_with_header(run_dir / "skipped_runs.csv", [], ["model_name", "seed", "fold_id", "train_subjects", "val_subjects", "test_subject", "status", "reason"])


def _read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _aggregate_confusion_rows_full(confusion_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, int, int], int] = {}
    for row in confusion_rows:
        if row.get("scope") != "fold":
            continue
        key = (str(row["model_name"]), int(row["true_class"]), int(row["pred_class"]))
        totals[key] = totals.get(key, 0) + int(row["count"])
    rows = []
    for (model_name, true_class, pred_class), count in sorted(totals.items()):
        rows.append(
            {
                "scope": "aggregate",
                "model_name": model_name,
                "seed": "all",
                "fold_id": "all",
                "test_subject": "all",
                "true_class": true_class,
                "pred_class": pred_class,
                "count": count,
            }
        )
    return rows


def _write_csv_with_header(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_deterministic(enabled: bool) -> None:
    if not enabled:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
