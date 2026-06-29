from __future__ import annotations

import csv
import hashlib
import json
import platform
import socket
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

import models.classical_features as classical_features
from analysis.feature_importance_analysis import (
    aggregate_importance_rows,
    build_feature_audit_rows,
    feature_definitions,
    validate_feature_audit_rows,
    xgboost_importance_rows,
)
from analysis.full_matrix_analysis import (
    aggregate_classwise_by_model,
    aggregate_full_matrix_by_model,
    aggregate_full_matrix_by_model_seed,
    aggregate_subjectwise_by_model,
    compute_bootstrap_confidence_intervals,
    compute_paired_model_differences,
)
from analysis.xgboost_completion_analysis import CONTROLLED_RESULT, write_xgboost_completion_analysis
from data.processed_loader import ProcessedTargetDataset
from training.classical_trainer import run_classical_fold
from training.literature_full_extension_runner import (
    _aggregate_confusion_rows,
    _config_for_seed,
    _extension_fold_metric_row,
    _failed_fold_row,
    _failed_row,
    _read_csv_dicts,
    _scaler_row,
    _split_plan_row,
    _validation_policy_row,
)
from training.loso_runner import _make_splits_for_config, build_scaler_fit_audit_row
from training.scalers import TrainOnlyStandardScaler
from training.splits import LOSOSmokeSplit
from training.supervised_trainer import append_run_log, create_run_dir, git_commit, git_status, manifest_checksums, resolve_path, write_csv
from utils.reproducibility import set_global_seed


def load_xgboost_completion_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def validate_xgboost_completion_safety(config: dict[str, Any]) -> None:
    split = config["split"]
    normalization = config["normalization"]
    safety = config["safety"]
    feature_policy = config["feature_policy"]
    xgboost = config["xgboost"]
    if config.get("models") != ["feature_xgboost_v1"]:
        raise ValueError("xgboost completion must run only feature_xgboost_v1")
    if config.get("seeds") != [42, 123, 2025]:
        raise ValueError("xgboost completion requires seeds [42, 123, 2025]")
    if split.get("type") != "loso_with_within_train_subject_stratified_validation":
        raise ValueError("xgboost completion requires final within-train stratified LOSO")
    if int(split.get("train_windows_per_subject_class", 0)) != 16 or int(split.get("val_windows_per_subject_class", 0)) != 4:
        raise ValueError("xgboost completion split must use 16 train and 4 validation windows per subject-class")
    if split.get("strict_subject_isolation_for_test") is not True or split.get("strict_index_disjoint_train_val") is not True:
        raise ValueError("split leakage guards must be enabled")
    for key in (
        "signal_derived_only",
        "forbid_metadata_features",
        "forbid_label_features",
        "forbid_subject_id_features",
        "forbid_window_boundary_features",
        "forbid_original_length_feature",
        "use_same_feature_set_as_random_forest",
    ):
        if feature_policy.get(key) is not True:
            raise ValueError(f"feature_policy.{key} must be true")
    if xgboost.get("use_sklearn_api") is not True:
        raise ValueError("xgboost completion requires sklearn API")
    if xgboost.get("no_hyperparameter_search") is not True or xgboost.get("no_early_stopping") is not True:
        raise ValueError("xgboost tuning and early stopping are forbidden")
    fixed = xgboost.get("fixed_params", {})
    if fixed.get("objective") != "multi:softprob" or int(fixed.get("num_class", 0)) != 5:
        raise ValueError("xgboost fixed params must use multi:softprob and num_class=5")
    if fixed.get("tree_method") != "hist" or fixed.get("random_state") != "use_seed":
        raise ValueError("xgboost fixed params must use tree_method=hist and random_state=use_seed")
    if normalization.get("global_standard_scaler") is not True or normalization.get("fit_scaler_on") != "train_indices_only":
        raise ValueError("scaler must fit on train indices only")
    if normalization.get("per_window_zscore") is not False:
        raise ValueError("per-window z-score is forbidden")
    for key in (
        "require_confirm_xgboost_completion",
        "require_clean_git_before_run",
        "forbid_ssl",
        "forbid_augmentation",
        "forbid_focal_loss",
        "forbid_balanced_sampling",
        "forbid_external_dataset",
        "forbid_hyperparameter_search",
        "forbid_modifying_controlled_results",
        "forbid_modifying_locked_matrix",
    ):
        if safety.get(key) is not True:
            raise ValueError(f"safety.{key} must be true")


def ensure_xgboost_dependency(*, require: bool) -> dict[str, Any]:
    if not classical_features.xgboost_available():
        if require:
            raise RuntimeError("dependency_missing: xgboost is not installed")
        return {"available": False, "version": ""}
    import xgboost

    return {"available": True, "version": str(xgboost.__version__)}


def build_xgboost_run_plan(config: dict[str, Any], splits: list[LOSOSmokeSplit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = 0
    for seed in config["seeds"]:
        for fold_id, split in enumerate(splits, start=1):
            run_id += 1
            rows.append(
                {
                    "run_id": run_id,
                    "model_name": "feature_xgboost_v1",
                    "seed": int(seed),
                    "fold_id": fold_id,
                    "test_subject": split.test_subject,
                    "status": "pending",
                    "output_subdir": f"feature_xgboost_v1/seed_{int(seed)}/fold_{fold_id:02d}",
                }
            )
    return rows


def run_xgboost_completion(
    config: dict[str, Any],
    *,
    project_root: Path,
    confirm_xgboost_completion: bool = False,
    dry_run: bool = False,
    resume_dir: str | Path | None = None,
) -> dict[str, Any]:
    validate_xgboost_completion_safety(config)
    dependency = ensure_xgboost_dependency(require=not dry_run)
    if not dry_run and not confirm_xgboost_completion:
        raise ValueError("refusing xgboost completion without --confirm-xgboost-completion")
    if not dry_run and _git_dirty(git_status(project_root)):
        raise ValueError("git status is dirty; commit before xgboost completion run")

    set_global_seed(int(config["seeds"][0]))
    dataset_dir = resolve_path(project_root, config["dataset_dir"])
    dataset = ProcessedTargetDataset(dataset_dir)
    run_dir = Path(resume_dir) if resume_dir else create_run_dir(project_root, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = config_digest(config)
    hash_path = run_dir / "config_hash.txt"
    if hash_path.exists():
        if hash_path.read_text(encoding="utf-8").strip() != config_hash:
            raise ValueError("resume config hash mismatch")
    else:
        _write_prelude(run_dir, project_root, config, config_hash, dataset_dir, dry_run, dependency)

    _write_feature_audit(run_dir)
    first_splits = _make_splits_for_config(_config_for_seed(config, int(config["seeds"][0])), dataset)
    run_plan = build_xgboost_run_plan(config, first_splits)
    completed = _completed_runs(run_dir) if (run_dir / "fold_metrics.csv").exists() else set()
    for row in run_plan:
        if (str(row["model_name"]), int(row["seed"]), int(row["fold_id"])) in completed:
            row["status"] = "skipped_completed"
    write_csv(run_dir / "run_plan.csv", run_plan)
    append_run_log(run_dir / "run_log.txt", [f"expected_runs={len(run_plan)}"])

    split_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []
    scaler_audit_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "fold_metrics.csv")
    classwise_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "classwise_metrics.csv")
    confusion_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "confusion_matrices.csv")
    prediction_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "predictions.csv")
    failed_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "failed_runs.csv")
    xgb_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "xgboost_feature_importance_by_fold.csv")

    run_status = {(str(row["model_name"]), int(row["seed"]), int(row["fold_id"])): str(row["status"]) for row in run_plan}
    for seed in config["seeds"]:
        seed_config = _config_for_seed(config, int(seed))
        splits = _make_splits_for_config(seed_config, dataset)
        for fold_id, split in enumerate(splits, start=1):
            split_rows.append(_split_plan_row(int(seed), fold_id, split, dataset.y))
            validation_rows.append(_validation_policy_row(int(seed), fold_id, split, dataset.y, dataset.subject_id))
            scaler = TrainOnlyStandardScaler().fit(dataset.X, train_idx=split.train_idx)
            X_scaled = scaler.transform(dataset.X)
            scaler_rows.append(_scaler_row(int(seed), fold_id, split, scaler))
            audit = build_scaler_fit_audit_row(fold_id=fold_id, model_name="feature_xgboost_v1", split=split, scaler=scaler, scaler_fit_indices=split.train_idx)
            audit["seed"] = int(seed)
            scaler_audit_rows.append(audit)
            key = ("feature_xgboost_v1", int(seed), fold_id)
            if run_status.get(key) == "skipped_completed":
                continue
            if not audit["scaler_leakage_check_passed"]:
                raise ValueError(f"scaler leakage detected for {key}")
            if dry_run:
                run_status[key] = "dry_run"
                _update_run_plan_status(run_plan, key, "dry_run")
                continue
            try:
                set_global_seed(int(seed))
                result = run_classical_fold(
                    model_name="feature_xgboost_v1",
                    seed=int(seed),
                    fold_id=fold_id,
                    X=X_scaled,
                    y=dataset.y,
                    subject_id=dataset.subject_id,
                    split=split,
                    return_estimator=True,
                )
                estimator = result.get("estimator")
                xgb_rows.extend(
                    xgboost_importance_rows(
                        estimator=estimator,
                        model_name="feature_xgboost_v1",
                        seed=int(seed),
                        fold_id=fold_id,
                        test_subject=split.test_subject,
                    )
                )
                fold_rows.append(_extension_fold_metric_row(result["fold_metric"], split, audit))
                classwise_rows.extend(result["classwise"])
                confusion_rows.extend(result["confusion"])
                prediction_rows.extend(result["predictions"])
                run_status[key] = "ok"
            except Exception as exc:  # noqa: BLE001 - preserve failure evidence.
                failed = _failed_row("feature_xgboost_v1", int(seed), fold_id, split, exc)
                failed_rows.append(failed)
                fold_rows.append(_failed_fold_row(failed))
                run_status[key] = "failed"
            _update_run_plan_status(run_plan, key, run_status[key])
            _write_intermediate(run_dir, run_plan, fold_rows, classwise_rows, confusion_rows, prediction_rows, failed_rows, xgb_rows)

    write_csv(run_dir / "split_plan.csv", split_rows)
    write_csv(run_dir / "validation_policy_summary.csv", validation_rows)
    write_csv(run_dir / "scaler_stats_by_fold.csv", scaler_rows)
    write_csv(run_dir / "scaler_fit_audit.csv", scaler_audit_rows)
    if dry_run:
        write_csv(run_dir / "run_plan.csv", run_plan)
        append_run_log(run_dir / "run_log.txt", ["status=dry_run_no_training"])
        return {"run_dir": str(run_dir), "dry_run": True, "expected_runs": len(run_plan), "xgboost_available": dependency["available"]}

    fold_confusion_rows = [row for row in confusion_rows if row.get("scope") == "fold"]
    confusion_rows = fold_confusion_rows + _aggregate_confusion_rows(fold_confusion_rows)
    statistics = config["statistics"]
    parameter_rows = [{"model_name": "feature_xgboost_v1", "total_params": 0, "encoder_params": 0, "aggregation_params": 0, "head_params": 0}]
    aggregate_seed_rows = aggregate_full_matrix_by_model_seed(fold_rows)
    aggregate_rows = aggregate_full_matrix_by_model(fold_rows, parameter_rows, bootstrap_n=int(statistics["bootstrap_n"]), seed=int(config["seeds"][0]))
    classwise_by_model = aggregate_classwise_by_model(classwise_rows)
    subjectwise_rows = aggregate_subjectwise_by_model(fold_rows, classwise_rows)
    combined_fold_rows = _controlled_reference_fold_rows(project_root, [str(item) for item in statistics["paired_reference_models"]]) + fold_rows
    paired_rows = [
        row
        for row in compute_paired_model_differences(
            combined_fold_rows,
            [str(item) for item in statistics["paired_reference_models"]],
            metric=str(statistics["primary_metric"]),
            bootstrap_n=int(statistics["bootstrap_n"]),
            seed=int(config["seeds"][0]),
        )
        if row.get("comparison_model") == "feature_xgboost_v1"
    ]
    ci_rows = compute_bootstrap_confidence_intervals(fold_rows, bootstrap_n=int(statistics["bootstrap_n"]), seed=int(config["seeds"][0]))

    _write_intermediate(run_dir, run_plan, fold_rows, classwise_rows, confusion_rows, prediction_rows, failed_rows, xgb_rows)
    write_csv(run_dir / "aggregate_metrics_by_model_seed.csv", aggregate_seed_rows)
    write_csv(run_dir / "aggregate_metrics_by_model.csv", aggregate_rows)
    write_csv(run_dir / "subjectwise_metrics_by_model.csv", subjectwise_rows)
    write_csv(run_dir / "classwise_metrics_by_model.csv", classwise_by_model)
    write_csv(run_dir / "bootstrap_confidence_intervals.csv", ci_rows)
    write_csv(run_dir / "paired_model_differences.csv", paired_rows)
    write_csv(run_dir / "xgboost_feature_importance_aggregate.csv", aggregate_importance_rows(xgb_rows))
    write_xgboost_completion_analysis(project_root=project_root, run_dir=run_dir)
    append_run_log(run_dir / "run_log.txt", [f"errors={len(failed_rows)}", "status=finished"])
    return {
        "run_dir": str(run_dir),
        "dry_run": False,
        "expected_runs": len(run_plan),
        "n_success_runs": sum(1 for status in run_status.values() if status in {"ok", "skipped_completed"}),
        "n_failed_runs": sum(1 for status in run_status.values() if status == "failed"),
        "xgboost_version": dependency["version"],
    }


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(yaml.safe_dump(config, allow_unicode=True, sort_keys=True).encode("utf-8")).hexdigest()


def _write_prelude(run_dir: Path, project_root: Path, config: dict[str, Any], config_hash: str, dataset_dir: Path, dry_run: bool, dependency: dict[str, Any]) -> None:
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "config_hash.txt").write_text(config_hash + "\n", encoding="utf-8")
    (run_dir / "git_status.txt").write_text(git_status(project_root) + "\n", encoding="utf-8")
    commit = git_commit(project_root)
    if commit:
        (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    else:
        (run_dir / "git_commit_unavailable.txt").write_text("git commit unavailable\n", encoding="utf-8")
    (run_dir / "environment_info.json").write_text(json.dumps(_environment_info(dependency), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "xgboost_version.txt").write_text((dependency.get("version") or "unavailable") + "\n", encoding="utf-8")
    (run_dir / "manifest_checksums.json").write_text(json.dumps(manifest_checksums(dataset_dir), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_run_log(
        run_dir / "run_log.txt",
        [
            f"experiment={config['experiment_name']}",
            f"mode={'dry_run_no_training' if dry_run else 'confirm_xgboost_completion'}",
            f"dataset_dir={dataset_dir}",
            "validation_policy=loso_with_within_train_subject_stratified_validation",
            "feature_policy=signal_derived_only",
            "scaler_fit=train_indices_only",
            "controlled_result=read_only",
        ],
    )


def _environment_info(dependency: dict[str, Any]) -> dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "sklearn_available": classical_features.sklearn_available(),
        "xgboost_available": bool(dependency.get("available")),
        "xgboost_version": dependency.get("version", ""),
    }
    if classical_features.sklearn_available():
        import sklearn

        info["sklearn"] = sklearn.__version__
    return info


def _write_feature_audit(run_dir: Path) -> None:
    definitions = feature_definitions(input_channels=18)
    audit = build_feature_audit_rows(input_channels=18)
    validate_feature_audit_rows(audit)
    write_csv(run_dir / "feature_definitions.csv", definitions)
    write_csv(run_dir / "feature_audit.csv", audit)


def _controlled_reference_fold_rows(project_root: Path, models: list[str]) -> list[dict[str, Any]]:
    path = project_root / CONTROLLED_RESULT / "fold_metrics.csv"
    return [row for row in _read_csv_dicts(path) if row.get("model_name") in set(models)]


def _completed_runs(run_dir: Path) -> set[tuple[str, int, int]]:
    return {(str(row["model_name"]), int(row["seed"]), int(row["fold_id"])) for row in _read_csv_dicts(run_dir / "fold_metrics.csv") if str(row.get("status")) == "ok"}


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
    confusion_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    xgb_rows: list[dict[str, Any]],
) -> None:
    write_csv(run_dir / "run_plan.csv", run_plan)
    _write_csv_with_header(run_dir / "fold_metrics.csv", fold_rows, ["model_name", "seed", "fold_id", "train_subjects", "val_subjects", "test_subject", "n_train", "n_val", "n_test", "best_epoch", "best_val_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1", "leakage_check_passed", "scaler_leakage_check_passed", "status"])
    write_csv(run_dir / "classwise_metrics.csv", classwise_rows)
    write_csv(run_dir / "confusion_matrices.csv", confusion_rows)
    write_csv(run_dir / "predictions.csv", prediction_rows)
    _write_csv_with_header(run_dir / "failed_runs.csv", failed_rows, ["model_name", "seed", "fold_id", "train_subjects", "val_subjects", "test_subject", "status", "error"])
    write_csv(run_dir / "xgboost_feature_importance_by_fold.csv", xgb_rows)


def _write_csv_with_header(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_dirty(status_text: str) -> bool:
    return str(status_text).strip() not in {"", "clean"}
