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

from analysis.controlled_comparison_analysis import write_controlled_analysis_outputs
from analysis.feature_importance_analysis import (
    aggregate_coefficient_rows,
    aggregate_importance_rows,
    build_feature_audit_rows,
    feature_definitions,
    linear_svm_coefficient_rows,
    random_forest_importance_rows,
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
from data.processed_loader import ProcessedTargetDataset
from models.classical_features import classical_model_available, is_classical_model, sklearn_available, xgboost_available
from models.controlled_models import CONTROLLED_NEURAL_MODEL_NAMES
from models.registry import build_model, count_parameter_groups
from training.classical_trainer import run_classical_fold
from training.literature_full_extension_runner import (
    _aggregate_confusion_rows,
    _config_for_seed,
    _extension_fold_metric_row,
    _failed_fold_row,
    _failed_row,
    _read_csv_dicts,
    _scaler_row,
    _skipped_row,
    _split_plan_row,
    _update_run_plan_status,
    _validation_policy_row,
    _with_seed,
)
from training.loso_runner import _make_splits_for_config, _run_model_fold, build_scaler_fit_audit_row
from training.scalers import TrainOnlyStandardScaler
from training.splits import LOSOSmokeSplit
from training.supervised_trainer import append_run_log, create_run_dir, git_commit, git_status, manifest_checksums, resolve_path, write_csv
from utils.reproducibility import set_global_seed


def load_controlled_comparison_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def controlled_model_names(config: dict[str, Any]) -> list[str]:
    groups = config["model_groups"]
    return [
        *[str(item) for item in groups.get("controlled_neural", [])],
        *[str(item) for item in groups.get("classical_practical", [])],
        *[str(item) for item in groups.get("literature_reference", [])],
    ]


def model_group_for(config: dict[str, Any], model_name: str) -> str:
    for group_name, names in config["model_groups"].items():
        if model_name in set(str(item) for item in names):
            return str(group_name)
    return "unknown"


def validate_controlled_comparison_safety(config: dict[str, Any]) -> None:
    training = config["training"]
    split = config["split"]
    normalization = config["normalization"]
    augmentation = config["augmentation"]
    safety = config["safety"]
    if training.get("controlled_comparison_mode") is not True:
        raise ValueError("controlled comparison requires training.controlled_comparison_mode=true")
    if training.get("allow_full_training") is not True:
        raise ValueError("controlled comparison requires allow_full_training=true")
    if config.get("seeds") != [42, 123, 2025]:
        raise ValueError("controlled comparison v1 requires seeds [42, 123, 2025]")
    if split.get("type") != "loso_with_within_train_subject_stratified_validation":
        raise ValueError("controlled comparison requires final within-train stratified LOSO")
    if int(split.get("train_windows_per_subject_class", 0)) != 16 or int(split.get("val_windows_per_subject_class", 0)) != 4:
        raise ValueError("controlled comparison split must use 16 train and 4 validation windows per subject-class")
    if split.get("strict_subject_isolation_for_test") is not True or split.get("strict_index_disjoint_train_val") is not True:
        raise ValueError("split leakage guards must be enabled")
    if training.get("loss") != "cross_entropy" or training.get("optimizer") != "adam":
        raise ValueError("controlled comparison uses fixed cross_entropy and adam settings")
    if training.get("mixed_precision") is not False:
        raise ValueError("mixed precision must remain disabled")
    if augmentation.get("enabled") is not False or safety.get("forbid_augmentation") is not True:
        raise ValueError("augmentation is forbidden")
    if safety.get("forbid_focal_loss") is not True or safety.get("forbid_balanced_sampling") is not True:
        raise ValueError("focal loss and balanced sampling are forbidden")
    if safety.get("forbid_ssl") is not True or safety.get("forbid_external_dataset") is not True:
        raise ValueError("SSL and external dataset adapters are forbidden")
    if safety.get("require_confirm_controlled_comparison") is not True:
        raise ValueError("controlled comparison requires explicit confirmation")
    if safety.get("require_clean_git_before_run") is not True:
        raise ValueError("controlled comparison requires clean git before run")
    for key in ("forbid_modifying_locked_matrix", "forbid_modifying_literature_extension", "forbid_modifying_v3_ablation"):
        if safety.get(key) is not True:
            raise ValueError(f"{key} must be true")
    if normalization.get("global_standard_scaler") is not True or normalization.get("fit_scaler_on") != "train_indices_only":
        raise ValueError("scaler must fit on train indices only")
    if normalization.get("per_window_zscore") is not False:
        raise ValueError("per-window z-score is forbidden")
    if int(config["common_representation"]["representation_dim"]) != 64:
        raise ValueError("controlled comparison v1 requires representation_dim=64")
    if config["common_classifier_head"] != {"hidden_dim": 64, "dropout": 0.1, "activation": "relu", "num_classes": 5}:
        raise ValueError("common classifier head must match the approved fixed architecture")
    if len(config["model_groups"]["controlled_neural"]) != 9:
        raise ValueError("controlled comparison v1 requires nine controlled neural models")
    if len(controlled_model_names(config)) != 14:
        raise ValueError("controlled comparison v1 requires fourteen configured models")


def build_controlled_run_plan(config: dict[str, Any], splits: list[LOSOSmokeSplit]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = 0
    for seed in config["seeds"]:
        for fold_id, split in enumerate(splits, start=1):
            for model_name in controlled_model_names(config):
                run_id += 1
                rows.append(
                    {
                        "run_id": run_id,
                        "model_name": model_name,
                        "model_group": model_group_for(config, model_name),
                        "seed": int(seed),
                        "fold_id": fold_id,
                        "test_subject": split.test_subject,
                        "status": "pending",
                        "output_subdir": f"{model_name}/seed_{int(seed)}/fold_{fold_id:02d}",
                    }
                )
    return rows


def run_controlled_comparison(
    config: dict[str, Any],
    *,
    project_root: Path,
    confirm_controlled_comparison: bool = False,
    dry_run: bool = False,
    device_name: str | None = None,
    resume_dir: str | Path | None = None,
) -> dict[str, Any]:
    validate_controlled_comparison_safety(config)
    if not dry_run and not confirm_controlled_comparison:
        raise ValueError("refusing controlled comparison without --confirm-controlled-comparison")
    if not dry_run and _git_dirty(git_status(project_root)):
        raise ValueError("git status is dirty; commit before controlled comparison run")

    set_global_seed(int(config["seeds"][0]))
    _configure_deterministic(bool(config.get("reproducibility", {}).get("deterministic", True)))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_dir = resolve_path(project_root, config["dataset_dir"])
    dataset = ProcessedTargetDataset(dataset_dir)
    run_dir = Path(resume_dir) if resume_dir else create_run_dir(project_root, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = config_digest(config)
    if resume_dir:
        existing_hash = (run_dir / "config_hash.txt").read_text(encoding="utf-8").strip()
        if existing_hash != config_hash:
            raise ValueError("resume config hash mismatch")
    else:
        _write_prelude(run_dir, project_root, config, config_hash, dataset_dir, device, dry_run)

    _write_feature_audit(run_dir)
    first_config = _config_for_seed(config, int(config["seeds"][0]))
    first_splits = _make_splits_for_config(first_config, dataset)
    run_plan = build_controlled_run_plan(config, first_splits)
    completed = _completed_runs(run_dir) if resume_dir else set()
    for row in run_plan:
        if (str(row["model_name"]), int(row["seed"]), int(row["fold_id"])) in completed:
            row["status"] = "skipped_completed"
    write_csv(run_dir / "run_plan.csv", run_plan)
    parameter_rows = _parameter_rows(config, device)
    write_csv(run_dir / "model_parameter_counts.csv", parameter_rows)
    write_csv(run_dir / "common_head_verification.csv", _common_head_verification_rows(parameter_rows))
    append_run_log(run_dir / "run_log.txt", [f"expected_runs={len(run_plan)}"])

    split_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []
    scaler_audit_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "fold_metrics.csv") if resume_dir else []
    classwise_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "classwise_metrics.csv") if resume_dir else []
    history_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "training_history.csv") if resume_dir else []
    confusion_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "confusion_matrices.csv") if resume_dir else []
    prediction_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "predictions.csv") if resume_dir else []
    failed_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "failed_runs.csv") if resume_dir else []
    skipped_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "skipped_runs.csv") if resume_dir else []
    rf_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "random_forest_feature_importance_by_fold.csv") if resume_dir else []
    xgb_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "xgboost_feature_importance_by_fold.csv") if resume_dir else []
    svm_rows: list[dict[str, Any]] = _read_csv_dicts(run_dir / "linear_svm_coefficients_by_fold.csv") if resume_dir else []

    run_status: dict[tuple[str, int, int], str] = {
        (str(row["model_name"]), int(row["seed"]), int(row["fold_id"])): str(row["status"])
        for row in run_plan
    }

    for seed in config["seeds"]:
        seed_config = _config_for_seed(config, int(seed))
        splits = _make_splits_for_config(seed_config, dataset)
        for fold_id, split in enumerate(splits, start=1):
            split_rows.append(_split_plan_row(int(seed), fold_id, split, dataset.y))
            validation_rows.append(_validation_policy_row(int(seed), fold_id, split, dataset.y, dataset.subject_id))
            scaler = TrainOnlyStandardScaler().fit(dataset.X, train_idx=split.train_idx)
            X_scaled = scaler.transform(dataset.X)
            scaler_rows.append(_scaler_row(int(seed), fold_id, split, scaler))
            for model_name in controlled_model_names(config):
                model_name = str(model_name)
                key = (model_name, int(seed), fold_id)
                audit = build_scaler_fit_audit_row(fold_id=fold_id, model_name=model_name, split=split, scaler=scaler, scaler_fit_indices=split.train_idx)
                audit["seed"] = int(seed)
                scaler_audit_rows.append(audit)
                if run_status.get(key) == "skipped_completed":
                    continue
                if not audit["scaler_leakage_check_passed"]:
                    raise ValueError(f"scaler leakage detected for {key}")
                if is_classical_model(model_name) and not classical_model_available(model_name):
                    skipped_rows.append(_skipped_row(model_name, int(seed), fold_id, split, _missing_dependency_reason(model_name)))
                    run_status[key] = "skipped"
                    _update_run_plan_status(run_plan, key, "skipped")
                    continue
                if dry_run:
                    run_status[key] = "forward_only"
                    _update_run_plan_status(run_plan, key, "forward_only")
                    continue
                try:
                    set_global_seed(int(seed))
                    if is_classical_model(model_name):
                        result = run_classical_fold(
                            model_name=model_name,
                            seed=int(seed),
                            fold_id=fold_id,
                            X=X_scaled,
                            y=dataset.y,
                            subject_id=dataset.subject_id,
                            split=split,
                            return_estimator=True,
                        )
                        estimator = result.get("estimator")
                        if model_name == "feature_random_forest_v1":
                            rf_rows.extend(random_forest_importance_rows(estimator=estimator, model_name=model_name, seed=int(seed), fold_id=fold_id, test_subject=split.test_subject))
                        if model_name == "feature_xgboost_v1":
                            xgb_rows.extend(xgboost_importance_rows(estimator=estimator, model_name=model_name, seed=int(seed), fold_id=fold_id, test_subject=split.test_subject))
                        if model_name == "feature_linear_svm_v1":
                            svm_rows.extend(linear_svm_coefficient_rows(estimator=estimator, model_name=model_name, seed=int(seed), fold_id=fold_id, test_subject=split.test_subject))
                    else:
                        model = build_model(model_name, input_length=512, input_channels=18, num_classes=5).to(device)
                        result = _run_model_fold(
                            model_name=model_name,
                            fold_id=fold_id,
                            model=model,
                            X=X_scaled,
                            y=dataset.y,
                            subject_id=dataset.subject_id,
                            split=split,
                            config=seed_config,
                            run_dir=run_dir / "runs" / model_name / f"seed_{int(seed)}",
                            device=device,
                            dry_run=False,
                        )
                    fold_rows.append(_extension_fold_metric_row(result["fold_metric"], split, audit))
                    classwise_rows.extend(result["classwise"])
                    history_rows.extend(_with_seed(result["history"], int(seed)))
                    confusion_rows.extend(result["confusion"])
                    prediction_rows.extend(result["predictions"])
                    run_status[key] = "ok"
                except Exception as exc:  # noqa: BLE001 - experiment failures are recorded.
                    failed = _failed_row(model_name, int(seed), fold_id, split, exc)
                    failed_rows.append(failed)
                    fold_rows.append(_failed_fold_row(failed))
                    run_status[key] = "failed"
                _update_run_plan_status(run_plan, key, run_status[key])
                _write_intermediate(run_dir, run_plan, fold_rows, classwise_rows, history_rows, confusion_rows, prediction_rows, failed_rows, skipped_rows, rf_rows, xgb_rows, svm_rows)

    write_csv(run_dir / "split_plan.csv", split_rows)
    write_csv(run_dir / "validation_policy_summary.csv", validation_rows)
    write_csv(run_dir / "scaler_stats_by_fold.csv", scaler_rows)
    write_csv(run_dir / "scaler_fit_audit.csv", scaler_audit_rows)
    if dry_run:
        _write_intermediate(run_dir, run_plan, fold_rows, classwise_rows, history_rows, confusion_rows, prediction_rows, failed_rows, skipped_rows, rf_rows, xgb_rows, svm_rows)
        append_run_log(run_dir / "run_log.txt", ["status=dry_run_no_training"])
        return {"run_dir": str(run_dir), "device": str(device), "dry_run": True, "expected_runs": len(run_plan)}

    fold_confusion_rows = [row for row in confusion_rows if row.get("scope") == "fold"]
    confusion_rows = fold_confusion_rows + _aggregate_confusion_rows(fold_confusion_rows)
    statistics = config["statistics"]
    aggregate_seed_rows = aggregate_full_matrix_by_model_seed(fold_rows)
    aggregate_rows = aggregate_full_matrix_by_model(fold_rows, parameter_rows, bootstrap_n=int(statistics["bootstrap_n"]), seed=int(config["seeds"][0]))
    classwise_by_model = aggregate_classwise_by_model(classwise_rows)
    subjectwise_rows = aggregate_subjectwise_by_model(fold_rows, classwise_rows)
    paired_refs = [str(item) for item in statistics["paired_reference_models"]]
    paired_rows = compute_paired_model_differences(fold_rows, paired_refs, metric=str(statistics["primary_metric"]), bootstrap_n=int(statistics["bootstrap_n"]), seed=int(config["seeds"][0]))
    ci_rows = compute_bootstrap_confidence_intervals(fold_rows, bootstrap_n=int(statistics["bootstrap_n"]), seed=int(config["seeds"][0]))

    _write_intermediate(run_dir, run_plan, fold_rows, classwise_rows, history_rows, confusion_rows, prediction_rows, failed_rows, skipped_rows, rf_rows, xgb_rows, svm_rows)
    write_csv(run_dir / "aggregate_metrics_by_model_seed.csv", aggregate_seed_rows)
    write_csv(run_dir / "aggregate_metrics_by_model.csv", aggregate_rows)
    write_csv(run_dir / "subjectwise_metrics_by_model.csv", subjectwise_rows)
    write_csv(run_dir / "classwise_metrics_by_model.csv", classwise_by_model)
    write_csv(run_dir / "paired_model_differences.csv", paired_rows)
    write_csv(run_dir / "bootstrap_confidence_intervals.csv", ci_rows)
    write_csv(run_dir / "random_forest_feature_importance_aggregate.csv", aggregate_importance_rows(rf_rows))
    write_csv(run_dir / "xgboost_feature_importance_aggregate.csv", aggregate_importance_rows(xgb_rows))
    write_csv(run_dir / "linear_svm_coefficients_aggregate.csv", aggregate_coefficient_rows(svm_rows))
    write_controlled_analysis_outputs(project_root=project_root, run_dir=run_dir)
    append_run_log(run_dir / "run_log.txt", [f"errors={len(failed_rows)}", f"skipped={len(skipped_rows)}", "status=finished"])
    return {
        "run_dir": str(run_dir),
        "device": str(device),
        "dry_run": False,
        "expected_runs": len(run_plan),
        "n_success_runs": sum(1 for status in run_status.values() if status in {"ok", "skipped_completed"}),
        "n_failed_runs": sum(1 for status in run_status.values() if status == "failed"),
        "n_skipped_runs": sum(1 for status in run_status.values() if status == "skipped"),
    }


def config_digest(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, allow_unicode=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_prelude(run_dir: Path, project_root: Path, config: dict[str, Any], config_hash: str, dataset_dir: Path, device: torch.device, dry_run: bool) -> None:
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run_dir / "config_hash.txt").write_text(config_hash + "\n", encoding="utf-8")
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
            f"mode={'dry_run_no_training' if dry_run else 'confirm_controlled_comparison'}",
            f"device={device}",
            f"dataset_dir={dataset_dir}",
            "validation_policy=loso_with_within_train_subject_stratified_validation",
            "augmentation=disabled",
            "per_window_zscore=false",
            "scaler_fit=train_indices_only",
            "locked_results=read_only",
        ],
    )


def _environment_info(device: torch.device) -> dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else "cpu",
        "sklearn_available": sklearn_available(),
        "xgboost_available": xgboost_available(),
    }
    if sklearn_available():
        import sklearn

        info["sklearn"] = sklearn.__version__
    if xgboost_available():
        import xgboost

        info["xgboost"] = xgboost.__version__
    return info


def _write_feature_audit(run_dir: Path) -> None:
    definitions = feature_definitions(input_channels=18)
    audit = build_feature_audit_rows(input_channels=18)
    validate_feature_audit_rows(audit)
    write_csv(run_dir / "feature_definitions.csv", definitions)
    write_csv(run_dir / "feature_audit.csv", audit)


def _parameter_rows(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x = torch.zeros(2, 512, 18, dtype=torch.float32, device=device)
    for model_name in controlled_model_names(config):
        group = model_group_for(config, model_name)
        if is_classical_model(model_name):
            rows.append(
                {
                    "model_name": model_name,
                    "model_group": group,
                    "total_params": "",
                    "encoder_params": "",
                    "aggregation_params": "",
                    "head_params": "",
                    "other_params": "",
                    "extractor_params": "",
                    "projection_params": "",
                    "common_head_params": "",
                    "residual_branch_params": "",
                    "identity_embedding_params": "",
                    "representation_dim": "not_applicable",
                    "common_head_signature": "not_applicable",
                    "input_shape": "(batch, 512, 18)",
                    "output_shape": "not_applicable",
                    "notes": f"classical baseline; available={classical_model_available(model_name)}",
                }
            )
            continue
        model = build_model(model_name, input_length=512, input_channels=18, num_classes=5).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        if group == "controlled_neural":
            rows.append(_controlled_parameter_row(model_name, group, model, logits))
        else:
            groups = count_parameter_groups(model)
            rows.append(
                {
                    "model_name": model_name,
                    "model_group": group,
                    "total_params": groups["total_params"],
                    "encoder_params": groups["encoder_params"],
                    "aggregation_params": groups["aggregation_params"],
                    "head_params": groups["head_params"],
                    "other_params": groups["other_params"],
                    "extractor_params": "",
                    "projection_params": "",
                    "common_head_params": "",
                    "residual_branch_params": "",
                    "identity_embedding_params": "",
                    "representation_dim": "",
                    "common_head_signature": "",
                    "input_shape": "(batch, 512, 18)",
                    "output_shape": str(tuple(logits.shape)),
                    "notes": "literature reference model",
                }
            )
    return rows


def _controlled_parameter_row(model_name: str, group: str, model: Any, logits: torch.Tensor) -> dict[str, Any]:
    extractor = model.extractor
    total = _module_params(model)
    head = _module_params(model.classifier)
    extractor_params = _module_params(extractor)
    projection_modules = [getattr(extractor, name) for name in ("projection", "fusion_projection") if hasattr(extractor, name) and getattr(extractor, name) is not None]
    residual_modules = [getattr(extractor, "residual_projection")] if hasattr(extractor, "residual_projection") and getattr(extractor, "residual_projection") is not None else []
    identity_modules = [getattr(extractor, "identity_embeddings")] if hasattr(extractor, "identity_embeddings") and len(getattr(extractor, "identity_embeddings")) > 0 else []
    projection = _unique_module_params(projection_modules)
    residual = _unique_module_params(residual_modules)
    identity = _unique_module_params(identity_modules)
    encoder = max(0, extractor_params - projection - residual - identity)
    return {
        "model_name": model_name,
        "model_group": group,
        "total_params": total,
        "encoder_params": encoder,
        "aggregation_params": identity + residual + projection,
        "head_params": head,
        "other_params": total - extractor_params - head,
        "extractor_params": extractor_params,
        "projection_params": projection,
        "common_head_params": head,
        "residual_branch_params": residual,
        "identity_embedding_params": identity,
        "representation_dim": model.representation_dim,
        "common_head_signature": "|".join(model.classifier.architecture_signature()),
        "input_shape": "(batch, 512, 18)",
        "output_shape": str(tuple(logits.shape)),
        "notes": "controlled neural model with common classifier head",
    }


def _common_head_verification_rows(parameter_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controlled = [row for row in parameter_rows if row.get("model_group") == "controlled_neural"]
    head_counts = {str(row.get("common_head_params")) for row in controlled}
    signatures = {str(row.get("common_head_signature")) for row in controlled}
    return [
        {
            "model_name": row["model_name"],
            "representation_dim": row["representation_dim"],
            "common_head_params": row["common_head_params"],
            "common_head_signature": row["common_head_signature"],
            "common_head_param_count_shared": len(head_counts) == 1,
            "common_head_signature_shared": len(signatures) == 1,
        }
        for row in controlled
    ]


def _module_params(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _unique_module_params(modules: list[torch.nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
    return total


def _missing_dependency_reason(model_name: str) -> str:
    if model_name == "feature_xgboost_v1":
        return "xgboost is not installed"
    return "scikit-learn is not installed"


def _completed_runs(run_dir: Path) -> set[tuple[str, int, int]]:
    return {(str(row["model_name"]), int(row["seed"]), int(row["fold_id"])) for row in _read_csv_dicts(run_dir / "fold_metrics.csv") if str(row.get("status")) == "ok"}


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
    rf_rows: list[dict[str, Any]],
    xgb_rows: list[dict[str, Any]],
    svm_rows: list[dict[str, Any]],
) -> None:
    write_csv(run_dir / "run_plan.csv", run_plan)
    _write_csv_with_header(run_dir / "fold_metrics.csv", fold_rows, ["model_name", "seed", "fold_id", "train_subjects", "val_subjects", "test_subject", "n_train", "n_val", "n_test", "best_epoch", "best_val_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1", "leakage_check_passed", "scaler_leakage_check_passed", "status"])
    write_csv(run_dir / "classwise_metrics.csv", classwise_rows)
    write_csv(run_dir / "training_history.csv", history_rows)
    write_csv(run_dir / "confusion_matrices.csv", confusion_rows)
    write_csv(run_dir / "predictions.csv", prediction_rows)
    _write_csv_with_header(run_dir / "failed_runs.csv", failed_rows, ["model_name", "seed", "fold_id", "train_subjects", "val_subjects", "test_subject", "status", "error"])
    _write_csv_with_header(run_dir / "skipped_runs.csv", skipped_rows, ["model_name", "seed", "fold_id", "train_subjects", "val_subjects", "test_subject", "status", "reason"])
    write_csv(run_dir / "random_forest_feature_importance_by_fold.csv", rf_rows)
    write_csv(run_dir / "xgboost_feature_importance_by_fold.csv", xgb_rows)
    write_csv(run_dir / "linear_svm_coefficients_by_fold.csv", svm_rows)


def _write_csv_with_header(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_dirty(status_text: str) -> bool:
    return str(status_text).strip() not in {"", "clean"}


def _configure_deterministic(enabled: bool) -> None:
    if not enabled:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
