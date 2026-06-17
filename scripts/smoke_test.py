#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.manifest import ManifestRecord, TargetDataManifest
from models.registry import build_model, count_parameters, list_models
from training.metrics import compute_classification_metrics
from training.results import git_commit_hash, result_payload, save_json
from utils.config import load_yaml
from utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic-data smoke checks for the IMU squat scaffold.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "smoke_test.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "smoke_test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config["seed"])
    set_global_seed(seed)

    shape_cfg = config["input_shape"]
    batch_size = int(shape_cfg["batch"])
    window_length = int(shape_cfg["window_length"])
    channels = int(shape_cfg["channels"])
    num_classes = 5
    num_subjects = int(config["num_subjects"])
    class_names = [config["class_names"][index] for index in range(num_classes)]

    rng = np.random.default_rng(seed)
    windows = rng.normal(size=(batch_size, window_length, channels)).astype(np.float32)
    labels = np.arange(batch_size, dtype=np.int64) % num_classes
    subject_ids = np.array([f"subject{(index % num_subjects) + 1}" for index in range(batch_size)])
    x = torch.from_numpy(windows)

    manifest = TargetDataManifest(
        dataset_name="synthetic-smoke-test",
        window_length=window_length,
        channels=channels,
        class_names=dict(config["class_names"]),
        records=[
            ManifestRecord(
                sample_id=f"synthetic_{index:03d}",
                path=f"synthetic://window/{index:03d}",
                subject_id=str(subject_ids[index]),
                label=int(labels[index]),
                source="synthetic",
            )
            for index in range(batch_size)
        ],
        notes="Synthetic manifest used only for smoke-test checksum plumbing.",
    )

    parameter_counts: dict[str, int] = {}
    model_predictions: dict[str, list[int]] = {}
    for model_name in list_models():
        model = build_model(model_name, input_length=window_length, input_channels=channels, num_classes=num_classes)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        if tuple(logits.shape) != (batch_size, num_classes):
            raise AssertionError(f"{model_name} returned {tuple(logits.shape)}, expected {(batch_size, num_classes)}")
        parameter_counts[model_name] = count_parameters(model)
        model_predictions[model_name] = logits.argmax(dim=1).cpu().numpy().astype(int).tolist()
        print(f"{model_name}: output_shape={tuple(logits.shape)} parameters={parameter_counts[model_name]}")

    reference_predictions = np.asarray(model_predictions["all_channel_conv1d"], dtype=np.int64)
    metrics = compute_classification_metrics(
        y_true=labels,
        y_pred=reference_predictions,
        subject_ids=subject_ids,
        num_classes=num_classes,
        class_names=class_names,
    )
    payload = result_payload(
        metrics=metrics,
        parameter_counts=parameter_counts,
        training_history={"smoke_test_only": True, "epochs": []},
        config_snapshot=config,
        git_commit=git_commit_hash(PROJECT_ROOT),
        data_manifest_checksum=manifest.checksum(),
        fold_confusion_matrices={"synthetic_smoke_fold": metrics["confusion_matrix"]},
    )
    payload["smoke_test"] = {
        "input_shape": [batch_size, window_length, channels],
        "output_shape": [batch_size, num_classes],
        "subject_ids_saved": sorted(set(subject_ids.tolist())),
        "labels_saved": labels.astype(int).tolist(),
        "models_checked": list_models(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.output_dir / "metrics.json")
    manifest.save(args.output_dir / "synthetic_manifest.json")
    print(f"saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"saved manifest: {args.output_dir / 'synthetic_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
