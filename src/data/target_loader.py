from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data.manifest import ManifestRecord, TargetDataManifest


class TargetWindowDataset(Dataset):
    """PyTorch dataset scaffold for manifest-backed target IMU windows."""

    def __init__(self, manifest: TargetDataManifest | str | Path, transform: Any | None = None) -> None:
        if isinstance(manifest, (str, Path)):
            manifest = TargetDataManifest.load(manifest)
        issues = manifest.validate()
        if issues:
            raise ValueError("invalid target data manifest:\n" + "\n".join(issues))
        self.manifest = manifest
        self.records = list(manifest.records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        window = load_window(record)
        if self.transform is not None:
            window = self.transform(window)
        return {
            "x": torch.as_tensor(window, dtype=torch.float32),
            "y": int(record.label),
            "subject_id": record.subject_id,
            "sample_id": record.sample_id,
            "path": record.path,
        }

    def subject_ids(self) -> list[str]:
        return [record.subject_id for record in self.records]

    def labels(self) -> list[int]:
        return [int(record.label) for record in self.records]


def load_window(record: ManifestRecord) -> np.ndarray:
    path = Path(record.path)
    if path.suffix.lower() == ".npy":
        window = np.load(path)
    elif path.suffix.lower() == ".csv":
        window = _load_csv_window(path)
    else:
        raise ValueError(f"unsupported window file extension: {path.suffix}")
    window = np.asarray(window, dtype=np.float32)
    expected_shape = (record.window_length, record.channels)
    if window.shape != expected_shape:
        raise ValueError(f"{path} has shape {window.shape}, expected {expected_shape}")
    return window


def _load_csv_window(path: Path) -> np.ndarray:
    raw = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if np.isnan(raw[0]).any():
        raw = raw[1:]
    return raw
