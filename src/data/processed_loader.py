from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedTargetDataset(Dataset):
    """Loader for converted target arrays. It performs no scaling or augmentation."""

    def __init__(self, root: str | Path) -> None:
        root_path = Path(root)
        self.root = root_path
        self.X = np.load(root_path / "X.npy")
        self.y = np.load(root_path / "y.npy")
        self.subject_id = np.load(root_path / "subject_id.npy")
        self.metadata = read_metadata(root_path / "metadata.csv")
        self._validate()

    @classmethod
    def from_arrays(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        subject_id: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> "ProcessedTargetDataset":
        obj = cls.__new__(cls)
        obj.root = None
        obj.X = np.asarray(X, dtype=np.float32)
        obj.y = np.asarray(y, dtype=np.int64)
        obj.subject_id = np.asarray(subject_id)
        obj.metadata = metadata or []
        obj._validate(require_metadata=False)
        return obj

    def _validate(self, require_metadata: bool = True) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"X must have shape (N, T, C), got {self.X.shape}")
        if self.X.shape[1:] != (512, 18):
            raise ValueError(f"expected X shape (N, 512, 18), got {self.X.shape}")
        if self.X.dtype != np.float32:
            raise ValueError(f"X dtype must be float32, got {self.X.dtype}")
        if self.y.dtype != np.int64:
            raise ValueError(f"y dtype must be int64, got {self.y.dtype}")
        if len(self.X) != len(self.y) or len(self.X) != len(self.subject_id):
            raise ValueError("X, y, and subject_id must have matching first dimension")
        if require_metadata and len(self.metadata) != len(self.X):
            raise ValueError("metadata.csv row count must match number of samples")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "x": torch.as_tensor(self.X[index], dtype=torch.float32),
            "y": int(self.y[index]),
            "subject_id": _subject_scalar(self.subject_id[index]),
            "metadata": self.metadata[index] if self.metadata else {},
        }


def make_loso_splits(subject_id: np.ndarray) -> dict[Any, dict[str, np.ndarray]]:
    subject_id = np.asarray(subject_id)
    splits: dict[Any, dict[str, np.ndarray]] = {}
    for held_out in sorted(np.unique(subject_id).tolist()):
        test_mask = subject_id == held_out
        train_mask = ~test_mask
        splits[_subject_scalar(held_out)] = {
            "train_idx": np.where(train_mask)[0],
            "test_idx": np.where(test_mask)[0],
        }
    return splits


def read_metadata(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _subject_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
