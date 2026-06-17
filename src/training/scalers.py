from __future__ import annotations

from typing import Any

import numpy as np


class TrainOnlyStandardScaler:
    """Global channel scaler fit only on explicitly supplied training indices."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.fit_sample_count_: int = 0
        self.fit_window_count_: int = 0

    def fit(self, X: np.ndarray, *, train_idx: np.ndarray) -> "TrainOnlyStandardScaler":
        X = np.asarray(X, dtype=np.float32)
        train_idx = np.asarray(train_idx, dtype=np.int64)
        if X.ndim != 3:
            raise ValueError(f"X must have shape (N, T, C), got {X.shape}")
        if train_idx.size == 0:
            raise ValueError("train_idx must not be empty")
        train = X[train_idx]
        self.mean_ = train.mean(axis=(0, 1)).astype(np.float32)
        scale = train.std(axis=(0, 1)).astype(np.float32)
        self.scale_ = np.where(scale < self.eps, 1.0, scale).astype(np.float32)
        self.fit_sample_count_ = int(train.shape[0])
        self.fit_window_count_ = int(train.shape[0] * train.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("TrainOnlyStandardScaler must be fit before transform")
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_scope": "train_subjects_only",
            "fit_sample_count": self.fit_sample_count_,
            "fit_window_count": self.fit_window_count_,
            "mean": [] if self.mean_ is None else self.mean_.astype(float).tolist(),
            "scale": [] if self.scale_ is None else self.scale_.astype(float).tolist(),
            "eps": self.eps,
        }
