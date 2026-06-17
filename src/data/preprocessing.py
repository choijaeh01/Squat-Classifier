from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PreprocessingLog:
    scaler_fit_scope: str
    held_out_subject: str | None
    per_window_normalization: bool
    augmentation_applied: bool
    random_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scaler_fit_scope": self.scaler_fit_scope,
            "held_out_subject": self.held_out_subject,
            "per_window_normalization": self.per_window_normalization,
            "augmentation_applied": self.augmentation_applied,
            "random_seed": self.random_seed,
        }


class LOSOStandardScaler:
    """Channel-wise standard scaler that can only be fit with a declared held-out subject."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.fit_subjects_: list[str] = []
        self.held_out_subject_: str | None = None

    def fit(self, windows: np.ndarray, subject_ids: np.ndarray, held_out_subject: str) -> "LOSOStandardScaler":
        windows = _as_windows(windows)
        subject_ids = np.asarray(subject_ids)
        if len(subject_ids) != windows.shape[0]:
            raise ValueError("subject_ids length must match number of windows")
        if held_out_subject is None:
            raise ValueError("held_out_subject is required to prevent scaler leakage")

        train_mask = subject_ids != held_out_subject
        if not np.any(train_mask):
            raise ValueError(f"no training windows remain after holding out {held_out_subject!r}")

        train_windows = windows[train_mask]
        self.mean_ = train_windows.mean(axis=(0, 1)).astype(np.float32)
        scale = train_windows.std(axis=(0, 1)).astype(np.float32)
        self.scale_ = np.where(scale < self.eps, 1.0, scale).astype(np.float32)
        self.fit_subjects_ = sorted(str(item) for item in np.unique(subject_ids[train_mask]))
        self.held_out_subject_ = str(held_out_subject)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        windows = _as_windows(windows)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("LOSOStandardScaler must be fit before transform")
        return ((windows - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, windows: np.ndarray, subject_ids: np.ndarray, held_out_subject: str) -> np.ndarray:
        return self.fit(windows, subject_ids, held_out_subject).transform(windows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
            "fit_subjects": self.fit_subjects_,
            "held_out_subject": self.held_out_subject_,
            "eps": self.eps,
        }


def per_window_normalize(windows: np.ndarray, enabled: bool, eps: float = 1e-8) -> tuple[np.ndarray, dict[str, Any]]:
    windows = _as_windows(windows)
    if not enabled:
        return windows.astype(np.float32, copy=True), {"per_window_normalization": False}
    mean = windows.mean(axis=1, keepdims=True)
    scale = windows.std(axis=1, keepdims=True)
    scale = np.where(scale < eps, 1.0, scale)
    normalized = ((windows - mean) / scale).astype(np.float32)
    return normalized, {"per_window_normalization": True, "eps": eps}


def augment_training_windows(
    windows: np.ndarray,
    *,
    is_training: bool,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    windows = _as_windows(windows)
    if not is_training or noise_std <= 0.0:
        return windows.astype(np.float32, copy=True), {
            "augmentation_applied": False,
            "reason": "not_training" if not is_training else "noise_std_is_zero",
        }
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=windows.shape).astype(np.float32)
    return (windows + noise).astype(np.float32), {
        "augmentation_applied": True,
        "noise_std": float(noise_std),
        "seed": seed,
    }


def _as_windows(windows: np.ndarray) -> np.ndarray:
    array = np.asarray(windows, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"windows must have shape (N, T, C), got {array.shape}")
    return array
