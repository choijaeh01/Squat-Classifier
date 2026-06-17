from __future__ import annotations

from typing import Any

import numpy as np


CLASSICAL_BASELINE_NAMES = {"feature_random_forest_v1", "feature_linear_svm_v1"}


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def is_classical_model(model_name: str) -> bool:
    return str(model_name) in CLASSICAL_BASELINE_NAMES


def extract_window_features(X: np.ndarray) -> np.ndarray:
    """Extract deterministic per-window, per-channel handcrafted features.

    Feature blocks are ordered by statistic, each containing all channels:
    mean, std, min, max, median, energy, RMS, peak-to-peak, dominant FFT bin.
    No label, subject, split, or test-set information is used.
    """

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"X must have shape (N, T, C), got {X.shape}")
    means = X.mean(axis=1)
    stds = X.std(axis=1)
    mins = X.min(axis=1)
    maxs = X.max(axis=1)
    medians = np.median(X, axis=1)
    energy = np.mean(np.square(X), axis=1)
    rms = np.sqrt(energy)
    ptp = np.ptp(X, axis=1)
    dominant = _dominant_frequency_bins(X)
    features = np.concatenate([means, stds, mins, maxs, medians, energy, rms, ptp, dominant], axis=1)
    return features.astype(np.float32)


def build_classical_estimator(model_name: str, seed: int) -> Any:
    if not sklearn_available():
        raise ModuleNotFoundError("scikit-learn is not installed")
    if model_name == "feature_random_forest_v1":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=200, max_depth=None, random_state=int(seed), n_jobs=1)
    if model_name == "feature_linear_svm_v1":
        from sklearn.svm import LinearSVC

        return LinearSVC(random_state=int(seed), max_iter=5000)
    raise KeyError(f"unknown classical model {model_name!r}")


def _dominant_frequency_bins(X: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(X, axis=1))
    if spectrum.shape[1] <= 1:
        return np.zeros((X.shape[0], X.shape[2]), dtype=np.float32)
    spectrum[:, 0, :] = 0.0
    dominant = np.argmax(spectrum, axis=1).astype(np.float32)
    return dominant / float(max(1, X.shape[1]))
