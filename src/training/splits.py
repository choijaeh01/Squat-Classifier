from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LOSOSmokeSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: list[int]
    val_subject: int
    test_subject: int
    leakage_check_passed: bool


def make_loso_smoke_split(
    *,
    subject_id: np.ndarray,
    y: np.ndarray,
    test_subject_id: int,
    val_subject_id: int,
    strict_subject_isolation: bool = True,
) -> LOSOSmokeSplit:
    subject_id = np.asarray(subject_id)
    y = np.asarray(y)
    if len(subject_id) != len(y):
        raise ValueError("subject_id and y must have matching lengths")
    if int(test_subject_id) == int(val_subject_id):
        raise ValueError("test_subject_id and val_subject_id must differ")
    available = set(int(item) for item in np.unique(subject_id))
    if int(test_subject_id) not in available or int(val_subject_id) not in available:
        raise ValueError("test and validation subjects must exist in subject_id")

    test_mask = subject_id == int(test_subject_id)
    val_mask = subject_id == int(val_subject_id)
    train_mask = ~(test_mask | val_mask)
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    train_subjects = sorted(int(item) for item in np.unique(subject_id[train_idx]))
    leakage_passed = (
        int(test_subject_id) not in train_subjects
        and int(val_subject_id) not in train_subjects
        and not np.any(subject_id[val_idx] == int(test_subject_id))
        and not np.any(subject_id[test_idx] == int(val_subject_id))
    )
    if strict_subject_isolation and not leakage_passed:
        raise ValueError("subject leakage detected in LOSO smoke split")
    return LOSOSmokeSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_subjects=train_subjects,
        val_subject=int(val_subject_id),
        test_subject=int(test_subject_id),
        leakage_check_passed=bool(leakage_passed),
    )


def make_cyclic_loso_splits(
    *,
    subject_id: np.ndarray,
    y: np.ndarray,
    subjects: list[int],
    strict_subject_isolation: bool = True,
) -> list[LOSOSmokeSplit]:
    if len(subjects) < 3:
        raise ValueError("cyclic LOSO requires at least three subjects")
    ordered_subjects = [int(item) for item in subjects]
    if len(set(ordered_subjects)) != len(ordered_subjects):
        raise ValueError("subjects must be unique")

    splits: list[LOSOSmokeSplit] = []
    for index, test_subject in enumerate(ordered_subjects):
        val_subject = ordered_subjects[(index + 1) % len(ordered_subjects)]
        splits.append(
            make_loso_smoke_split(
                subject_id=subject_id,
                y=y,
                test_subject_id=test_subject,
                val_subject_id=val_subject,
                strict_subject_isolation=strict_subject_isolation,
            )
        )
    return splits


def summarize_split(
    split: LOSOSmokeSplit,
    *,
    y: np.ndarray,
    subject_id: np.ndarray,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    y = np.asarray(y)
    subject_id = np.asarray(subject_id)
    return {
        "model_name": model_name,
        "seed": int(seed),
        "train_subjects": "|".join(str(item) for item in split.train_subjects),
        "val_subject": str(split.val_subject),
        "test_subject": str(split.test_subject),
        "n_train": int(len(split.train_idx)),
        "n_val": int(len(split.val_idx)),
        "n_test": int(len(split.test_idx)),
        "train_class_counts": json.dumps(class_counts(y[split.train_idx]), sort_keys=True),
        "val_class_counts": json.dumps(class_counts(y[split.val_idx]), sort_keys=True),
        "test_class_counts": json.dumps(class_counts(y[split.test_idx]), sort_keys=True),
        "leakage_check_passed": bool(split.leakage_check_passed),
    }


def class_counts(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)
    return {str(label): int(np.sum(labels == label)) for label in sorted(np.unique(labels).tolist())}
