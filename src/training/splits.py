from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LOSOSmokeSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: list[int]
    val_subject: Any
    test_subject: int
    leakage_check_passed: bool
    test_subject_isolated: bool = True
    train_val_index_disjoint: bool = True
    train_class_counts: dict[str, int] = field(default_factory=dict)
    val_class_counts: dict[str, int] = field(default_factory=dict)
    test_class_counts: dict[str, int] = field(default_factory=dict)


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
        test_subject_isolated=bool(leakage_passed),
        train_val_index_disjoint=True,
        train_class_counts=class_counts(y[train_idx]),
        val_class_counts=class_counts(y[val_idx]),
        test_class_counts=class_counts(y[test_idx]),
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


def make_final_protocol_loso_splits(
    *,
    subject_id: np.ndarray,
    y: np.ndarray,
    metadata: list[dict[str, Any]],
    subjects: list[int],
    train_windows_per_subject_class: int,
    val_windows_per_subject_class: int,
    seed: int,
    strict_subject_isolation_for_test: bool = True,
    strict_index_disjoint_train_val: bool = True,
) -> list[LOSOSmokeSplit]:
    """LOSO with within-training-subject stratified validation.

    For each held-out test subject, every remaining subject contributes an
    equal number of windows per class to train and validation. The test subject
    is never used for train, validation, or scaler fitting.
    """

    subject_id = np.asarray(subject_id)
    y = np.asarray(y, dtype=np.int64)
    if len(subject_id) != len(y):
        raise ValueError("subject_id and y must have matching lengths")
    if len(metadata) != len(y):
        raise ValueError("metadata row count must match labels")
    ordered_subjects = [int(item) for item in subjects]
    if len(ordered_subjects) < 3 or len(set(ordered_subjects)) != len(ordered_subjects):
        raise ValueError("final protocol LOSO requires unique subjects")
    required_per_group = int(train_windows_per_subject_class) + int(val_windows_per_subject_class)
    if required_per_group <= 0:
        raise ValueError("train and validation windows per subject-class must be positive")

    _validate_metadata_matches_arrays(subject_id, y, metadata)
    class_ids = sorted(int(item) for item in np.unique(y).tolist())
    splits: list[LOSOSmokeSplit] = []
    for fold_index, test_subject in enumerate(ordered_subjects):
        candidate_subjects = [subject for subject in ordered_subjects if subject != test_subject]
        rng = np.random.default_rng(int(seed) + fold_index)
        train_idx: list[int] = []
        val_idx: list[int] = []
        for train_subject in candidate_subjects:
            for class_id in class_ids:
                group = np.where((subject_id.astype(int) == train_subject) & (y == class_id))[0]
                group = np.asarray(sorted(group.tolist(), key=lambda idx: _metadata_sort_key(metadata[idx], idx)), dtype=np.int64)
                if len(group) != required_per_group:
                    raise ValueError(
                        f"expected {required_per_group} windows for subject={train_subject}, class={class_id}; got {len(group)}"
                    )
                shuffled = group.copy()
                rng.shuffle(shuffled)
                val_idx.extend(shuffled[: int(val_windows_per_subject_class)].astype(int).tolist())
                train_idx.extend(
                    shuffled[
                        int(val_windows_per_subject_class) : int(val_windows_per_subject_class)
                        + int(train_windows_per_subject_class)
                    ]
                    .astype(int)
                    .tolist()
                )
        test_idx = np.where(subject_id.astype(int) == test_subject)[0].astype(np.int64)
        train_arr = np.asarray(sorted(train_idx), dtype=np.int64)
        val_arr = np.asarray(sorted(val_idx), dtype=np.int64)
        train_val_disjoint = not bool(set(train_arr.tolist()) & set(val_arr.tolist()))
        test_isolated = (
            test_subject not in set(subject_id[train_arr].astype(int).tolist())
            and test_subject not in set(subject_id[val_arr].astype(int).tolist())
            and set(subject_id[test_idx].astype(int).tolist()) == {test_subject}
        )
        leakage_passed = bool(train_val_disjoint and test_isolated)
        if strict_subject_isolation_for_test and not test_isolated:
            raise ValueError("test subject leakage detected in final protocol split")
        if strict_index_disjoint_train_val and not train_val_disjoint:
            raise ValueError("train/validation index overlap detected in final protocol split")
        splits.append(
            LOSOSmokeSplit(
                train_idx=train_arr,
                val_idx=val_arr,
                test_idx=test_idx,
                train_subjects=sorted(candidate_subjects),
                val_subject="within_train_stratified",
                test_subject=test_subject,
                leakage_check_passed=leakage_passed,
                test_subject_isolated=bool(test_isolated),
                train_val_index_disjoint=bool(train_val_disjoint),
                train_class_counts=class_counts(y[train_arr]),
                val_class_counts=class_counts(y[val_arr]),
                test_class_counts=class_counts(y[test_idx]),
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


def _validate_metadata_matches_arrays(
    subject_id: np.ndarray,
    y: np.ndarray,
    metadata: list[dict[str, Any]],
) -> None:
    for index, row in enumerate(metadata):
        if int(row["subject_id"]) != int(subject_id[index]):
            raise ValueError(f"metadata subject_id mismatch at sample {index}")
        if int(row["class_id"]) != int(y[index]):
            raise ValueError(f"metadata class_id mismatch at sample {index}")
        if "window_index" not in row:
            raise ValueError("metadata must include window_index")


def _metadata_sort_key(row: dict[str, Any], index: int) -> tuple[int, int, str, int]:
    return (
        int(row.get("set_id", 0) or 0),
        int(row.get("window_index", 0) or 0),
        str(row.get("sample_id", "")),
        int(index),
    )
