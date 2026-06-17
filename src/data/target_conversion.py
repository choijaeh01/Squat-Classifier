from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


METADATA_RE = re.compile(r"subject(?P<subject>\d+)_set(?P<set>\d+)_metadata\.json$")
RAW_RE = re.compile(r"subject(?P<subject>\d+)_set(?P<set>\d+)\.csv$")
REFERENCE_RE = re.compile(r"subject(?P<subject>\d+)_set(?P<set>\d+)_window(?P<window>\d+)\.csv$")


@dataclass(frozen=True)
class ConversionConfig:
    config_path: Path
    project_root: Path
    raw_root: Path
    metadata_root: Path
    reference_window_root: Path
    output_dir: Path
    seed: int
    class_mapping: dict[str, dict[str, Any]]
    sensor_mapping: dict[str, str]
    channel_order: list[str]
    csv_column_mapping: dict[str, str]
    target_length: int
    resampling: str
    normalization: str
    expected_num_subjects: int
    expected_num_classes: int
    expected_total_windows: int
    expected_windows_per_subject_class: int
    strict_mode: bool
    make_report: bool
    make_plots: bool
    raw_payload: dict[str, Any]


def load_conversion_config(path: str | Path) -> ConversionConfig:
    config_path = Path(path).resolve()
    project_root = config_path.parent.parent
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    channel_order = list(payload["channel_order"])
    csv_column_mapping = dict(payload["csv_column_mapping"])
    missing = [channel for channel in channel_order if channel not in csv_column_mapping]
    if missing:
        raise ValueError(f"csv_column_mapping is missing channels: {missing}")
    if payload.get("normalization") != "none":
        raise ValueError("target conversion must not apply normalization")
    if payload.get("resampling") != "linear_phase_interpolation":
        raise ValueError("only linear_phase_interpolation is supported for target v1")
    return ConversionConfig(
        config_path=config_path,
        project_root=project_root,
        raw_root=_resolve_project_path(project_root, payload["raw_root"]),
        metadata_root=_resolve_project_path(project_root, payload["metadata_root"]),
        reference_window_root=_resolve_project_path(project_root, payload["reference_window_root"]),
        output_dir=_resolve_project_path(project_root, payload["output_dir"]),
        seed=int(payload.get("seed", 20260617)),
        class_mapping={str(key): dict(value) for key, value in payload["class_mapping"].items()},
        sensor_mapping={str(key): str(value) for key, value in payload["sensor_mapping"].items()},
        channel_order=channel_order,
        csv_column_mapping=csv_column_mapping,
        target_length=int(payload["target_length"]),
        resampling=str(payload["resampling"]),
        normalization=str(payload["normalization"]),
        expected_num_subjects=int(payload["expected_num_subjects"]),
        expected_num_classes=int(payload["expected_num_classes"]),
        expected_total_windows=int(payload["expected_total_windows"]),
        expected_windows_per_subject_class=int(payload["expected_windows_per_subject_class"]),
        strict_mode=bool(payload.get("strict_mode", True)),
        make_report=bool(payload.get("make_report", False)),
        make_plots=bool(payload.get("make_plots", False)),
        raw_payload=payload,
    )


def convert_target_dataset(config: ConversionConfig, overwrite: bool = False) -> dict[str, Any]:
    validate_config_inputs(config)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output_dir already contains files: {config.output_dir}")
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    original_for_plots: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    warnings: list[str] = []

    raw_cache: dict[Path, dict[str, Any]] = {}
    metadata_files = discover_metadata_files(config)
    for metadata_path in metadata_files:
        meta_info = parse_metadata_identity(metadata_path, config)
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_path = raw_path_for_metadata(config, meta_info, payload)
        raw_data = raw_cache.get(raw_path)
        if raw_data is None:
            raw_data = load_raw_csv(raw_path, config)
            raw_cache[raw_path] = raw_data

        windows = payload.get("windows")
        if not isinstance(windows, list):
            raise ValueError(f"{metadata_path} has no windows list")
        for window in windows:
            sample_index = len(rows)
            window_id = int(window["window_id"])
            start = int(window["start_sample"])
            end = int(window["end_sample"])
            raw_window = extract_window(raw_data["values"], start, end, raw_path, window_id)
            timestamp_slice = raw_data["timestamps"][start:end]
            if not np.isfinite(raw_window).all():
                raise ValueError(f"NaN or Inf found in {raw_path} window {window_id}")
            original_length = int(raw_window.shape[0])
            expected_length = int(window.get("num_samples", original_length))
            boundary_status = "ok" if original_length == expected_length else "length_mismatch"
            if boundary_status != "ok":
                warnings.append(
                    f"{metadata_path.name} window{window_id}: boundary length {original_length} != metadata {expected_length}"
                )

            resampled, time_basis = resample_window(
                raw_window,
                target_length=config.target_length,
                timestamps=timestamp_slice,
                return_time_basis=True,
            )
            if not np.isfinite(resampled).all():
                raise ValueError(f"NaN or Inf produced after resampling {raw_path} window {window_id}")

            reference = compare_reference_window(config, meta_info, window_id, raw_data, start, end)
            reference_rows.append(reference)
            if reference["reference_status"] != "ok":
                warnings.append(f"reference mismatch: {reference['sample_id']} {reference['reference_status']}")

            sample_id = f"subject{meta_info['subject_id']}_class{meta_info['class_id']}_set{meta_info['set_id']}_window{window_id}"
            metadata_row = {
                "sample_id": sample_id,
                "subject_id": meta_info["subject_id"],
                "class_id": meta_info["class_id"],
                "class_name": meta_info["class_name"],
                "set_id": meta_info["set_id"],
                "window_index": window_id,
                "source_file": relative_to_project(raw_path, config.project_root),
                "metadata_file": relative_to_project(metadata_path, config.project_root),
                "reference_window_file": reference["reference_window_file"],
                "start_sample": start,
                "end_sample": end,
                "original_length": original_length,
                "metadata_num_samples": expected_length,
                "target_length": config.target_length,
                "resampling_method": config.resampling,
                "resampling_time_basis": time_basis,
                "channel_order": "|".join(config.channel_order),
                "sensor_mapping": json.dumps(config.sensor_mapping, ensure_ascii=False, sort_keys=True),
                "validation_status": "ok" if boundary_status == "ok" else boundary_status,
                "reference_status": reference["reference_status"],
                "normalization": config.normalization,
            }
            rows.append(resampled.astype(np.float32))
            labels.append(int(meta_info["class_id"]))
            subjects.append(int(meta_info["subject_id"]))
            metadata_rows.append(metadata_row)
            if int(meta_info["class_id"]) not in original_for_plots:
                original_for_plots[int(meta_info["class_id"])] = (raw_window.copy(), resampled.copy(), metadata_row)

    X = np.stack(rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    subject_id = np.asarray(subjects, dtype=np.int64)
    duplicate_count = count_duplicate_samples(X)
    summary = build_summary(config, X, y, subject_id, metadata_rows, metadata_files, raw_cache, warnings, duplicate_count)
    write_outputs(config, X, y, subject_id, metadata_rows, reference_rows, summary, original_for_plots)
    return summary


def validate_config_inputs(config: ConversionConfig) -> None:
    if not config.raw_root.exists():
        raise FileNotFoundError(config.raw_root)
    if not config.metadata_root.exists():
        raise FileNotFoundError(config.metadata_root)
    for class_key in config.class_mapping:
        if not re.fullmatch(r"class[0-4]", class_key):
            raise ValueError(f"unexpected class key: {class_key}")
    if len(config.channel_order) != 18:
        raise ValueError(f"channel_order must contain 18 channels, got {len(config.channel_order)}")
    if len(set(config.channel_order)) != len(config.channel_order):
        raise ValueError("channel_order contains duplicates")


def discover_metadata_files(config: ConversionConfig) -> list[Path]:
    files: list[Path] = []
    for class_key in sorted(config.class_mapping):
        class_dir = config.metadata_root / class_key
        files.extend(sorted(class_dir.glob("*_metadata.json")))
    return files


def parse_metadata_identity(metadata_path: Path, config: ConversionConfig) -> dict[str, Any]:
    class_key = metadata_path.parent.name
    if class_key not in config.class_mapping:
        raise ValueError(f"metadata path is outside configured classes: {metadata_path}")
    match = METADATA_RE.fullmatch(metadata_path.name)
    if not match:
        raise ValueError(f"metadata filename does not match subject/set pattern: {metadata_path.name}")
    class_payload = config.class_mapping[class_key]
    return {
        "class_key": class_key,
        "class_id": int(class_payload["id"]),
        "class_name": str(class_payload["name"]),
        "subject_id": int(match.group("subject")),
        "set_id": int(match.group("set")),
    }


def raw_path_for_metadata(config: ConversionConfig, meta_info: dict[str, Any], payload: dict[str, Any]) -> Path:
    expected = config.raw_root / meta_info["class_key"] / f"subject{meta_info['subject_id']}_set{meta_info['set_id']}.csv"
    source_file = payload.get("source_file")
    if source_file:
        source = Path(str(source_file))
        parts = source.parts
        parsed_class = None
        for part in parts:
            if re.fullmatch(r"class[0-4]", part):
                parsed_class = part
                break
        raw_match = RAW_RE.fullmatch(source.name)
        if parsed_class and parsed_class != meta_info["class_key"]:
            raise ValueError(f"metadata source class conflicts with path: {source_file}")
        if raw_match:
            subject = int(raw_match.group("subject"))
            set_id = int(raw_match.group("set"))
            if subject != meta_info["subject_id"] or set_id != meta_info["set_id"]:
                raise ValueError(f"metadata source subject/set conflicts with filename: {source_file}")
    if not expected.exists():
        raise FileNotFoundError(expected)
    return expected


def load_raw_csv(path: Path, config: ConversionConfig) -> dict[str, Any]:
    mapped_columns = [config.csv_column_mapping[channel] for channel in config.channel_order]
    rows: list[list[float]] = []
    timestamps: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = [column for column in mapped_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing mapped channel columns: {missing}")
        for row in reader:
            timestamps.append(row.get("timestamp", ""))
            rows.append([float(row[column]) for column in mapped_columns])
    values = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"NaN or Inf found in raw CSV {path}")
    return {
        "path": path,
        "values": values,
        "timestamps": timestamps,
        "columns": mapped_columns,
    }


def extract_window(values: np.ndarray, start: int, end: int, raw_path: Path, window_id: int) -> np.ndarray:
    if start < 0 or end <= start or end > values.shape[0]:
        raise ValueError(f"invalid boundary for {raw_path} window {window_id}: start={start}, end={end}")
    return values[start:end]


def resample_window(
    window: np.ndarray,
    target_length: int,
    timestamps: list[str] | None = None,
    return_time_basis: bool = False,
) -> np.ndarray | tuple[np.ndarray, str]:
    window = np.asarray(window, dtype=np.float32)
    if window.ndim != 2:
        raise ValueError(f"window must be 2D, got {window.shape}")
    if window.shape[0] == 0:
        raise ValueError("cannot resample empty window")
    if window.shape[0] == 1:
        out = np.repeat(window, target_length, axis=0).astype(np.float32)
        return (out, "single_sample_repeat") if return_time_basis else out
    x_old, basis = normalized_time_axis(timestamps, window.shape[0])
    x_new = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    out = np.empty((target_length, window.shape[1]), dtype=np.float32)
    for channel_index in range(window.shape[1]):
        out[:, channel_index] = np.interp(x_new, x_old, window[:, channel_index]).astype(np.float32)
    return (out, basis) if return_time_basis else out


def normalized_time_axis(timestamps: list[str] | None, length: int) -> tuple[np.ndarray, str]:
    if timestamps and len(timestamps) == length:
        parsed = parse_timestamps(timestamps)
        if parsed is not None:
            diffs = np.diff(parsed)
            if np.all(np.isfinite(parsed)) and np.all(diffs > 0) and parsed[-1] > parsed[0]:
                return ((parsed - parsed[0]) / (parsed[-1] - parsed[0]), "timestamp")
    return (np.linspace(0.0, 1.0, length, dtype=np.float64), "sample_index")


def parse_timestamps(timestamps: list[str]) -> np.ndarray | None:
    parsed: list[float] = []
    for value in timestamps:
        try:
            parsed.append(datetime.fromisoformat(value).timestamp())
        except ValueError:
            return None
    return np.asarray(parsed, dtype=np.float64)


def compare_reference_window(
    config: ConversionConfig,
    meta_info: dict[str, Any],
    window_id: int,
    raw_data: dict[str, Any],
    start: int,
    end: int,
) -> dict[str, Any]:
    sample_id = f"subject{meta_info['subject_id']}_class{meta_info['class_id']}_set{meta_info['set_id']}_window{window_id}"
    reference_path = (
        config.reference_window_root
        / meta_info["class_key"]
        / f"subject{meta_info['subject_id']}_set{meta_info['set_id']}_window{window_id}.csv"
    )
    result = {
        "sample_id": sample_id,
        "reference_window_file": relative_to_project(reference_path, config.project_root),
        "reference_status": "missing",
        "reference_rows": "",
        "raw_rows": end - start,
        "first_timestamp_match": "",
        "last_timestamp_match": "",
        "column_order_match": "",
        "first_values_max_abs_diff": "",
        "last_values_max_abs_diff": "",
    }
    if not reference_path.exists():
        return result
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    result["reference_rows"] = len(rows)
    mapped_columns = [config.csv_column_mapping[channel] for channel in config.channel_order]
    column_order_match = all(column in fieldnames for column in mapped_columns)
    result["column_order_match"] = column_order_match
    if len(rows) != end - start or not column_order_match:
        result["reference_status"] = "row_or_column_mismatch"
        return result
    ref_first_ts = rows[0].get("timestamp", "")
    ref_last_ts = rows[-1].get("timestamp", "")
    raw_first_ts = raw_data["timestamps"][start]
    raw_last_ts = raw_data["timestamps"][end - 1]
    result["first_timestamp_match"] = ref_first_ts == raw_first_ts
    result["last_timestamp_match"] = ref_last_ts == raw_last_ts
    ref_first_values = np.asarray([float(rows[0][column]) for column in mapped_columns], dtype=np.float32)
    ref_last_values = np.asarray([float(rows[-1][column]) for column in mapped_columns], dtype=np.float32)
    raw_first_values = raw_data["values"][start]
    raw_last_values = raw_data["values"][end - 1]
    first_diff = float(np.max(np.abs(ref_first_values - raw_first_values)))
    last_diff = float(np.max(np.abs(ref_last_values - raw_last_values)))
    result["first_values_max_abs_diff"] = first_diff
    result["last_values_max_abs_diff"] = last_diff
    if ref_first_ts == raw_first_ts and ref_last_ts == raw_last_ts and first_diff < 1e-5 and last_diff < 1e-5:
        result["reference_status"] = "ok"
    else:
        result["reference_status"] = "endpoint_mismatch"
    return result


def build_summary(
    config: ConversionConfig,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    metadata_rows: list[dict[str, Any]],
    metadata_files: list[Path],
    raw_cache: dict[Path, dict[str, Any]],
    warnings: list[str],
    duplicate_count: int,
) -> dict[str, Any]:
    original_lengths = np.asarray([int(row["original_length"]) for row in metadata_rows], dtype=np.float64)
    subject_values = sorted(np.unique(subject_id).astype(int).tolist())
    class_values = sorted(np.unique(y).astype(int).tolist())
    subject_counts = {str(subject): int(np.sum(subject_id == subject)) for subject in subject_values}
    class_counts = {str(label): int(np.sum(y == label)) for label in class_values}
    matrix = {
        str(subject): {str(label): int(np.sum((subject_id == subject) & (y == label))) for label in class_values}
        for subject in subject_values
    }
    expectation_mismatches: list[str] = []
    if len(subject_values) != config.expected_num_subjects:
        expectation_mismatches.append(f"subject 수 {len(subject_values)} != expected {config.expected_num_subjects}")
    if len(class_values) != config.expected_num_classes:
        expectation_mismatches.append(f"class 수 {len(class_values)} != expected {config.expected_num_classes}")
    if len(X) != config.expected_total_windows:
        expectation_mismatches.append(f"window 수 {len(X)} != expected {config.expected_total_windows}")
    for subject in subject_values:
        for label in class_values:
            value = matrix[str(subject)][str(label)]
            if value != config.expected_windows_per_subject_class:
                expectation_mismatches.append(
                    f"subject{subject} class{label} windows {value} != expected {config.expected_windows_per_subject_class}"
                )
    return {
        "num_samples": int(len(X)),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "subject_id_shape": list(subject_id.shape),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
        "subject_id_dtype": str(subject_id.dtype),
        "num_subjects": int(len(subject_values)),
        "num_classes": int(len(class_values)),
        "subject_counts": subject_counts,
        "class_counts": class_counts,
        "subject_class_matrix": matrix,
        "original_length_min": int(original_lengths.min()),
        "original_length_mean": float(original_lengths.mean()),
        "original_length_max": int(original_lengths.max()),
        "original_length_std": float(original_lengths.std()),
        "all_resampled_length_512": bool(X.shape[1] == config.target_length),
        "channel_order": config.channel_order,
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
        "class_raw_file_counts": class_raw_file_counts(raw_cache),
        "metadata_json_count": len(metadata_files),
        "boundary_issue_count": sum(1 for row in metadata_rows if row["validation_status"] != "ok"),
        "duplicate_sample_count": duplicate_count,
        "excluded_sample_count": 0,
        "expectation_mismatches": expectation_mismatches,
        "warnings": warnings,
        "git_commit_hash": git_commit_hash(config.project_root),
        "git_status": git_status(config.project_root),
        "manifest_checksum": metadata_manifest_checksum(metadata_rows),
    }


def write_outputs(
    config: ConversionConfig,
    X: np.ndarray,
    y: np.ndarray,
    subject_id: np.ndarray,
    metadata_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    original_for_plots: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> None:
    summary["reference_total"] = len(reference_rows)
    summary["reference_status_counts"] = count_values(reference_rows, "reference_status")
    summary["reference_first_timestamp_match_counts"] = count_values(reference_rows, "first_timestamp_match")
    summary["reference_last_timestamp_match_counts"] = count_values(reference_rows, "last_timestamp_match")
    np.save(config.output_dir / "X.npy", X)
    np.save(config.output_dir / "y.npy", y)
    np.save(config.output_dir / "subject_id.npy", subject_id)
    write_csv(config.output_dir / "metadata.csv", metadata_rows)
    write_csv(config.output_dir / "reference_validation.csv", reference_rows)
    write_dataset_summary(config.output_dir / "dataset_summary.csv", summary)
    write_subject_class_matrix(config.output_dir / "subject_class_matrix.csv", summary)
    write_channel_summary(config.output_dir / "channel_summary.csv", X, config.channel_order)
    (config.output_dir / "conversion_config.yaml").write_text(
        yaml.safe_dump(config.raw_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    checksums = build_checksums(config.output_dir)
    checksums["manifest_checksum"] = summary["manifest_checksum"]
    (config.output_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_conversion_log(config.output_dir / "conversion_log.txt", summary)
    if config.make_report:
        write_report(config.project_root / "docs" / "target_conversion_v1_report.md", config, summary)
    if config.make_plots:
        make_sanity_plots(config, original_for_plots, summary)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_summary(path: Path, summary: dict[str, Any]) -> None:
    rows = [
        {"metric": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
        if key not in {"subject_class_matrix"}
    ]
    write_csv(path, rows)


def write_subject_class_matrix(path: Path, summary: dict[str, Any]) -> None:
    matrix = summary["subject_class_matrix"]
    class_labels = sorted({label for values in matrix.values() for label in values}, key=int)
    rows = []
    for subject, values in sorted(matrix.items(), key=lambda item: int(item[0])):
        row = {"subject_id": subject}
        row.update({f"class{label}": values[label] for label in class_labels})
        row["total"] = sum(values[label] for label in class_labels)
        rows.append(row)
    write_csv(path, rows)


def write_channel_summary(path: Path, X: np.ndarray, channel_order: list[str]) -> None:
    rows = []
    flat = X.reshape(-1, X.shape[-1])
    for index, channel in enumerate(channel_order):
        values = flat[:, index]
        rows.append(
            {
                "channel": channel,
                "min": float(values.min()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "max": float(values.max()),
            }
        )
    write_csv(path, rows)


def write_conversion_log(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "target conversion v1",
        f"num_samples={summary['num_samples']}",
        f"X_shape={summary['X_shape']}",
        f"nan_count={summary['nan_count']}",
        f"inf_count={summary['inf_count']}",
        f"boundary_issue_count={summary['boundary_issue_count']}",
        f"duplicate_sample_count={summary['duplicate_sample_count']}",
        f"excluded_sample_count={summary['excluded_sample_count']}",
        f"git_commit_hash={summary['git_commit_hash']}",
        f"git_status={summary['git_status']}",
        "warnings:",
    ]
    lines.extend(f"- {warning}" for warning in summary.get("warnings", []))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, config: ConversionConfig, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Target Conversion v1 Report",
        "",
        "## 요약",
        "",
        f"- output_dir: `{relative_to_project(config.output_dir, config.project_root)}`",
        f"- X shape: `{tuple(summary['X_shape'])}`, dtype `{summary['X_dtype']}`",
        f"- y shape: `{tuple(summary['y_shape'])}`, dtype `{summary['y_dtype']}`",
        f"- subject_id shape: `{tuple(summary['subject_id_shape'])}`, dtype `{summary['subject_id_dtype']}`",
        f"- 전체 window 수: {summary['num_samples']}",
        f"- subject 수: {summary['num_subjects']}",
        f"- class 수: {summary['num_classes']}",
        f"- NaN count: {summary['nan_count']}",
        f"- Inf count: {summary['inf_count']}",
        f"- 제외 sample 수: {summary['excluded_sample_count']}",
        "",
        "## 공식 Source 선택",
        "",
        "`datasets/raw/labeled`의 raw continuous CSV를 공식 source로 사용하고, `datasets/manually_labeled/class0~4`의 metadata JSON boundary를 적용했다. 기존 `*_window*.csv`는 공식 입력이 아니라 raw slicing 검증용 reference로만 사용했다.",
        "",
        "## 센서 매핑",
        "",
        "| sensor | location |",
        "|---|---|",
    ]
    for sensor, location in config.sensor_mapping.items():
        lines.append(f"| `{sensor}` | {location} |")
    lines.extend(
        [
            "",
            "## Channel Order",
            "",
            "`" + ", ".join(config.channel_order) + "`",
            "",
            "## Resampling",
            "",
            "각 manually labeled variable-length window를 phase-normalized linear interpolation으로 512 time steps로 변환했다. timestamp가 엄격히 증가하는 window는 timestamp 기준 상대 시간축을 사용하고, 그렇지 않으면 sample index 기준 시간축을 사용한다. Padding, 단순 truncate, z-score, StandardScaler, clipping, augmentation은 conversion 단계에서 적용하지 않았다.",
            "",
            "## LOSO Leakage 방지",
            "",
            "conversion 산출물에는 normalization을 적용하지 않았다. 향후 LOSO 학습에서는 held-out subject를 제외한 training subject에 대해서만 scaler를 fit해야 하며, held-out subject는 scaler, augmentation, validation tuning, SSL pretraining에 사용하지 않는다.",
            "",
            "## Subject-Class Matrix",
            "",
            "| subject | class0 | class1 | class2 | class3 | class4 | total |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    matrix = summary["subject_class_matrix"]
    for subject, values in sorted(matrix.items(), key=lambda item: int(item[0])):
        total = sum(values.get(str(label), 0) for label in range(5))
        lines.append(
            f"| {subject} | {values.get('0', 0)} | {values.get('1', 0)} | {values.get('2', 0)} | {values.get('3', 0)} | {values.get('4', 0)} | {total} |"
        )
    lines.extend(
        [
            "",
            "## Original Length",
            "",
            f"- min: {summary['original_length_min']}",
            f"- mean: {summary['original_length_mean']:.2f}",
            f"- max: {summary['original_length_max']}",
            f"- std: {summary['original_length_std']:.2f}",
            "",
            "## Reference Window 검증",
            "",
            "reference window CSV는 raw+metadata 추출 window와 row count, column presence, first/last timestamp, first/last 18채널 값을 비교했다. 상세 결과는 `reference_validation.csv`에 저장했다.",
            "",
            f"- reference row 수: {summary.get('reference_total', 0)}",
            f"- reference status: `{json.dumps(summary.get('reference_status_counts', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- first timestamp match: `{json.dumps(summary.get('reference_first_timestamp_match_counts', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- last timestamp match: `{json.dumps(summary.get('reference_last_timestamp_match_counts', {}), ensure_ascii=False, sort_keys=True)}`",
            "",
            "## 남은 불확실성",
            "",
            "- sensor 위치 매핑은 사용자 승인값을 config에 기록했으며, 원본 CSV 자체에는 위치명이 없다.",
            "- 512 변환은 interpolation 정책의 결과이므로 원본 sample 수가 512였던 데이터가 아니다.",
            "- git initial commit은 사용자 identity 미설정으로 실패했으며, conversion에는 현재 git 상태를 함께 기록했다.",
            "",
            "## 다음 단계",
            "",
            "다음 단계에서는 shared encoder 계열의 aggregation head를 줄이는 model capacity correction 계획을 승인한 뒤, real-data smoke training은 full LOSO training과 분리해 아주 작은 forward/overfit 점검으로만 설계해야 한다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def make_sanity_plots(
    config: ConversionConfig,
    original_for_plots: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    summary: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = config.project_root / "results" / "data_validation" / "target_v1"
    output.mkdir(parents=True, exist_ok=True)
    s1_gz_index = config.channel_order.index("s1_gz")
    for class_id, (original, resampled, metadata_row) in sorted(original_for_plots.items()):
        plt.figure(figsize=(8, 4))
        plt.plot(np.linspace(0, 1, original.shape[0]), original[:, s1_gz_index], label="original")
        plt.plot(np.linspace(0, 1, resampled.shape[0]), resampled[:, s1_gz_index], label="resampled")
        plt.title(f"class{class_id} s1_gz original vs resampled")
        plt.xlabel("normalized phase")
        plt.ylabel("s1_gz")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"class{class_id}_s1_gz_original_vs_resampled.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        offset = 0.0
        for channel_index, channel in enumerate(config.channel_order):
            values = resampled[:, channel_index]
            scale = float(values.std()) or 1.0
            plt.plot((values - values.mean()) / scale + offset, linewidth=0.6)
            offset += 4.0
        plt.title(f"class{class_id} 18-channel resampled overview")
        plt.xlabel("time index")
        plt.ylabel("normalized channels with offsets")
        plt.tight_layout()
        plt.savefig(output / f"class{class_id}_18ch_resampled_overview.png")
        plt.close()

    matrix = summary["subject_class_matrix"]
    subjects = sorted(matrix, key=int)
    classes = [str(index) for index in range(5)]
    values = np.asarray([[matrix[subject].get(label, 0) for label in classes] for subject in subjects])
    plt.figure(figsize=(6, 4))
    plt.imshow(values, aspect="auto")
    plt.xticks(range(len(classes)), [f"class{label}" for label in classes])
    plt.yticks(range(len(subjects)), [f"subject{subject}" for subject in subjects])
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            plt.text(col, row, str(values[row, col]), ha="center", va="center")
    plt.title("subject-class matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output / "subject_class_matrix_heatmap.png")
    plt.close()


def build_checksums(output_dir: Path) -> dict[str, Any]:
    files = [
        "X.npy",
        "y.npy",
        "subject_id.npy",
        "metadata.csv",
        "reference_validation.csv",
        "dataset_summary.csv",
        "subject_class_matrix.csv",
        "channel_summary.csv",
        "conversion_config.yaml",
    ]
    return {"files": {name: sha256_file(output_dir / name) for name in files if (output_dir / name).exists()}}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_manifest_checksum(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def count_duplicate_samples(X: np.ndarray) -> int:
    digests = [hashlib.sha256(np.ascontiguousarray(sample).tobytes()).hexdigest() for sample in X]
    return len(digests) - len(set(digests))


def class_raw_file_counts(raw_cache: dict[Path, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in raw_cache:
        class_key = path.parent.name
        counts[class_key] = counts.get(class_key, 0) + 1
    return dict(sorted(counts.items()))


def git_commit_hash(project_root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return "git-commit-unavailable"
    return result.stdout.strip()


def git_status(project_root: Path) -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=project_root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return "git-status-unavailable"
    return result.stdout.strip() or "clean"


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare target squat dataset v1 from raw CSV and metadata windows.")
    parser.add_argument("--config", type=Path, default=Path("configs/target_conversion_v1.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_conversion_config(args.config)
    summary = convert_target_dataset(config, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
