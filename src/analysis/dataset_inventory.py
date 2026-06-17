from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHANNEL_COLUMNS = [
    "s0_ax",
    "s0_ay",
    "s0_az",
    "s0_gx",
    "s0_gy",
    "s0_gz",
    "s1_ax",
    "s1_ay",
    "s1_az",
    "s1_gx",
    "s1_gy",
    "s1_gz",
    "s2_ax",
    "s2_ay",
    "s2_az",
    "s2_gx",
    "s2_gy",
    "s2_gz",
]


@dataclass
class InventoryRow:
    path: str
    extension: str
    size_bytes: int
    inferred_role: str
    shape: str
    columns_or_keys: str
    has_subject_id: bool
    has_label: bool
    has_channel_names: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "inferred_role": self.inferred_role,
            "shape": self.shape,
            "columns_or_keys": self.columns_or_keys,
            "has_subject_id": self.has_subject_id,
            "has_label": self.has_label,
            "has_channel_names": self.has_channel_names,
            "notes": self.notes,
        }


def build_inventory(datasets_root: Path) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for path in sorted(datasets_root.rglob("*")):
        if not path.is_file():
            continue
        rows.append(inspect_file(datasets_root, path))
    return rows


def inspect_file(datasets_root: Path, path: Path) -> InventoryRow:
    rel = path.relative_to(datasets_root.parent).as_posix()
    extension = path.suffix.lower() or path.name
    size_bytes = path.stat().st_size
    has_subject = bool(re.search(r"subject\d+", rel))
    has_label = bool(re.search(r"/class[0-4]/", rel))
    has_set = bool(re.search(r"_set\d+", rel))
    has_window = bool(re.search(r"_window\d+", rel))
    shape = ""
    columns_or_keys = ""
    has_channel_names = False
    notes: list[str] = []

    if extension == ".csv":
        csv_info = inspect_csv(path)
        shape = csv_info["shape"]
        columns = csv_info["columns"]
        columns_or_keys = ",".join(columns[:24])
        has_channel_names = all(column in columns for column in CHANNEL_COLUMNS)
        role = infer_csv_role(rel)
        notes.extend(csv_info["notes"])
        if has_set:
            notes.append("set/repetition id in filename")
        if has_window:
            notes.append("window id in filename")
        if has_channel_names:
            notes.append("18 IMU channel columns present")
    elif extension == ".json":
        json_info = inspect_json(path)
        shape = json_info["shape"]
        columns_or_keys = ",".join(json_info["keys"])
        has_channel_names = False
        role = "csv metadata"
        notes.extend(json_info["notes"])
        if has_set:
            notes.append("set/repetition id in filename")
        if "windows" in json_info["keys"]:
            notes.append("window boundary metadata")
    elif extension == ".png":
        role = "result file"
        shape = "image"
        notes.append("plot/visualization sidecar; not training input")
        if has_set:
            notes.append("set/repetition id in filename")
    elif extension == ".ds_store" or path.name == ".DS_Store":
        role = "unknown"
        notes.append("macOS folder metadata; ignore for experiments")
    elif extension in {".npy", ".npz"}:
        role = "numpy array"
        notes.append("numpy artifact; shape inspection not needed because none expected in current dataset")
    else:
        role = "unknown"

    return InventoryRow(
        path=rel,
        extension=extension,
        size_bytes=size_bytes,
        inferred_role=role,
        shape=shape,
        columns_or_keys=columns_or_keys,
        has_subject_id=has_subject,
        has_label=has_label,
        has_channel_names=has_channel_names,
        notes="; ".join(notes),
    )


def inspect_csv(path: Path) -> dict[str, Any]:
    notes: list[str] = []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {"shape": "0x0", "columns": [], "notes": ["empty csv"]}
        row_count = sum(1 for _ in reader)
    ncols = len(header)
    rows_excluding_header = row_count
    if rows_excluding_header != 512:
        notes.append(f"row count is {rows_excluding_header}, not 512")
    imu_count = sum(1 for column in header if re.match(r"s[0-2]_[ag][xyz]$", column))
    if imu_count == 18:
        notes.append("18 sensor-value columns")
    if ncols == 24:
        notes.append("24 total columns including timestamp/seq/millis/q0-q2")
    return {
        "shape": f"{rows_excluding_header}x{ncols}",
        "columns": header,
        "notes": notes,
    }


def inspect_json(path: Path) -> dict[str, Any]:
    notes: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return {"shape": "invalid-json", "keys": [], "notes": [f"json decode error: {exc}"]}
    if isinstance(payload, dict):
        keys = sorted(str(key) for key in payload)
        windows = payload.get("windows")
        if isinstance(windows, list):
            notes.append(f"num_windows={len(windows)}")
            sample_counts = [item.get("num_samples") for item in windows if isinstance(item, dict)]
            numeric_counts = [int(item) for item in sample_counts if isinstance(item, int)]
            if numeric_counts:
                notes.append(f"window_samples_min={min(numeric_counts)}")
                notes.append(f"window_samples_max={max(numeric_counts)}")
        if "source_file" in payload:
            notes.append(f"source_file={payload['source_file']}")
        if "sampling_rate" in payload:
            notes.append(f"sampling_rate={payload['sampling_rate']}")
        shape = f"dict[{len(keys)}]"
    else:
        keys = []
        shape = type(payload).__name__
    return {"shape": shape, "keys": keys, "notes": notes}


def infer_csv_role(rel: str) -> str:
    if "/raw/labeled/" in rel:
        return "raw continuous session"
    if "/raw/unlabeled/" in rel:
        return "raw continuous session"
    if "/manually_labeled/ssl/" in rel:
        return "manually labeled window"
    if "/manually_labeled/class" in rel and "_window" in rel:
        return "manually labeled window"
    return "unknown"


def write_inventory(rows: list[InventoryRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "extension",
        "size_bytes",
        "inferred_role",
        "shape",
        "columns_or_keys",
        "has_subject_id",
        "has_label",
        "has_channel_names",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a read-only inventory of the local datasets folder.")
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("squat_imu_experiments/data/manifests/data_inventory.csv"),
    )
    args = parser.parse_args()
    rows = build_inventory(args.datasets_root)
    write_inventory(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
