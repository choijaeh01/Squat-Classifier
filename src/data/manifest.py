from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CLASS_NAMES = {
    0: "Correct",
    1: "Knee Valgus",
    2: "Butt Wink",
    3: "Excessive Lean",
    4: "Partial Squat",
}


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    path: str
    subject_id: str
    label: int
    source: str = "target"
    window_start: int | None = None
    window_length: int = 512
    channels: int = 18
    file_checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "subject_id": self.subject_id,
            "label": int(self.label),
            "source": self.source,
            "window_start": self.window_start,
            "window_length": int(self.window_length),
            "channels": int(self.channels),
            "file_checksum": self.file_checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManifestRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            path=str(payload["path"]),
            subject_id=str(payload["subject_id"]),
            label=int(payload["label"]),
            source=str(payload.get("source", "target")),
            window_start=payload.get("window_start"),
            window_length=int(payload.get("window_length", 512)),
            channels=int(payload.get("channels", 18)),
            file_checksum=payload.get("file_checksum"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class TargetDataManifest:
    dataset_name: str
    window_length: int = 512
    channels: int = 18
    class_names: dict[int, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_NAMES))
    records: list[ManifestRecord] = field(default_factory=list)
    version: str = "1"
    notes: str = ""

    def to_dict(self, include_checksum: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "dataset_name": self.dataset_name,
            "window_length": int(self.window_length),
            "channels": int(self.channels),
            "class_names": {str(k): v for k, v in sorted(self.class_names.items())},
            "records": [record.to_dict() for record in self.records],
            "notes": self.notes,
        }
        if include_checksum:
            payload["manifest_checksum"] = self.checksum()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TargetDataManifest":
        class_names = {int(k): str(v) for k, v in payload.get("class_names", DEFAULT_CLASS_NAMES).items()}
        return cls(
            dataset_name=str(payload["dataset_name"]),
            window_length=int(payload.get("window_length", 512)),
            channels=int(payload.get("channels", 18)),
            class_names=class_names,
            records=[ManifestRecord.from_dict(item) for item in payload.get("records", [])],
            version=str(payload.get("version", "1")),
            notes=str(payload.get("notes", "")),
        )

    def checksum(self) -> str:
        canonical = json.dumps(self.to_dict(include_checksum=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(include_checksum=True), indent=2, sort_keys=True) + "\n")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "TargetDataManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> list[str]:
        issues: list[str] = []
        expected_labels = set(DEFAULT_CLASS_NAMES)
        declared_labels = set(self.class_names)
        if self.window_length != 512:
            issues.append(f"window_length must be 512, got {self.window_length}")
        if self.channels != 18:
            issues.append(f"channels must be 18, got {self.channels}")
        if declared_labels != expected_labels:
            issues.append(f"class_names must define labels {sorted(expected_labels)}, got {sorted(declared_labels)}")
        for index, record in enumerate(self.records):
            if record.label not in expected_labels:
                issues.append(f"records[{index}] label {record.label} is outside 0..4")
            if not record.subject_id:
                issues.append(f"records[{index}] has empty subject_id")
            if record.window_length != self.window_length:
                issues.append(f"records[{index}] window_length {record.window_length} differs from manifest")
            if record.channels != self.channels:
                issues.append(f"records[{index}] channels {record.channels} differs from manifest")
            if not Path(record.path).exists():
                issues.append(f"records[{index}] path does not exist: {record.path}")
        return issues


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
