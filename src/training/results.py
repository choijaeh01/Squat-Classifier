from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    return save_json(metrics, path)


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def git_commit_hash(repo_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return "git-unavailable"
    if result.returncode != 0:
        return "not-a-git-repository"
    return result.stdout.strip()


def result_payload(
    *,
    metrics: dict[str, Any],
    parameter_counts: dict[str, int],
    training_history: dict[str, Any],
    config_snapshot: dict[str, Any],
    git_commit: str,
    data_manifest_checksum: str,
    fold_confusion_matrices: dict[str, list[list[int]]] | None = None,
) -> dict[str, Any]:
    return {
        "metrics": metrics,
        "model_parameter_counts": parameter_counts,
        "training_history": training_history,
        "config_snapshot": config_snapshot,
        "git_commit_hash": git_commit,
        "data_manifest_checksum": data_manifest_checksum,
        "fold_confusion_matrices": fold_confusion_matrices or {},
        "aggregated_confusion_matrix": metrics.get("confusion_matrix", []),
    }
