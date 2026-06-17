from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from models.registry import build_model, count_parameter_groups
from utils.reproducibility import set_global_seed


DEFAULT_NETRON_MODELS = [
    "all_channel_conv1d_v1",
    "all_channel_conv1d_small",
    "cnn2d_baseline_v1",
    "channel_shared_meanpool_v2",
    "channel_shared_attentionpool_v2",
    "channel_shared_posres_attention_v3",
    "modality_shared_sensorattn_v3",
]


def export_torchscript_models(
    output_dir: Path,
    model_names: list[str] | None = None,
    batch_size: int = 1,
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
    seed: int = 20260618,
) -> list[dict[str, Any]]:
    """Export initialized registry models as TorchScript files for Netron inspection.

    These exports are architecture visualizations only. They do not load trained
    checkpoints and must not be interpreted as trained model artifacts.
    """

    output_dir = Path(output_dir)
    model_dir = output_dir / "torchscript"
    model_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    models = list(model_names or DEFAULT_NETRON_MODELS)
    example_input = torch.randn(batch_size, input_length, input_channels)

    rows: list[dict[str, Any]] = []
    for model_name in models:
        model = build_model(
            model_name,
            input_length=input_length,
            input_channels=input_channels,
            num_classes=num_classes,
        )
        model.eval()
        with torch.no_grad():
            logits = model(example_input)
            traced = torch.jit.trace(model, example_input, strict=False)
            traced = torch.jit.freeze(traced.eval())

        artifact_path = model_dir / f"{model_name}.torchscript.pt"
        torch.jit.save(traced, str(artifact_path))
        groups = count_parameter_groups(model)
        rows.append(
            {
                "model_name": model_name,
                "artifact_path": str(artifact_path),
                "file_format": "TorchScript",
                "input_shape": str(tuple(example_input.shape)),
                "output_shape": str(tuple(logits.shape)),
                "total_params": groups["total_params"],
                "encoder_params": groups["encoder_params"],
                "aggregation_params": groups["aggregation_params"],
                "head_params": groups["head_params"],
                "netron_command": f"netron {artifact_path}",
                "notes": "Initialized architecture export for Netron; no trained weights loaded.",
            }
        )

    _write_index(output_dir / "netron_model_index.csv", rows)
    _write_manifest(
        output_dir / "netron_export_manifest.json",
        rows=rows,
        batch_size=batch_size,
        input_length=input_length,
        input_channels=input_channels,
        num_classes=num_classes,
        seed=seed,
    )
    _write_readme(output_dir / "README.md", rows)
    return rows


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model_name",
        "artifact_path",
        "file_format",
        "input_shape",
        "output_shape",
        "total_params",
        "encoder_params",
        "aggregation_params",
        "head_params",
        "netron_command",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    batch_size: int,
    input_length: int,
    input_channels: int,
    num_classes: int,
    seed: int,
) -> None:
    payload = {
        "purpose": "Netron architecture visualization",
        "trained_weights_loaded": False,
        "batch_size": batch_size,
        "input_length": input_length,
        "input_channels": input_channels,
        "num_classes": num_classes,
        "seed": seed,
        "models": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_readme(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Netron Model Exports",
        "",
        "이 폴더는 Netron 구조 도식화용 TorchScript export를 보관한다.",
        "학습된 checkpoint가 아니라 registry 모델을 초기화한 구조 확인용 산출물이다.",
        "",
        "Netron 실행 예:",
        "",
        "```bash",
        "pip install netron",
        "netron torchscript/all_channel_conv1d_v1.torchscript.pt",
        "```",
        "",
        "| model | file |",
        "|---|---|",
    ]
    for row in rows:
        artifact = Path(str(row["artifact_path"]))
        lines.append(f"| {row['model_name']} | `{artifact.name}` |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
