#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.registry import build_model, count_parameter_groups
from utils.reproducibility import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit v3 component ablation parameter counts.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_capacity_config(args.config)
    rows = audit_v3_ablation_capacity(config)
    output_dir = PROJECT_ROOT / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_capacity_table.csv"
    _write_csv(output_path, rows)
    report_path = PROJECT_ROOT / "docs" / f"{config['experiment_name']}_report.md"
    report_path.write_text(_capacity_report(rows), encoding="utf-8")
    for row in rows:
        print(
            f"{row['model_name']}: total={row['total_params']} encoder={row['encoder_params']} "
            f"identity={row['identity_params']} attention={row['attention_params']} residual={row['residual_params']} "
            f"output={row['output_shape']}"
        )
    print(f"saved: {output_path}")
    print(f"report: {report_path}")
    return 0


def load_capacity_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def audit_v3_ablation_capacity(config: dict[str, Any]) -> list[dict[str, Any]]:
    set_global_seed(int(config.get("seed", 20260618)))
    input_cfg = config["input_shape"]
    batch = int(input_cfg["batch"])
    window_length = int(input_cfg["window_length"])
    channels = int(input_cfg["channels"])
    num_classes = int(config["num_classes"])
    x = torch.randn(batch, window_length, channels)
    rows: list[dict[str, Any]] = []
    for entry in config["models"]:
        model_name = str(entry["name"])
        model = build_model(model_name, input_length=window_length, input_channels=channels, num_classes=num_classes)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        groups = count_parameter_groups(model)
        component_groups = count_v3_component_groups(model)
        rows.append(
            {
                "model_name": model_name,
                "total_params": groups["total_params"],
                "encoder_params": groups["encoder_params"],
                "aggregation_params": groups["aggregation_params"],
                "identity_params": component_groups["identity_params"],
                "attention_params": component_groups["attention_params"],
                "residual_params": component_groups["residual_params"],
                "head_params": groups["head_params"],
                "other_params": groups["total_params"]
                - groups["encoder_params"]
                - component_groups["identity_params"]
                - component_groups["attention_params"]
                - component_groups["residual_params"]
                - groups["head_params"],
                "input_shape": f"({batch}, {window_length}, {channels})",
                "output_shape": str(tuple(logits.shape)),
                "removed_component": str(entry.get("removed_component", "")),
                "notes": str(entry.get("notes", "")),
            }
        )
    return rows


def count_v3_component_groups(model: torch.nn.Module) -> dict[str, int]:
    identity_modules = []
    attention_modules = []
    residual_modules = []
    aggregation = getattr(model, "aggregation", None)
    if isinstance(aggregation, torch.nn.ModuleDict):
        for name in ("channel_embedding", "sensor_embedding", "modality_embedding", "axis_embedding"):
            if name in aggregation:
                identity_modules.append(aggregation[name])
        for name in ("attention_pool", "acc_attention_pool", "gyro_attention_pool"):
            if name in aggregation:
                attention_modules.append(aggregation[name])
        if "residual_projection" in aggregation:
            residual_modules.append(aggregation["residual_projection"])
    if hasattr(model, "residual_projection"):
        residual_modules.append(getattr(model, "residual_projection"))
    return {
        "identity_params": _count_unique_module_parameters(identity_modules),
        "attention_params": _count_unique_module_parameters(attention_modules),
        "residual_params": _count_unique_module_parameters(residual_modules),
    }


def _count_unique_module_parameters(modules: list[torch.nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
    return total


def _capacity_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# v3_component_ablation_capacity_v1 Report",
        "",
        "## 범위",
        "",
        "이 보고서는 `channel_shared_posres_attention_v3`와 component ablation 모델의 parameter count를 비교한다. 학습은 수행하지 않았다.",
        "",
        "## Parameter Count",
        "",
        "| model | total | encoder | identity | attention | residual | head | removed component |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['total_params']} | {row['encoder_params']} | {row['identity_params']} | "
            f"{row['attention_params']} | {row['residual_params']} | {row['head_params']} | {row['removed_component']} |"
        )
    lines.extend(
        [
            "",
            "## 해석 주의점",
            "",
            "- 이 표는 capacity 확인용이며 성능 해석을 포함하지 않는다.",
            "- `no_identity`는 token identity embedding만 제거하며, residual branch는 여전히 channel order 기반 summary를 포함한다.",
            "- `residual_only_mlp`는 v3 residual branch와 동일한 mean/std/min/max channel summary만 사용한다.",
            "- 결과가 낮거나 높아도 model width, learning rate, split, preprocessing을 변경하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
