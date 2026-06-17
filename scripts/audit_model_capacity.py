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
    parser = argparse.ArgumentParser(description="Audit parameter budgets for model capacity v2 variants.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "model_capacity_v2.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    set_global_seed(int(config.get("seed", 20260617)))

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
        rows.append(
            {
                "model_name": model_name,
                "total_params": groups["total_params"],
                "encoder_params": groups["encoder_params"],
                "aggregation_params": groups["aggregation_params"],
                "head_params": groups["head_params"],
                "input_shape": f"({batch}, {window_length}, {channels})",
                "output_shape": str(tuple(logits.shape)),
                "notes": str(entry.get("notes", "")),
            }
        )

    output_dir = PROJECT_ROOT / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_capacity_table.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "total_params",
                "encoder_params",
                "aggregation_params",
                "head_params",
                "input_shape",
                "output_shape",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['model_name']}: total={row['total_params']} encoder={row['encoder_params']} "
            f"aggregation={row['aggregation_params']} head={row['head_params']} output={row['output_shape']}"
        )
    print(f"saved: {output_path}")
    return 0


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


if __name__ == "__main__":
    raise SystemExit(main())
