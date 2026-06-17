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

from models.classical_features import is_classical_model, sklearn_available
from models.registry import build_model, count_parameter_groups
from utils.reproducibility import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit literature temporal baseline parameter counts.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = _load_yaml(args.config)
    rows = audit_literature_capacity(config)
    output_dir = PROJECT_ROOT / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_capacity_table.csv"
    _write_csv(output_path, rows)
    report_path = PROJECT_ROOT / "docs" / f"{config['experiment_name']}_report.md"
    report_path.write_text(_capacity_report(rows, config), encoding="utf-8")
    for row in rows:
        print(f"{row['model_name']}: total={row['total_params']} output={row['output_shape']} notes={row['notes']}")
    print(f"saved: {output_path}")
    print(f"report: {report_path}")
    return 0


def audit_literature_capacity(config: dict[str, Any]) -> list[dict[str, Any]]:
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
        if is_classical_model(model_name):
            rows.append(
                {
                    "model_name": model_name,
                    "total_params": "",
                    "encoder_params": "",
                    "aggregation_params": "",
                    "head_params": "",
                    "other_params": "",
                    "input_shape": f"({batch}, {window_length}, {channels})",
                    "output_shape": "not_applicable",
                    "notes": "classical baseline; parameter count not applicable; training requires scikit-learn",
                }
            )
            continue
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
                "other_params": groups["other_params"],
                "input_shape": f"({batch}, {window_length}, {channels})",
                "output_shape": str(tuple(logits.shape)),
                "notes": str(entry.get("notes", "")),
            }
        )
    for model_name in config.get("locked_reference_models", []):
        model = build_model(str(model_name), input_length=window_length, input_channels=channels, num_classes=num_classes)
        groups = count_parameter_groups(model)
        rows.append(
            {
                "model_name": str(model_name),
                "total_params": groups["total_params"],
                "encoder_params": groups["encoder_params"],
                "aggregation_params": groups["aggregation_params"],
                "head_params": groups["head_params"],
                "other_params": groups["other_params"],
                "input_shape": f"({batch}, {window_length}, {channels})",
                "output_shape": "reference_locked_matrix",
                "notes": "locked full matrix reference model",
            }
        )
    return rows


def _capacity_report(rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    experiment_name = str(config.get("experiment_name", "literature_baseline_capacity"))
    lines = [
        f"# {experiment_name} Report",
        "",
        "## 범위",
        "",
        "이 보고서는 literature temporal baseline extension의 parameter count audit이다. 학습은 수행하지 않았다.",
        "",
        "## Parameter Count",
        "",
        "| model | total params | encoder | aggregation | head | notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['total_params']} | {row['encoder_params']} | "
            f"{row['aggregation_params']} | {row['head_params']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 주의점",
            "",
            "- CNN-LSTM/GRU 계열은 IMU 문헌에서 널리 쓰이는 family representative baseline이며 특정 외부 논문의 exact reproduction이 아니다.",
            "- lee_style_cnn_lstm_2d_v1은 512-step 입력을 모델 내부에서 deterministic 40-step representation으로 downsample한 뒤 2D CNN과 LSTM을 적용하는 adapted clean-room baseline이다.",
            "- Lee-style 모델은 Lee et al.의 exact reproduction이 아니며, 현재 센서 수, channel 수, label, LOSO protocol에 맞춘 reviewer-facing baseline이다.",
            "- rescnn_bigru_attention_lite_v1은 이전 졸업논문 결과 재사용이 아니라 동일 protocol에서 새로 평가할 clean-room reference baseline이다.",
            "- classical baseline은 scikit-learn 설치 여부에 따라 실행 또는 skipped 처리된다.",
            "- 이 audit은 capacity 확인용이며 성능 해석을 포함하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
