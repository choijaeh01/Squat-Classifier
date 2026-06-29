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
from training.controlled_comparison_runner import _parameter_rows
from utils.reproducibility import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit controlled feature extractor capacity.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = _load_yaml(args.config)
    rows = audit_controlled_capacity(config)
    output_dir = PROJECT_ROOT / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_capacity_table.csv"
    _write_csv(output_path, rows)
    report_path = PROJECT_ROOT / "docs" / f"{config['experiment_name']}_report.md"
    report_path.write_text(_capacity_report(rows, config), encoding="utf-8")
    for row in rows:
        print(
            f"{row['model_name']}: group={row.get('model_group', '')} "
            f"total={row.get('total_params', '')} head={row.get('head_params', '')} "
            f"output={row.get('output_shape', '')}"
        )
    print(f"saved: {output_path}")
    print(f"report: {report_path}")
    return 0


def audit_controlled_capacity(config: dict[str, Any]) -> list[dict[str, Any]]:
    set_global_seed(int(config.get("seed", 20260629)))
    device = torch.device("cpu")
    rows = _parameter_rows(config, device)
    rows.extend(_locked_reference_rows(config, device))
    return rows


def _locked_reference_rows(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    input_cfg = config["input_shape"]
    batch = int(input_cfg["batch"])
    window_length = int(input_cfg["window_length"])
    channels = int(input_cfg["channels"])
    num_classes = int(config["num_classes"])
    x = torch.zeros(batch, window_length, channels, dtype=torch.float32, device=device)
    rows: list[dict[str, Any]] = []
    for model_name in config.get("locked_reference_models", []):
        model = build_model(str(model_name), input_length=window_length, input_channels=channels, num_classes=num_classes).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        groups = count_parameter_groups(model)
        rows.append(
            {
                "model_name": str(model_name),
                "model_group": "locked_reference",
                "total_params": groups["total_params"],
                "encoder_params": groups["encoder_params"],
                "aggregation_params": groups["aggregation_params"],
                "head_params": groups["head_params"],
                "other_params": groups["other_params"],
                "extractor_params": "",
                "projection_params": "",
                "common_head_params": "not_applicable",
                "residual_branch_params": "",
                "identity_embedding_params": "",
                "representation_dim": "",
                "common_head_signature": "not_applicable",
                "input_shape": f"({batch}, {window_length}, {channels})",
                "output_shape": str(tuple(logits.shape)),
                "notes": "read-only locked matrix reference architecture",
            }
        )
    return rows


def _capacity_report(rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    controlled = [row for row in rows if row.get("model_group") == "controlled_neural"]
    head_counts = sorted({str(row.get("common_head_params")) for row in controlled})
    signatures = sorted({str(row.get("common_head_signature")) for row in controlled})
    lines = [
        f"# {config['experiment_name']} Report",
        "",
        "## 목적",
        "",
        "이 보고서는 controlled feature extractor comparison v1의 capacity audit이다. 학습, 역전파, optimizer step은 수행하지 않았다.",
        "",
        "## 공통 Head 검증",
        "",
        f"- controlled neural 모델 수: {len(controlled)}",
        f"- 공통 head parameter count 후보: {', '.join(head_counts)}",
        f"- 공통 head 구조 후보: {', '.join(_escape_signature(item) for item in signatures)}",
        "- 모든 controlled neural 모델은 64차원 representation을 같은 MLP head에 입력하도록 설계했다.",
        "",
        "## Parameter Count",
        "",
        "| model | group | total params | extractor params | common head params | representation dim | notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model_name} | {model_group} | {total_params} | {extractor_params} | {common_head_params} | "
            "{representation_dim} | {notes} |".format(**{key: _md_cell(value) for key, value in row.items()})
        )
    lines.extend(
        [
            "",
            "## 해석 범위",
            "",
            "- 이 단계는 extractor 구조만 통제 비교하기 위한 사전 점검이다.",
            "- common head가 같기 때문에 controlled neural 내부 비교에서는 classifier head 차이를 주요 confounder에서 제외할 수 있다.",
            "- classical baseline은 train fold 내부 IMU signal-derived feature만 사용하며 parameter count는 신경망 parameter count와 직접 비교하지 않는다.",
            "- XGBoost는 설치되어 있지 않으면 training 단계에서 skipped 처리한다.",
            "- locked reference 모델은 기존 결과 디렉터리를 수정하지 않고 구조 비교 참고용으로만 포함했다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _escape_signature(value: str) -> str:
    return value.replace("|", " &#124; ")


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "&#124;")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
