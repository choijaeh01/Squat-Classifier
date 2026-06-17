#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.netron_exports import DEFAULT_NETRON_MODELS, export_torchscript_models


def main() -> int:
    parser = argparse.ArgumentParser(description="Export registry models as TorchScript files for Netron visualization.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "netron_model_export_v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config) if args.config.exists() else {}
    input_shape = config.get("input_shape", {})
    output_dir = args.output_dir or PROJECT_ROOT / str(config.get("output_dir", "results/model_diagrams/netron_exports_v1"))
    model_names = args.models or list(config.get("models", DEFAULT_NETRON_MODELS))
    rows = export_torchscript_models(
        output_dir=output_dir,
        model_names=model_names,
        batch_size=int(input_shape.get("batch", 1)),
        input_length=int(input_shape.get("window_length", 512)),
        input_channels=int(input_shape.get("channels", 18)),
        num_classes=int(config.get("num_classes", 5)),
        seed=int(config.get("seed", 20260618)),
    )

    for row in rows:
        print(
            f"{row['model_name']}: {row['file_format']} {row['artifact_path']} "
            f"params={row['total_params']} output={row['output_shape']}"
        )
    print(f"saved index: {Path(output_dir) / 'netron_model_index.csv'}")
    return 0


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


if __name__ == "__main__":
    raise SystemExit(main())
