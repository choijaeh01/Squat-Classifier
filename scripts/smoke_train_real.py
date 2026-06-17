#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.supervised_trainer import load_smoke_training_config, run_real_smoke_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strictly limited real-data smoke training.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "real_smoke_training_v1.yaml")
    parser.add_argument("--confirm-smoke", action="store_true", help="Required to run optimizer/backward smoke steps.")
    parser.add_argument("--dry-run-forward-only", action="store_true", help="Only load data, scale, and forward pass; no optimizer/backward.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    config = load_smoke_training_config(args.config)
    result = run_real_smoke_training(
        config,
        project_root=PROJECT_ROOT,
        confirm_smoke=args.confirm_smoke,
        forward_only=args.dry_run_forward_only,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
