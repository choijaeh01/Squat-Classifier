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

from training.controlled_comparison_runner import (
    controlled_model_names,
    load_controlled_comparison_config,
    run_controlled_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled feature extractor comparison v1.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirm-controlled-comparison", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; no optimizer, backward, or epoch training.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    config = load_controlled_comparison_config(args.config)
    expected_runs = len(config["seeds"]) * len(config["split"]["subjects"]) * len(controlled_model_names(config))
    print(f"expected_runs={expected_runs}")
    result = run_controlled_comparison(
        config,
        project_root=PROJECT_ROOT,
        confirm_controlled_comparison=args.confirm_controlled_comparison,
        dry_run=args.dry_run,
        device_name=args.device,
        resume_dir=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
