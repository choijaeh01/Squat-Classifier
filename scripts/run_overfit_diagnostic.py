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

from training.overfit_diagnostic import load_overfit_config, run_overfit_diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description="Run small-subset overfit diagnostic.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "overfit_diagnostic_v1.yaml")
    parser.add_argument("--confirm-diagnostic", action="store_true", help="Required to run optimizer/backward diagnostic training.")
    parser.add_argument("--dry-run", action="store_true", help="Forward-only dry run; no optimizer/backward.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    config = load_overfit_config(args.config)
    result = run_overfit_diagnostic(
        config,
        project_root=PROJECT_ROOT,
        confirm_diagnostic=args.confirm_diagnostic,
        dry_run=args.dry_run,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
