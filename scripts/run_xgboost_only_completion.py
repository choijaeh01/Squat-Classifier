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

from training.xgboost_completion_runner import load_xgboost_completion_config, run_xgboost_completion


def main() -> int:
    parser = argparse.ArgumentParser(description="Run XGBoost-only completion for controlled feature extractor comparison.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirm-xgboost-completion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    config = load_xgboost_completion_config(args.config)
    expected_runs = len(config["seeds"]) * len(config["split"]["subjects"]) * len(config["models"])
    print(f"expected_runs={expected_runs}")
    result = run_xgboost_completion(
        config,
        project_root=PROJECT_ROOT,
        confirm_xgboost_completion=args.confirm_xgboost_completion,
        dry_run=args.dry_run,
        resume_dir=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
