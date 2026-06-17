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

from training.full_matrix_runner import load_full_matrix_config, run_full_supervised_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full supervised LOSO matrix.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirm-full-matrix", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Forward-only full matrix path; no optimizer/backward.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    config = load_full_matrix_config(args.config)
    expected_runs = len(config["seeds"]) * len(config["split"]["subjects"]) * len(config["models"])
    print(f"expected_runs={expected_runs}")
    result = run_full_supervised_matrix(
        config,
        project_root=PROJECT_ROOT,
        confirm_full_matrix=args.confirm_full_matrix,
        dry_run=args.dry_run,
        device_name=args.device,
        resume_dir=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
