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

from training.literature_full_extension_runner import load_literature_full_extension_config, run_literature_full_extension


def main() -> int:
    parser = argparse.ArgumentParser(description="Run literature baseline 3-seed full extension.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirm-full-extension", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; no optimizer, backward, or epoch training.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    config = load_literature_full_extension_config(args.config)
    expected_runs = len(config["seeds"]) * len(config["split"]["subjects"]) * len(config["models"])
    print(f"expected_runs={expected_runs}")
    result = run_literature_full_extension(
        config,
        project_root=PROJECT_ROOT,
        confirm_full_extension=args.confirm_full_extension,
        dry_run=args.dry_run,
        device_name=args.device,
        resume_dir=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
