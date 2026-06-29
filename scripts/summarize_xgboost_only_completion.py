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

from analysis.xgboost_completion_analysis import write_xgboost_completion_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize XGBoost-only completion results.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "requires_input": bool(args.input)}, ensure_ascii=False, indent=2))
        return 0
    run_dir = args.input.resolve() if args.input else _latest_run_dir()
    write_xgboost_completion_analysis(project_root=PROJECT_ROOT, run_dir=run_dir)
    print(json.dumps({"run_dir": str(run_dir), "status": "summarized"}, ensure_ascii=False, indent=2))
    return 0


def _latest_run_dir() -> Path:
    root = PROJECT_ROOT / "results" / "xgboost_only_completion"
    candidates = sorted(path for path in root.glob("*_xgboost_only_completion_v1") if path.is_dir())
    if not candidates:
        raise SystemExit("--input is required because no xgboost completion run directory exists")
    return candidates[-1]


if __name__ == "__main__":
    raise SystemExit(main())
