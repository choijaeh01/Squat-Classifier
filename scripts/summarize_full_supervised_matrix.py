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

from training.full_matrix_runner import summarize_full_matrix_run


def _latest_run() -> Path | None:
    root = PROJECT_ROOT / "results" / "full_supervised_matrix"
    if not root.exists():
        return None
    runs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.endswith("_full_supervised_matrix_v1"))
    return runs[-1] if runs else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize full supervised matrix outputs.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    args = parser.parse_args()
    run_dir = args.input or _latest_run()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "input": None if run_dir is None else str(run_dir)}, ensure_ascii=False, indent=2))
        return 0
    if run_dir is None:
        raise SystemExit("no full supervised matrix run found")
    result = summarize_full_matrix_run(run_dir, bootstrap_n=args.bootstrap_n)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
