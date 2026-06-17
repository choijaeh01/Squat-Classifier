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

from analysis.paper_result_lock import lock_paper_results, validate_result_integrity


DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "full_supervised_matrix" / "20260617_144309_full_supervised_matrix_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lock supervised matrix results for manuscript drafting.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    integrity = validate_result_integrity(args.input)
    if args.check_only:
        print(json.dumps(integrity, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if integrity["all_checks_passed"] else 1
    output_dir = lock_paper_results(args.input)
    print(json.dumps({"paper_lock_dir": str(output_dir), **integrity}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if integrity["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
