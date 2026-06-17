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

from analysis.literature_full_extension_analysis import generate_literature_extension_figures, merge_full_extension_with_locked_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize literature full extension results.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "requires_input": bool(args.input)}, indent=2))
        return 0
    if args.input is None:
        raise SystemExit("--input is required unless --dry-run is used")
    run_dir = args.input.resolve()
    merged = merge_full_extension_with_locked_matrix(project_root=PROJECT_ROOT, extension_run_dir=run_dir)
    generate_literature_extension_figures(run_dir)
    print(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
