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

from analysis.model_architecture_audit import ARCHITECTURE_MODELS, run_architecture_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward-only architecture audit for professor report v2.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "professor_report_v2" / "tables")
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT / "docs" / "professor_report_v2" / "display_name_mapping.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = (PROJECT_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    mapping = (PROJECT_ROOT / args.mapping).resolve() if not args.mapping.is_absolute() else args.mapping
    if args.dry_run:
        print(json.dumps({"dry_run": True, "output": str(output), "mapping": str(mapping), "models": ARCHITECTURE_MODELS}, ensure_ascii=False, indent=2))
        return 0
    result = run_architecture_audit(output_dir=output, mapping_path=mapping)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
