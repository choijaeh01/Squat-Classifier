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

from analysis.professor_report_assets import (  # noqa: E402
    ProfessorReportPaths,
    build_professor_report_assets,
    check_professor_report_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build professor-facing Obsidian report assets from read-only result CSVs.")
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-docs", type=Path, default=PROJECT_ROOT / "docs" / "professor_report_v1")
    parser.add_argument("--output-obsidian", type=Path, default=None)
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT / "docs" / "professor_report_v1" / "display_name_mapping.yaml")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.input_root.resolve()
    status = check_professor_report_inputs(project_root)
    if args.check_only or args.dry_run:
        print(json.dumps({"check_only": args.check_only, "dry_run": args.dry_run, **status}, ensure_ascii=False, indent=2))
        return 0 if status["all_present"] else 1
    if not status["all_present"]:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 1

    paths = ProfessorReportPaths(
        project_root=project_root,
        output_docs=(project_root / args.output_docs).resolve() if not args.output_docs.is_absolute() else args.output_docs,
        output_obsidian=args.output_obsidian,
        mapping_path=(project_root / args.mapping).resolve() if not args.mapping.is_absolute() else args.mapping,
    )
    result = build_professor_report_assets(paths)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
