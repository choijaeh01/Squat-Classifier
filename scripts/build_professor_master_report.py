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

from analysis.professor_master_report import (  # noqa: E402
    ProfessorMasterReportPaths,
    build_professor_master_report,
    check_professor_master_report_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the professor master report from read-only result artifacts.")
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-docs", type=Path, default=PROJECT_ROOT / "docs" / "professor_master_report_v1")
    parser.add_argument("--output-obsidian", type=Path, default=None)
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.input_root.resolve()
    output_docs = (project_root / args.output_docs).resolve() if not args.output_docs.is_absolute() else args.output_docs
    default_mapping = project_root / "docs" / "professor_master_report_v1" / "display_name_mapping.yaml"
    mapping_arg = args.mapping or default_mapping
    mapping_path = (project_root / mapping_arg).resolve() if not mapping_arg.is_absolute() else mapping_arg

    status = check_professor_master_report_inputs(project_root)
    status["mapping_path"] = str(mapping_path)
    status["mapping_exists"] = mapping_path.exists()
    status["all_present"] = bool(status["all_present"] and mapping_path.exists())
    if args.check_only or args.dry_run:
        print(json.dumps({"check_only": args.check_only, "dry_run": args.dry_run, **status}, ensure_ascii=False, indent=2))
        return 0 if status["all_present"] else 1
    if not status["all_present"]:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 1

    paths = ProfessorMasterReportPaths(
        project_root=project_root,
        output_docs=output_docs,
        output_obsidian=args.output_obsidian,
        mapping_path=mapping_path,
    )
    result = build_professor_master_report(paths)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
