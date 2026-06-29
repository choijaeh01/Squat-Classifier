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

from analysis.normalization_protocol_audit import run_normalization_protocol_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit normalization/scaler/feature extraction protocol for professor report v2.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs" / "professor_report_v2" / "tables")
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "docs" / "professor_report_v2" / "figures")
    parser.add_argument("--diagrams", type=Path, default=PROJECT_ROOT / "docs" / "professor_report_v2" / "diagrams")
    args = parser.parse_args()

    output = (PROJECT_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    figures = (PROJECT_ROOT / args.figures).resolve() if not args.figures.is_absolute() else args.figures
    diagrams = (PROJECT_ROOT / args.diagrams).resolve() if not args.diagrams.is_absolute() else args.diagrams
    result = run_normalization_protocol_audit(project_root=PROJECT_ROOT, output_dir=output, figure_dir=figures, diagram_dir=diagrams)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
