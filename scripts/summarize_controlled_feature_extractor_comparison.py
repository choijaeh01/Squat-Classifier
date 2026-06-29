#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.controlled_comparison_analysis import write_controlled_analysis_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize controlled feature extractor comparison results.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "requires_input": bool(args.input)}, ensure_ascii=False, indent=2))
        return 0
    run_dir = args.input.resolve() if args.input else _latest_run_dir()
    write_controlled_analysis_outputs(project_root=PROJECT_ROOT, run_dir=run_dir)
    rows = _read_csv(run_dir / "aggregate_metrics_by_model.csv")
    ranking = sorted(rows, key=lambda row: float(row.get("mean_macro_f1", 0.0) or 0.0), reverse=True)
    print(json.dumps({"run_dir": str(run_dir), "top_models": ranking[:5]}, ensure_ascii=False, indent=2))
    return 0


def _latest_run_dir() -> Path:
    root = PROJECT_ROOT / "results" / "controlled_feature_extractor_comparison"
    candidates = sorted(path for path in root.glob("*_controlled_feature_extractor_comparison_v1") if path.is_dir())
    if not candidates:
        raise SystemExit("--input is required because no controlled comparison run directory exists")
    return candidates[-1]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    raise SystemExit(main())
