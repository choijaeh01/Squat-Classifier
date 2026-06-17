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

from analysis.pilot_diagnostics import analyze_pilot_loso_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze pilot LOSO result CSV files.")
    parser.add_argument("--input", type=Path, required=True, help="Pilot LOSO run directory.")
    args = parser.parse_args()
    result = analyze_pilot_loso_run(args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
