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

from training.loso_runner import generate_figures


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate pilot LOSO summary figures for an existing run directory.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    generate_figures(args.run_dir)
    print(json.dumps({"run_dir": str(args.run_dir), "figures": str(args.run_dir / "figures")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
