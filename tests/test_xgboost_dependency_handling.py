from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.xgboost_completion_runner import ensure_xgboost_dependency


class XGBoostDependencyHandlingTests(unittest.TestCase):
    def test_missing_xgboost_raises_dependency_missing_in_confirmed_mode(self) -> None:
        with mock.patch("models.classical_features.xgboost_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "dependency_missing"):
                ensure_xgboost_dependency(require=True)

    def test_dry_run_can_report_missing_dependency_without_raising(self) -> None:
        with mock.patch("models.classical_features.xgboost_available", return_value=False):
            info = ensure_xgboost_dependency(require=False)
        self.assertFalse(info["available"])
        self.assertEqual(info["version"], "")


if __name__ == "__main__":
    unittest.main()
