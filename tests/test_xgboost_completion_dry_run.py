from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.xgboost_completion_runner import load_xgboost_completion_config, run_xgboost_completion


class XGBoostCompletionDryRunTests(unittest.TestCase):
    def test_dry_run_writes_xgboost_only_plan_without_dependency_requirement(self) -> None:
        config = load_xgboost_completion_config(PROJECT_ROOT / "configs" / "xgboost_only_completion_v1.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_xgboost_completion(
                config,
                project_root=PROJECT_ROOT,
                dry_run=True,
                confirm_xgboost_completion=False,
                resume_dir=Path(tmp) / "xgboost_dry_run",
            )
            run_dir = Path(result["run_dir"])
            with (run_dir / "run_plan.csv").open("r", encoding="utf-8") as handle:
                plan = list(csv.DictReader(handle))
            self.assertEqual(result["expected_runs"], 18)
            self.assertEqual(len(plan), 18)
            self.assertEqual({row["model_name"] for row in plan}, {"feature_xgboost_v1"})
            self.assertEqual({row["status"] for row in plan}, {"dry_run"})
            self.assertTrue((run_dir / "feature_audit.csv").exists())
            self.assertTrue((run_dir / "scaler_fit_audit.csv").exists())
            self.assertFalse((run_dir / "xgboost_feature_importance_by_fold.csv").exists())


if __name__ == "__main__":
    unittest.main()
