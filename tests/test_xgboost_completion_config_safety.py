from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.xgboost_completion_runner import validate_xgboost_completion_safety


class XGBoostCompletionConfigSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        with (PROJECT_ROOT / "configs" / "xgboost_only_completion_v1.yaml").open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

    def test_config_matches_xgboost_only_contract(self) -> None:
        validate_xgboost_completion_safety(self.config)
        self.assertEqual(self.config["models"], ["feature_xgboost_v1"])
        self.assertEqual(self.config["seeds"], [42, 123, 2025])
        self.assertTrue(self.config["feature_policy"]["use_same_feature_set_as_random_forest"])
        self.assertEqual(self.config["normalization"]["fit_scaler_on"], "train_indices_only")
        self.assertFalse(self.config["normalization"]["per_window_zscore"])
        self.assertTrue(self.config["safety"]["forbid_modifying_controlled_results"])

    def test_rejects_non_xgboost_or_tuning_config(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["models"] = ["feature_xgboost_v1", "feature_random_forest_v1"]
        with self.assertRaises(ValueError):
            validate_xgboost_completion_safety(bad)
        bad = copy.deepcopy(self.config)
        bad["xgboost"]["no_hyperparameter_search"] = False
        with self.assertRaises(ValueError):
            validate_xgboost_completion_safety(bad)


if __name__ == "__main__":
    unittest.main()
