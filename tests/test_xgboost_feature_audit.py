from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.feature_importance_analysis import build_feature_audit_rows, validate_feature_audit_rows
from models.classical_features import extract_window_features


class XGBoostFeatureAuditTests(unittest.TestCase):
    def test_xgboost_uses_same_signal_derived_feature_set_as_random_forest(self) -> None:
        audit = build_feature_audit_rows(input_channels=18)
        validate_feature_audit_rows(audit)
        self.assertEqual(len(audit), 162)
        self.assertTrue(all(row["allowed"] for row in audit))
        self.assertTrue(all(row["is_signal_derived"] for row in audit))
        self.assertFalse(any(row["uses_metadata"] or row["uses_label"] or row["uses_subject_id"] for row in audit))

    def test_feature_count_matches_extractor_output(self) -> None:
        import numpy as np

        X = np.zeros((3, 512, 18), dtype=np.float32)
        features = extract_window_features(X)
        self.assertEqual(features.shape, (3, 162))


if __name__ == "__main__":
    unittest.main()
