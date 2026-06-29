import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ControlledFeatureAuditTests(unittest.TestCase):
    def test_signal_feature_audit_allows_no_metadata_or_subject_features(self):
        from analysis.feature_importance_analysis import build_feature_audit_rows, feature_definitions, validate_feature_audit_rows

        definitions = feature_definitions(input_channels=18)
        audit = build_feature_audit_rows(input_channels=18)
        self.assertEqual(len(definitions), 162)
        self.assertEqual(len(audit), 162)
        validate_feature_audit_rows(audit)
        for row in audit:
            self.assertTrue(row["allowed"])
            self.assertTrue(row["is_signal_derived"])
            self.assertFalse(row["uses_metadata"])
            self.assertFalse(row["uses_label"])
            self.assertFalse(row["uses_subject_id"])
            self.assertFalse(row["uses_window_boundary"])


if __name__ == "__main__":
    unittest.main()
