import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FeatureImportanceAuditTests(unittest.TestCase):
    def test_feature_audit_allows_only_signal_derived_features(self):
        from analysis.feature_importance_analysis import build_feature_audit_rows, validate_feature_audit_rows

        rows = build_feature_audit_rows(input_channels=18)
        self.assertEqual(len(rows), 18 * 9)
        self.assertTrue(all(row["is_signal_derived"] for row in rows))
        self.assertTrue(all(row["allowed"] for row in rows))
        validate_feature_audit_rows(rows)

    def test_feature_names_do_not_contain_metadata_terms(self):
        from analysis.feature_importance_analysis import feature_definitions

        banned = {"subject", "label", "class", "window_index", "source", "boundary", "filename"}
        for row in feature_definitions(input_channels=18):
            lower = row["feature_name"].lower()
            self.assertFalse(any(term in lower for term in banned), row["feature_name"])


if __name__ == "__main__":
    unittest.main()
