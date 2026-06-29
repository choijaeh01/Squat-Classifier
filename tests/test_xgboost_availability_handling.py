import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class XGBoostAvailabilityHandlingTests(unittest.TestCase):
    def test_xgboost_model_is_classical_and_availability_is_explicit(self):
        from models.classical_features import is_classical_model, xgboost_available

        self.assertTrue(is_classical_model("feature_xgboost_v1"))
        self.assertIsInstance(xgboost_available(), bool)

    def test_xgboost_estimator_builds_or_raises_module_not_found(self):
        from models.classical_features import build_classical_estimator, xgboost_available

        if xgboost_available():
            estimator = build_classical_estimator("feature_xgboost_v1", seed=42)
            self.assertTrue(hasattr(estimator, "fit"))
            self.assertTrue(hasattr(estimator, "predict"))
        else:
            with self.assertRaises(ModuleNotFoundError):
                build_classical_estimator("feature_xgboost_v1", seed=42)


if __name__ == "__main__":
    unittest.main()
