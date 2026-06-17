import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ClassicalFeatureExtractionTests(unittest.TestCase):
    def test_feature_extraction_is_deterministic_and_per_window(self):
        from models.classical_features import extract_window_features

        X = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
        first = extract_window_features(X)
        second = extract_window_features(X.copy())
        self.assertEqual(first.shape, (2, 27))
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(first[0, :3], X[0].mean(axis=0))
        np.testing.assert_allclose(first[1, 6:9], X[1].min(axis=0))

    def test_sklearn_availability_check_is_boolean(self):
        from models.classical_features import sklearn_available

        self.assertIsInstance(sklearn_available(), bool)


if __name__ == "__main__":
    unittest.main()
