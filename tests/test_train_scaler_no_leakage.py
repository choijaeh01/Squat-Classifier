import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TrainOnlyScalerTests(unittest.TestCase):
    def test_scaler_fit_uses_train_indices_only(self):
        from training.scalers import TrainOnlyStandardScaler

        X = np.concatenate(
            [
                np.zeros((2, 4, 3), dtype=np.float32),
                np.full((1, 4, 3), 100.0, dtype=np.float32),
                np.full((1, 4, 3), -50.0, dtype=np.float32),
            ],
            axis=0,
        )
        train_idx = np.array([0, 1])
        scaler = TrainOnlyStandardScaler()
        scaler.fit(X, train_idx=train_idx)

        self.assertTrue(np.allclose(scaler.mean_, np.zeros(3)))
        self.assertTrue(np.allclose(scaler.scale_, np.ones(3)))
        transformed = scaler.transform(X)
        self.assertTrue(np.allclose(transformed[:2], 0.0))
        self.assertTrue(np.allclose(transformed[2], 100.0))
        self.assertTrue(np.allclose(transformed[3], -50.0))
        self.assertEqual(scaler.fit_sample_count_, 2)
        self.assertEqual(scaler.fit_window_count_, 8)


if __name__ == "__main__":
    unittest.main()
