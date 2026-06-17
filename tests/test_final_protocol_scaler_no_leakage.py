import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FinalProtocolScalerLeakageTests(unittest.TestCase):
    def test_scaler_audit_rejects_val_or_test_indices(self):
        from training.loso_runner import build_scaler_fit_audit_row
        from training.scalers import TrainOnlyStandardScaler
        from training.splits import LOSOSmokeSplit

        split = LOSOSmokeSplit(
            train_idx=np.asarray([0, 1, 2, 3]),
            val_idx=np.asarray([4, 5]),
            test_idx=np.asarray([6, 7]),
            train_subjects=[1, 2],
            val_subject="within_train_stratified",
            test_subject=3,
            leakage_check_passed=True,
        )
        X = np.arange(8 * 512 * 18, dtype=np.float32).reshape(8, 512, 18)
        scaler = TrainOnlyStandardScaler().fit(X, train_idx=split.train_idx)
        row = build_scaler_fit_audit_row(
            fold_id=1,
            model_name="dummy",
            split=split,
            scaler=scaler,
            scaler_fit_indices=split.train_idx,
        )
        self.assertFalse(row["val_indices_used_in_scaler"])
        self.assertFalse(row["test_indices_used_in_scaler"])
        self.assertTrue(row["scaler_leakage_check_passed"])

        bad = build_scaler_fit_audit_row(
            fold_id=1,
            model_name="dummy",
            split=split,
            scaler=scaler,
            scaler_fit_indices=np.asarray([0, 4]),
        )
        self.assertTrue(bad["val_indices_used_in_scaler"])
        self.assertFalse(bad["scaler_leakage_check_passed"])


if __name__ == "__main__":
    unittest.main()
