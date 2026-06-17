import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class CyclicLOSOSplitTests(unittest.TestCase):
    def test_cyclic_validation_policy_matches_protocol(self):
        from training.splits import make_cyclic_loso_splits

        subject_id = np.repeat(np.arange(1, 7), 10)
        y = np.tile(np.arange(5), 12)
        splits = make_cyclic_loso_splits(
            subject_id=subject_id,
            y=y,
            subjects=[1, 2, 3, 4, 5, 6],
            strict_subject_isolation=True,
        )

        observed = [
            (split.test_subject, split.val_subject, split.train_subjects)
            for split in splits
        ]
        self.assertEqual(
            observed,
            [
                (1, 2, [3, 4, 5, 6]),
                (2, 3, [1, 4, 5, 6]),
                (3, 4, [1, 2, 5, 6]),
                (4, 5, [1, 2, 3, 6]),
                (5, 6, [1, 2, 3, 4]),
                (6, 1, [2, 3, 4, 5]),
            ],
        )
        for split in splits:
            self.assertTrue(split.leakage_check_passed)
            self.assertFalse(set(subject_id[split.train_idx]) & set(subject_id[split.val_idx]))
            self.assertFalse(set(subject_id[split.train_idx]) & set(subject_id[split.test_idx]))
            self.assertFalse(set(subject_id[split.val_idx]) & set(subject_id[split.test_idx]))


if __name__ == "__main__":
    unittest.main()
