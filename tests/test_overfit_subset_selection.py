import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class OverfitSubsetSelectionTests(unittest.TestCase):
    def test_selects_balanced_subset_from_requested_subjects_only(self):
        from training.overfit_diagnostic import select_balanced_overfit_subset

        subject_id = np.repeat(np.arange(1, 7), 50)
        y = np.tile(np.repeat(np.arange(5), 10), 6)

        indices = select_balanced_overfit_subset(
            y=y,
            subject_id=subject_id,
            subject_ids=[3, 4, 5, 6],
            samples_per_class=8,
            seed=42,
        )

        self.assertEqual(len(indices), 40)
        self.assertEqual(set(subject_id[indices].tolist()), {3, 4, 5, 6})
        counts = {label: int(np.sum(y[indices] == label)) for label in range(5)}
        self.assertEqual(counts, {0: 8, 1: 8, 2: 8, 3: 8, 4: 8})

        repeated = select_balanced_overfit_subset(
            y=y,
            subject_id=subject_id,
            subject_ids=[3, 4, 5, 6],
            samples_per_class=8,
            seed=42,
        )
        self.assertTrue(np.array_equal(indices, repeated))

    def test_raises_when_class_has_too_few_samples(self):
        from training.overfit_diagnostic import select_balanced_overfit_subset

        subject_id = np.array([3, 3, 3, 3, 3])
        y = np.array([0, 1, 1, 2, 3])
        with self.assertRaises(ValueError):
            select_balanced_overfit_subset(
                y=y,
                subject_id=subject_id,
                subject_ids=[3],
                samples_per_class=2,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
