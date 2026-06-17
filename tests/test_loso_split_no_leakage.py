import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class LOSOSmokeSplitTests(unittest.TestCase):
    def test_test_and_val_subjects_are_excluded_from_train(self):
        from training.splits import make_loso_smoke_split, summarize_split

        subject_id = np.repeat(np.arange(1, 7), 10)
        y = np.tile(np.arange(5), 12)
        split = make_loso_smoke_split(
            subject_id=subject_id,
            y=y,
            test_subject_id=1,
            val_subject_id=2,
            strict_subject_isolation=True,
        )

        self.assertEqual(set(subject_id[split.train_idx].tolist()), {3, 4, 5, 6})
        self.assertEqual(set(subject_id[split.val_idx].tolist()), {2})
        self.assertEqual(set(subject_id[split.test_idx].tolist()), {1})
        self.assertTrue(split.leakage_check_passed)

        summary = summarize_split(split, y=y, subject_id=subject_id, model_name="dummy", seed=42)
        self.assertEqual(summary["train_subjects"], "3|4|5|6")
        self.assertEqual(summary["val_subject"], "2")
        self.assertEqual(summary["test_subject"], "1")
        self.assertEqual(summary["leakage_check_passed"], True)


if __name__ == "__main__":
    unittest.main()
