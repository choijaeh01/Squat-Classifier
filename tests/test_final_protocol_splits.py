import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _synthetic_balanced_metadata():
    subject_ids = []
    labels = []
    metadata = []
    for subject in range(1, 7):
        for class_id in range(5):
            for window_index in range(20):
                subject_ids.append(subject)
                labels.append(class_id)
                metadata.append(
                    {
                        "subject_id": str(subject),
                        "class_id": str(class_id),
                        "set_id": "1",
                        "window_index": str(window_index),
                        "sample_id": f"subject{subject}_class{class_id}_window{window_index}",
                    }
                )
    return np.asarray(subject_ids), np.asarray(labels, dtype=np.int64), metadata


class FinalProtocolSplitTests(unittest.TestCase):
    def test_within_train_subject_stratified_validation_counts_and_isolation(self):
        from training.splits import make_final_protocol_loso_splits

        subject_id, y, metadata = _synthetic_balanced_metadata()
        splits = make_final_protocol_loso_splits(
            subject_id=subject_id,
            y=y,
            metadata=metadata,
            subjects=[1, 2, 3, 4, 5, 6],
            train_windows_per_subject_class=16,
            val_windows_per_subject_class=4,
            seed=42,
        )

        self.assertEqual(len(splits), 6)
        for split in splits:
            self.assertEqual(len(split.train_idx), 400)
            self.assertEqual(len(split.val_idx), 100)
            self.assertEqual(len(split.test_idx), 100)
            self.assertNotIn(split.test_subject, set(subject_id[split.train_idx]))
            self.assertNotIn(split.test_subject, set(subject_id[split.val_idx]))
            self.assertEqual(set(subject_id[split.test_idx]), {split.test_subject})
            self.assertFalse(set(split.train_idx.tolist()) & set(split.val_idx.tolist()))
            self.assertTrue(split.leakage_check_passed)
            self.assertTrue(split.test_subject_isolated)
            self.assertTrue(split.train_val_index_disjoint)
            self.assertEqual({str(k): v for k, v in split.train_class_counts.items()}, {str(k): 80 for k in range(5)})
            self.assertEqual({str(k): v for k, v in split.val_class_counts.items()}, {str(k): 20 for k in range(5)})
            self.assertEqual({str(k): v for k, v in split.test_class_counts.items()}, {str(k): 20 for k in range(5)})

    def test_split_is_seed_deterministic(self):
        from training.splits import make_final_protocol_loso_splits

        subject_id, y, metadata = _synthetic_balanced_metadata()
        kwargs = {
            "subject_id": subject_id,
            "y": y,
            "metadata": metadata,
            "subjects": [1, 2, 3, 4, 5, 6],
            "train_windows_per_subject_class": 16,
            "val_windows_per_subject_class": 4,
            "seed": 42,
        }
        first = make_final_protocol_loso_splits(**kwargs)
        second = make_final_protocol_loso_splits(**kwargs)

        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.train_idx, b.train_idx)
            np.testing.assert_array_equal(a.val_idx, b.val_idx)
            np.testing.assert_array_equal(a.test_idx, b.test_idx)


if __name__ == "__main__":
    unittest.main()
