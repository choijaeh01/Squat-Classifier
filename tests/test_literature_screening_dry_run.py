import pathlib
import sys
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class LiteratureScreeningDryRunTests(unittest.TestCase):
    def test_run_plan_contains_one_seed_six_fold_nine_model_screening(self):
        from training.literature_screening_runner import build_literature_screening_run_plan, load_literature_screening_config
        from training.splits import make_final_protocol_loso_splits

        config = load_literature_screening_config(PROJECT_ROOT / "configs" / "literature_baseline_screening_v1.yaml")
        subject_id, y, metadata = _balanced_arrays()
        splits = make_final_protocol_loso_splits(
            subject_id=subject_id,
            y=y,
            metadata=metadata,
            subjects=config["split"]["subjects"],
            train_windows_per_subject_class=16,
            val_windows_per_subject_class=4,
            seed=42,
        )
        plan = build_literature_screening_run_plan(config, splits)
        self.assertEqual(len(plan), 54)
        self.assertEqual({int(row["seed"]) for row in plan}, {42})
        self.assertEqual({int(row["fold_id"]) for row in plan}, {1, 2, 3, 4, 5, 6})
        self.assertEqual(plan[0]["status"], "pending")


def _balanced_arrays():
    subject_id = []
    y = []
    metadata = []
    for subject in range(1, 7):
        for class_id in range(5):
            for window_index in range(20):
                subject_id.append(subject)
                y.append(class_id)
                metadata.append(
                    {
                        "subject_id": str(subject),
                        "class_id": str(class_id),
                        "set_id": "1",
                        "window_index": str(window_index),
                        "sample_id": f"s{subject}_c{class_id}_w{window_index}",
                    }
                )
    return np.asarray(subject_id), np.asarray(y, dtype=np.int64), metadata


if __name__ == "__main__":
    unittest.main()
