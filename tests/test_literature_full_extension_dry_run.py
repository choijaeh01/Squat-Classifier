import pathlib
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class LiteratureFullExtensionDryRunTests(unittest.TestCase):
    def test_run_plan_contains_three_seed_six_fold_eight_model_extension(self):
        from training.literature_full_extension_runner import build_full_extension_run_plan, load_literature_full_extension_config
        from training.splits import make_final_protocol_loso_splits

        config = load_literature_full_extension_config(PROJECT_ROOT / "configs" / "literature_baseline_full_extension_v1.yaml")
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
        plan = build_full_extension_run_plan(config, splits)
        self.assertEqual(len(plan), 144)
        self.assertEqual({int(row["seed"]) for row in plan}, {42, 123, 2025})
        self.assertEqual({int(row["fold_id"]) for row in plan}, {1, 2, 3, 4, 5, 6})
        self.assertEqual(plan[0]["status"], "pending")

    def test_dry_run_writes_split_and_scaler_audit_files_without_training(self):
        from training.literature_full_extension_runner import load_literature_full_extension_config, run_literature_full_extension

        output_dir = PROJECT_ROOT / "data" / "target" / "processed" / "v1_manual_windows_resample512"
        required = [output_dir / "X.npy", output_dir / "y.npy", output_dir / "subject_id.npy", output_dir / "metadata.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("processed target v1 dataset is ignored and not present in this checkout")

        config = load_literature_full_extension_config(PROJECT_ROOT / "configs" / "literature_baseline_full_extension_v1.yaml")
        with tempfile.TemporaryDirectory() as temp_dir:
            config["output_root"] = temp_dir
            result = run_literature_full_extension(config, project_root=PROJECT_ROOT, dry_run=True, device_name="cpu")
            run_dir = pathlib.Path(result["run_dir"])
            self.assertEqual(result["expected_runs"], 144)
            self.assertTrue((run_dir / "split_plan.csv").exists())
            self.assertTrue((run_dir / "validation_policy_summary.csv").exists())
            self.assertTrue((run_dir / "scaler_fit_audit.csv").exists())
            self.assertGreater((run_dir / "split_plan.csv").stat().st_size, 0)
            self.assertGreater((run_dir / "validation_policy_summary.csv").stat().st_size, 0)
            self.assertGreater((run_dir / "scaler_fit_audit.csv").stat().st_size, 0)


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
