import pathlib
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ControlledComparisonDryRunTests(unittest.TestCase):
    def test_run_plan_has_expected_252_runs(self):
        from training.controlled_comparison_runner import (
            build_controlled_run_plan,
            load_controlled_comparison_config,
        )
        from training.splits import make_final_protocol_loso_splits

        config = load_controlled_comparison_config(PROJECT_ROOT / "configs" / "controlled_feature_extractor_comparison_v1.yaml")
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
        plan = build_controlled_run_plan(config, splits)
        self.assertEqual(len(plan), 252)
        self.assertEqual({int(row["seed"]) for row in plan}, {42, 123, 2025})
        self.assertEqual({int(row["fold_id"]) for row in plan}, {1, 2, 3, 4, 5, 6})
        self.assertIn("model_group", plan[0])

    def test_dry_run_writes_required_preflight_files_without_training(self):
        from training.controlled_comparison_runner import load_controlled_comparison_config, run_controlled_comparison

        dataset_dir = PROJECT_ROOT / "data" / "target" / "processed" / "v1_manual_windows_resample512"
        required = [dataset_dir / "X.npy", dataset_dir / "y.npy", dataset_dir / "subject_id.npy", dataset_dir / "metadata.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("processed target v1 dataset is ignored and not present in this checkout")
        config = load_controlled_comparison_config(PROJECT_ROOT / "configs" / "controlled_feature_extractor_comparison_v1.yaml")
        with tempfile.TemporaryDirectory() as temp_dir:
            config["output_root"] = temp_dir
            result = run_controlled_comparison(config, project_root=PROJECT_ROOT, dry_run=True, device_name="cpu")
            run_dir = pathlib.Path(result["run_dir"])
            self.assertEqual(result["expected_runs"], 252)
            for filename in ["run_plan.csv", "split_plan.csv", "validation_policy_summary.csv", "scaler_fit_audit.csv", "common_head_verification.csv"]:
                self.assertTrue((run_dir / filename).exists(), filename)
                self.assertGreater((run_dir / filename).stat().st_size, 0, filename)


def _balanced_arrays():
    subject_id = []
    y = []
    metadata = []
    for subject in range(1, 7):
        for class_id in range(5):
            for window_index in range(20):
                subject_id.append(subject)
                y.append(class_id)
                metadata.append({"subject_id": str(subject), "class_id": str(class_id), "window_index": str(window_index)})
    return np.asarray(subject_id), np.asarray(y, dtype=np.int64), metadata


if __name__ == "__main__":
    unittest.main()
