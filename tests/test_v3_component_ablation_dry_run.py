import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class V3ComponentAblationDryRunTests(unittest.TestCase):
    def test_run_plan_contains_three_seed_six_fold_four_model_ablation(self):
        from data.processed_loader import ProcessedTargetDataset
        from training.loso_runner import _make_splits_for_config
        from training.v3_ablation_runner import build_v3_ablation_run_plan, load_v3_ablation_config

        output_dir = PROJECT_ROOT / "data" / "target" / "processed" / "v1_manual_windows_resample512"
        required = [output_dir / "X.npy", output_dir / "y.npy", output_dir / "subject_id.npy", output_dir / "metadata.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("processed target v1 dataset is ignored and not present in this checkout")

        config = load_v3_ablation_config(PROJECT_ROOT / "configs" / "v3_component_ablation_v1.yaml")
        seed_config = dict(config)
        seed_config["seed"] = 42
        splits = _make_splits_for_config(seed_config, ProcessedTargetDataset(output_dir))
        plan = build_v3_ablation_run_plan(config, splits)
        self.assertEqual(len(plan), 72)
        self.assertEqual({int(row["seed"]) for row in plan}, {42, 123, 2025})
        self.assertEqual({int(row["fold_id"]) for row in plan}, {1, 2, 3, 4, 5, 6})

    def test_dry_run_writes_required_audit_files_without_training(self):
        from training.v3_ablation_runner import load_v3_ablation_config, run_v3_component_ablation

        output_dir = PROJECT_ROOT / "data" / "target" / "processed" / "v1_manual_windows_resample512"
        required = [output_dir / "X.npy", output_dir / "y.npy", output_dir / "subject_id.npy", output_dir / "metadata.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("processed target v1 dataset is ignored and not present in this checkout")

        config = load_v3_ablation_config(PROJECT_ROOT / "configs" / "v3_component_ablation_v1.yaml")
        with tempfile.TemporaryDirectory() as temp_dir:
            config["output_root"] = temp_dir
            result = run_v3_component_ablation(config, project_root=PROJECT_ROOT, dry_run=True, device_name="cpu")
            run_dir = pathlib.Path(result["run_dir"])
            self.assertEqual(result["expected_runs"], 72)
            for filename in ("run_plan.csv", "split_plan.csv", "validation_policy_summary.csv", "scaler_fit_audit.csv"):
                self.assertTrue((run_dir / filename).exists(), filename)
                self.assertGreater((run_dir / filename).stat().st_size, 0, filename)


if __name__ == "__main__":
    unittest.main()
