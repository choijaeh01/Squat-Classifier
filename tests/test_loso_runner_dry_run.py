import csv
import pathlib
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class LOSORunnerDryRunTests(unittest.TestCase):
    def test_dry_run_writes_fold_outputs_without_training(self):
        from training.loso_runner import run_pilot_loso

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            X = np.random.default_rng(123).normal(size=(60, 512, 18)).astype(np.float32)
            y = np.tile(np.arange(5, dtype=np.int64), 12)
            subject_id = np.repeat(np.arange(1, 7, dtype=np.int64), 10)
            np.save(dataset_dir / "X.npy", X)
            np.save(dataset_dir / "y.npy", y)
            np.save(dataset_dir / "subject_id.npy", subject_id)
            with (dataset_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_index", "subject", "class"])
                writer.writeheader()
                for index in range(len(X)):
                    writer.writerow({"sample_index": index, "subject": int(subject_id[index]), "class": int(y[index])})

            config = {
                "experiment_name": "pilot_loso_test",
                "dataset_dir": str(dataset_dir),
                "output_root": str(root / "out"),
                "seed": 42,
                "split": {
                    "type": "loso_with_cyclic_subject_validation",
                    "subjects": [1, 2, 3, 4, 5, 6],
                    "validation_policy": "next_subject_cyclic",
                    "strict_subject_isolation": True,
                },
                "models": ["all_channel_conv1d_small"],
                "training": {
                    "pilot_mode": True,
                    "allow_full_training": False,
                    "max_epochs": 1,
                    "batch_size": 8,
                    "loss": "cross_entropy",
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "early_stopping": {
                        "enabled": True,
                        "monitor": "val_macro_f1",
                        "mode": "max",
                        "patience": 1,
                        "min_delta": 0.0001,
                        "restore_best": True,
                    },
                    "gradient_clip_norm": None,
                    "mixed_precision": False,
                },
                "normalization": {
                    "global_standard_scaler": True,
                    "fit_scaler_on": "train_subjects_only",
                    "per_window_zscore": False,
                    "save_scaler_stats": True,
                },
                "augmentation": {"enabled": False},
                "checkpointing": {
                    "save_best_checkpoint": True,
                    "save_last_checkpoint": False,
                    "save_model_state_dict": True,
                },
                "metrics": {
                    "accuracy": True,
                    "macro_f1": True,
                    "class_wise_precision_recall_f1": True,
                    "confusion_matrix": True,
                    "subject_wise_metrics": True,
                },
                "reproducibility": {
                    "deterministic": True,
                    "save_config_snapshot": True,
                    "save_git_status": True,
                    "save_manifest_checksum": True,
                    "save_model_parameter_count": True,
                },
                "safety": {
                    "max_allowed_epochs_for_pilot": 30,
                    "max_allowed_seeds_for_pilot": 1,
                    "forbid_ssl": True,
                    "forbid_augmentation": True,
                    "forbid_focal_loss": True,
                    "forbid_external_dataset": True,
                },
            }

            result = run_pilot_loso(config, project_root=PROJECT_ROOT, dry_run=True, device_name="cpu")

            self.assertTrue(result["dry_run"])
            run_dir = pathlib.Path(result["run_dir"])
            self.assertTrue((run_dir / "split_plan.csv").exists())
            self.assertTrue((run_dir / "fold_metrics.csv").exists())
            self.assertTrue((run_dir / "aggregate_metrics_by_model.csv").exists())
            with (run_dir / "fold_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["status"] == "forward_only" for row in rows))
            self.assertTrue(all(row["leakage_check_passed"] == "True" for row in rows))
            self.assertTrue(all(row["scaler_fit_subjects"] == row["train_subjects"] for row in rows))


if __name__ == "__main__":
    unittest.main()
