import pathlib
import sys
import tempfile
import unittest

import yaml


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class RealSmokeTrainingConfigTests(unittest.TestCase):
    def test_config_loads_and_enforces_smoke_safety(self):
        from training.supervised_trainer import load_smoke_training_config, validate_smoke_safety

        config = load_smoke_training_config(PROJECT_ROOT / "configs" / "real_smoke_training_v1.yaml")
        self.assertEqual(config["experiment_name"], "real_smoke_training_v1")
        self.assertTrue(config["training"]["smoke_mode"])
        self.assertFalse(config["training"]["allow_full_training"])
        self.assertEqual(config["training"]["max_epochs"], 1)
        self.assertLessEqual(config["training"]["max_train_batches"], 5)
        self.assertFalse(config["augmentation"]["enabled"])
        validate_smoke_safety(config)

    def test_unsafe_config_is_rejected_before_training(self):
        from training.supervised_trainer import load_smoke_training_config, validate_smoke_safety

        unsafe = {
            "experiment_name": "unsafe",
            "dataset_dir": "data/target/processed/v1_manual_windows_resample512",
            "output_root": "results/smoke_training",
            "seed": 42,
            "split": {
                "type": "loso_smoke",
                "test_subject_id": 1,
                "val_subject_id": 2,
                "train_subject_policy": "all_remaining",
                "strict_subject_isolation": True,
            },
            "models": ["all_channel_conv1d_v1"],
            "training": {
                "smoke_mode": True,
                "allow_full_training": False,
                "max_epochs": 3,
                "max_train_batches": 3,
                "max_val_batches": 3,
                "max_test_batches": 3,
                "batch_size": 16,
                "loss": "cross_entropy",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
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
            "metrics": {"accuracy": True},
            "reproducibility": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "unsafe.yaml"
            path.write_text(yaml.safe_dump(unsafe), encoding="utf-8")
            loaded = load_smoke_training_config(path)
        with self.assertRaises(ValueError):
            validate_smoke_safety(loaded)


if __name__ == "__main__":
    unittest.main()
