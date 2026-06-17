import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ManifestTests(unittest.TestCase):
    def test_manifest_round_trip_and_checksum_are_stable(self):
        from data.manifest import ManifestRecord, TargetDataManifest

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            sample = root / "subject1_class0_window0.npy"
            np.save(sample, np.zeros((512, 18), dtype=np.float32))

            manifest = TargetDataManifest(
                dataset_name="synthetic-target",
                window_length=512,
                channels=18,
                class_names={
                    0: "Correct",
                    1: "Knee Valgus",
                    2: "Butt Wink",
                    3: "Excessive Lean",
                    4: "Partial Squat",
                },
                records=[
                    ManifestRecord(
                        sample_id="subject1_class0_window0",
                        path=str(sample),
                        subject_id="subject1",
                        label=0,
                        source="synthetic",
                    )
                ],
            )

            out = root / "manifest.json"
            manifest.save(out)
            loaded = TargetDataManifest.load(out)

            self.assertEqual(loaded.records[0].subject_id, "subject1")
            self.assertEqual(loaded.records[0].label, 0)
            self.assertEqual(loaded.checksum(), manifest.checksum())
            self.assertEqual(loaded.validate(), [])


class PreprocessingTests(unittest.TestCase):
    def test_loso_scaler_fits_only_training_subjects(self):
        from data.preprocessing import LOSOStandardScaler

        windows = np.concatenate(
            [
                np.full((2, 4, 3), 1.0, dtype=np.float32),
                np.full((2, 4, 3), 100.0, dtype=np.float32),
            ],
            axis=0,
        )
        subject_ids = np.array(["train_subject", "train_subject", "heldout", "heldout"])

        scaler = LOSOStandardScaler()
        scaler.fit(windows, subject_ids=subject_ids, held_out_subject="heldout")

        self.assertTrue(np.allclose(scaler.mean_, np.ones((3,), dtype=np.float32)))
        self.assertTrue(np.allclose(scaler.scale_, np.ones((3,), dtype=np.float32)))
        transformed = scaler.transform(windows)
        self.assertTrue(np.allclose(transformed[:2], 0.0))
        self.assertTrue(np.allclose(transformed[2:], 99.0))


class ModelRegistryTests(unittest.TestCase):
    def test_every_registered_model_accepts_target_shape(self):
        from models.registry import list_models, build_model, count_parameters

        expected = {
            "classical_feature_baseline",
            "all_channel_conv1d",
            "channel_shared_1d_encoder",
            "modality_shared_acc_gyro_encoder",
            "cnn2d_baseline",
        }
        self.assertTrue(expected.issubset(set(list_models())))

        x = torch.randn(2, 512, 18)
        for name in sorted(expected):
            with self.subTest(model=name):
                model = build_model(name, input_length=512, input_channels=18, num_classes=5)
                model.eval()
                with torch.no_grad():
                    out = model(x)
                self.assertEqual(tuple(out.shape), (2, 5))
                self.assertGreater(count_parameters(model), 0)

    def test_shared_encoder_models_reuse_encoder_objects(self):
        from models.registry import build_model

        channel_model = build_model(
            "channel_shared_1d_encoder",
            input_length=512,
            input_channels=18,
            num_classes=5,
        )
        self.assertIs(channel_model.channel_encoder_refs[0], channel_model.shared_encoder)
        self.assertTrue(all(ref is channel_model.shared_encoder for ref in channel_model.channel_encoder_refs))

        modality_model = build_model(
            "modality_shared_acc_gyro_encoder",
            input_length=512,
            input_channels=18,
            num_classes=5,
        )
        self.assertIsNot(modality_model.acc_encoder, modality_model.gyro_encoder)
        self.assertTrue(all(ref is modality_model.acc_encoder for ref in modality_model.acc_encoder_refs))
        self.assertTrue(all(ref is modality_model.gyro_encoder for ref in modality_model.gyro_encoder_refs))


class MetricsAndSmokeTests(unittest.TestCase):
    def test_metrics_payload_contains_required_future_outputs(self):
        from training.metrics import compute_classification_metrics
        from training.results import save_metrics

        y_true = np.array([0, 1, 2, 3, 4, 4])
        y_pred = np.array([0, 1, 2, 2, 4, 3])
        subject_ids = np.array(["s1", "s1", "s2", "s2", "s3", "s3"])
        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            subject_ids=subject_ids,
            num_classes=5,
            class_names=["Correct", "Knee Valgus", "Butt Wink", "Excessive Lean", "Partial Squat"],
        )

        required = {
            "accuracy",
            "macro_f1",
            "class_wise",
            "subject_wise_macro_f1",
            "confusion_matrix",
        }
        self.assertTrue(required.issubset(metrics))

        with tempfile.TemporaryDirectory() as tmp:
            path = save_metrics(metrics, pathlib.Path(tmp) / "metrics.json")
            saved = json.loads(path.read_text())
        self.assertIn("accuracy", saved)

    def test_smoke_script_writes_dummy_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_copy = pathlib.Path(tmp) / "squat_imu_experiments"
            ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "results")
            shutil.copytree(PROJECT_ROOT, project_copy, ignore=ignore)
            script = project_copy / "scripts" / "smoke_test.py"
            result = subprocess.run(
                [sys.executable, str(script), "--output-dir", str(project_copy / "results" / "smoke_test")],
                cwd=project_copy,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
            metrics_path = project_copy / "results" / "smoke_test" / "metrics.json"
            self.assertTrue(metrics_path.exists())
            metrics = json.loads(metrics_path.read_text())
            self.assertEqual(metrics["smoke_test"]["input_shape"], [16, 512, 18])
            self.assertTrue({
                "classical_feature_baseline",
                "all_channel_conv1d",
                "channel_shared_1d_encoder",
                "modality_shared_acc_gyro_encoder",
                "cnn2d_baseline",
            }.issubset(set(metrics["model_parameter_counts"])))


if __name__ == "__main__":
    unittest.main()
