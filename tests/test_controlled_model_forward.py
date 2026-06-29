import pathlib
import sys
import unittest

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CONTROLLED_MODELS = [
    "controlled_flatten_mlp",
    "controlled_stats_mlp",
    "controlled_all_channel_1d_cnn",
    "controlled_all_channel_1d_cnn_small",
    "controlled_shared_1d",
    "controlled_shared_1d_identity",
    "controlled_shared_1d_residual",
    "controlled_shared_1d_residual_identity",
    "controlled_2d_cnn",
]


class ControlledModelForwardTests(unittest.TestCase):
    def test_controlled_models_forward_on_synthetic_input(self):
        from models.registry import build_model, list_models

        registered = set(list_models())
        self.assertTrue(set(CONTROLLED_MODELS).issubset(registered))
        x = torch.randn(2, 512, 18)
        for model_name in CONTROLLED_MODELS:
            with self.subTest(model=model_name):
                model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, 5))

    def test_controlled_models_forward_on_real_processed_batch_if_available(self):
        from data.processed_loader import ProcessedTargetDataset
        from models.registry import build_model

        dataset_dir = PROJECT_ROOT / "data" / "target" / "processed" / "v1_manual_windows_resample512"
        required = [dataset_dir / "X.npy", dataset_dir / "y.npy", dataset_dir / "subject_id.npy", dataset_dir / "metadata.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("processed target v1 dataset is ignored and not present in this checkout")
        dataset = ProcessedTargetDataset(dataset_dir)
        x = torch.as_tensor(dataset.X[:2], dtype=torch.float32)
        for model_name in CONTROLLED_MODELS:
            with self.subTest(model=model_name):
                model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
