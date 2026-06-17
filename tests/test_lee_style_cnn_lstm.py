import pathlib
import sys
import unittest

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class LeeStyleCNNLSTMTests(unittest.TestCase):
    def test_lee_style_model_forward_and_parameter_budget(self):
        from models.registry import build_model, count_parameter_groups, list_models

        self.assertIn("lee_style_cnn_lstm_2d_v1", set(list_models()))
        model = build_model("lee_style_cnn_lstm_2d_v1", input_length=512, input_channels=18, num_classes=5)
        model.eval()
        x = torch.randn(3, 512, 18)
        with torch.no_grad():
            logits = model(x)
            downsampled = model.downsample_to_40(x)
        self.assertEqual(tuple(logits.shape), (3, 5))
        self.assertEqual(tuple(downsampled.shape), (3, 40, 18))
        groups = count_parameter_groups(model)
        self.assertGreaterEqual(groups["total_params"], 30_000)
        self.assertLessEqual(groups["total_params"], 150_000)

    def test_lee_style_forward_on_real_processed_batch_if_available(self):
        from data.processed_loader import ProcessedTargetDataset
        from models.registry import build_model

        output_dir = PROJECT_ROOT / "data" / "target" / "processed" / "v1_manual_windows_resample512"
        required = [output_dir / "X.npy", output_dir / "y.npy", output_dir / "subject_id.npy", output_dir / "metadata.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("processed target v1 dataset is ignored and not present in this checkout")

        dataset = ProcessedTargetDataset(output_dir)
        batch = torch.as_tensor(dataset.X[:2], dtype=torch.float32)
        model = build_model("lee_style_cnn_lstm_2d_v1", input_length=512, input_channels=18, num_classes=5)
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
