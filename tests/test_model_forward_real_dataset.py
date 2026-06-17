import pathlib
import sys
import unittest

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ModelForwardRealDatasetTests(unittest.TestCase):
    def test_v2_models_forward_one_real_processed_batch_without_training(self):
        from data.processed_loader import ProcessedTargetDataset
        from models.registry import build_model

        output_dir = PROJECT_ROOT / "data" / "target" / "processed" / "v1_manual_windows_resample512"
        required = [output_dir / "X.npy", output_dir / "y.npy", output_dir / "subject_id.npy", output_dir / "metadata.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("processed target v1 dataset is ignored and not present in this checkout")

        dataset = ProcessedTargetDataset(output_dir)
        batch = torch.as_tensor(dataset.X[:4], dtype=torch.float32)
        for model_name in [
            "all_channel_conv1d_v1",
            "all_channel_conv1d_small",
            "channel_shared_meanpool_v2",
            "channel_shared_attentionpool_v2",
            "modality_shared_meanpool_v2",
            "cnn2d_baseline_v1",
        ]:
            with self.subTest(model=model_name):
                model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
                model.eval()
                with torch.no_grad():
                    logits = model(batch)
                self.assertEqual(tuple(logits.shape), (4, 5))


if __name__ == "__main__":
    unittest.main()
