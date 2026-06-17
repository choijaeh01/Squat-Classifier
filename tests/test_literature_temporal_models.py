import pathlib
import sys
import unittest

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


NEURAL_LITERATURE_MODELS = [
    "cnn_lstm_literature_v1",
    "cnn_gru_literature_v1",
    "rescnn_bigru_attention_lite_v1",
    "tcn_literature_v1",
    "lstm_only_literature_v1",
    "gru_only_literature_v1",
    "transformer_encoder_lite_v1",
]


class LiteratureTemporalModelTests(unittest.TestCase):
    def test_registry_models_forward_on_synthetic_target_shape(self):
        from models.registry import build_model, list_models

        registered = set(list_models())
        self.assertTrue(set(NEURAL_LITERATURE_MODELS).issubset(registered))
        x = torch.randn(2, 512, 18)
        for model_name in NEURAL_LITERATURE_MODELS:
            with self.subTest(model=model_name):
                model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                self.assertEqual(tuple(logits.shape), (2, 5))

    def test_registry_models_forward_on_real_processed_batch_if_available(self):
        from data.processed_loader import ProcessedTargetDataset
        from models.registry import build_model

        output_dir = PROJECT_ROOT / "data" / "target" / "processed" / "v1_manual_windows_resample512"
        required = [output_dir / "X.npy", output_dir / "y.npy", output_dir / "subject_id.npy", output_dir / "metadata.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("processed target v1 dataset is ignored and not present in this checkout")

        dataset = ProcessedTargetDataset(output_dir)
        batch = torch.as_tensor(dataset.X[:2], dtype=torch.float32)
        for model_name in NEURAL_LITERATURE_MODELS:
            with self.subTest(model=model_name):
                model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
                model.eval()
                with torch.no_grad():
                    logits = model(batch)
                self.assertEqual(tuple(logits.shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
