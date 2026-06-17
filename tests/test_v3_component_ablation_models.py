import pathlib
import sys
import unittest

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


ABLATION_MODELS = [
    "channel_shared_posres_attention_v3_no_residual",
    "channel_shared_posres_attention_v3_residual_only_mlp",
    "channel_shared_posres_attention_v3_no_identity",
    "channel_shared_posres_meanpool_v3_no_attention",
]


class V3ComponentAblationModelTests(unittest.TestCase):
    def test_ablation_models_forward_and_registry(self):
        from models.registry import build_model, list_models

        available = set(list_models())
        for model_name in ABLATION_MODELS:
            self.assertIn(model_name, available)
            model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
            model.eval()
            with torch.no_grad():
                logits = model(torch.randn(2, 512, 18))
            self.assertEqual(tuple(logits.shape), (2, 5), model_name)

    def test_ablation_component_removals_are_structural(self):
        from models.registry import build_model

        no_residual = build_model("channel_shared_posres_attention_v3_no_residual", input_length=512, input_channels=18, num_classes=5)
        self.assertTrue(hasattr(no_residual, "shared_encoder"))
        self.assertIn("attention_pool", no_residual.aggregation)
        self.assertNotIn("residual_projection", no_residual.aggregation)

        residual_only = build_model("channel_shared_posres_attention_v3_residual_only_mlp", input_length=512, input_channels=18, num_classes=5)
        self.assertFalse(hasattr(residual_only, "shared_encoder"))
        self.assertFalse(hasattr(residual_only, "aggregation"))

        no_identity = build_model("channel_shared_posres_attention_v3_no_identity", input_length=512, input_channels=18, num_classes=5)
        self.assertTrue(hasattr(no_identity, "shared_encoder"))
        self.assertIn("attention_pool", no_identity.aggregation)
        self.assertIn("residual_projection", no_identity.aggregation)
        for name in ("channel_embedding", "sensor_embedding", "modality_embedding", "axis_embedding"):
            self.assertNotIn(name, no_identity.aggregation)

        no_attention = build_model("channel_shared_posres_meanpool_v3_no_attention", input_length=512, input_channels=18, num_classes=5)
        self.assertTrue(hasattr(no_attention, "shared_encoder"))
        self.assertIn("residual_projection", no_attention.aggregation)
        self.assertNotIn("attention_pool", no_attention.aggregation)

    def test_ablation_models_forward_on_real_processed_batch_if_available(self):
        from data.processed_loader import ProcessedTargetDataset
        from models.registry import build_model

        output_dir = PROJECT_ROOT / "data" / "target" / "processed" / "v1_manual_windows_resample512"
        required = [output_dir / "X.npy", output_dir / "y.npy", output_dir / "subject_id.npy", output_dir / "metadata.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("processed target v1 dataset is ignored and not present in this checkout")

        dataset = ProcessedTargetDataset(output_dir)
        batch = torch.as_tensor(dataset.X[:2], dtype=torch.float32)
        for model_name in ABLATION_MODELS:
            model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
            model.eval()
            with torch.no_grad():
                logits = model(batch)
            self.assertEqual(tuple(logits.shape), (2, 5), model_name)


if __name__ == "__main__":
    unittest.main()
