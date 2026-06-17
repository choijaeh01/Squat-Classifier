import pathlib
import sys
import unittest

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


V3_MODELS = {
    "channel_shared_posres_attention_v3",
    "modality_shared_sensorattn_v3",
}


class ModelCapacityV3Tests(unittest.TestCase):
    def test_v3_registry_models_forward_on_synthetic_target_shape(self):
        from models.registry import build_model, count_parameters, list_models

        registered = set(list_models())
        self.assertTrue(V3_MODELS.issubset(registered))
        x = torch.randn(3, 512, 18)
        for model_name in sorted(V3_MODELS):
            with self.subTest(model=model_name):
                model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                self.assertEqual(tuple(logits.shape), (3, 5))
                params = count_parameters(model)
                self.assertGreaterEqual(params, 8000)
                self.assertLessEqual(params, 32000)

    def test_v3_shared_encoder_identity(self):
        from models.registry import build_model

        channel_model = build_model("channel_shared_posres_attention_v3", input_length=512, input_channels=18, num_classes=5)
        self.assertTrue(all(ref is channel_model.shared_encoder for ref in channel_model.channel_encoder_refs))
        self.assertEqual(len({id(ref) for ref in channel_model.channel_encoder_refs}), 1)

        modality_model = build_model("modality_shared_sensorattn_v3", input_length=512, input_channels=18, num_classes=5)
        self.assertIsNot(modality_model.acc_encoder, modality_model.gyro_encoder)
        self.assertTrue(all(ref is modality_model.acc_encoder for ref in modality_model.acc_encoder_refs))
        self.assertTrue(all(ref is modality_model.gyro_encoder for ref in modality_model.gyro_encoder_refs))
        self.assertEqual(len({id(ref) for ref in modality_model.acc_encoder_refs}), 1)
        self.assertEqual(len({id(ref) for ref in modality_model.gyro_encoder_refs}), 1)

    def test_v3_parameter_groups_are_counted(self):
        from models.registry import build_model, count_parameter_groups

        for model_name in sorted(V3_MODELS):
            with self.subTest(model=model_name):
                model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
                groups = count_parameter_groups(model)
                self.assertGreater(groups["encoder_params"], 0)
                self.assertGreater(groups["aggregation_params"], 0)
                self.assertGreater(groups["head_params"], 0)
                self.assertGreaterEqual(groups["total_params"], 8000)
                self.assertLessEqual(groups["total_params"], 32000)


if __name__ == "__main__":
    unittest.main()
