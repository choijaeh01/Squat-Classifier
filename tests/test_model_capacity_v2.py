import pathlib
import sys
import unittest

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


V2_MODELS = {
    "all_channel_conv1d_v1",
    "all_channel_conv1d_small",
    "channel_shared_meanpool_v2",
    "channel_shared_attentionpool_v2",
    "modality_shared_meanpool_v2",
    "cnn2d_baseline_v1",
}


class ModelCapacityV2Tests(unittest.TestCase):
    def test_v2_registry_models_forward_on_synthetic_target_shape(self):
        from models.registry import build_model, count_parameters, list_models

        registered = set(list_models())
        self.assertTrue(V2_MODELS.issubset(registered))
        x = torch.randn(3, 512, 18)
        for model_name in sorted(V2_MODELS):
            with self.subTest(model=model_name):
                model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                self.assertEqual(tuple(logits.shape), (3, 5))
                self.assertGreater(count_parameters(model), 0)

    def test_v2_shared_encoder_object_identity(self):
        from models.registry import build_model

        channel_model = build_model("channel_shared_meanpool_v2", input_length=512, input_channels=18, num_classes=5)
        self.assertTrue(all(ref is channel_model.shared_encoder for ref in channel_model.channel_encoder_refs))
        self.assertEqual(len({id(ref) for ref in channel_model.channel_encoder_refs}), 1)

        attention_model = build_model("channel_shared_attentionpool_v2", input_length=512, input_channels=18, num_classes=5)
        self.assertTrue(all(ref is attention_model.shared_encoder for ref in attention_model.channel_encoder_refs))
        self.assertEqual(len({id(ref) for ref in attention_model.channel_encoder_refs}), 1)

        modality_model = build_model("modality_shared_meanpool_v2", input_length=512, input_channels=18, num_classes=5)
        self.assertIsNot(modality_model.acc_encoder, modality_model.gyro_encoder)
        self.assertTrue(all(ref is modality_model.acc_encoder for ref in modality_model.acc_encoder_refs))
        self.assertTrue(all(ref is modality_model.gyro_encoder for ref in modality_model.gyro_encoder_refs))
        self.assertEqual(len({id(ref) for ref in modality_model.acc_encoder_refs}), 1)
        self.assertEqual(len({id(ref) for ref in modality_model.gyro_encoder_refs}), 1)

    def test_v2_parameter_budgets_reduce_flatten_head_capacity(self):
        from models.registry import build_model, count_parameter_groups, count_parameters

        all_channel_v1 = build_model("all_channel_conv1d_v1", input_length=512, input_channels=18, num_classes=5)
        old_shared = build_model("channel_shared_1d_encoder", input_length=512, input_channels=18, num_classes=5)
        meanpool = build_model("channel_shared_meanpool_v2", input_length=512, input_channels=18, num_classes=5)
        attention = build_model("channel_shared_attentionpool_v2", input_length=512, input_channels=18, num_classes=5)
        modality = build_model("modality_shared_meanpool_v2", input_length=512, input_channels=18, num_classes=5)

        self.assertLess(count_parameters(meanpool), count_parameters(old_shared))
        self.assertLessEqual(count_parameters(meanpool), count_parameters(all_channel_v1))
        self.assertLessEqual(count_parameters(attention), count_parameters(all_channel_v1))
        self.assertLessEqual(count_parameters(modality), count_parameters(all_channel_v1))

        groups = count_parameter_groups(meanpool)
        self.assertEqual(groups["total_params"], count_parameters(meanpool))
        self.assertGreater(groups["encoder_params"], 0)
        self.assertEqual(groups["aggregation_params"], 0)
        self.assertGreater(groups["head_params"], 0)

        attention_groups = count_parameter_groups(attention)
        self.assertGreater(attention_groups["aggregation_params"], 0)
        self.assertGreater(attention_groups["head_params"], 0)


if __name__ == "__main__":
    unittest.main()
