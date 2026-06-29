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


class ControlledFeatureExtractorTests(unittest.TestCase):
    def test_extractors_return_common_representation_dimension(self):
        from models.registry import build_model

        x = torch.randn(3, 512, 18)
        for model_name in CONTROLLED_MODELS:
            with self.subTest(model=model_name):
                model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
                representation = model.extract_features(x)
                self.assertEqual(tuple(representation.shape), (3, 64))
                self.assertEqual(model.representation_dim, 64)

    def test_shared_extractors_reuse_single_encoder_object(self):
        from models.registry import build_model

        for model_name in [
            "controlled_shared_1d",
            "controlled_shared_1d_identity",
            "controlled_shared_1d_residual",
            "controlled_shared_1d_residual_identity",
        ]:
            with self.subTest(model=model_name):
                model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
                refs = model.extractor.channel_encoder_refs
                self.assertEqual(len(refs), 18)
                self.assertEqual(len({id(ref) for ref in refs}), 1)
                self.assertTrue(all(ref is model.extractor.shared_encoder for ref in refs))

    def test_identity_extractors_use_approved_channel_metadata(self):
        from models.channel_metadata import CHANNEL_ORDER, axis_indices, modality_indices, sensor_indices
        from models.registry import build_model

        self.assertEqual(CHANNEL_ORDER[0:6], ["s0_ax", "s0_ay", "s0_az", "s0_gx", "s0_gy", "s0_gz"])
        model = build_model("controlled_shared_1d_residual_identity", input_length=512, input_channels=18, num_classes=5)
        extractor = model.extractor
        self.assertEqual(extractor.channel_ids.tolist(), list(range(18)))
        self.assertEqual(extractor.sensor_ids.tolist(), sensor_indices())
        self.assertEqual(extractor.modality_ids.tolist(), modality_indices())
        self.assertEqual(extractor.axis_ids.tolist(), axis_indices())

    def test_residual_branch_uses_signal_summary_only(self):
        from models.controlled_extractors import raw_summary_features

        x = torch.randn(2, 512, 18)
        features = raw_summary_features(x)
        self.assertEqual(tuple(features.shape), (2, 72))
        shifted = x + 1.0
        shifted_features = raw_summary_features(shifted)
        self.assertFalse(torch.allclose(features, shifted_features))


if __name__ == "__main__":
    unittest.main()
