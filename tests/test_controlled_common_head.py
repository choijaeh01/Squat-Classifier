import pathlib
import sys
import unittest

import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ControlledCommonHeadTests(unittest.TestCase):
    def test_common_head_outputs_logits_and_has_fixed_architecture(self):
        from models.common_head import CommonHeadConfig, CommonMLPClassifierHead

        config = CommonHeadConfig(representation_dim=64, hidden_dim=64, dropout=0.1, activation="relu", num_classes=5)
        head = CommonMLPClassifierHead(config)
        logits = head(torch.randn(4, 64))
        self.assertEqual(tuple(logits.shape), (4, 5))
        self.assertEqual(head.architecture_signature(), ("linear:64->64", "relu", "dropout:0.1", "linear:64->5"))

    def test_all_controlled_models_share_common_head_signature_and_param_count(self):
        from models.controlled_models import CONTROLLED_NEURAL_MODEL_NAMES
        from models.registry import build_model

        signatures = set()
        head_param_counts = set()
        for model_name in CONTROLLED_NEURAL_MODEL_NAMES:
            model = build_model(model_name, input_length=512, input_channels=18, num_classes=5)
            signatures.add(model.classifier.architecture_signature())
            head_param_counts.add(sum(parameter.numel() for parameter in model.classifier.parameters()))
        self.assertEqual(len(signatures), 1)
        self.assertEqual(len(head_param_counts), 1)
        self.assertEqual(next(iter(head_param_counts)), 4485)


if __name__ == "__main__":
    unittest.main()
