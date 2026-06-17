import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


EXPECTED_RANGES = {
    "cnn_lstm_literature_v1": (10_000, 80_000),
    "cnn_gru_literature_v1": (10_000, 80_000),
    "rescnn_bigru_attention_lite_v1": (30_000, 150_000),
    "tcn_literature_v1": (10_000, 100_000),
    "lstm_only_literature_v1": (5_000, 80_000),
    "gru_only_literature_v1": (5_000, 80_000),
    "transformer_encoder_lite_v1": (20_000, 150_000),
}


class LiteratureBaselineCapacityTests(unittest.TestCase):
    def test_neural_literature_parameter_ranges(self):
        from models.registry import build_model, count_parameter_groups

        for model_name, (lower, upper) in EXPECTED_RANGES.items():
            with self.subTest(model=model_name):
                groups = count_parameter_groups(build_model(model_name, input_length=512, input_channels=18, num_classes=5))
                self.assertGreaterEqual(groups["total_params"], lower)
                self.assertLessEqual(groups["total_params"], upper)
                self.assertGreater(groups["head_params"], 0)


if __name__ == "__main__":
    unittest.main()
