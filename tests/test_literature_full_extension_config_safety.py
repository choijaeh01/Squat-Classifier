import copy
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class LiteratureFullExtensionConfigSafetyTests(unittest.TestCase):
    def test_full_extension_config_loads_and_matches_contract(self):
        from training.literature_full_extension_runner import load_literature_full_extension_config, validate_literature_full_extension_safety

        config = load_literature_full_extension_config(PROJECT_ROOT / "configs" / "literature_baseline_full_extension_v1.yaml")
        validate_literature_full_extension_safety(config)
        self.assertEqual(config["seeds"], [42, 123, 2025])
        self.assertEqual(len(config["models"]), 8)
        self.assertIn("lee_style_cnn_lstm_2d_v1", config["models"])
        self.assertTrue(config["safety"]["forbid_modifying_locked_matrix"])

    def test_unsafe_full_extension_config_is_rejected(self):
        from training.literature_full_extension_runner import load_literature_full_extension_config, validate_literature_full_extension_safety

        config = load_literature_full_extension_config(PROJECT_ROOT / "configs" / "literature_baseline_full_extension_v1.yaml")
        bad = copy.deepcopy(config)
        bad["augmentation"]["enabled"] = True
        with self.assertRaises(ValueError):
            validate_literature_full_extension_safety(bad)

        bad = copy.deepcopy(config)
        bad["normalization"]["per_window_zscore"] = True
        with self.assertRaises(ValueError):
            validate_literature_full_extension_safety(bad)

        bad = copy.deepcopy(config)
        bad["safety"]["forbid_modifying_locked_matrix"] = False
        with self.assertRaises(ValueError):
            validate_literature_full_extension_safety(bad)


if __name__ == "__main__":
    unittest.main()
