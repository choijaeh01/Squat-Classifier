import copy
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class V3ComponentAblationConfigSafetyTests(unittest.TestCase):
    def test_config_loads_and_matches_contract(self):
        from training.v3_ablation_runner import load_v3_ablation_config, validate_v3_ablation_safety

        config = load_v3_ablation_config(PROJECT_ROOT / "configs" / "v3_component_ablation_v1.yaml")
        validate_v3_ablation_safety(config)
        self.assertEqual(config["seeds"], [42, 123, 2025])
        self.assertEqual(len(config["models"]), 4)
        self.assertTrue(config["safety"]["forbid_modifying_locked_matrix"])
        self.assertTrue(config["safety"]["forbid_modifying_literature_extension"])
        self.assertEqual(config["training"]["loss"], "cross_entropy")

    def test_unsafe_config_is_rejected(self):
        from training.v3_ablation_runner import load_v3_ablation_config, validate_v3_ablation_safety

        config = load_v3_ablation_config(PROJECT_ROOT / "configs" / "v3_component_ablation_v1.yaml")
        bad = copy.deepcopy(config)
        bad["augmentation"]["enabled"] = True
        with self.assertRaises(ValueError):
            validate_v3_ablation_safety(bad)

        bad = copy.deepcopy(config)
        bad["normalization"]["per_window_zscore"] = True
        with self.assertRaises(ValueError):
            validate_v3_ablation_safety(bad)

        bad = copy.deepcopy(config)
        bad["safety"]["forbid_modifying_literature_extension"] = False
        with self.assertRaises(ValueError):
            validate_v3_ablation_safety(bad)


if __name__ == "__main__":
    unittest.main()
