import copy
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ControlledComparisonConfigSafetyTests(unittest.TestCase):
    def test_controlled_comparison_config_matches_contract(self):
        from training.controlled_comparison_runner import (
            controlled_model_names,
            load_controlled_comparison_config,
            validate_controlled_comparison_safety,
        )

        config = load_controlled_comparison_config(PROJECT_ROOT / "configs" / "controlled_feature_extractor_comparison_v1.yaml")
        validate_controlled_comparison_safety(config)
        self.assertEqual(config["seeds"], [42, 123, 2025])
        self.assertEqual(len(controlled_model_names(config)), 14)
        self.assertEqual(len(config["model_groups"]["controlled_neural"]), 9)
        self.assertEqual(config["common_representation"]["representation_dim"], 64)
        self.assertTrue(config["safety"]["forbid_modifying_locked_matrix"])
        self.assertTrue(config["safety"]["forbid_modifying_literature_extension"])
        self.assertTrue(config["safety"]["forbid_modifying_v3_ablation"])

    def test_unsafe_controlled_comparison_config_is_rejected(self):
        from training.controlled_comparison_runner import load_controlled_comparison_config, validate_controlled_comparison_safety

        config = load_controlled_comparison_config(PROJECT_ROOT / "configs" / "controlled_feature_extractor_comparison_v1.yaml")
        for path, value in [
            (("augmentation", "enabled"), True),
            (("normalization", "per_window_zscore"), True),
            (("training", "loss"), "focal_loss"),
            (("safety", "forbid_modifying_locked_matrix"), False),
        ]:
            bad = copy.deepcopy(config)
            bad[path[0]][path[1]] = value
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_controlled_comparison_safety(bad)


if __name__ == "__main__":
    unittest.main()
