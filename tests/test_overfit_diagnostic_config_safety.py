import copy
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class OverfitDiagnosticSafetyTests(unittest.TestCase):
    def test_config_loads_and_enforces_diagnostic_safety(self):
        from training.overfit_diagnostic import load_overfit_config, validate_overfit_safety

        config = load_overfit_config(PROJECT_ROOT / "configs" / "overfit_diagnostic_v1.yaml")
        self.assertEqual(config["experiment_name"], "overfit_diagnostic_v1")
        self.assertTrue(config["training"]["diagnostic_mode"])
        self.assertFalse(config["training"]["allow_generalization_claims"])
        self.assertEqual(config["data"]["samples_per_class"], 8)
        self.assertEqual(config["data"]["total_samples"], 40)
        self.assertFalse(config["augmentation"]["enabled"])
        validate_overfit_safety(config)

    def test_unsafe_overfit_config_is_rejected(self):
        from training.overfit_diagnostic import load_overfit_config, validate_overfit_safety

        config = load_overfit_config(PROJECT_ROOT / "configs" / "overfit_diagnostic_v1.yaml")
        unsafe = copy.deepcopy(config)
        unsafe["augmentation"]["enabled"] = True
        with self.assertRaises(ValueError):
            validate_overfit_safety(unsafe)

        unsafe = copy.deepcopy(config)
        unsafe["training"]["loss"] = "focal_loss"
        with self.assertRaises(ValueError):
            validate_overfit_safety(unsafe)

        unsafe = copy.deepcopy(config)
        unsafe["training"]["allow_generalization_claims"] = True
        with self.assertRaises(ValueError):
            validate_overfit_safety(unsafe)


if __name__ == "__main__":
    unittest.main()
