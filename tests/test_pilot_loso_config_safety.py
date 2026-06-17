import copy
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class PilotLOSOSafetyTests(unittest.TestCase):
    def test_config_loads_and_matches_pilot_limits(self):
        from training.loso_runner import load_pilot_loso_config, validate_pilot_loso_safety

        config = load_pilot_loso_config(PROJECT_ROOT / "configs" / "pilot_loso_v1.yaml")
        self.assertEqual(config["experiment_name"], "pilot_loso_v1")
        self.assertEqual(config["split"]["validation_policy"], "next_subject_cyclic")
        self.assertEqual(config["split"]["subjects"], [1, 2, 3, 4, 5, 6])
        self.assertTrue(config["training"]["pilot_mode"])
        self.assertFalse(config["training"]["allow_full_training"])
        self.assertEqual(config["training"]["max_epochs"], 30)
        self.assertFalse(config["augmentation"]["enabled"])
        validate_pilot_loso_safety(config)

    def test_unsafe_pilot_config_is_rejected(self):
        from training.loso_runner import load_pilot_loso_config, validate_pilot_loso_safety

        config = load_pilot_loso_config(PROJECT_ROOT / "configs" / "pilot_loso_v1.yaml")
        unsafe = copy.deepcopy(config)
        unsafe["training"]["max_epochs"] = 31
        with self.assertRaises(ValueError):
            validate_pilot_loso_safety(unsafe)

        unsafe = copy.deepcopy(config)
        unsafe["augmentation"]["enabled"] = True
        with self.assertRaises(ValueError):
            validate_pilot_loso_safety(unsafe)

        unsafe = copy.deepcopy(config)
        unsafe["training"]["loss"] = "focal_loss"
        with self.assertRaises(ValueError):
            validate_pilot_loso_safety(unsafe)


if __name__ == "__main__":
    unittest.main()
