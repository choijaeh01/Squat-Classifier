import copy
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FullMatrixConfigSafetyTests(unittest.TestCase):
    def test_config_loads_and_matches_full_matrix_contract(self):
        from training.full_matrix_runner import load_full_matrix_config, validate_full_matrix_safety

        config = load_full_matrix_config(PROJECT_ROOT / "configs" / "full_supervised_matrix_v1.yaml")
        self.assertEqual(config["experiment_name"], "full_supervised_matrix_v1")
        self.assertEqual(config["seeds"], [42, 123, 2025])
        self.assertEqual(len(config["models"]), 7)
        self.assertEqual(config["split"]["type"], "loso_with_within_train_subject_stratified_validation")
        self.assertTrue(config["training"]["full_matrix_mode"])
        self.assertTrue(config["training"]["allow_full_training"])
        self.assertEqual(config["training"]["max_epochs"], 120)
        self.assertEqual(config["normalization"]["fit_scaler_on"], "train_indices_only")
        validate_full_matrix_safety(config)

    def test_unsafe_full_matrix_config_is_rejected(self):
        from training.full_matrix_runner import load_full_matrix_config, validate_full_matrix_safety

        config = load_full_matrix_config(PROJECT_ROOT / "configs" / "full_supervised_matrix_v1.yaml")
        unsafe = copy.deepcopy(config)
        unsafe["augmentation"]["enabled"] = True
        with self.assertRaises(ValueError):
            validate_full_matrix_safety(unsafe)

        unsafe = copy.deepcopy(config)
        unsafe["training"]["loss"] = "focal_loss"
        with self.assertRaises(ValueError):
            validate_full_matrix_safety(unsafe)

        unsafe = copy.deepcopy(config)
        unsafe["safety"]["require_confirm_full_matrix"] = False
        with self.assertRaises(ValueError):
            validate_full_matrix_safety(unsafe)


if __name__ == "__main__":
    unittest.main()
