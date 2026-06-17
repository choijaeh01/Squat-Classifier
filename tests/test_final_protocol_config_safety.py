import copy
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FinalProtocolPilotSafetyTests(unittest.TestCase):
    def test_config_loads_and_matches_final_protocol_limits(self):
        from training.loso_runner import load_pilot_loso_config, validate_pilot_loso_safety

        config = load_pilot_loso_config(PROJECT_ROOT / "configs" / "final_protocol_pilot_v1.yaml")
        self.assertEqual(config["experiment_name"], "final_protocol_pilot_v1")
        self.assertEqual(config["split"]["type"], "loso_with_within_train_subject_stratified_validation")
        self.assertEqual(config["split"]["train_windows_per_subject_class"], 16)
        self.assertEqual(config["split"]["val_windows_per_subject_class"], 4)
        self.assertEqual(config["training"]["max_epochs"], 60)
        self.assertEqual(config["normalization"]["fit_scaler_on"], "train_indices_only")
        validate_pilot_loso_safety(config)

    def test_unsafe_final_protocol_config_is_rejected(self):
        from training.loso_runner import load_pilot_loso_config, validate_pilot_loso_safety

        config = load_pilot_loso_config(PROJECT_ROOT / "configs" / "final_protocol_pilot_v1.yaml")

        unsafe = copy.deepcopy(config)
        unsafe["training"]["max_epochs"] = 61
        with self.assertRaises(ValueError):
            validate_pilot_loso_safety(unsafe)

        unsafe = copy.deepcopy(config)
        unsafe["split"]["strict_index_disjoint_train_val"] = False
        with self.assertRaises(ValueError):
            validate_pilot_loso_safety(unsafe)

        unsafe = copy.deepcopy(config)
        unsafe["safety"]["forbid_full_matrix"] = False
        with self.assertRaises(ValueError):
            validate_pilot_loso_safety(unsafe)


if __name__ == "__main__":
    unittest.main()
