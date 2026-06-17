import copy
import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class LiteratureBaselineConfigSafetyTests(unittest.TestCase):
    def test_screening_config_loads_and_enforces_one_seed(self):
        from training.literature_screening_runner import load_literature_screening_config, validate_literature_screening_safety

        config = load_literature_screening_config(PROJECT_ROOT / "configs" / "literature_baseline_screening_v1.yaml")
        validate_literature_screening_safety(config)
        self.assertEqual(config["seeds"], [42])
        self.assertFalse(config["augmentation"]["enabled"])
        self.assertTrue(config["safety"]["forbid_full_matrix"])

    def test_unsafe_screening_config_is_rejected(self):
        from training.literature_screening_runner import load_literature_screening_config, validate_literature_screening_safety

        config = load_literature_screening_config(PROJECT_ROOT / "configs" / "literature_baseline_screening_v1.yaml")
        bad = copy.deepcopy(config)
        bad["seeds"] = [42, 123]
        with self.assertRaises(ValueError):
            validate_literature_screening_safety(bad)

        bad = copy.deepcopy(config)
        bad["augmentation"]["enabled"] = True
        with self.assertRaises(ValueError):
            validate_literature_screening_safety(bad)

        bad = copy.deepcopy(config)
        bad["training"]["allow_full_training"] = True
        with self.assertRaises(ValueError):
            validate_literature_screening_safety(bad)


if __name__ == "__main__":
    unittest.main()
