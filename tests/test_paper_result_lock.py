import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

RUN_DIR = PROJECT_ROOT / "results" / "full_supervised_matrix" / "20260617_144309_full_supervised_matrix_v1"


class PaperResultLockTests(unittest.TestCase):
    def test_integrity_check_accepts_full_supervised_matrix_v1(self):
        from analysis.paper_result_lock import validate_result_integrity

        if not RUN_DIR.exists():
            self.skipTest("full supervised matrix result is not available")
        result = validate_result_integrity(RUN_DIR)
        self.assertTrue(result["all_checks_passed"])
        self.assertEqual(result["run_plan_total"], 126)
        self.assertEqual(result["fold_metric_success_runs"], 126)
        self.assertEqual(result["failed_run_count"], 0)
        self.assertTrue(result["all_leakage_checks_passed"])
        self.assertTrue(result["all_scaler_leakage_checks_passed"])

    def test_lock_outputs_include_main_tables(self):
        from analysis.paper_result_lock import lock_paper_results

        if not RUN_DIR.exists():
            self.skipTest("full supervised matrix result is not available")
        output = lock_paper_results(RUN_DIR)
        self.assertTrue((output / "result_integrity_check.json").exists())
        self.assertTrue((output / "tables" / "table3_main_model_comparison.csv").exists())
        self.assertTrue((output / "figures" / "fig1_model_macro_f1_ci.png").exists())
        self.assertTrue((output / "figure_captions.md").exists())


if __name__ == "__main__":
    unittest.main()
