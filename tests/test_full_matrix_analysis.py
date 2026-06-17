import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FullMatrixAnalysisTests(unittest.TestCase):
    def test_aggregate_and_paired_difference_outputs_expected_columns(self):
        from analysis.full_matrix_analysis import (
            aggregate_full_matrix_by_model,
            aggregate_full_matrix_by_model_seed,
            compute_paired_model_differences,
        )

        fold_rows = [
            {
                "model_name": "ref",
                "seed": seed,
                "fold_id": fold,
                "test_accuracy": 0.7 + 0.01 * fold,
                "test_macro_f1": 0.6 + 0.01 * fold,
                "test_weighted_f1": 0.65 + 0.01 * fold,
                "best_epoch": 5,
                "status": "ok",
            }
            for seed in [1, 2]
            for fold in [1, 2]
        ] + [
            {
                "model_name": "cmp",
                "seed": seed,
                "fold_id": fold,
                "test_accuracy": 0.8 + 0.01 * fold,
                "test_macro_f1": 0.7 + 0.01 * fold,
                "test_weighted_f1": 0.75 + 0.01 * fold,
                "best_epoch": 6,
                "status": "ok",
            }
            for seed in [1, 2]
            for fold in [1, 2]
        ]
        parameter_rows = [
            {"model_name": "ref", "total_params": 10, "encoder_params": 6, "aggregation_params": 0, "head_params": 4},
            {"model_name": "cmp", "total_params": 20, "encoder_params": 8, "aggregation_params": 8, "head_params": 4},
        ]

        by_seed = aggregate_full_matrix_by_model_seed(fold_rows)
        by_model = aggregate_full_matrix_by_model(fold_rows, parameter_rows, bootstrap_n=100, seed=0)
        diffs = compute_paired_model_differences(fold_rows, ["ref"], metric="macro_f1", bootstrap_n=100, seed=0)

        self.assertEqual(len(by_seed), 4)
        self.assertEqual(len(by_model), 2)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["reference_model"], "ref")
        self.assertEqual(diffs[0]["comparison_model"], "cmp")
        self.assertGreater(float(diffs[0]["mean_difference"]), 0.0)
        self.assertIn("macro_f1_ci_low", by_model[0])


if __name__ == "__main__":
    unittest.main()
