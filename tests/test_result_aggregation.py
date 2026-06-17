import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ResultAggregationTests(unittest.TestCase):
    def test_aggregate_fold_metrics_by_model(self):
        from training.aggregation import aggregate_metrics_by_model

        fold_rows = [
            {
                "model_name": "m1",
                "seed": 42,
                "test_accuracy": 0.5,
                "test_macro_f1": 0.4,
                "best_epoch": 2,
                "status": "ok",
            },
            {
                "model_name": "m1",
                "seed": 42,
                "test_accuracy": 0.7,
                "test_macro_f1": 0.6,
                "best_epoch": 4,
                "status": "ok",
            },
            {
                "model_name": "m2",
                "seed": 42,
                "test_accuracy": "",
                "test_macro_f1": "",
                "best_epoch": "",
                "status": "failed",
            },
        ]
        parameter_rows = [
            {
                "model_name": "m1",
                "total_params": 10,
                "encoder_params": 6,
                "aggregation_params": 1,
                "head_params": 3,
            },
            {
                "model_name": "m2",
                "total_params": 20,
                "encoder_params": 12,
                "aggregation_params": 2,
                "head_params": 6,
            },
        ]

        rows = aggregate_metrics_by_model(fold_rows, parameter_rows)
        by_model = {row["model_name"]: row for row in rows}
        self.assertAlmostEqual(by_model["m1"]["mean_test_accuracy"], 0.6)
        self.assertAlmostEqual(by_model["m1"]["std_test_accuracy"], 0.1)
        self.assertAlmostEqual(by_model["m1"]["mean_test_macro_f1"], 0.5)
        self.assertEqual(by_model["m1"]["n_success_folds"], 2)
        self.assertEqual(by_model["m1"]["n_failed_folds"], 0)
        self.assertEqual(by_model["m1"]["total_params"], 10)
        self.assertEqual(by_model["m2"]["n_success_folds"], 0)
        self.assertEqual(by_model["m2"]["n_failed_folds"], 1)


if __name__ == "__main__":
    unittest.main()
