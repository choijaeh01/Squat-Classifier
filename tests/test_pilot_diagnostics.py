import csv
import pathlib
import sys
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class PilotDiagnosticsTests(unittest.TestCase):
    def test_analyze_pilot_run_writes_collapse_and_class3_outputs(self):
        from analysis.pilot_diagnostics import analyze_pilot_loso_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp)
            _write_csv(
                run_dir / "fold_metrics.csv",
                [
                    {
                        "model_name": "shared",
                        "seed": 42,
                        "fold_id": 1,
                        "train_subjects": "3|4|5|6",
                        "val_subject": 2,
                        "test_subject": 1,
                        "n_train": 8,
                        "n_val": 4,
                        "n_test": 10,
                        "best_epoch": 2,
                        "best_val_macro_f1": 0.2,
                        "test_accuracy": 0.2,
                        "test_macro_f1": 0.1,
                        "leakage_check_passed": True,
                        "scaler_fit_subjects": "3|4|5|6",
                        "status": "ok",
                    }
                ],
            )
            _write_csv(
                run_dir / "training_history.csv",
                [
                    {
                        "model_name": "shared",
                        "seed": 42,
                        "fold_id": 1,
                        "epoch": 2,
                        "train_loss": 0.1,
                        "val_loss": 1.0,
                        "train_accuracy": 1.0,
                        "train_macro_f1": 1.0,
                        "val_accuracy": 0.2,
                        "val_macro_f1": 0.2,
                        "is_best": True,
                    }
                ],
            )
            predictions = []
            labels = [0, 1, 2, 3, 4] * 2
            for index, label in enumerate(labels):
                predictions.append(
                    {
                        "model_name": "shared",
                        "seed": 42,
                        "fold_id": 1,
                        "test_subject": 1,
                        "sample_index": index,
                        "subject_id": 1,
                        "y_true": label,
                        "y_pred": 1,
                    }
                )
            _write_csv(run_dir / "predictions.csv", predictions)
            _write_csv(run_dir / "classwise_metrics.csv", [])
            _write_csv(run_dir / "subjectwise_metrics_by_model.csv", [])

            result = analyze_pilot_loso_run(run_dir)

            diagnostics_dir = pathlib.Path(result["diagnostics_dir"])
            self.assertTrue((diagnostics_dir / "prediction_distribution_by_model.csv").exists())
            self.assertTrue((diagnostics_dir / "class3_error_analysis.csv").exists())
            self.assertTrue((diagnostics_dir / "collapse_diagnosis.csv").exists())
            collapse_rows = _read_csv(diagnostics_dir / "collapse_diagnosis.csv")
            self.assertEqual(collapse_rows[0]["collapse_flag"], "True")
            self.assertEqual(collapse_rows[0]["dominant_pred_class"], "1")
            class3_rows = _read_csv(diagnostics_dir / "class3_error_analysis.csv")
            pred1_row = next(row for row in class3_rows if row["model_name"] == "shared" and row["pred_class"] == "1")
            self.assertEqual(pred1_row["count"], "2")


def _write_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
