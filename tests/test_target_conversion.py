import csv
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import yaml


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CHANNEL_ORDER = [
    "s0_ax",
    "s0_ay",
    "s0_az",
    "s0_gx",
    "s0_gy",
    "s0_gz",
    "s1_ax",
    "s1_ay",
    "s1_az",
    "s1_gx",
    "s1_gy",
    "s1_gz",
    "s2_ax",
    "s2_ay",
    "s2_az",
    "s2_gx",
    "s2_gy",
    "s2_gz",
]


class TargetConversionTests(unittest.TestCase):
    def test_linear_phase_resampling_preserves_endpoints(self):
        from data.target_conversion import resample_window

        window = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]], dtype=np.float32)
        out = resample_window(window, target_length=5)

        self.assertEqual(out.shape, (5, 2))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.allclose(out[0], window[0]))
        self.assertTrue(np.allclose(out[-1], window[-1]))
        self.assertTrue(np.allclose(out[:, 0], np.linspace(0.0, 2.0, 5)))

    def test_conversion_writes_arrays_metadata_and_checksums(self):
        from data.target_conversion import convert_target_dataset, load_conversion_config

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            raw_root = root / "raw" / "labeled"
            metadata_root = root / "manually_labeled"
            reference_root = metadata_root
            output_dir = root / "processed"
            raw_dir = raw_root / "class0"
            meta_dir = metadata_root / "class0"
            raw_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)

            raw_path = raw_dir / "subject1_set1.csv"
            write_synthetic_raw_csv(raw_path, rows=12)
            metadata_path = meta_dir / "subject1_set1_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "source_file": "raw/labeled/class0/subject1_set1.csv",
                        "num_windows": 2,
                        "sampling_rate": 100,
                        "method": "manual",
                        "windows": [
                            {
                                "window_id": 0,
                                "start_sample": 0,
                                "end_sample": 4,
                                "num_samples": 4,
                                "start_time": 0.0,
                                "end_time": 0.03,
                            },
                            {
                                "window_id": 1,
                                "start_sample": 4,
                                "end_sample": 10,
                                "num_samples": 6,
                                "start_time": 0.04,
                                "end_time": 0.09,
                            },
                        ],
                    }
                )
            )
            write_reference_window(meta_dir / "subject1_set1_window0.csv", start=0, rows=4)
            write_reference_window(meta_dir / "subject1_set1_window1.csv", start=4, rows=6)

            config_path = root / "conversion.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "raw_root": str(raw_root),
                        "metadata_root": str(metadata_root),
                        "reference_window_root": str(reference_root),
                        "output_dir": str(output_dir),
                        "class_mapping": {"class0": {"id": 0, "name": "Correct"}},
                        "sensor_mapping": {
                            "s0": "lower back / waist / 허리",
                            "s1": "right thigh / 오른쪽 허벅지",
                            "s2": "right calf / 오른쪽 종아리",
                        },
                        "channel_order": CHANNEL_ORDER,
                        "csv_column_mapping": {column: column for column in CHANNEL_ORDER},
                        "target_length": 8,
                        "resampling": "linear_phase_interpolation",
                        "normalization": "none",
                        "expected_num_subjects": 1,
                        "expected_num_classes": 1,
                        "expected_total_windows": 2,
                        "expected_windows_per_subject_class": 2,
                        "strict_mode": True,
                    },
                    allow_unicode=True,
                    sort_keys=False,
                )
            )

            summary = convert_target_dataset(load_conversion_config(config_path))

            self.assertEqual(summary["num_samples"], 2)
            X = np.load(output_dir / "X.npy")
            y = np.load(output_dir / "y.npy")
            subject_id = np.load(output_dir / "subject_id.npy")
            self.assertEqual(X.shape, (2, 8, 18))
            self.assertEqual(X.dtype, np.float32)
            self.assertEqual(y.dtype, np.int64)
            self.assertEqual(subject_id.dtype, np.int64)
            self.assertTrue((y == np.array([0, 0])).all())
            self.assertTrue((subject_id == np.array([1, 1])).all())
            self.assertTrue((output_dir / "metadata.csv").exists())
            self.assertTrue((output_dir / "checksums.json").exists())
            self.assertTrue((output_dir / "conversion_config.yaml").exists())
            self.assertTrue((output_dir / "conversion_log.txt").exists())

            with (output_dir / "metadata.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["validation_status"], "ok")
            self.assertEqual(rows[0]["resampling_method"], "linear_phase_interpolation")
            self.assertIn("s0_ax", rows[0]["channel_order"])


def write_synthetic_raw_csv(path: pathlib.Path, rows: int) -> None:
    header = ["timestamp", "seq", "millis", "q0", "q1", "q2"] + CHANNEL_ORDER
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(rows):
            values = [
                f"2025-01-01T00:00:{index:02d}.000",
                index,
                index * 10,
                0,
                0,
                0,
            ]
            values.extend(index + channel_index / 100.0 for channel_index in range(len(CHANNEL_ORDER)))
            writer.writerow(values)


def write_reference_window(path: pathlib.Path, start: int, rows: int) -> None:
    header = ["timestamp", "seq", "millis", "q0", "q1", "q2"] + CHANNEL_ORDER
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for offset in range(rows):
            index = start + offset
            values = [
                f"2025-01-01T00:00:{index:02d}.000",
                index,
                index * 10,
                0,
                0,
                0,
            ]
            values.extend(index + channel_index / 100.0 for channel_index in range(len(CHANNEL_ORDER)))
            writer.writerow(values)


if __name__ == "__main__":
    unittest.main()
