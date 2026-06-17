import pathlib
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ProcessedTargetLoaderTests(unittest.TestCase):
    def test_processed_dataset_and_loso_splits_from_arrays(self):
        from data.processed_loader import ProcessedTargetDataset, make_loso_splits

        with self.subTest("synthetic processed arrays"):
            X = np.zeros((6, 512, 18), dtype=np.float32)
            y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
            subject_id = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
            dataset = ProcessedTargetDataset.from_arrays(X, y, subject_id)

            self.assertEqual(len(dataset), 6)
            item = dataset[0]
            self.assertEqual(tuple(item["x"].shape), (512, 18))
            self.assertEqual(item["x"].dtype, torch.float32)
            self.assertEqual(item["y"], 0)
            self.assertEqual(item["subject_id"], 1)

            splits = make_loso_splits(subject_id)
            self.assertEqual(set(splits), {1, 2, 3})
            for held_out, split in splits.items():
                train_subjects = set(subject_id[split["train_idx"]].tolist())
                test_subjects = set(subject_id[split["test_idx"]].tolist())
                self.assertNotIn(held_out, train_subjects)
                self.assertEqual(test_subjects, {held_out})

    def test_real_processed_dataset_smoke_if_available(self):
        from data.processed_loader import ProcessedTargetDataset, make_loso_splits
        from models.registry import build_model

        output_dir = PROJECT_ROOT / "data" / "target" / "processed" / "v1_manual_windows_resample512"
        required = [output_dir / "X.npy", output_dir / "y.npy", output_dir / "subject_id.npy", output_dir / "metadata.csv"]
        if not all(path.exists() for path in required):
            self.skipTest("processed target v1 dataset has not been generated yet")

        dataset = ProcessedTargetDataset(output_dir)
        self.assertEqual(dataset.X.shape[1:], (512, 18))
        self.assertEqual(dataset.X.dtype, np.float32)
        self.assertEqual(dataset.y.dtype, np.int64)
        self.assertEqual(dataset.subject_id.shape, dataset.y.shape)
        self.assertEqual(set(dataset.y.tolist()), {0, 1, 2, 3, 4})

        splits = make_loso_splits(dataset.subject_id)
        self.assertEqual(set(splits), set(np.unique(dataset.subject_id).tolist()))
        for held_out, split in splits.items():
            self.assertFalse(np.any(dataset.subject_id[split["train_idx"]] == held_out))
            self.assertTrue(np.all(dataset.subject_id[split["test_idx"]] == held_out))

        model = build_model("all_channel_conv1d", input_length=512, input_channels=18, num_classes=5)
        model.eval()
        batch = torch.as_tensor(dataset.X[:4], dtype=torch.float32)
        with torch.no_grad():
            logits = model(batch)
        self.assertEqual(tuple(logits.shape), (4, 5))


if __name__ == "__main__":
    unittest.main()
