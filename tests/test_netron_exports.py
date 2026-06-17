from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class NetronExportTests(unittest.TestCase):
    def test_default_model_list_matches_full_matrix_models(self):
        from analysis.netron_exports import DEFAULT_NETRON_MODELS

        self.assertEqual(
            DEFAULT_NETRON_MODELS,
            [
                "all_channel_conv1d_v1",
                "all_channel_conv1d_small",
                "cnn2d_baseline_v1",
                "channel_shared_meanpool_v2",
                "channel_shared_attentionpool_v2",
                "channel_shared_posres_attention_v3",
                "modality_shared_sensorattn_v3",
            ],
        )

    def test_export_one_torchscript_model_for_netron(self):
        from analysis.netron_exports import export_torchscript_models

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            rows = export_torchscript_models(
                output_dir=output_dir,
                model_names=["all_channel_conv1d_small"],
                batch_size=1,
                input_length=512,
                input_channels=18,
                num_classes=5,
                seed=20260618,
            )

            self.assertEqual(len(rows), 1)
            artifact_path = Path(rows[0]["artifact_path"])
            self.assertTrue(artifact_path.exists())
            self.assertEqual(artifact_path.name, "all_channel_conv1d_small.torchscript.pt")
            loaded = torch.jit.load(str(artifact_path))
            loaded.eval()
            with torch.no_grad():
                logits = loaded(torch.randn(1, 512, 18))
            self.assertEqual(tuple(logits.shape), (1, 5))

            index_path = output_dir / "netron_model_index.csv"
            self.assertTrue(index_path.exists())
            with index_path.open("r", encoding="utf-8", newline="") as handle:
                index_rows = list(csv.DictReader(handle))
            self.assertEqual(index_rows[0]["model_name"], "all_channel_conv1d_small")
            self.assertEqual(index_rows[0]["file_format"], "TorchScript")


if __name__ == "__main__":
    unittest.main()
