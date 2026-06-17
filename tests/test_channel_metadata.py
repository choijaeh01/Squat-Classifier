import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class ChannelMetadataTests(unittest.TestCase):
    def test_target_channel_metadata_matches_approved_order(self):
        from models.channel_metadata import (
            CHANNEL_ORDER,
            TARGET_CHANNELS,
            axis_indices,
            modality_indices,
            sensor_indices,
            validate_channel_order,
        )

        self.assertEqual(
            CHANNEL_ORDER,
            [
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
            ],
        )
        self.assertEqual(len(TARGET_CHANNELS), 18)
        self.assertEqual(sensor_indices(), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        self.assertEqual(modality_indices(), [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        self.assertEqual(axis_indices(), [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        validate_channel_order(CHANNEL_ORDER)
        with self.assertRaises(ValueError):
            validate_channel_order(CHANNEL_ORDER[:-1] + ["s2_bad"])


if __name__ == "__main__":
    unittest.main()
