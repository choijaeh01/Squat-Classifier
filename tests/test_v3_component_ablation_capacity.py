import pathlib
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class V3ComponentAblationCapacityTests(unittest.TestCase):
    def test_capacity_audit_reports_component_groups(self):
        from scripts.audit_v3_component_ablation_capacity import audit_v3_ablation_capacity, load_capacity_config

        config = load_capacity_config(PROJECT_ROOT / "configs" / "v3_component_ablation_capacity_v1.yaml")
        rows = audit_v3_ablation_capacity(config)
        by_name = {row["model_name"]: row for row in rows}
        self.assertIn("channel_shared_posres_attention_v3", by_name)
        self.assertEqual(int(by_name["channel_shared_posres_attention_v3_no_residual"]["residual_params"]), 0)
        self.assertEqual(int(by_name["channel_shared_posres_attention_v3_residual_only_mlp"]["encoder_params"]), 0)
        self.assertEqual(int(by_name["channel_shared_posres_attention_v3_no_identity"]["identity_params"]), 0)
        self.assertEqual(int(by_name["channel_shared_posres_meanpool_v3_no_attention"]["attention_params"]), 0)


if __name__ == "__main__":
    unittest.main()
