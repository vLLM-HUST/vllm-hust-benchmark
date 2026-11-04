import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import tp2_pilot
from raw_tp2_audit import exact_period


class PilotGuardTests(unittest.TestCase):
    def run_guard(self, env, backend="FFTS+"):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "capture"
            args = [
                "pilot",
                "--runtime",
                tmp,
                "--output",
                str(out),
                "--model",
                tmp,
                "--backend",
                backend,
            ]
            with patch.dict(os.environ, env, clear=True), patch("sys.argv", args):
                with self.assertRaises(RuntimeError):
                    tp2_pilot.main()
            self.assertFalse(out.exists())

    def test_no_lease_rejected_before_import_or_output(self):
        self.run_guard({})

    def test_single_device_lease_rejected(self):
        self.run_guard({"NPU_SHARE_LEASE_ID": "test", "NPU_SHARE_DEVICE_COUNT": "1"})

    def test_backend_mismatch_rejected(self):
        self.run_guard(
            {
                "NPU_SHARE_LEASE_ID": "test",
                "NPU_SHARE_DEVICE_COUNT": "2",
                "HCCL_OP_EXPANSION_MODE": "AIV",
            }
        )

    def test_protocol_has_no_profiled_speedup_claim(self):
        protocol = json.loads(Path(__file__).with_name("tp2_protocol.json").read_text())
        self.assertEqual(
            protocol["formal_order_if_qualified"], ["FFTS+", "AIV", "AIV", "FFTS+"]
        )
        self.assertIn(
            "no service speedup claim from profile durations", protocol["exclusions"]
        )

    def test_exact_period_preserves_complete_population(self):
        self.assertEqual(exact_period([0, 1, 1, 2] * 6), 4)
        self.assertIsNone(exact_period([0, 1, 1, 2] * 6 + [0]))

    def test_exact_period_needs_three_repeats(self):
        self.assertIsNone(exact_period([0, 1, 2] * 2))
        self.assertIsNone(exact_period([]))

    def test_new_protocol_keeps_backend_failure(self):
        protocol = json.loads(
            Path(__file__).with_name("tp2_protocol_v2.json").read_text()
        )
        self.assertEqual(protocol["pilot_order"], ["HOST", "AIV"])
        self.assertIn("FFTS+", protocol["reason"])


if __name__ == "__main__":
    unittest.main()
