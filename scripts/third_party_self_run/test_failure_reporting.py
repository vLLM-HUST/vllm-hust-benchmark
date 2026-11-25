"""Fail-closed integration checks; these fixtures are never model measurements."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class FailureReporting(unittest.TestCase):
    def command(self, action, plan, digest, out):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "run_vllm_hust_tests.py"),
                action,
                "--plan",
                str(plan),
                "--sha256",
                digest,
                "--out",
                str(out),
            ],
            text=True,
            capture_output=True,
            timeout=60,
        )

    def test_rejected_plan_is_reported_and_sealed_without_execution(self):
        plan = ROOT / "execution-plan.example.json"
        digest = hashlib.sha256(plan.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            run = self.command("run", plan, digest, out)
            self.assertEqual(run.returncode, 2, run.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(summary["preflight_rejected"])
            self.assertFalse(summary["complete_plan"])
            self.assertEqual(set(summary["targets"].values()), {"unassessable"})
            self.assertEqual(len(list((out / "charts").glob("*.png"))), 4)
            self.assertEqual(
                json.loads((out / "preflight-error.json").read_text())[
                    "adapters_launched"
                ],
                0,
            )
            self.assertEqual(self.command("review", plan, digest, out).returncode, 1)
            with (out / "report.html").open("a") as f:
                f.write("tamper")
            review = self.command("review", plan, digest, out)
            self.assertEqual(review.returncode, 2)
            self.assertIn("digest differs", review.stderr)

    def test_wrong_plan_hash_creates_no_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "run"
            run = self.command(
                "run", ROOT / "execution-plan.example.json", "0" * 64, out
            )
            self.assertEqual(run.returncode, 2)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
