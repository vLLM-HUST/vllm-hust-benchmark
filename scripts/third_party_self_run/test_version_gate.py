"""Version admission uses the named test, not local availability or a main label."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import version_gate


class VersionAdmission(unittest.TestCase):
    def test_reconstruction_detects_source_change_after_freezing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sdk = root / "synthetic-sdk"
            sdk.write_bytes(b"unit fixture, not a runtime")
            files = {}
            for name in [
                "vllm/model_executor/models/qwen3_5.py",
                "vllm_ascend/ops/gdn.py",
            ]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            digest = hashlib.sha256(sdk.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "sdk": str(sdk),
                        "sdk_sha256": digest,
                        "site_packages": str(root),
                        "files": files,
                    }
                )
            )
            profile = {
                "reconstruction_manifest": str(manifest),
                "reconstruction_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
            }
            observed = {
                "origins": {
                    name: str(root / name / "__init__.py")
                    for name in ["vllm", "vllm_ascend"]
                }
            }
            with patch("rebuild_runtime.SDK_SHA", digest):
                self.assertEqual(
                    version_gate.check_reconstruction(profile, observed), []
                )
                (root / "vllm_ascend/ops/gdn.py").write_text("changed\n")
                errors = version_gate.check_reconstruction(profile, observed)
            self.assertEqual(
                [e["component"] for e in errors], ["reconstructed_source_file"]
            )

    def test_august_and_upstream023_cannot_be_the_tested_b1(self):
        for version in ["0.23.0", "0.23.0+empty"]:
            errors = version_gate.compare_packages(
                {"vllm": version}, {"vllm": ["0.28.1.post1.dev143+gf18cf803c.empty"]}
            )
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["observed"], version)

    def test_same_package_labels_do_not_replace_source_and_mod_proof(self):
        authority = json.loads(
            (version_gate.ROOT / "version-authority/tested-runtime.json").read_text()
        )
        packages = {
            "vllm": "0.28.1.post1.dev143+gf18cf803c.empty",
            "vllm-ascend": "0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909",
            "torch": "2.13.0+cpu",
            "torch-npu": "2.13.0rc1",
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "version-authority").mkdir()
            (root / "version-authority/tested-runtime.json").write_text(
                json.dumps(authority)
            )
            with patch.object(version_gate.subprocess, "run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = json.dumps({"packages": packages})
                result = version_gate.check(root, ["B1"])
            self.assertFalse(result["passed"])
            self.assertFalse(result["model_started"])
            self.assertFalse(result["patches_applied"])
            self.assertEqual(
                {e["component"] for e in result["arms"]["B1"]["errors"]},
                {"tested_implementation_capsule", "installed_source_manifest"},
            )

    def test_cli_stops_before_runtime_bootstrap_when_admission_fails(self):
        import one_click

        with (
            patch.object(version_gate, "check", return_value={"passed": False}),
            patch.object(one_click.subprocess, "run") as run,
            patch("sys.argv", ["one_click.py", "--legacy-diagnostic", "--verify-only"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(one_click.main(), 2)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
