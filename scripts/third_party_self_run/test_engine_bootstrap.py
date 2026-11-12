"""Bootstrap routing/integrity fixtures, never inference measurements."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

import engine_bootstrap as engine
import one_click


class EngineBootstrapTests(unittest.TestCase):
    def test_default_entry_routes_to_fixed_engine(self):
        with (
            patch("sys.argv", ["one_click.py", "--prepare-only"]),
            patch.object(engine, "main", return_value=17) as run,
        ):
            self.assertEqual(one_click.main(), 17)
            run.assert_called_once_with()

    def test_dotenv_is_parsed_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "must-not-exist"
            env = root / ".env"
            env.write_text(
                f'UNRELATED=$(touch {marker})\nexport GITHUB_TOKEN="synthetic-read-only-key" # fixture\n'
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(engine.token(env), "synthetic-read-only-key")
            self.assertFalse(marker.exists())

    def test_manifest_cannot_self_approve_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "main.py"
            source.write_text("original synthetic source")
            manifest = root / "DEPLOYMENT-TOOLS.json"
            value = {
                "repository": engine.REPOSITORY,
                "commit": engine.COMMIT,
                "files": {"main.py": engine.sha(source)},
            }
            manifest.write_text(json.dumps(value))
            digest = engine.sha(manifest)
            with patch.object(engine, "MANIFEST_SHA256", digest):
                self.assertEqual(engine.verify_tools(root), root)
                source.write_text("changed synthetic source")
                value["files"]["main.py"] = engine.sha(source)
                manifest.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "externally fixed"):
                    engine.verify_tools(root)

    def test_archive_rejects_parent_paths_and_symbolic_links(self):
        for name, kind in (
            (engine.PREFIX + "/../escape", tarfile.REGTYPE),
            (engine.PREFIX + "/link", tarfile.SYMTYPE),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive = root / "fixture.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    item = tarfile.TarInfo(name)
                    item.type = kind
                    if kind == tarfile.SYMTYPE:
                        item.linkname = "/etc/passwd"
                        tar.addfile(item)
                    else:
                        item.size = 1
                        tar.addfile(item, io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    engine.extract(archive, root / "out")
                self.assertFalse((root / "escape").exists())

    def test_cross_host_redirect_drops_authorization(self):
        req = urllib.request.Request(
            "https://api.github.com/example",
            headers={"Authorization": "Bearer synthetic-key"},
        )
        result = engine.SafeRedirect().redirect_request(
            req,
            None,
            302,
            "Found",
            {},
            "https://release-assets.githubusercontent.com/example",
        )
        self.assertIsNone(result.get_header("Authorization"))

    def test_old_pip_baseline_and_ten_request_prepare_are_disabled(self):
        root = Path(__file__).resolve().parent
        env = dict(os.environ)
        env.pop("VLLM_HUST_ALLOW_LEGACY_DIAGNOSTIC", None)
        for args in (
            [str(root / "bootstrap_runtime.py"), "baseline"],
            [str(root / "pipeline/prepare.py")],
        ):
            result = subprocess.run(
                [sys.executable, *args],
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("disabled", result.stderr)


if __name__ == "__main__":
    unittest.main()
