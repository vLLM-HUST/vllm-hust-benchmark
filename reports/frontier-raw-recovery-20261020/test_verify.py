import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frontier_verify", ROOT / "verify.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ArchiveVerificationTest(unittest.TestCase):
    def test_all_56_runs(self):
        self.assertEqual(MODULE.verify(ROOT)["runs"], 56)

    def reject_modified_manifest(self, field, value):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        first = manifest["runs"][0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "raw").mkdir()
            for part in first["archive_parts"]:
                shutil.copyfile(ROOT / part["path"], root / part["path"])
            first[field] = value
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):
                MODULE.verify(root)

    def test_rejects_archive_hash_mismatch(self):
        self.reject_modified_manifest("archive_sha256", "0" * 64)

    def test_rejects_wrong_throughput(self):
        self.reject_modified_manifest("expected_tps", 0)

    def test_rejects_wrong_concurrency(self):
        self.reject_modified_manifest("concurrency", 999)


if __name__ == "__main__":
    unittest.main()
