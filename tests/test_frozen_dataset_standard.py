"""The public standard must reject edited membership and generation budgets."""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs/dataset-standards/five-datasets-v1"
spec = importlib.util.spec_from_file_location(
    "frozen_dataset_standard", BUNDLE / "verify.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DatasetStandardTests(unittest.TestCase):
    def test_frozen_bundle(self):
        standard = module.validate(BUNDLE)
        self.assertEqual(
            [d["requests_per_arm"] for d in standard["datasets"]],
            [200, 200, 160, 64, 48],
        )

    def test_tamper_and_extra_file_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            shutil.copytree(BUNDLE, root, ignore=shutil.ignore_patterns("__pycache__"))
            path = root / "hle-fixed200.json"
            original = path.read_text()
            path.write_text(
                original.replace(
                    '"solver_max_tokens": 16384', '"solver_max_tokens": 256'
                )
            )
            with self.assertRaises(ValueError):
                module.validate(root)
            path.write_text(original)
            (root / "extra.json").write_text("{}")
            with self.assertRaises(ValueError):
                module.validate(root)
