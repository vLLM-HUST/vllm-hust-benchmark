"""Inert fixture checks, not candidate execution or scoring."""

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "bcb_static_assets", SCRIPTS / "audit_bigcodebench_scoring_assets.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reference_continuation_is_not_executed():
    row = {
        "complete_prompt": "import random\ndef f():\n",
        "canonical_solution": "    import statistics\n    raise RuntimeError('not executed')\n",
        "test": "import unittest\nfrom collections import Counter\n",
    }
    assert module.original_imports(row) == [
        "collections",
        "random",
        "statistics",
        "unittest",
    ]


def test_relative_import_requires_review():
    with pytest.raises(ValueError, match="relative import"):
        module.original_imports(
            {
                "complete_prompt": "from .local import x\n",
                "canonical_solution": "",
                "test": "",
            }
        )


def test_source_checksum_required(tmp_path):
    path = tmp_path / "fixed.py"
    path.write_bytes(b"raise RuntimeError('never import')\n")
    pins = {"fixed.py": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert module.verify_source(tmp_path, pins) == pins
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.verify_source(tmp_path, pins)


def test_static_report_cannot_complete_source(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "verify_source", lambda root: {})
    rows = [
        {
            "task_id": f"BigCodeBench/{i}",
            "complete_prompt": "",
            "canonical_solution": "",
            "test": "",
        }
        for i in range(1140)
    ]
    result = module.audit(tmp_path, rows)
    assert result["source_checkbox_eligible"] is False
    assert result["environment_qualified"] is False
    assert result["official_functional_scoring_executed"] is False
    rows.reverse()
    with pytest.raises(ValueError, match="ordered"):
        module.audit(tmp_path, rows)


def test_cli_reads_original_parquet_without_importing_publisher(
    tmp_path, monkeypatch, capsys
):
    data = tmp_path / "data"
    data.mkdir()
    parquet = data / "v0.1.4-00000-of-00001.parquet"
    parquet.write_bytes(b"inert parquet fixture")
    monkeypatch.setattr(
        module, "DATA_SHA", hashlib.sha256(parquet.read_bytes()).hexdigest()
    )
    fake = ModuleType("pyarrow.parquet")
    fake.read_table = lambda path: SimpleNamespace(to_pylist=lambda: ["inert row"])
    parent = ModuleType("pyarrow")
    parent.parquet = fake
    monkeypatch.setitem(sys.modules, "pyarrow", parent)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake)
    monkeypatch.setattr(
        module, "audit", lambda root, rows: {"rows": rows, "executed": False}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "--official-code", str(tmp_path), "--data-root", str(tmp_path)],
    )
    module.main()
    assert '"executed": false' in capsys.readouterr().out
    parquet.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        module.main()
