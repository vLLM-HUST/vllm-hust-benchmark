"""Inert fixture checks, not candidate execution or scoring."""

import hashlib
import importlib.util
import json
import subprocess
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


def test_bigcodebench_acquisition_manifest_matches_auditor_identities():
    pins = json.loads(
        (
            SCRIPTS.parent / "configs/workload-native/bigcodebench-author-source.json"
        ).read_text("utf-8")
    )
    assert pins["revision"] == module.CODE_REVISION
    assert pins["repository"] == "bigcode-project/bigcodebench"
    assert {item["path"]: item["sha256"] for item in pins["files"]} == module.PINS
    assert len(pins["files"]) == 32
    assert sum(item["bytes"] for item in pins["files"]) == 109939
    assert all(len(item["git_blob"]) == 40 for item in pins["files"])


def test_public_bigcodebench_source_plan_needs_no_runtime_dependencies(tmp_path):
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_jsonschemabench_author_source.py"),
            "--source",
            "bigcodebench",
            "--author-root",
            str(output),
            "--action",
            "plan",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["source_id"] == "BIGCODEBENCH"
    assert receipt["files"] == 32 and receipt["bytes"] == 109939
    assert receipt["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert not receipt["source_checkbox_eligible"]
    assert not output.exists()


def test_public_bigcodebench_source_rejects_insufficient_budget_before_write(tmp_path):
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_jsonschemabench_author_source.py"),
            "--source",
            "bigcodebench",
            "--author-root",
            str(output),
            "--action",
            "prepare",
            "--max-bytes",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "109939" in result.stderr
    assert not output.exists()


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


def test_default_manifest_covers_imported_scoring_modules_and_rejects_extra(tmp_path):
    assert len([name for name in module.PINS if name.endswith(".py")]) == 29
    assert "bigcodebench/eval/utils.py" in module.PINS
    assert "bigcodebench/syncheck.py" in module.PINS
    assert "bigcodebench/provider/__init__.py" in module.PINS
    assert "Docker/Evaluate.Dockerfile" in module.PINS
    assert "Requirements/requirements.txt" in module.PINS
    (tmp_path / "bigcodebench").mkdir()
    (tmp_path / "bigcodebench/unexpected.py").write_text(
        "raise RuntimeError('never import')"
    )
    with pytest.raises(ValueError, match="source tree required"):
        module.verify_source(tmp_path)


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
    environment = result["environment_preparation"]
    assert not environment["recipe_is_fully_pinned"]
    assert not environment["image_built_or_qualified"]
    assert len(environment["floating_inputs_requiring_freeze"]) == 6
    assert "immutable final image ID" in environment["required_build_record"]
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
