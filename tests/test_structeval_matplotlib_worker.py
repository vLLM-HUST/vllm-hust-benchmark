"""Worker safety/software fixtures; never execute generated code on host."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/run_structeval_matplotlib_worker.py"


def module():
    spec = importlib.util.spec_from_file_location("structeval_worker", ENTRY)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_plan_has_no_runtime_data_requirement(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--task-id",
            "000800",
            "--renderer",
            str(tmp_path / "absent.py"),
            "--code-file",
            str(tmp_path / "absent.txt"),
            "--code-sha256",
            "0" * 64,
            "--output",
            str(tmp_path / "not-created"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("task_id", ["../escape", "a/b", "'injection", "", "a" * 65])
def test_task_path_injection_rejected(task_id):
    with pytest.raises(ValueError, match="identifier"):
        module().execute(SimpleNamespace(task_id=task_id, execute=False))


def test_changed_original_source_rejected_without_import(tmp_path):
    source = tmp_path / "changed.py"
    source.write_text('raise RuntimeError("must never import")\n')
    with pytest.raises(ValueError, match="unchanged pinned original"):
        module().verify_inputs(source, tmp_path / "absent.txt", "0" * 64)


def test_execute_rejects_host_before_reading_any_input(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--task-id",
            "000800",
            "--renderer",
            str(tmp_path / "absent.py"),
            "--code-file",
            str(tmp_path / "absent.txt"),
            "--code-sha256",
            "0" * 64,
            "--output",
            str(tmp_path / "not-created"),
            "--execute",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "worker rejected" in result.stderr
    assert not list(tmp_path.iterdir())
