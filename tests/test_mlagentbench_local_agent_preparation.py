"""Exact source/port fixtures; no Agent tasks, SDK imports or model requests."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "mlagent_local_prepare", ROOT / "scripts/prepare_mlagentbench_local_agent.py"
)
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def test_plan_does_not_read_or_execute_original_code():
    assert MODULE.prepare(SimpleNamespace(action="plan"))["model_calls"] == 0


def test_lazy_optional_import_port_preserves_original_arguments():
    original = (
        MODULE.OPTIONAL_IMPORT
        + b'\nif __name__ == "__main__":\n    parser.add_argument("--max-steps", default=50)\n    parser.add_argument("--max-time", default=18000)\n'
        + MODULE.RUN
        + b"\n"
    )
    ported = MODULE.port_runner(original)

    def arguments(raw):
        return [
            ast.dump(node)
            for node in ast.walk(ast.parse(raw))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ]

    assert arguments(original) == arguments(ported)
    assert b'if args.agent_type == "LangChainAgent"' in ported
    with pytest.raises(ValueError, match="exact pinned"):
        MODULE.port_runner(b"different runner")


@pytest.fixture
def task(tmp_path, monkeypatch):
    author = tmp_path / "original"
    package = author / "MLAgentBench"
    package.mkdir(parents=True)
    runner = (
        MODULE.OPTIONAL_IMPORT + b'\nif __name__ == "__main__":\n' + MODULE.RUN + b"\n"
    )
    sources = {
        "MLAgentBench/runner.py": runner,
        "MLAgentBench/LLM.py": b"# original LLM fixture, never executed\n",
        "task.txt": b"unchanged original task fixture",
    }
    for name, raw in sources.items():
        (author / name).write_bytes(raw)
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps({"files": [{"path": name} for name in sources]}), encoding="utf-8"
    )
    monkeypatch.setattr(MODULE, "PINS", pins)
    monkeypatch.setattr(MODULE, "verify_source", lambda root, pins: root.resolve())
    return SimpleNamespace(
        action="prepare", author_root=author, output=tmp_path / "port"
    )


def test_fresh_workspace_preserves_originals_and_discloses_ports(task):
    result = MODULE.prepare(task)
    assert result["original_files"] == 3
    assert not result["task_or_prompt_modified"]
    assert not result["source_checkbox_eligible"]
    assert (task.output / "task.txt").read_bytes() == (
        task.author_root / "task.txt"
    ).read_bytes()
    assert (task.output / "originals/MLAgentBench/LLM.py.source").read_bytes() == (
        task.author_root / "MLAgentBench/LLM.py"
    ).read_bytes()
    assert (task.output / "ports/MLAgentBench/runner.py.diff").is_file()
    with pytest.raises(FileExistsError):
        MODULE.prepare(task)
    task.action = "verify"
    assert MODULE.prepare(task) == result


def test_unregistered_workspace_modification_rejected(task):
    MODULE.prepare(task)
    (task.output / "extra.py").write_text("# extra", encoding="utf-8")
    task.action = "verify"
    with pytest.raises(ValueError, match="unexpected"):
        MODULE.prepare(task)


def test_original_checkout_cannot_be_overwritten(task):
    task.output = task.author_root
    with pytest.raises(ValueError, match="separate"):
        MODULE.prepare(task)


def test_actual_cli_qualification_receipt_matches_public_entry_and_sdk_lock():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/mlagentbench-local-agent-preparation-20261101.json"
        ).read_text("utf-8")
    )
    assert (
        receipt["entry_sha256"]
        == hashlib.sha256(
            (ROOT / receipt["public_entry"]).read_text("utf-8").encode()
        ).hexdigest()
    )
    sdk = receipt["sdk"]
    assert (
        sdk["requirements_sha256"]
        == hashlib.sha256(
            (ROOT / sdk["requirements"]).read_text("utf-8").encode()
        ).hexdigest()
    )
    assert receipt["original_runner_cli"]["exit_code"] == 0
    assert receipt["original_runner_cli"]["torch_imported"] is False
    assert receipt["native_agent_tasks_run"] == 0
    assert receipt["source_checkbox_eligible"] is False
