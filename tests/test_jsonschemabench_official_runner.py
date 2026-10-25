"""Fake official-interface integration only; no dataset, network or NPU usage."""

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_jsonschemabench_official")


def test_task_selection_does_not_double_count_default():
    assert runner.select_tasks(["A", "B", "default"]) == ["A", "B"]
    assert runner.select_tasks(["A", "default"], ["default"]) == ["default"]
    for tasks in (["A", "default"], ["A", "A"], ["unknown"], []):
        with pytest.raises(ValueError):
            runner.select_tasks(["A", "B", "default"], tasks)


@pytest.mark.parametrize(
    "value",
    [
        "https://localhost",
        "http://remote.invalid",
        "http://localhost/v1",
        "http://user:pass@localhost",  # pragma: allowlist secret - rejected inert fixture credentials
        "http://localhost/?key=fixture",
    ],
)
def test_nonlocal_or_credential_routes_rejected(value):
    with pytest.raises(ValueError):
        runner.origin(value)


def test_clean_source_identity_required(tmp_path, monkeypatch):
    dirty = False

    def git(command, **kwargs):
        if "--show-toplevel" in command:
            return str(tmp_path)
        if "HEAD" in command:
            return runner.AUTHOR_REVISION
        return " M core/dataset.py" if dirty else ""

    monkeypatch.setattr(runner.subprocess, "check_output", git)
    assert runner.validate_source(tmp_path) == tmp_path.resolve()
    dirty = True
    with pytest.raises(ValueError, match="clean fixed"):
        runner.validate_source(tmp_path)


@dataclass
class FakeOutput:
    generation: str = "fixture answer"


def test_checkpoint_keeps_failed_attempt_without_replacement(tmp_path):
    class Engine:
        def generate(self, *args):
            raise ValueError("fixture failure")

    receipt = {"attempted_samples": 0, "returned_samples": 0}
    saved = []
    engine = runner.checkpoint_engine(
        Engine(), tmp_path, receipt, lambda: saved.append(dict(receipt))
    )
    with pytest.raises(ValueError):
        engine.generate(
            "A", [{"role": "user", "content": "fixture"}], {"type": "object"}
        )
    record = json.loads((tmp_path / "attempt-000000.private.json").read_text())
    assert record["status"] == "GENERATION_FAILED_NOT_REPLACED"
    assert receipt == {"attempted_samples": 1, "returned_samples": 0}
    assert len(saved) == 2


def test_official_bench_bridge_preserves_location_and_defaults(tmp_path, monkeypatch):
    source, snapshot, out = (
        tmp_path / name for name in ["author", "snapshot", "output"]
    )
    source.mkdir()
    snapshot.mkdir()
    monkeypatch.setattr(runner, "validate_source", lambda p: Path(p).resolve())
    monkeypatch.setattr(runner, "validate_snapshot", lambda p, pins: Path(p).resolve())
    dataset = NS(
        DATASET_HUGGINGFACE_PATH="original",
        Dataset=lambda config: [1, 2, 3],
        DatasetConfig=lambda task: task,
    )
    observed = []

    class Engine:
        def generate(self, task, messages, schema):
            return FakeOutput()

    def bench(**kwargs):
        observed.append(kwargs)
        assert "messages_formatter" not in kwargs
        assert kwargs["save_outputs"] is True
        assert dataset.DATASET_HUGGINGFACE_PATH == str(snapshot.resolve())
        return [
            [
                kwargs["engine"].generate(
                    task,
                    [{"role": "user", "content": "original fixture"}],
                    {"type": "object"},
                )
                for _ in range(kwargs["limit"])
            ]
            for task in kwargs["tasks"]
        ]

    modules = {
        "core.dataset": dataset,
        "core.bench": NS(bench=bench),
        "core.engine": NS(),
        "core.types": NS(),
    }
    real_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else real_import(name),
    )
    monkeypatch.delitem(sys.modules, "core", raising=False)
    sdk = ModuleType("openai")
    sdk.OpenAI = lambda **kwargs: NS(close=lambda: None)
    monkeypatch.setitem(sys.modules, "openai", sdk)
    monkeypatch.setattr(runner, "create_engine", lambda **kwargs: Engine())
    args = NS(
        official_source=source,
        snapshot_root=snapshot,
        output=out,
        tasks=["Github_easy"],
        full=False,
        limit=3,
        model="fixture-local",
        max_tokens=128,
        context_capacity=4096,
        origin="http://localhost:19466",
    )
    runner.run(args)
    receipt = json.loads((out / "receipt-public.json").read_text())
    assert len(observed) == 1
    assert receipt["status"] == "ORIGINAL_BENCH_AND_EVALUATOR_RETURNED"
    assert receipt["attempted_samples"] == receipt["returned_samples"] == 3
    assert receipt["schema_rewriting"] is False
    assert receipt["source_checkbox_eligible"] is False
    assert len(list(out.glob("attempt-*.private.json"))) == 3
    assert dataset.DATASET_HUGGINGFACE_PATH == "original"
