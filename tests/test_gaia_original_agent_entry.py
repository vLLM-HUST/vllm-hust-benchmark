"""Invented protocol controls, no model calls or original GAIA tasks."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "gaia_agent_entry", Path(__file__).parents[1] / "scripts/run_gaia_original_agent.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def task():
    return {
        "task_id": "invented-id",
        "input": {
            "question": "Unchanged invented question",
            "file_name": "",
            "file_path": "",
            "attachment_asset": None,
        },
    }


def test_selection_does_not_read_references(monkeypatch):
    class Record:
        task = task()

        def outcome_record(self):
            raise AssertionError("references must not reach Agent")

    monkeypatch.setattr(module, "iter_workload", lambda *a, **k: iter([Record()]))
    assert module.selected_tasks("unused-fixture-root", 0, 1) == [task()]
    with pytest.raises(ValueError, match="exceeds"):
        module.selected_tasks("unused-fixture-root", 0, 2)


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("model_timeout", [600, 1800])
def test_full_framework_call_or_retained_failure(
    tmp_path, monkeypatch, fails, model_timeout
):
    calls = []
    monkeypatch.setattr(module, "selected_tasks", lambda *args: [task()])
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "1.26.0")

    def factory(inputs, directory):
        assert inputs == task()["input"]
        return ["invented-nonexecuting-tool"]

    monkeypatch.setattr(module, "load_factory", lambda *args: factory)

    class Agent:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.memory = SimpleNamespace(get_full_steps=lambda: [{"fixture": True}])

        def run(self, question, additional_args):
            assert question == task()["input"]["question"] and additional_args is None
            if fails:
                raise RuntimeError("invented failed attempt")
            return "invented prediction"

    monkeypatch.setitem(
        sys.modules,
        "smolagents",
        SimpleNamespace(
            OpenAIModel=lambda **kwargs: kwargs,
            ToolCallingAgent=Agent,
        ),
    )
    data = tmp_path / "data"
    data.mkdir()
    out = tmp_path / "private-output"
    result = module.run(
        data_root=data,
        output=out,
        factory_file="unused-fixture",
        factory_sha256="fixture",
        endpoint="http://127.0.0.1:19503/v1",
        model="fixture-not-a-model-run",
        start=0,
        count=1,
        max_steps=20,
        max_output_tokens=4096,
        model_timeout=model_timeout,
    )
    assert calls[0]["max_steps"] == 20 and calls[0]["max_tool_threads"] == 1
    assert calls[0]["model"]["max_tokens"] == 4096
    assert calls[0]["model"]["client_kwargs"]["max_retries"] == 0
    assert calls[0]["model"]["client_kwargs"]["timeout"] == model_timeout
    assert result["model_timeout_seconds"] == model_timeout
    assert result["attempted_original_ids"] == ["invented-id"]
    assert result["failed_original_ids"] == (["invented-id"] if fails else [])
    prediction = json.loads((out / "predictions-private.jsonl").read_text())
    assert prediction == {
        "task_id": "invented-id",
        "prediction": None if fails else "invented prediction",
    }
    assert result["source_checkbox_eligible"] is False


def test_remote_model_transport_rejected():
    with pytest.raises(ValueError, match="loopback"):
        module.run(
            data_root="unused",
            output="unused",
            factory_file="unused",
            factory_sha256="unused",
            endpoint="http://public.example/v1",
            model="unused",
            start=0,
            count=1,
            max_steps=20,
            max_output_tokens=4096,
        )


@pytest.mark.parametrize("timeout", [True, 0, -1, 86401, 1.5, "1800"])
def test_invalid_model_timeout_rejected_before_assets(timeout):
    with pytest.raises(ValueError, match="model timeout"):
        module.run(
            data_root="unused",
            output="unused",
            factory_file="unused",
            factory_sha256="unused",
            endpoint="http://127.0.0.1:19503/v1",
            model="fixture",
            start=0,
            count=1,
            max_steps=20,
            max_output_tokens=4096,
            model_timeout=timeout,
        )


def test_changed_factory_rejected(tmp_path):
    path = tmp_path / "factory.py"
    path.write_text("raise AssertionError('not executed')")
    with pytest.raises(ValueError, match="digest"):
        module.load_factory(path, "changed")


def test_default_plan_does_not_execute(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_gaia_original_agent.py"])
    module.main()
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_ONLY"
