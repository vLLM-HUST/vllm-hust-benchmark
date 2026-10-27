"""Software fixtures only; no publisher tasks, NPU requests or native score claims."""

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import pytest

spec = importlib.util.spec_from_file_location(
    "docvqa_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_docvqa_original.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(
    "url",
    [
        "https://paid.example/v1",
        "http://127.0.0.1/api",
        "http://user@localhost/v1",
        "http://localhost/v1?x=1",
    ],
)
def test_only_explicit_local_endpoint(url):
    with pytest.raises(ValueError):
        runner.local_endpoint(url)


@pytest.mark.parametrize("fail_sample", [False, True])
def test_native_fixed_selection_transport_guard_and_failure_receipt(
    tmp_path, monkeypatch, fail_sample
):
    monkeypatch.setenv("XDG_CACHE_HOME", "fixture-old-cache")
    monkeypatch.setenv("XDG_DATA_HOME", "fixture-old-data")
    monkeypatch.setenv("XDG_CONFIG_HOME", "fixture-old-config")
    monkeypatch.setenv("INSPECT_EVALS_CACHE_DIR", "fixture-old-image-cache")
    monkeypatch.setattr(
        runner, "verify_originals", lambda *a: {"files": [{"rows": 5349}]}
    )
    monkeypatch.setattr(runner, "digest", lambda *a: runner.TASK_SHA256)
    monkeypatch.setattr(runner, "idle", lambda *a: True)
    native = ModuleType("inspect_evals.docvqa.docvqa")
    native.__file__ = "fixture-task.py"

    class Dataset:
        def __len__(self):
            return 5349

        def __getitem__(self, index):
            assert index == slice(None, 3)
            return [SimpleNamespace(id=value) for value in runner.FIXED_IDS]

    def factory(*, shuffle):
        assert shuffle is False
        return SimpleNamespace(dataset=Dataset())

    native.docvqa = factory
    monkeypatch.setitem(sys.modules, "inspect_evals.docvqa.docvqa", native)
    api_calls = []

    async def generate(*args, **kwargs):
        api_calls.append((args, kwargs))
        return "fixture-result-not-real-native-output"

    model = SimpleNamespace(api=SimpleNamespace(generate=generate))
    model_module = ModuleType("inspect_ai.model")
    model_module.GenerateConfig = lambda **values: values

    def get_model(name, **kwargs):
        assert name == "openai-api/local/Qwen/fixture"
        assert kwargs["config"]["max_retries"] == 0
        assert kwargs["max_retries"] == 0
        assert kwargs["config"]["max_connections"] == 1
        return model

    model_module.get_model = get_model
    monkeypatch.setitem(sys.modules, "inspect_ai.model", model_module)
    inspect_module = ModuleType("inspect_ai")

    def fake_eval(task, **kwargs):
        assert kwargs["sample_id"] == runner.FIXED_IDS
        assert kwargs["retry_on_error"] == 0 and kwargs["epochs"] == 1
        assert kwargs["log_model_api"] is True
        for _ in runner.FIXED_IDS:
            asyncio.run(model.api.generate(input=["fixture-prompt"], config={}))
        return [
            SimpleNamespace(
                status="success",
                samples=[
                    SimpleNamespace(
                        id=i,
                        error="fixture-error" if fail_sample else None,
                        scores={"fixture-score": SimpleNamespace(value=0)},
                    )
                    for i in runner.FIXED_IDS
                ],
            )
        ]

    inspect_module.eval = fake_eval
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_module)
    args = SimpleNamespace(
        endpoint="http://127.0.0.1:19466/v1",
        model="Qwen/fixture",
        output=tmp_path / "output",
        data_root=tmp_path / "original",
        framework_source_root=tmp_path / "framework",
        max_tokens=16384,
        inspect_cache_root=tmp_path / "cache",
        image_cache_root=tmp_path / "images",
        api_server_pid=1,
        npu_engine_pid=2,
    )
    if fail_sample:
        with pytest.raises(ValueError, match="flow incomplete"):
            runner.execute(args)
    else:
        result = runner.execute(args)
        assert (
            result["status"]
            == "FIXED3_NATIVE_TASKS_SCORED_SAVED_AUDIT_AND_CI_HANDOFF_PENDING"
        )
    receipt = json.loads((args.output / "native-public.json").read_text())
    assert not receipt["source_checkbox_eligible"]
    assert len(api_calls) == 3
    assert receipt["original_fixed_question_ids"] == runner.FIXED_IDS
    assert receipt["status"] == (
        "FAILED_NOT_COMPLETE"
        if fail_sample
        else "FIXED3_NATIVE_TASKS_SCORED_SAVED_AUDIT_AND_CI_HANDOFF_PENDING"
    )
