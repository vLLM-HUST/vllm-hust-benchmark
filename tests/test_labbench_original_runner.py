"""Model-free fixtures; not publisher task or NPU execution evidence."""

import hashlib
import importlib.util
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "labbench_runner", ROOT / "scripts/run_labbench_original.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://remote.example/v1",
        "http://user:secret@localhost/v1",  # pragma: allowlist secret - deliberately rejected dummy fixture
        "http://localhost/v1?key=x",
    ],
)
def test_reject_external_or_credentialed_endpoint(endpoint):
    with pytest.raises(ValueError):
        RUNNER.local_endpoint(endpoint)


def test_fixed_original_order_per_subset():
    originals = [("a", object()), ("b", object()), ("a", object()), ("b", object())]
    assert RUNNER.select_instances(originals, 1) == originals[:2]
    assert RUNNER.select_instances(originals, None) == originals
    with pytest.raises(ValueError):
        RUNNER.select_instances(originals, 0)


def test_verified_archive_and_unchanged_source_required(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "task.py").write_bytes(b"# original fixture\n")
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.write(source / "task.py", "publisher/task.py")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (
        RUNNER.verify_source(source, archive, expected_sha=sha, expected_files=1) == 1
    )
    (source / "task.py").write_bytes(b"# modified\n")
    with pytest.raises(ValueError, match="modified"):
        RUNNER.verify_source(source, archive, expected_sha=sha, expected_files=1)


def test_unregistered_executable_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "task.py").write_bytes(b"# original fixture\n")
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.write(source / "task.py", "publisher/task.py")
    (source / "injected.py").write_bytes(b"# unexpected\n")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unexpected"):
        RUNNER.verify_source(source, archive, expected_sha=sha, expected_files=1)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("disable_thinking", [False, True])
def test_native_wrapper_retains_attempts_and_only_marks_scored_after_completion(
    tmp_path, monkeypatch, fail, disable_thinking
):
    """Fake harness/transport: protocol plumbing only, never official run evidence."""
    source = tmp_path / "source"
    instance = SimpleNamespace(id="fixture-id", model_dump=lambda: {"id": "fixture-id"})
    original_calls = []

    async def create(**kwargs):
        original_calls.append(kwargs)
        if fail:
            raise RuntimeError("fixture transport failure")
        return SimpleNamespace(
            model_dump=lambda: {
                "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}]
            }
        )

    class Agent:
        def __init__(self, **kwargs):
            assert kwargs["use_cot"] is True and kwargs["open_answer"] is False
            self.kwargs = kwargs["model_kwargs"]
            self.task_buffer = []
            self.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            self.client.with_options = lambda **options: self.client

        async def run_task(self, inp):
            await self.client.chat.completions.create(
                **self.kwargs,
                messages=[{"role": "user", "content": "fixture original input"}],
            )
            return "A"

    class Evaluator:
        def __init__(self, category):
            self.eval_set = SimpleNamespace(instances=[("fixture-subset", instance)])

        async def score_agent(self, agent_fn, n_threads):
            assert n_threads == 1
            answer = await agent_fn(instance)
            return {
                "metrics_all": {"accuracy": 1.0},
                "results": {
                    "fixture-id": {
                        "instance": instance,
                        "input": instance,
                        "agent_output": answer,
                    }
                },
            }

    fake = SimpleNamespace(
        __file__=str(source / "labbench/__init__.py"),
        Eval=lambda category: category,
        Evaluator=Evaluator,
        OpenAIZeroShotAgent=Agent,
    )
    monkeypatch.setitem(sys.modules, "labbench", fake)
    monkeypatch.setattr(RUNNER, "verify_source", lambda *args: 553)
    # execute changes these only for a loopback client; restore them after test.
    for key in ("PUBLIC_RELEASE", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "fixture-before")
    args = SimpleNamespace(
        source_root=source,
        archive=tmp_path / "unused.zip",
        output=tmp_path / "output",
        endpoint="http://127.0.0.1:19466/v1",
        model="fixture-npu-model",
        max_tokens=42,
        seed=1,
        per_subset=1,
        categories=["TableQA"],
        timeout=60,
        disable_thinking=disable_thinking,
    )
    if fail:
        with pytest.raises(RuntimeError):
            asyncio.run(RUNNER.execute(args))
    else:
        asyncio.run(RUNNER.execute(args))
    receipt = json.loads((args.output / "receipt-private.json").read_text())
    attempt = json.loads(
        (args.output / "TableQA-attempt-00000.private.json").read_text()
    )
    assert attempt["request"] == original_calls[0]
    assert attempt["request"]["max_tokens"] == 42
    assert receipt["thinking_override"] is (False if disable_thinking else None)
    if disable_thinking:
        assert attempt["request"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    else:
        assert "extra_body" not in attempt["request"]
    assert receipt["categories"][0]["selected_ids"] == ["fixture-id"]
    assert receipt["categories"][0]["model_calls"] == 1
    assert receipt["source_checkbox_eligible"] is False
    if fail:
        assert receipt["status"] == "FAILED_NOT_COMPLETE"
        assert "response" not in attempt and attempt["error_type"] == "RuntimeError"
    else:
        assert receipt["status"] == "FIXED_ORIGINAL_TASKS_SCORED_PENDING_DELIVERY_AUDIT"
        assert "response" in attempt
