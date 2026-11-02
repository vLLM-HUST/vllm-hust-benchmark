"""Fixture-only orchestration checks; never scientific tasks or native results."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "naturebench_original_runner", ROOT / "scripts/run_naturebench_original.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
TASKS = json.loads(runner.RELEASE.read_text("utf-8"))["official_cpu_tasks"]


@pytest.fixture
def args(tmp_path, monkeypatch):
    author = tmp_path / "author"
    runtime = tmp_path / "runtime"
    data = tmp_path / "data"
    for path in [author, runtime / "source", data]:
        path.mkdir(parents=True)
    (runtime / "source/eval_env_mapping.json").write_text("{}")
    monkeypatch.setattr(runner, "verify_runtime", lambda a, r: r / "source")
    monkeypatch.setattr(
        runner,
        "original_cpu_packages",
        lambda root, task_ids: ({"task_id": task} for task in task_ids),
    )
    return SimpleNamespace(
        author_root=author,
        runtime=runtime,
        data_root=data,
        output=tmp_path / "fresh-output",
        model="fixture-local-model",
        endpoint="http://127.0.0.1:9999",
        judge_model=None,
        judge_endpoint=None,
        task_id=None,
        eval_port=8321,
        eval_mapping=None,
    )


def test_plan_preserves_original_cpu_group_and_full_flow(args):
    result = runner.plan(args)
    assert result["task_ids"] == TASKS
    assert result["task_budget_seconds"] == result["setup_budget_seconds"] == 14400
    assert result["skip_judge"] is False
    command = result["native_command"]
    assert "--build-task-images" in command
    assert "--skip-judge" not in command and "--skip-build" not in command
    assert "--skip-download" in command
    assert command[command.index("--agent") + 1] == "claude"
    assert not args.output.exists()


@pytest.mark.parametrize("tasks", [[TASKS[0], TASKS[0]], ["invented-task"]])
def test_no_duplicate_or_replacement_tasks(args, tasks):
    args.task_id = tasks
    with pytest.raises(ValueError, match="original CPU"):
        runner.plan(args)
    assert not args.output.exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@example.org",  # pragma: allowlist secret - synthetic reserved-domain rejection fixture
        "https://example.org?token=secret",
        "file:///tmp/socket",
    ],
)
def test_endpoint_credentials_and_non_http_rejected(url):
    with pytest.raises(ValueError):
        runner.endpoint(url)


def test_missing_explicit_credentials_fails_before_launch(args, monkeypatch):
    prepared = runner.plan(args)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="explicit"):
        runner.execute(args, prepared)
    assert not args.output.exists()


class Health:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        return b'{"status":"ok"}'


def fake_execution(monkeypatch, code):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-agent-secret")
    monkeypatch.setenv("JUDGE_API_KEY", "fixture-judge-secret")
    monkeypatch.setenv("JUDGE_OPENAI_BASE_URL", "https://unrelated.example")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "fixture-unrelated-secret")
    monkeypatch.setattr(runner, "urlopen", lambda *a, **k: Health())
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0 if command[0] == "docker" else code)

    monkeypatch.setattr(runner.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("code", [0, 7])
def test_native_exit_never_promotes_source_or_discloses_keys(args, monkeypatch, code):
    prepared = runner.plan(args)
    calls = fake_execution(monkeypatch, code)
    assert runner.execute(args, prepared) == code
    assert calls[0][0][:2] == ["docker", "info"]
    environment = calls[1][1]["env"]
    assert "JUDGE_OPENAI_BASE_URL" not in environment
    assert "ANTHROPIC_AUTH_TOKEN" not in environment
    assert environment["ANTHROPIC_BASE_URL"] == args.endpoint
    receipt = (args.output / "execution-public.json").read_text()
    result = json.loads(receipt)
    assert result["source_checkbox_eligible"] is False
    assert result["native_exit_code"] == code
    assert (
        "fixture-agent-secret" not in receipt and "fixture-judge-secret" not in receipt
    )
    assert (
        args.output / "selected-original-taskset.txt"
    ).read_text().splitlines() == TASKS


def test_original_input_verification_failure_is_not_replaced(args, monkeypatch):
    def fail(*a, **k):
        raise ValueError("fixture original asset checksum differs")

    monkeypatch.setattr(runner, "original_cpu_packages", fail)
    with pytest.raises(ValueError, match="checksum"):
        runner.plan(args)
    assert not args.output.exists()
