import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from vllm_hust_benchmark import swe_bench_pro as runner
from vllm_hust_benchmark.swe_bench_pro_timing import instrument

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads(
        (ROOT / "docs/examples/swe-bench-pro/single-agent.json").read_text()
    )


def test_standalone_without_controller_or_deployment(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(configuration()))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm_hust_benchmark.swe_bench_pro",
            "check",
            "--settings",
            str(settings),
            "--output",
            str(tmp_path / "unused"),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "configuration": "PASS",
        "program_instances": 1,
    }
    assert runner.hooks(configuration()) is None


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("batch", "arrival_time", 1),
        ("batch", "client_concurrency_limit", 1),
        ("agent", "step_limit", 0),
        ("agent", "automatic_retries", 1),
        ("agent", "task_instructions", "  "),
        ("agent", "task_instructions", ["run tests"]),
        ("batch", "program_instances", 2),
        ("model", "api_base", "local"),
    ],
)
def test_invalid_protocol(section, key, value):
    c = configuration()
    c[section][key] = value
    with pytest.raises(ValueError):
        runner.validate(c)


def test_initial_repository_checks_real_git_state(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "source.txt"
    tracked.write_text("original\n")
    subprocess.run(["git", "add", "source.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()

    def actual_state():
        result = subprocess.run(
            ["sh", "-c", "git rev-parse HEAD && git status --porcelain"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        return {"returncode": result.returncode, "output": result.stdout}

    runner.validate_initial_repository(actual_state(), commit)
    (tmp_path / "appendonly.aof").write_text("runtime data\n")
    with pytest.raises(ValueError):
        runner.validate_initial_repository(actual_state(), commit)
    runner.validate_initial_repository(actual_state(), commit, ["appendonly.aof"])
    tracked.write_text("changed source\n")
    with pytest.raises(ValueError):
        runner.validate_initial_repository(actual_state(), commit, ["appendonly.aof"])
    tracked.write_text("original\n")
    (tmp_path / "unexpected.txt").write_text("unexpected\n")
    with pytest.raises(ValueError):
        runner.validate_initial_repository(actual_state(), commit, ["appendonly.aof"])
    with pytest.raises(ValueError):
        runner.validate_initial_repository(
            actual_state(), "wrong", ["appendonly.aof", "unexpected.txt"]
        )


@pytest.mark.parametrize(
    "allowed", [["*"], ["../data"], ["/tmp/data"], [".git/config"], "data"]
)
def test_initial_untracked_allowlist_rejects_broad_or_unsafe_paths(allowed):
    c = configuration()
    c["container"]["allowed_initial_untracked"] = allowed
    with pytest.raises(ValueError):
        runner.validate(c)


def test_query_instrumentation_preserves_result_timeout_and_headers(tmp_path):
    calls = []
    response = {
        "content": "action",
        "extra": {"response": {"id": "one", "usage": {"total_tokens": 7}}},
    }

    def query(messages, **kwargs):
        calls.append((messages, kwargs))
        return response

    model = SimpleNamespace(
        config=SimpleNamespace(
            model_kwargs={"extra_headers": {"X-Session-ID": "program"}}
        ),
        query=query,
    )
    tool_result = {"returncode": 0, "output": "complete output"}
    env = SimpleNamespace(execute=lambda *args, **kwargs: tool_result)
    instrument(model, env, tmp_path, "program")
    marker = object()
    assert model.query([{"role": "user"}], timeout=marker) is response
    assert calls[0][1]["timeout"] is marker
    assert calls[0][1]["extra_headers"] == {
        "X-Session-ID": "program",
        "X-Delivery-Request-ID": "program:query-001",
    }
    assert env.execute("test command") is tool_result
    events = [
        json.loads(x)
        for x in (tmp_path / "agent-events-private.jsonl").read_text().splitlines()
    ]
    assert [x["event"] for x in events] == [
        "agent_query_start",
        "agent_query_return",
        "tool_start",
        "tool_return",
    ]
    assert events[-1]["output"] == "complete output"


def test_query_error_remains_error(tmp_path):
    def query(*args, **kwargs):
        raise TimeoutError("wall budget")

    model = SimpleNamespace(config=SimpleNamespace(model_kwargs={}), query=query)
    env = SimpleNamespace(execute=lambda command: None)
    instrument(model, env, tmp_path, "program")
    with pytest.raises(TimeoutError, match="wall budget"):
        model.query([])
    assert (
        "agent_query_exception" in (tmp_path / "agent-events-private.jsonl").read_text()
    )


def test_input_reuse_revalidates_before_creating_output(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError("changed image")

    monkeypatch.setattr(runner, "verify_inputs", reject)
    with pytest.raises(ValueError, match="changed image"):
        runner.adopt_inputs(configuration(), tmp_path, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_source_archive_cannot_escape_destination(tmp_path):
    extract = runpy.run_path(
        str(ROOT / "scripts/prepare_swe_bench_pro_environment.py")
    )["extract_source"]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("upstream/../../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="escapes"):
        extract(buffer.getvalue(), tmp_path / "source")
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "mutation", [None, "dataset", "source", "row", "image", "task_count", "selection"]
)
def test_verified_inputs_fail_closed(tmp_path, monkeypatch, mutation):
    from types import ModuleType

    c = configuration()
    row = {"instance_id": "task", "repo": "NodeBB/NodeBB"}
    c["batch"]["instance_ids"] = ["other" if mutation == "selection" else "task"]
    (tmp_path / "cases-private.jsonl").write_text(json.dumps(row) + "\n")
    case_hash = runner.digest(tmp_path / "cases-private.jsonl")
    (tmp_path / "input-selection.json").write_text(
        json.dumps(
            {
                "status": "PREPARED",
                "cases_sha256": case_hash,
                "tasks": [{"instance_id": "task", "image_id": "immutable"}],
            }
        )
    )
    c["paths"]["dataset_parquet"] = "/dataset"
    c["paths"]["agent_source"] = "/agent"
    lock = {
        "dataset": {"parquet_sha256": "dataset"},
        "agent_source_files": {"file.py": "source"},
    }
    original_read = runner.read
    monkeypatch.setattr(
        runner,
        "read",
        lambda path: lock if path == runner.LOCK_PATH else original_read(path),
    )

    def digest(path):
        if str(path) == "/dataset":
            return "wrong" if mutation == "dataset" else "dataset"
        if str(path) == "/agent/file.py":
            return "wrong" if mutation == "source" else "source"
        return case_hash

    monkeypatch.setattr(runner, "digest", digest)
    parquet = ModuleType("pyarrow.parquet")
    parquet.read_table = lambda path: SimpleNamespace(
        to_pylist=lambda: [dict(row, changed=True)] if mutation == "row" else [row]
    )
    arrow = ModuleType("pyarrow")
    arrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args: json.dumps(
            [
                {
                    "Architecture": "amd64" if mutation == "image" else "arm64",
                    "Id": "immutable",
                }
            ]
        ).encode(),
    )
    monkeypatch.setattr(runner, "validate_grading_sources", lambda *args: {})
    if mutation == "task_count":
        c["batch"]["size"] = 2
    if mutation:
        with pytest.raises(ValueError):
            runner.verify_inputs(c, tmp_path)
    else:
        assert runner.verify_inputs(c, tmp_path)["official_sources"] == "PASS"


@pytest.mark.parametrize("phase", ["query", "tool"])
def test_real_wall_alarm_bypasses_agent_error_recovery_and_closes_event(
    tmp_path, phase
):
    """A real timer must abort rather than become a recoverable tool/HTTP error."""
    code = """
import json, signal, time, sys
from pathlib import Path
from types import SimpleNamespace
from vllm_hust_benchmark.swe_bench_pro import AgentWallBudgetExceeded, _wall_budget_expired
from vllm_hust_benchmark.swe_bench_pro_timing import instrument
folder, phase = Path(sys.argv[1]), sys.argv[2]
def blocked(*args, **kwargs):
    time.sleep(5)
model = SimpleNamespace(config=SimpleNamespace(model_kwargs={}), query=blocked)
env = SimpleNamespace(execute=blocked)
instrument(model, env, folder, 'budget-program')
signal.signal(signal.SIGALRM, _wall_budget_expired)
signal.setitimer(signal.ITIMER_REAL, 0.05)
try:
    try:
        model.query([]) if phase == 'query' else env.execute('blocked tool')
    except Exception:
        print('RECOVERED_AND_CONTINUED')
except AgentWallBudgetExceeded as exc:
    print(type(exc).__name__, str(exc))
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    (folder / 'cleanup-ran').write_text('yes')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), phase],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert "AgentWallBudgetExceeded Frozen Agent wall budget reached" in result.stdout
    assert "RECOVERED_AND_CONTINUED" not in result.stdout
    assert (tmp_path / "cleanup-ran").read_text() == "yes"
    events = [
        json.loads(line)
        for line in (tmp_path / "agent-events-private.jsonl").read_text().splitlines()
    ]
    assert len(events) == 2
    assert events[-1]["event"] == (
        "agent_query_exception" if phase == "query" else "tool_exception"
    )
    assert events[-1]["error"] == "AgentWallBudgetExceeded"


@pytest.mark.parametrize("wall,expected", [(1200, 2400), (3600, 4200), (7200, 7800)])
def test_sandbox_outlives_agent_and_preserves_tool_contract(wall, expected):
    c = configuration()
    c["agent"]["wall_budget_seconds"] = wall
    captured = []
    marker = object()

    def actual_constructor(**kwargs):
        captured.append(kwargs)
        return marker

    assert (
        runner.create_task_environment(
            c, {"image_id": "immutable-image"}, actual_constructor
        )
        is marker
    )
    options = captured[0]
    assert options["container_timeout"] == str(expected) + "s"
    assert expected >= wall + 600
    assert options["timeout"] == c["agent"]["tool_timeout_seconds"]
    assert options["image"] == "immutable-image"
    assert options["run_args"] == [
        "--rm",
        "--entrypoint",
        "",
        "--network",
        c["container"]["network"],
        "--cpus",
        str(c["container"]["cpus"]),
        "--memory",
        c["container"]["memory"],
    ]
    assert c["agent"]["wall_budget_seconds"] == wall


@pytest.mark.parametrize("wall", [True, 0, -1, None])
def test_invalid_sandbox_budget_rejected_before_creation(wall):
    c = configuration()
    c["agent"]["wall_budget_seconds"] = wall

    def forbidden(**kwargs):
        raise AssertionError("No container may be created")

    with pytest.raises(ValueError):
        runner.create_task_environment(c, {"image_id": "image"}, forbidden)
