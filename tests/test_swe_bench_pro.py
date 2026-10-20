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
        ("batch", "program_instances", 2),
        ("model", "api_base", "local"),
    ],
)
def test_invalid_protocol(section, key, value):
    c = configuration()
    c[section][key] = value
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
