"""Inert local record/config fixtures; no Docker, publisher runtime or model."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "swe_original_agent", SCRIPTS / "run_swebench_original_agent.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def records():
    return [
        {
            "instance_id": f"fixture__repo-{number}",
            "repo": "fixture/repo",
            "base_commit": "fixture-base",
            "version": "fixture-version",
            "problem_statement": "Inert issue\nwith exact whitespace.  ",
            "patch": "inert reference",
            "test_patch": "inert reference tests",
            "FAIL_TO_PASS": '["test_failure"]',
            "PASS_TO_PASS": '["test_pass"]',
            "hints_text": "inert original hint",
            "additional_original_field": number,
        }
        for number in range(5)
    ]


def config():
    return {
        "agent": {
            "step_limit": 250,
            "cost_limit": 3,
            "system_template": "inert",
            "instance_template": "{{ task }}",
        },
        "environment": {
            "cwd": "/testbed",
            "timeout": 60,
            "interpreter": ["bash", "-c"],
            "env": {"BASH_ENV": "/root/.bashrc"},
        },
        "model": {
            "model_name": "original-fixture",
            "model_kwargs": {"parallel_tool_calls": True, "drop_params": True},
        },
    }


def test_original_physical_order_and_all_fields_preserved(tmp_path):
    rows = records()
    path = tmp_path / "inert.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert runner.select_records(path) == rows[:3]
    assert runner.select_records(path, full=True) == rows
    assert path.read_text(encoding="utf-8") == json.dumps(rows)


def test_original_parquet_retains_order_and_all_fields_without_json_export(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    original = records()
    path = tmp_path / "inert-original.parquet"
    pq.write_table(pa.Table.from_pylist(original), path, row_group_size=2)
    before = path.read_bytes()
    assert runner.select_records(path) == original[:3]
    assert runner.select_records(path, full=True) == original
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad", [[], {}, [{"instance_id": "incomplete"}]])
def test_incomplete_original_records_rejected(tmp_path, bad):
    path = tmp_path / "inert.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.select_records(path)


def test_duplicate_original_ids_rejected_not_replaced(tmp_path):
    rows = records()
    rows[1]["instance_id"] = rows[0]["instance_id"]
    path = tmp_path / "inert.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        runner.select_records(path)


def test_deployment_bindings_do_not_change_original_agent_or_tool_budgets():
    original = config()
    before = copy.deepcopy(original)
    actual = runner.execution_config(
        original, "http://127.0.0.1:19501/v1", "fixture-model"
    )
    assert original == before
    assert actual["agent"] == original["agent"]
    for key, value in original["environment"].items():
        assert actual["environment"][key] == value
    assert actual["environment"]["run_args"] == runner.RUN_ARGS
    assert "--network=none" in runner.RUN_ARGS and "--cap-drop=ALL" in runner.RUN_ARGS
    assert actual["model"]["model_kwargs"]["parallel_tool_calls"] is True
    assert "max_tokens" not in actual["model"]["model_kwargs"]
    assert actual["model"]["cost_tracking"] == "ignore_errors"


@pytest.mark.parametrize(
    "identity", ["../outside", "/absolute", "..", "nested/path", "nested\\path"]
)
def test_external_instance_ids_cannot_escape_result_directory(tmp_path, identity):
    rows = records()
    rows[0]["instance_id"] = identity
    path = tmp_path / "inert.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.select_records(path)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.example.com/v1",
        "http://10.0.0.1/v1",
        "http://user:pass@127.0.0.1/v1",  # pragma: allowlist secret - inert rejected authentication fixture
        "http://127.0.0.1/v1?key=fixture",
        "http://127.0.0.1/not-v1",
    ],
)
def test_remote_or_credential_bearing_endpoints_rejected(endpoint):
    with pytest.raises(ValueError):
        runner.execution_config(config(), endpoint, "fixture-model")


def test_forwarding_credentials_into_generated_code_rejected():
    original = config()
    original["environment"]["forward_env"] = ["OPENAI_API_KEY"]
    with pytest.raises(ValueError, match="credentials"):
        runner.execution_config(original, "http://127.0.0.1/v1", "fixture-model")


def test_plan_never_imports_missing_author_runtime_or_creates_results(tmp_path):
    path = tmp_path / "inert.json"
    path.write_text(json.dumps(records()), encoding="utf-8")
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_swebench_original_agent.py"),
            "--source-id",
            "SWE_VERIFIED",
            "--records-json",
            str(path),
            "--agent-root",
            str(tmp_path / "absent-agent"),
            "--harness-root",
            str(tmp_path / "absent-harness"),
            "--output",
            str(output),
            "--endpoint",
            "http://127.0.0.1:19501/v1",
            "--model",
            "fixture-model",
            "--docker-url",
            "unix:///fixture.sock",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY" and report["model_calls"] == 0
    assert not report["source_checkbox_eligible"] and not output.exists()
    assert report["original_instance_ids"] == [r["instance_id"] for r in records()[:3]]
