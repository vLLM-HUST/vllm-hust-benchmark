"""Candidate configuration integrity; no model, sandbox, or task execution."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
spec = importlib.util.spec_from_file_location(
    "audit_pyramidkv_terminal", SCRIPTS / "audit_pyramidkv_terminal.py"
)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
sys.modules[spec.name] = source
spec = importlib.util.spec_from_file_location(
    "terminal_plan", SCRIPTS / "prepare_pyramidkv_terminal.py"
)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def test_matched_arms_only_change_artifact_destinations():
    jobs = [
        plan.job_config(arm, ["task-a", "task-b"], "http://127.0.0.1:8000/v1")
        for arm in ("b0", "b1")
    ]
    for job in jobs:
        del job["job_name"], job["jobs_dir"]
    assert jobs[0] == jobs[1]
    job = jobs[0]
    assert job["retry"]["max_retries"] == 0
    assert job["n_attempts"] == 1
    assert job["n_concurrent_trials"] == 1
    assert job["environment"]["delete"] is True
    assert job["environment"]["force_build"] is False
    assert job["verifier"]["disable"] is False
    options = job["agents"][0]["kwargs"]
    assert options["enable_summarize"] is False
    assert (
        options["llm_call_kwargs"]["extra_body"]["chat_template_kwargs"][
            "enable_thinking"
        ]
        is False
    )
    assert not any(key.startswith("override_") for key in job["environment"])


@pytest.mark.parametrize(
    "url",
    [
        "file:///v1",
        "http:///v1",
        "https://user:synthetic@example.com/v1",  # pragma: allowlist secret
        "https://example.com/v1?key=synthetic",
        "https://example.com/v1#fragment",
        "https://example.com",
    ],
)
def test_reject_credentials_and_invalid_endpoint(url):
    with pytest.raises(ValueError, match="base URL"):
        plan.job_config("b0", ["task"], url)


@pytest.mark.parametrize(
    "tasks", [[], ["same", "same"], ["../escape"], ["/absolute"], ["has/slash"]]
)
def test_reject_changed_or_unsafe_task_inventory(tasks):
    with pytest.raises(ValueError, match="Task IDs"):
        plan.job_config("b0", tasks, "http://localhost:8000/v1")


def test_image_pin_preserves_task_limits_and_instruction_metadata():
    import tomllib

    text = """schema_version="1.1"
[task]
name="terminal-bench/example"
[environment]
docker_image = "alexgshaw/example:20260430"
cpus=4
memory_mb=8192
allow_internet=true
[agent]
timeout_sec=12000.0
[verifier]
timeout_sec=360.0
"""
    pinned = "alexgshaw/example@sha256:" + "a" * 64
    result = plan.pin_task_image(text, "alexgshaw/example:20260430", pinned)
    parsed = tomllib.loads(result)
    assert parsed["environment"]["docker_image"] == pinned
    parsed["environment"]["docker_image"] = "alexgshaw/example:20260430"
    assert parsed == tomllib.loads(text)


def test_refuse_unexpected_image_mapping():
    with pytest.raises(ValueError, match="audited mapping"):
        plan.pin_task_image(
            '[environment]\ndocker_image="different:tag"\n',
            "expected:tag",
            "expected@sha256:" + "b" * 64,
        )


def test_refuse_existing_plan_before_reading_assets(tmp_path):
    with pytest.raises(ValueError, match="existing plan"):
        plan.prepare(tmp_path / "nonexistent", tmp_path, "http://localhost:8000/v1")
