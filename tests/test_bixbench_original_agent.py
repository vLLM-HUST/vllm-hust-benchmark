"""Original Agent entry contracts only, not task execution or scoring evidence."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ENTRY = Path(__file__).resolve().parents[1] / "scripts/run_bixbench_original_agent.py"


def module():
    spec = importlib.util.spec_from_file_location("bixbench_agent", ENTRY)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_plan_without_data_dependencies_docker_or_model(tmp_path):
    command = [sys.executable, str(ENTRY)]
    for name in ("author-root", "data-root", "prepared", "output", "workspace"):
        command += ["--" + name, str(tmp_path / name)]
    command += [
        "--model",
        "openai/local-model",
        "--api-base",
        "http://127.0.0.1:1234/v1",
        "--docker-host",
        "unix:///not-present.sock",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY" and report["model_calls"] == 0
    assert not report["source_checkbox_eligible"] and not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://provider.example/v1",
        "http://0.0.0.0:123/v1",
        "http://user:password@localhost:123/v1",  # pragma: allowlist secret - rejection fixture, not credentials
        "http://localhost/v1",
        "http://localhost:123/v1?redirect=remote",
        "http://localhost:123/v1#fragment",
    ],
)
def test_nonlocal_or_ambiguous_model_binding_rejected(endpoint):
    with pytest.raises(ValueError, match="loopback"):
        module().validate_options(
            SimpleNamespace(
                api_base=endpoint,
                model="openai/local",
                limit=3,
                run_name="fixed-original",
            )
        )


def test_container_policy_preserves_original_command_and_scoped_bind(tmp_path):
    m = module()
    original = {
        "Image": m.IMAGE,
        "Cmd": ["sleep", "infinity"],
        "HostConfig": {"Binds": [f"{tmp_path / 'task'}:/workspace"]},
        "WorkingDir": "/workspace",
        "Tty": True,
    }
    actual = m.notebook_policy(original, tmp_path)
    assert actual["Cmd"] == original["Cmd"] and "User" not in original
    assert actual["User"] == "65534:65534"
    assert actual["HostConfig"]["NetworkMode"] == "none"
    assert actual["HostConfig"]["ReadonlyRootfs"]
    assert actual["HostConfig"]["CapDrop"] == ["ALL"]
    assert actual["HostConfig"]["Binds"] == original["HostConfig"]["Binds"]


@pytest.mark.parametrize("timeout", [0, 59, 1801, float("nan")])
def test_invalid_transport_timeout_rejected(timeout):
    with pytest.raises(ValueError, match="timeout"):
        module().validate_options(
            SimpleNamespace(
                request_timeout=timeout,
                api_base="http://127.0.0.1:1234/v1",
                model="openai/local",
                limit=3,
                run_name="fixed",
            )
        )


def test_container_policy_rejects_extra_device_or_mount(tmp_path):
    m = module()
    config = {
        "Image": m.IMAGE,
        "HostConfig": {
            "Binds": [f"{tmp_path}:/workspace"],
            "Devices": [{"PathOnHost": "/dev/fixture"}],
        },
    }
    with pytest.raises(ValueError):
        m.notebook_policy(config, tmp_path)


def test_container_policy_rejects_bind_escape(tmp_path):
    m = module()
    config = {
        "Image": m.IMAGE,
        "HostConfig": {"Binds": [f"{tmp_path.parent}:/workspace"]},
    }
    with pytest.raises(ValueError, match="escapes"):
        m.notebook_policy(config, tmp_path)


def recovered_prefix(tmp_path):
    config = {
        "run_name": "old-run",
        "paths": {"workspace_dir": "old-private-path"},
        "base_prompt": "Question: {question}; language: {language}",
        "notebook": {"language": "python"},
        "agent": {
            "agent_kwargs": {
                "llm_model": {
                    "name": "openai/local",
                    "timeout": 600,
                    "max_tokens": 4096,
                }
            }
        },
        "rollout": {"max_steps": 40},
    }
    rows = [
        {"question_id": f"q{i}", "question": f"question{i}", "ideal": f"reference{i}"}
        for i in range(3)
    ]
    (tmp_path / "runtime-binding.private.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "state.private.json").write_text(
        json.dumps(
            {
                "source_id": "BIXBENCH",
                "limit": 3,
                "actual_actor_model": "openai/local",
                "completed": ["q0"],
            }
        )
    )
    folder = tmp_path / "trajectories" / "old-run"
    folder.mkdir(parents=True)
    result = folder / "q0.json"
    result.write_text(
        json.dumps(
            {
                "problem_id": "q0",
                "ideal_answer": "reference0",
                "problem": "Question: question0; language: python",
                "model": "openai/local",
                "agent_answer": None,
                "metadata": rows[0],
            }
        )
    )
    result.with_suffix(".jsonl").write_text('{"original_transition":true}\n')
    return config, rows, result


def test_reuse_completed_prefix_keeps_no_answer_and_all_trace_bytes(tmp_path):
    config, rows, result = recovered_prefix(tmp_path)
    config["run_name"] = "new-run"
    config["paths"] = {"workspace_dir": "new-private-path"}
    llm = config["agent"]["agent_kwargs"]["llm_model"]
    llm.pop("timeout")
    llm["router_kwargs"] = {"timeout": 600}
    completed, files = module().reusable_trajectories(
        tmp_path, rows, "openai/local", config
    )
    assert completed == ["q0"]
    assert files == [result, result.with_suffix(".jsonl")]
    assert json.loads(result.read_text())["agent_answer"] is None


@pytest.mark.parametrize(
    "change", ["reference", "budget", "missing_trace", "nonprefix"]
)
def test_reuse_rejects_changed_or_partial_original_evidence(tmp_path, change):
    config, rows, result = recovered_prefix(tmp_path)
    if change == "reference":
        rows[0]["ideal"] = "different"
    elif change == "budget":
        config["rollout"]["max_steps"] = 3
    elif change == "missing_trace":
        result.with_suffix(".jsonl").unlink()
    else:
        state = json.loads((tmp_path / "state.private.json").read_text())
        state["completed"] = ["q1"]
        (tmp_path / "state.private.json").write_text(json.dumps(state))
    with pytest.raises((ValueError, FileNotFoundError)):
        module().reusable_trajectories(tmp_path, rows, "openai/local", config)
