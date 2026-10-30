"""Original Agent entry contracts only, not task execution or scoring evidence."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

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
