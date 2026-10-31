"""Controller command/ownership fixtures only; no Docker or candidate execution."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def module():
    spec = importlib.util.spec_from_file_location(
        "matplotlib_controller", ROOT / "scripts/run_structeval_matplotlib_isolated.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_worker_command_has_only_private_readonly_data_and_fixed_bounds():
    command = module().worker_command(
        "sha256:" + "0" * 64,
        "owned-fixture",
        "software",
        "inputs",
        "000800",
        "0" * 64,
        "outputs",
    )
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--user") + 1] == "65534:65534"
    assert command[command.index("--memory") + 1] == "2g"
    assert command[command.index("--pids-limit") + 1] == "256"
    assert "software:/software:ro" in command and "inputs:/input:ro" in command
    assert "outputs:/output" in command
    assert "--cap-add" not in command and "--privileged" not in command
    assert "--device" not in command and "--rm" not in command


@pytest.mark.parametrize("wrong", ["label", "id", "name"])
def test_timeout_cannot_stop_other_container(monkeypatch, wrong):
    loaded = module()
    actual = {
        "Id": "owned-id",
        "Name": "/owned-worker",
        "Config": {"Labels": {loaded.OWNER_LABEL: "owned"}},
    }
    if wrong == "label":
        actual["Config"]["Labels"][loaded.OWNER_LABEL] = "another"
    elif wrong == "id":
        actual["Id"] = "another-id"
    else:
        actual["Name"] = "/another-worker"
    calls = []

    def fake(*args):
        calls.append(args)
        return json.dumps([actual])

    monkeypatch.setattr(loaded, "docker", fake)
    with pytest.raises(ValueError, match="exact owned identity"):
        loaded.stop_owned("owned-id", "owned")
    assert calls == [("inspect", "owned-id")]


def test_plan_has_no_docker_data_or_model_requirement(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_structeval_matplotlib_isolated.py"),
            "--image",
            "sha256:" + "0" * 64,
            "--task-id",
            "000800",
            "--renderer",
            str(tmp_path / "absent.py"),
            "--code-file",
            str(tmp_path / "absent.txt"),
            "--code-sha256",
            "0" * 64,
            "--output",
            str(tmp_path / "not-created"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY"
    assert not list(tmp_path.iterdir())
