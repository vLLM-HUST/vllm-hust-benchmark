"""Tests for the watchdog-owned NPU container launcher script.

These tests cover argparse, RUNNER_NAME resolution, exit-code propagation,
preflight failures, and the cleanup branch (keep_container=False) using
subprocess monkeypatching — no real docker daemon is required.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run-watchdog-owned-npu-container.py"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "run_watchdog_owned_npu_container", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCompleted:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


@pytest.fixture
def launcher():
    return _load_launcher()


def _run_args(name: str = "test", image: str = "img", **extra) -> list[str]:
    args = ["--name", name, "--image", image]
    for key, value in extra.items():
        if key == "volumes":
            for vol in value:
                args.extend(["--volume", vol])
        elif key == "envs":
            for env in value:
                args.extend(["--env", env])
        elif key == "keep":
            args.append("--keep-container")
    args.extend(["python", "run.py"])
    return args


def test_build_parser_accepts_required_flags(launcher) -> None:
    parser = launcher.build_parser()
    args = parser.parse_args(
        ["--name", "test", "--image", "img", "--keep-container", "python", "run.py"]
    )
    assert args.name == "test"
    assert args.image == "img"
    assert args.keep_container is True
    assert args.command == ["python", "run.py"]


def test_build_parser_accepts_repeated_volume_and_env(launcher) -> None:
    parser = launcher.build_parser()
    args = parser.parse_args(
        [
            "--name",
            "test",
            "--image",
            "img",
            "--volume",
            "/a:/workspace",
            "--volume",
            "/b:/tmp",
            "--env",
            "A=1",
            "python",
            "run.py",
        ]
    )
    assert args.volume == ["/a:/workspace", "/b:/tmp"]
    assert args.env == ["A=1"]


def test_run_returns_2_when_runner_name_unset(launcher, monkeypatch) -> None:
    monkeypatch.delenv("RUNNER_NAME", raising=False)
    rc = launcher.run(_run_args())
    assert rc == 2


def test_run_returns_2_on_preflight_failure(launcher, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_NAME", "invalid-runner")
    rc = launcher.run(_run_args())
    assert rc == 2


def test_run_returns_2_on_forbidden_volume(launcher, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_NAME", "poy-180-21rc-npu0")
    rc = launcher.run(_run_args(volumes=["/dev:/dev"]))
    assert rc == 2


def test_run_returns_2_on_reserved_env(launcher, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_NAME", "poy-180-21rc-npu0")
    rc = launcher.run(_run_args(envs=["ASCEND_RT_VISIBLE_DEVICES=0"]))
    assert rc == 2


def test_run_returns_container_exit_code_on_success(launcher, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_NAME", "poy-180-21rc-npu0")
    fake_container_id = "abc123"

    def fake_run(cmd, **kwargs):
        if "create" in cmd:
            return _FakeCompleted(stdout=fake_container_id, returncode=0)
        if "inspect" in cmd:
            return _FakeCompleted(stdout="[]", returncode=0)
        if "start" in cmd:
            return _FakeCompleted(returncode=0)
        if "wait" in cmd:
            return _FakeCompleted(stdout="0\n", returncode=0)
        if "logs" in cmd:
            return _FakeCompleted(returncode=0)
        if "rm" in cmd:
            return _FakeCompleted(returncode=0)
        return _FakeCompleted(returncode=0)

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("json.loads", return_value=[{"Config": {}, "HostConfig": {}}]),
        patch.object(launcher, "validate_container_inspect"),
    ):
        rc = launcher.run(_run_args())
    assert rc == 0


def test_run_returns_nonzero_when_container_fails(launcher, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_NAME", "poy-180-21rc-npu0")
    fake_container_id = "abc123"

    def fake_run(cmd, **kwargs):
        if "create" in cmd:
            return _FakeCompleted(stdout=fake_container_id, returncode=0)
        if "inspect" in cmd:
            return _FakeCompleted(stdout="[]", returncode=0)
        if "start" in cmd:
            return _FakeCompleted(returncode=0)
        if "wait" in cmd:
            return _FakeCompleted(stdout="42\n", returncode=0)
        if "logs" in cmd:
            return _FakeCompleted(returncode=0)
        if "rm" in cmd:
            return _FakeCompleted(returncode=0)
        return _FakeCompleted(returncode=0)

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("json.loads", return_value=[{"Config": {}, "HostConfig": {}}]),
        patch.object(launcher, "validate_container_inspect"),
    ):
        rc = launcher.run(_run_args())
    assert rc == 42


def test_run_returns_1_on_docker_create_failure(launcher, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_NAME", "poy-180-21rc-npu0")

    def fake_run(cmd, **kwargs):
        if "create" in cmd:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return _FakeCompleted(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        rc = launcher.run(_run_args())
    assert rc == 1


def test_run_cleans_up_container_when_not_keep_container(launcher, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_NAME", "poy-180-21rc-npu0")
    fake_container_id = "abc123"
    rm_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if "create" in cmd:
            return _FakeCompleted(stdout=fake_container_id, returncode=0)
        if "inspect" in cmd:
            return _FakeCompleted(stdout="[]", returncode=0)
        if "start" in cmd:
            return _FakeCompleted(returncode=0)
        if "wait" in cmd:
            return _FakeCompleted(stdout="0\n", returncode=0)
        if "logs" in cmd:
            return _FakeCompleted(returncode=0)
        if "rm" in cmd:
            rm_calls.append(list(cmd))
            return _FakeCompleted(returncode=0)
        return _FakeCompleted(returncode=0)

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("json.loads", return_value=[{"Config": {}, "HostConfig": {}}]),
        patch.object(launcher, "validate_container_inspect"),
    ):
        launcher.run(_run_args())
    assert any("rm" in call and "--force" in call for call in rm_calls)


def test_run_skips_cleanup_when_keep_container(launcher, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_NAME", "poy-180-21rc-npu0")
    fake_container_id = "abc123"
    rm_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if "create" in cmd:
            return _FakeCompleted(stdout=fake_container_id, returncode=0)
        if "inspect" in cmd:
            return _FakeCompleted(stdout="[]", returncode=0)
        if "start" in cmd:
            return _FakeCompleted(returncode=0)
        if "wait" in cmd:
            return _FakeCompleted(stdout="0\n", returncode=0)
        if "logs" in cmd:
            return _FakeCompleted(returncode=0)
        if "rm" in cmd:
            rm_calls.append(list(cmd))
            return _FakeCompleted(returncode=0)
        return _FakeCompleted(returncode=0)

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("json.loads", return_value=[{"Config": {}, "HostConfig": {}}]),
        patch.object(launcher, "validate_container_inspect"),
    ):
        launcher.run(_run_args(keep=True))
    assert not rm_calls


def test_run_cleans_up_even_on_validation_failure(launcher, monkeypatch) -> None:
    monkeypatch.setenv("RUNNER_NAME", "poy-180-21rc-npu0")
    fake_container_id = "abc123"
    rm_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        if "create" in cmd:
            return _FakeCompleted(stdout=fake_container_id, returncode=0)
        if "inspect" in cmd:
            return _FakeCompleted(stdout="[]", returncode=0)
        if "start" in cmd:
            return _FakeCompleted(returncode=0)
        if "wait" in cmd:
            return _FakeCompleted(stdout="0\n", returncode=0)
        if "logs" in cmd:
            return _FakeCompleted(returncode=0)
        if "rm" in cmd:
            rm_calls.append(list(cmd))
            return _FakeCompleted(returncode=0)
        return _FakeCompleted(returncode=0)

    def fake_validate(*args, **kwargs):
        raise ValueError("validation failed")

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("json.loads", return_value=[{"Config": {}, "HostConfig": {}}]),
        patch.object(launcher, "validate_container_inspect", side_effect=fake_validate),
    ):
        rc = launcher.run(_run_args())
    assert rc == 1
    assert any("rm" in call and "--force" in call for call in rm_calls)
