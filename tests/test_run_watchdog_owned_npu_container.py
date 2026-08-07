"""Tests for the watchdog-owned NPU container launcher script.

Review feedback (PR #148): the launcher script had zero tests.  These tests
cover argparse, exit-code propagation, preflight failures, and the cleanup
branch (keep_container=False) using subprocess monkeypatching — no real
docker daemon is required.
"""

from __future__ import annotations

import importlib.util
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


def test_build_parser_accepts_init_flag(launcher) -> None:
    parser = launcher.build_parser()
    args = parser.parse_args(
        [
            "--runner-name",
            "poy-180-21rc-npu0",
            "--name",
            "test",
            "--image",
            "img",
            "--init",
            "--",
            "python",
            "run.py",
        ]
    )
    assert args.init is True
    assert args.name == "test"
    assert args.image == "img"


def test_build_parser_defaults_init_false(launcher) -> None:
    parser = launcher.build_parser()
    args = parser.parse_args(["--name", "test", "--image", "img", "python", "run.py"])
    assert args.init is False


def test_run_returns_2_on_preflight_failure(launcher) -> None:
    """Invalid runner name triggers preflight ValueError -> exit code 2."""
    rc = launcher.run(
        [
            "--runner-name",
            "invalid-runner",
            "--name",
            "test",
            "--image",
            "img",
            "python",
            "run.py",
        ]
    )
    assert rc == 2


def test_run_returns_2_on_forbidden_volume(launcher) -> None:
    """Forbidden volume spec triggers preflight ValueError -> exit code 2."""
    rc = launcher.run(
        [
            "--runner-name",
            "poy-180-21rc-npu0",
            "--name",
            "test",
            "--image",
            "img",
            "--volume",
            "/dev:/dev",
            "python",
            "run.py",
        ]
    )
    assert rc == 2


def test_run_returns_container_exit_code_on_success(launcher) -> None:
    """When docker wait returns exit code N, the launcher returns N."""
    fake_container_id = "abc123"

    def fake_run(cmd, **kwargs):
        if "create" in cmd:
            return _FakeCompleted(stdout=fake_container_id, returncode=0)
        if "inspect" in cmd:
            # Minimal valid inspect payload (device mapping will fail
            # validation, but we patch validate_container_inspect below).
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
        rc = launcher.run(
            [
                "--runner-name",
                "poy-180-21rc-npu0",
                "--name",
                "test",
                "--image",
                "img",
                "python",
                "run.py",
            ]
        )
    assert rc == 0


def test_run_returns_nonzero_when_container_fails(launcher) -> None:
    """When docker wait returns non-zero exit code, the launcher returns it."""
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
        rc = launcher.run(
            [
                "--runner-name",
                "poy-180-21rc-npu0",
                "--name",
                "test",
                "--image",
                "img",
                "python",
                "run.py",
            ]
        )
    assert rc == 42


def test_run_returns_1_on_docker_create_failure(launcher) -> None:
    """When docker create fails (CalledProcessError), the launcher returns 1."""

    def fake_run(cmd, **kwargs):
        import subprocess

        if "create" in cmd:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return _FakeCompleted(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        rc = launcher.run(
            [
                "--runner-name",
                "poy-180-21rc-npu0",
                "--name",
                "test",
                "--image",
                "img",
                "python",
                "run.py",
            ]
        )
    assert rc == 1


def test_run_cleans_up_container_when_not_keep_container(launcher) -> None:
    """When --keep-container is not set, docker rm --force is called."""
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
        launcher.run(
            [
                "--runner-name",
                "poy-180-21rc-npu0",
                "--name",
                "test",
                "--image",
                "img",
                "python",
                "run.py",
            ]
        )
    assert any("rm" in call and "--force" in call for call in rm_calls)


def test_run_skips_cleanup_when_keep_container(launcher) -> None:
    """When --keep-container is set, docker rm is NOT called."""
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
        launcher.run(
            [
                "--runner-name",
                "poy-180-21rc-npu0",
                "--name",
                "test",
                "--image",
                "img",
                "--keep-container",
                "python",
                "run.py",
            ]
        )
    assert not rm_calls


def test_run_cleans_up_even_on_validation_failure(launcher) -> None:
    """finally block runs docker rm even when validate_container_inspect raises."""
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
        rc = launcher.run(
            [
                "--runner-name",
                "poy-180-21rc-npu0",
                "--name",
                "test",
                "--image",
                "img",
                "python",
                "run.py",
            ]
        )
    assert rc == 1
    assert any("rm" in call and "--force" in call for call in rm_calls)
