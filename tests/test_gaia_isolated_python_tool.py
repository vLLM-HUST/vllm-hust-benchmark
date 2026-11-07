"""Command contracts only; real kernel controls are separate server evidence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "gaia_sandbox", Path(__file__).parents[1] / "scripts/gaia_isolated_python_tool.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_guest_has_no_host_network_or_socket_mount():
    argv = module.command(
        docker_host="unix:///run/task/docker.sock",
        image="sha256:" + "a" * 64,
        framework_venv="/task/framework/venv",
        name="fixture",
    )
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--user") + 1] == "65534:65534"
    assert argv[argv.index("--pull") + 1] == "never"
    assert argv.count("-v") == 1
    assert argv[argv.index("-v") + 1] == "/task/framework/venv:/work/venv:ro"
    assert "--read-only" in argv and "no-new-privileges" in argv
    assert "sys.stdin.read()" in argv[-1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("image", "unfixed:latest"),
        ("docker_host", "tcp://remote:2375"),
        ("framework_venv", "relative"),
        ("framework_venv", "/task/a:/etc"),
    ],
)
def test_unsafe_operator_binding_rejected(field, value):
    args = dict(
        docker_host="unix:///run/task/docker.sock",
        image="sha256:" + "a" * 64,
        framework_venv="/task/venv",
        name="fixture",
    )
    args[field] = value
    with pytest.raises(ValueError):
        module.command(**args)


def test_generated_code_is_only_stdin_and_cleanup_is_owned(tmp_path, monkeypatch):
    calls = []

    def execute(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "prlimit":
            assert argv[1] == "--fsize=1048576:1048576"
            assert kwargs["input"] == b"print('fixture')"
            assert "print('fixture')" not in argv
            kwargs["stdout"].write(json.dumps({"tool_output": "fixture"}).encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", execute)
    out = tmp_path / "private"
    assert (
        module.run_code(
            "print('fixture')",
            docker_host="unix:///run/task/docker.sock",
            image="sha256:" + "a" * 64,
            framework_venv="/task/venv",
            output=out,
        )
        == "fixture"
    )
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "-H", "unix:///run/task/docker.sock", "kill", name]
    assert json.loads((out / "receipt-private.json").read_text())["returncode"] == 0
