from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify-owned-runtime-and-exec.py"


def _load_preflight():
    spec = importlib.util.spec_from_file_location("owned_runtime_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_attests_runtime_before_exact_runner_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_preflight()
    expected = tmp_path / "expected.json"
    expected.write_text(
        json.dumps(
            {
                "core_commit": "core",
                "backend_commit": "backend",
                "packages": {"vllm": "0.18.0"},
            }
        ),
        encoding="utf-8",
    )
    expected_sha256 = module.hashlib.sha256(expected.read_bytes()).hexdigest()
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "startup_instance_id": "startup",
                "container_id": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "actual.json"
    runner = [
        "/usr/bin/env",
        "KEY=value with spaces",
        "bash",
        "-c",
        "literal;not-a-wrapper-shell",
        "$(not-expanded)",
    ]
    monkeypatch.setattr(
        module,
        "git_identity",
        lambda path: {
            "path": str(path),
            "commit": "core" if path == module.SOURCE_PATHS["core"] else "backend",
            "clean": True,
        },
    )
    monkeypatch.setattr(module, "package_versions", lambda names: {"vllm": "0.18.0"})
    observed: list[list[str]] = []

    class ExecReached(RuntimeError):
        pass

    def fake_exec(file: str, argv: list[str]) -> None:
        assert output.is_file()
        observed.append([file, *argv])
        raise ExecReached

    monkeypatch.setattr(module.os, "execvp", fake_exec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--expected",
            str(expected),
            "--expected-sha256",
            expected_sha256,
            "--container-identity",
            str(identity),
            "--output",
            str(output),
            "--startup-instance-id",
            "startup",
            "--",
            *runner,
        ],
    )
    with pytest.raises(ExecReached):
        module.main()
    assert observed == [[runner[0], *runner]]
    actual = json.loads(output.read_text(encoding="utf-8"))
    assert actual["startup_instance_id"] == "startup"
    assert actual["container_id"] == "c" * 64
    assert actual["expected_contract_sha256"] == expected_sha256
