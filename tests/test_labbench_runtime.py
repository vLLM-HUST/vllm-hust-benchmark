"""Configuration-only tests, not installed official environment evidence."""

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "labbench_runtime", ROOT / "scripts/prepare_labbench_runtime.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_uses_original_frozen_lock_no_system_install():
    command = MODULE.install_command("uv", "/python", "/publisher")
    assert command == [
        "uv",
        "sync",
        "--frozen",
        "--no-dev",
        "--python",
        "/python",
        "--project",
        "/publisher",
    ]


def test_rejects_system_site_packages(tmp_path):
    (tmp_path / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    with pytest.raises(ValueError, match="disabled"):
        MODULE.isolated_python(tmp_path)


def test_requires_explicit_environment(tmp_path):
    with pytest.raises(ValueError, match="isolated"):
        MODULE.isolated_python(tmp_path)


def test_compat_is_pinned_isolated_and_does_not_resolve_other_dependencies(monkeypatch):
    monkeypatch.setattr(
        MODULE, "isolated_python", lambda environment: "/isolated/bin/python"
    )
    calls = []
    monkeypatch.setattr(
        MODULE.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    MODULE.install_compat("uv", "/isolated")
    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "--no-deps",
            "--python",
            "/isolated/bin/python",
            "setuptools==75.1.0",
        ]
    ]
