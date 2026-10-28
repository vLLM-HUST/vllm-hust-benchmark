"""Worker safety/software fixtures; never execute generated code on host."""

import importlib.util
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/run_structeval_matplotlib_worker.py"


def module():
    spec = importlib.util.spec_from_file_location("structeval_worker", ENTRY)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_compiler_diagnostic_retains_exact_command_and_return():
    result = object()
    command = ["compiler", "original.typ", "original.pdf"]

    def original(actual, **kwargs):
        assert actual is command
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return result

    assert (
        module().run_retaining_errors(
            original, command, check=True, capture_output=True, text=True
        )
        is result
    )


def test_compiler_diagnostic_retains_same_exception_without_retry(capfd):
    error = subprocess.CalledProcessError(
        1, ["compiler"], stderr="inert compiler fixture"
    )
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        raise error

    with pytest.raises(subprocess.CalledProcessError) as caught:
        module().run_retaining_errors(original, ["compiler"], check=True)
    assert caught.value is error and len(calls) == 1
    assert "inert compiler fixture" in capfd.readouterr().err


def test_svg_relative_module_identity_checked_before_any_import(tmp_path, monkeypatch):
    loaded = module()
    renderer = tmp_path / "render_svg.py"
    renderer.write_bytes(b"inert renderer fixture")
    helper = tmp_path / "render_utils.py"
    helper.write_bytes(b"inert helper fixture")
    code = tmp_path / "code.txt"
    code.write_bytes(b"inert SVG payload")
    monkeypatch.setattr(
        loaded,
        "RENDERERS",
        {"svg": (hashlib.sha256(renderer.read_bytes()).hexdigest(), "unused")},
    )
    monkeypatch.setattr(
        loaded,
        "EXTRA_MODULES",
        {"svg": {"render_utils.py": hashlib.sha256(helper.read_bytes()).hexdigest()}},
    )
    digest = hashlib.sha256(code.read_bytes()).hexdigest()
    assert loaded.verify_inputs(renderer, code, digest, "svg")[2] == "inert SVG payload"
    helper.write_bytes(b"changed helper")
    with pytest.raises(ValueError, match="relative module"):
        loaded.verify_inputs(renderer, code, digest, "svg")


def test_browser_asset_guard_preserves_factory_arguments_and_inline_resources():
    loaded = module()
    registered = []
    factory_calls = []
    requested = []

    class Context:
        async def route(self, pattern, callback):
            assert pattern == "**/*"
            registered.append(callback)

    context = Context()
    original = (object(), context, object(), object())

    async def start(*args, **kwargs):
        factory_calls.append((args, kwargs))
        return original

    async def exercise():
        result = await loaded.browser_without_external_assets(
            start, requested, headless=True
        )
        assert result == original
        for url, expected in [
            ("https://example.invalid/a.svg", "abort"),
            ("data:image/png;base64,fixture", "continue"),
        ]:
            operations = []

            class Route:
                request = SimpleNamespace(url=url)

                async def abort(self):
                    operations.append("abort")

                async def continue_(self):
                    operations.append("continue")

            await registered[0](Route())
            assert operations == [expected]

    asyncio.run(exercise())
    assert factory_calls == [((), {"headless": True})]
    assert requested == ["https://example.invalid/a.svg"]


def test_plan_has_no_runtime_data_requirement(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
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


@pytest.mark.parametrize("task_id", ["../escape", "a/b", "'injection", "", "a" * 65])
def test_task_path_injection_rejected(task_id):
    with pytest.raises(ValueError, match="identifier"):
        module().execute(SimpleNamespace(task_id=task_id, execute=False))


def test_changed_original_source_rejected_without_import(tmp_path):
    source = tmp_path / "changed.py"
    source.write_text('raise RuntimeError("must never import")\n')
    with pytest.raises(ValueError, match="unchanged pinned original"):
        module().verify_inputs(source, tmp_path / "absent.txt", "0" * 64)


def test_execute_rejects_host_before_reading_any_input(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
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
            "--execute",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "worker rejected" in result.stderr
    assert not list(tmp_path.iterdir())
