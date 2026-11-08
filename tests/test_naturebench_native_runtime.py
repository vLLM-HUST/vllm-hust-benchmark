"""Software-only fixtures, not original scientific tasks or neural execution."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "naturebench_native_runtime", ROOT / "scripts/prepare_naturebench_native_runtime.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

SOLVE = b"""def launch(container_name):
    docker_cmd = [
            "docker", "run",
            "--name", container_name,
    ]
    docker_cmd.extend(["--add-host=host.docker.internal:host-gateway"])
    env_keys = (
        "ANTHROPIC_API_BASE", "ANTHROPIC_MODEL",
    )
    return docker_cmd
"""


@pytest.fixture
def original(tmp_path, monkeypatch):
    root = tmp_path / "original"
    root.mkdir()
    files = {
        "solve.py": SOLVE,
        "evaluator.py": b"ORIGINAL_SCORER = True\n",
        "LICENSE": b"fixture license\n",
    }
    pins = []
    for name, raw in files.items():
        (root / name).write_bytes(raw)
        pins.append(
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest(),
            }
        )
    manifest = tmp_path / "pins.json"
    manifest.write_text(
        json.dumps({"revision": "fixture-original-revision", "files": pins})
    )
    monkeypatch.setattr(runner, "PINS", manifest)
    return root


@pytest.mark.parametrize("profile", runner.PROFILES)
def test_exact_profile_and_original_preservation(original, tmp_path, profile):
    output = tmp_path / "prepared"
    before = {p.name: p.read_bytes() for p in original.iterdir()}
    result = runner.prepare(original, output, profile)
    assert runner.verify_runtime(original, output) == output / "source"
    assert {p.name: p.read_bytes() for p in original.iterdir()} == before
    assert (output / "source/evaluator.py").read_bytes() == before["evaluator.py"]
    assert result["source_checkbox_eligible"] is False
    assert result["model_calls"] == 0
    assert result["task_data_copied"] is False
    expected = [] if profile == "original" else ["solve.py"]
    assert result["changed_original_files"] == expected
    if profile.startswith("owned-hostnet"):
        raw = (output / "source/solve.py").read_bytes()
        assert b'"--network", "host"' in raw
        assert b"host.docker.internal:127.0.0.1" in raw
    assert result["task_container_context_capacity_forwarded"] == profile.endswith(
        "-context"
    )
    if profile.endswith("-context"):
        assert (
            b"CLAUDE_CODE_MAX_CONTEXT_TOKENS"
            in (output / "source/solve.py").read_bytes()
        )


def test_unknown_or_changed_original_network_fails_closed(original, tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        runner.prepare(original, tmp_path / "unknown", "unknown")
    with pytest.raises(ValueError, match="no heuristic"):
        runner.network_profile(b"changed original Docker construction", "owned-hostnet")
    assert not (tmp_path / "unknown").exists()


def test_existing_output_and_source_nested_output_rejected(original, tmp_path):
    output = tmp_path / "prepared"
    runner.prepare(original, output)
    before = (output / "runtime-profile.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.prepare(original, output)
    assert (output / "runtime-profile.json").read_bytes() == before
    with pytest.raises(ValueError, match="separate"):
        runner.prepare(original, original / "nested")


@pytest.mark.parametrize("tamper", ["source", "manifest", "extra_module"])
def test_runtime_tampering_rejected(original, tmp_path, tamper):
    output = tmp_path / "prepared"
    runner.prepare(original, output, "owned-hostnet")
    if tamper == "source":
        (output / "source/evaluator.py").write_bytes(b"CHANGED_SCORER = True\n")
    elif tamper == "manifest":
        path = output / "runtime-profile.json"
        data = json.loads(path.read_text())
        data["files"][0]["path"] = "../other.py"
        path.write_text(json.dumps(data))
    else:
        (output / "source/extra.py").write_text(
            "raise RuntimeError('fixture never executed')"
        )
    with pytest.raises(ValueError):
        runner.verify_runtime(original, output)


def test_cli_offline_plan_creates_nothing(original, tmp_path, monkeypatch, capsys):
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--author-root",
            str(original),
            "--output",
            str(output),
            "--profile",
            "owned-hostnet",
        ],
    )
    runner.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "OFFLINE_RUNTIME_PROFILE_PLAN"
    assert not output.exists()


def test_changed_original_is_not_accepted(original, tmp_path):
    (original / "evaluator.py").write_text("CHANGED = True")
    with pytest.raises(ValueError):
        runner.prepare(original, tmp_path / "not-created")
