"""Offline build contracts, not native replay or NPU evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "trace_build", ROOT / "scripts/build_tracelab_original.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_plan_never_accesses_source_toolchain_or_output(monkeypatch, tmp_path, capsys):
    output = tmp_path / "absent"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            "--official-source",
            "absent",
            "--cargo",
            "absent",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **kw: pytest.fail("No execution in plan")
    )
    module.main()
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0


def test_pinned_original_source_identity(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Cargo.lock").write_bytes(b"original-lock")
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "source_commit": "fixed",
                "files": [
                    {
                        "path": "Cargo.lock",
                        "bytes": 13,
                        "sha256": hashlib.sha256(b"original-lock").hexdigest(),
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(module, "PINS", pins)
    assert module.verify_source(source) == "fixed"
    (source / "Cargo.lock").write_bytes(b"modified-lock")
    with pytest.raises(ValueError):
        module.verify_source(source)


def test_verify_only_does_not_turn_build_into_replay(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "build"
    binary = output / "target/release/session_runner"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native-build-artifact")
    (output / "build-public.json").write_text(
        json.dumps(
            {
                "status": "ORIGINAL_LOCKED_BUILD_COMPLETE_NOT_REPLAY",
                "source_commit": "fixed",
                "binary_sha256": module.digest(binary),
            }
        )
    )
    monkeypatch.setattr(module, "verify_source", lambda root: "fixed")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            "--official-source",
            str(source),
            "--cargo",
            "unused",
            "--output",
            str(output),
            "--verify-only",
        ],
    )
    module.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ORIGINAL_BINARY_AND_SOURCE_VERIFIED_READ_ONLY"
    assert result["model_calls"] == 0
    binary.write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.main()
