import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "auxiliary_probe",
    Path(__file__).resolve().parents[1] / "scripts/probe_marble_auxiliary_client.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_plan_never_imports_model_or_installs(tmp_path, capsys):
    assert MODULE.main(["--client-python", str(tmp_path / "missing")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PLAN_ONLY"
    assert not report["model_loaded"] and not report["packages_installed"]
    assert not list(tmp_path.iterdir())


def test_vendor_torch_and_original_auxiliary_libraries_are_explicit():
    pins = MODULE.requirements()
    assert pins["torch"] == "2.10.0+cpu"
    assert pins["torch-npu"] == "2.10.0.post4"
    assert pins["sentence-transformers"] == "3.4.1"
    assert pins["transformers"] == "4.49.0"
    assert len(pins) == 11


def test_venv_interpreter_symlink_is_not_resolved_to_base(
    tmp_path, monkeypatch, capsys
):
    link = tmp_path / "client-python"
    try:
        link.symlink_to(sys.executable)
    except OSError:
        pytest.skip("host cannot create interpreter symlink")
    calls = []

    def fixture_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"versions": MODULE.requirements(), "npu_initialized": False}
            ),
            stderr="",
        )

    monkeypatch.setattr(MODULE.subprocess, "run", fixture_run)
    assert MODULE.main(["--client-python", str(link), "--execute"]) == 0
    assert len(calls) == 2 and all(
        command[0] == str(link.absolute()) for command in calls
    )
    assert not json.loads(capsys.readouterr().out)["inference_executed"]
