"""No Torch import or reader assets are needed for software plan fixtures."""

import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
entry = importlib.import_module("verify_longmemeval_v2_processor")


def test_plan_has_no_asset_or_model_side_effects(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            entry.__file__,
            "--python",
            str(tmp_path / "missing-python"),
            "--reader-assets",
            str(tmp_path / "missing-assets"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY"
    assert not list(tmp_path.iterdir())


def test_neural_forward_weight_loading_and_gpu_are_guarded():
    ast.parse(entry.PROBE)
    for guard in (
        "torch.nn.Module._call_impl=forbidden",
        "PreTrainedModel.from_pretrained=forbidden",
        "torch.load=forbidden",
        "torch.version.cuda is None",
        "local_files_only=True",
    ):
        assert guard in entry.PROBE


def test_venv_interpreter_symlink_is_not_resolved_to_base_python(
    tmp_path, monkeypatch, capsys
):
    target = tmp_path / "base-python"
    target.write_text("fixture")
    alias = tmp_path / "venv-python"
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("symlink privilege unavailable")
    monkeypatch.setattr(entry, "verify_metadata", lambda root: 12)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--python",
            str(alias),
            "--reader-assets",
            str(tmp_path),
            "--execute",
        ],
    )

    def run(command, **kwargs):
        assert command[0] == str(alias.absolute())
        assert command[0] != str(target.absolute())
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "status": "ORIGINAL_READER_PROCESSOR_FIXTURE_PASSED_NOT_NATIVE_RAG",
                    "new_model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )

    monkeypatch.setattr(entry.subprocess, "run", run)
    entry.main()
    assert (
        json.loads(capsys.readouterr().out)[
            "original_metadata_files_verified_before_and_after"
        ]
        == 12
    )
