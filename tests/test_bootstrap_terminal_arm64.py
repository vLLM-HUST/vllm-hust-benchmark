"""Safety and reproducibility boundaries for the single-task entry."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/bootstrap_terminal_arm64.py"
spec = importlib.util.spec_from_file_location("terminal_bootstrap", SOURCE)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_started_attempt_refuses_before_environment_or_credentials(tmp_path):
    (tmp_path / "model-started.json").write_text("{}")
    with patch.object(bootstrap, "verify_preparation") as check:
        with pytest.raises(RuntimeError, match="already started"):
            bootstrap.run_once(tmp_path, "model", {"api_key_file": "/must/not/read"})
    check.assert_not_called()


def test_bad_download_never_persists(tmp_path):
    target = tmp_path / "tool"
    with patch.object(bootstrap, "fetch", return_value=b"wrong artifact"):
        with pytest.raises(RuntimeError, match="hash mismatch"):
            bootstrap.checked_download("https://example.invalid/tool", target, "0" * 64)
    assert not target.exists()


def test_existing_bad_artifact_is_not_replaced(tmp_path):
    target = tmp_path / "tool"
    target.write_bytes(b"original bad artifact")
    with patch.object(bootstrap, "fetch") as fetch:
        with pytest.raises(RuntimeError, match="hash mismatch"):
            bootstrap.checked_download("https://example.invalid/tool", target, "0" * 64)
    fetch.assert_not_called()
    assert target.read_bytes() == b"original bad artifact"


def test_job_keeps_real_verifier_and_does_not_expose_credential(tmp_path):
    job = bootstrap.config(
        tmp_path,
        "model",
        {
            "api_base": "http://model.invalid/v1",
            "model": "qwen3.5-35b",
            "api_key_file": "/private/credential",
        },
    )
    encoded = json.dumps(job)
    assert "/private/credential" not in encoded
    assert "verifier" not in job or not job["verifier"].get("disable", False)
    assert job["tasks"] == [{"path": str(tmp_path / "task")}]
    assert job["n_attempts"] == 1
    assert job["retry"]["max_retries"] == 0
    assert job["agents"][0]["kwargs"]["enable_summarize"] is False
    assert not job.get("extra_instructions")
    assert not job["agents"][0].get("skills")


def test_identity_change_blocks_execution(tmp_path):
    (tmp_path / "task").mkdir()
    (tmp_path / "task/instruction.md").write_text("altered")
    (tmp_path / "prepared.json").write_text(
        json.dumps({"bindings": {"task/instruction.md": "0" * 64}})
    )
    with patch.object(bootstrap, "docker") as docker:
        with pytest.raises(RuntimeError, match="source changed"):
            bootstrap.verify_preparation(tmp_path)
    docker.assert_not_called()
