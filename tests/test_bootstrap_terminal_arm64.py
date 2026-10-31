"""Safety and reproducibility boundaries for the single-task entry."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/bootstrap_terminal_arm64.py"
spec = importlib.util.spec_from_file_location("terminal_bootstrap", SOURCE)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_started_attempt_refuses_before_environment_or_credentials(tmp_path):
    (tmp_path / "model-started.json").write_text("{}")
    with (
        patch.object(bootstrap, "verify_preparation") as check,
        pytest.raises(RuntimeError, match="already started"),
    ):
        bootstrap.run_once(tmp_path, "model", {"api_key_file": "/must/not/read"})
    check.assert_not_called()


def test_bad_download_never_persists(tmp_path):
    target = tmp_path / "tool"
    with (
        patch.object(bootstrap, "fetch", return_value=b"wrong artifact"),
        pytest.raises(RuntimeError, match="hash mismatch"),
    ):
        bootstrap.checked_download("https://example.invalid/tool", target, "0" * 64)
    assert not target.exists()


def test_existing_bad_artifact_is_not_replaced(tmp_path):
    target = tmp_path / "tool"
    target.write_bytes(b"original bad artifact")
    with (
        patch.object(bootstrap, "fetch") as fetch,
        pytest.raises(RuntimeError, match="hash mismatch"),
    ):
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
    with (
        patch.object(bootstrap, "docker") as docker,
        pytest.raises(RuntimeError, match="source changed"),
    ):
        bootstrap.verify_preparation(tmp_path)
    docker.assert_not_called()


def connection(**kwargs):
    options = {
        "profile": None,
        "model_api": "https://model.invalid/v1/",
        "model_name": "openai/example",
        "api_key_env": None,
        "no_api_key": False,
    }
    options.update(kwargs)
    return Namespace(**options)


def test_direct_connection_requires_no_private_profile():
    variable_name = "MODEL_API_KEY"
    assert bootstrap.model_profile(connection(api_key_env=variable_name)) == {
        "api_base": "https://model.invalid/v1",
        "model": "example",
        "api_key_env": variable_name,
    }


def test_explicit_unauthenticated_service():
    profile = bootstrap.model_profile(connection(no_api_key=True))
    assert profile["no_api_key"] is True
    assert "api_key_env" not in profile
    with pytest.raises(ValueError, match="either"):
        bootstrap.model_profile(
            connection(no_api_key=True, api_key_env="KEY")  # pragma: allowlist secret
        )  # pragma: allowlist secret - non-secret test or unauthenticated-service placeholder


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/api",
        "https://key@model.invalid/v1",
        "https://model.invalid/v1?api_key=secret",
    ],
)
def test_connection_rejects_credentials_in_url(url):
    with pytest.raises(ValueError):
        bootstrap.model_profile(connection(model_api=url))


def test_private_profile_remains_supported(tmp_path):
    path = tmp_path / "model.json"
    profile = {
        "api_base": "https://model.invalid/v1",
        "model": "example",
        "api_key_file": "/private/key",
    }
    path.write_text(json.dumps(profile))
    args = connection(profile=path, model_api=None, model_name=None)
    assert bootstrap.model_profile(args) == profile
    with pytest.raises(ValueError, match="either"):
        bootstrap.model_profile(connection(profile=path))


def test_mirror_cannot_change_pinned_base(tmp_path):
    with (
        patch.object(bootstrap, "docker") as docker,
        pytest.raises(ValueError, match="pinned ARM64 manifest"),
    ):
        bootstrap.prepare(tmp_path, None, base_image_ref="mirror.invalid/python:latest")
    docker.assert_not_called()
