"""Dependency context fixtures; no build, package install or model requests."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "discovery_client_context",
    Path(__file__).resolve().parents[1]
    / "scripts/prepare_discoverybench_client_context.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_recipe_preserves_original_install_steps():
    image = "sha256:" + "0" * 64
    result = entry.dockerfile(image, "/opt/miniconda3/bin/python")
    assert result.startswith("FROM " + image + "\n")
    assert "-m venv /opt/discovery-client" in result
    assert "-r /opt/discovery-original-requirements.txt" in result
    assert result.endswith("-m pip check\n")


@pytest.mark.parametrize("image", ["latest", "python:3.10", "sha256:" + "x" * 64])
def test_mutable_or_invalid_base_rejected(image):
    with pytest.raises(ValueError, match="immutable"):
        entry.dockerfile(image, "/usr/local/bin/python")


@pytest.mark.parametrize(
    "path", ["python", "/bin/python;id", "/bin/../python", "/bin/python\nRUN id"]
)
def test_base_python_cannot_inject_dockerfile(path):
    with pytest.raises(ValueError, match="Python path"):
        entry.dockerfile("sha256:" + "0" * 64, path)
