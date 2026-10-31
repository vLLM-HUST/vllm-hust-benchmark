"""Dependency metadata fixtures only; no installation, imports or model calls."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "dragon_lock", ROOT / "scripts/prepare_dragon_model_client_lock.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def plan(
    name="sentence-transformers",
    url="https://files.pythonhosted.org/packages/inert.whl",
):
    return {
        "install": [
            {
                "metadata": {"name": name, "version": "1.0"},
                "download_info": {
                    "url": url,
                    "archive_info": {"hashes": {"sha256": "0" * 64}},
                },
            }
        ]
    }


def test_full_extra_and_exact_hash_preserved():
    loaded = module()
    lines = loaded.hashed_requirements(plan(), {"torch": "protected"})
    assert lines == ["sentence-transformers==1.0 --hash=sha256:" + "0" * 64]


def test_protected_stack_cannot_be_replaced():
    with pytest.raises(ValueError, match="protected"):
        module().hashed_requirements(plan("torch_npu"), {"torch-npu": "protected"})


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/artifact.whl",
        "http://files.pythonhosted.org/a.whl",
        "https://user:secret@files.pythonhosted.org/a.whl",  # pragma: allowlist secret - intentionally invalid fixture, not credentials
        "https://files.pythonhosted.org/a.whl?token=inert",
    ],
)
def test_nonpublic_or_credential_urls_rejected(url):
    with pytest.raises(ValueError, match="official PyPI"):
        module().hashed_requirements(plan(url=url), {})


def test_missing_embedding_dependency_rejected():
    with pytest.raises(ValueError, match="sentence-transformers"):
        module().hashed_requirements(plan("other"), {})


def test_missing_hash_rejected():
    value = plan()
    value["install"][0]["download_info"]["archive_info"]["hashes"] = {}
    with pytest.raises(ValueError, match="SHA256"):
        module().hashed_requirements(value, {})


def test_observed_lock_keeps_numpy1_compatible_data_pair():
    lines = (
        (
            ROOT
            / "configs/workload-native/dragon-npu-model-client-py312-observed.lock.txt"
        )
        .read_text()
        .splitlines()
    )
    requirements = [line for line in lines if line and not line.startswith("#")]
    assert len(requirements) == 56
    assert all(" --hash=sha256:" in line for line in requirements)
    assert any(line.startswith("pyarrow==22.0.0 ") for line in requirements)
    assert any(line.startswith("datasets==4.8.5 ") for line in requirements)
    assert not any(
        line.startswith(
            (
                "torch==",
                "torch-npu==",
                "numpy==",
                "transformers==",
                "tokenizers==",
                "vllm==",
            )
        )
        for line in requirements
    )
