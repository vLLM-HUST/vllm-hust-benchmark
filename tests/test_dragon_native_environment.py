"""Software dependency fixtures only; no imports of model stacks or NPU use."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_downloaded_embedding_artifact_is_not_native_execution():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/dragon-original-preparation-20261028.json"
        ).read_text(encoding="utf-8")
    )["official_embedding_artifact_acquisition_20261029"]
    assert receipt["original_files_verified"] == len(receipt["included_paths"]) == 11
    assert receipt["original_lfs_sha256_and_small_file_git_blobs_verified"]
    assert receipt["all_files_independently_sha256_reverified_on_server"]
    assert receipt["original_transformer_pooling_normalize_modules_preserved"]
    assert receipt["safetensors_offsets_and_tensor_sizes_cover_data_exactly"]
    assert not receipt["model_weights_loaded"]
    assert not receipt["npu_embedding_execution_verified"]
    assert not receipt["cpu_model_inference"]
    assert receipt["model_calls"] == receipt["source_checkbox_increment"] == 0


def load():
    spec = importlib.util.spec_from_file_location(
        "dragon_probe", ROOT / "scripts/probe_dragon_native_environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_presence_never_claims_native_execution():
    module = load()
    calls = []

    def locate(name):
        calls.append(name)
        return SimpleNamespace(origin="inert fixture, never imported")

    report = module.probe(locate=locate)
    assert calls == list(module.MODEL_MODULES + module.PIPELINE_MODULES)
    assert not report["missing_modules"]
    assert not report["native_npu_model_execution_verified"]
    assert not report["native_benchmark_completed"]
    assert not report["source_checkbox_eligible"]
    assert report["model_calls"] == 0
    assert not report["model_weights_loaded"]
    assert not report["cpu_embedding_fallback_permitted"]


def test_missing_dependencies_remain_explicit():
    module = load()
    missing = {"sentence_transformers", "langchain_huggingface"}
    report = module.probe(
        locate=lambda name: None if name in missing else SimpleNamespace()
    )
    assert set(report["missing_modules"]) == missing
    assert report["native_npu_adapter_required"]
    assert not report["service_changed"]


def test_inherited_profile_never_silently_replaces_versions():
    loaded = load()
    profile = {
        "schema_version": "dragon-inherited-runtime/v1",
        "protected_versions": {"torch-npu": "fixed"},
    }
    assert loaded.verify_versions(profile, version=lambda _: "fixed")
    with pytest.raises(ValueError, match="resolve separately"):
        loaded.verify_versions(profile, version=lambda _: "different")
