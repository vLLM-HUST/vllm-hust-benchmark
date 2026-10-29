"""Software dependency fixtures only; no imports of model stacks or NPU use."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


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
