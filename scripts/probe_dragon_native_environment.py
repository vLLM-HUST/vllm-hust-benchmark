"""Inspect DRAGON model-client dependencies without importing/loading models.

Presence of torch_npu is not proof of NPU execution. This software probe neither
installs anything nor calls a model, changes a service or qualifies a benchmark.
"""

import argparse
from importlib import metadata
import importlib.util
import json
from pathlib import Path

MODEL_MODULES = (
    "torch",
    "torch_npu",
    "transformers",
    "sentence_transformers",
    "langchain_huggingface",
)
PIPELINE_MODULES = ("datasets", "langchain", "langchain_chroma")


def probe(*, locate=importlib.util.find_spec):
    present = {}
    for name in MODEL_MODULES + PIPELINE_MODULES:
        spec = locate(name)
        present[name] = spec is not None
    return {
        "source_id": "DRAGON_PUBLIC",
        "status": "SOFTWARE_PRESENCE_ONLY_NOT_NATIVE_QUALIFICATION",
        "modules_present": present,
        "missing_modules": [name for name, found in present.items() if not found],
        "official_example_requires_cuda_specific_initialization": True,
        "native_npu_adapter_required": True,
        "native_npu_model_execution_verified": False,
        "native_benchmark_completed": False,
        "model_calls": 0,
        "model_weights_loaded": False,
        "service_changed": False,
        "cpu_embedding_fallback_permitted": False,
        "source_checkbox_eligible": False,
        "next_gate": "Qualify isolated original pipeline dependencies plus explicit NPU embedding and existing-reader transport. Preserve original corpus/chunking/MMR/prompt/scoring; do not run CUDA example or substitute random/CPU embeddings.",
    }


def verify_versions(profile, *, version=metadata.version):
    if profile.get("schema_version") != "dragon-inherited-runtime/v1":
        raise ValueError("explicit inherited runtime profile required")
    expected = profile.get("protected_versions", {})
    if not expected or any(version(name) != value for name, value in expected.items()):
        raise ValueError(
            "inherited runtime differs; resolve separately, do not replace serving stack"
        )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-profile", type=Path)
    args = parser.parse_args()
    report = probe()
    if args.runtime_profile:
        report["inherited_profile_versions_match"] = verify_versions(
            json.loads(args.runtime_profile.read_bytes())
        )
    print(json.dumps(report, sort_keys=True))
