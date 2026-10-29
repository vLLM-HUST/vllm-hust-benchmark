"""Inspect DRAGON model-client dependencies without importing/loading models.

Presence of torch_npu is not proof of NPU execution. This software probe neither
installs anything nor calls a model, changes a service or qualifies a benchmark.
"""

import importlib.util
import json

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


if __name__ == "__main__":
    print(json.dumps(probe(), sort_keys=True))
