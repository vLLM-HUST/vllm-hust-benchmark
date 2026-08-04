"""Validate specialty spec server_parameters map to real vLLM CLI args.

Issue #104 review feedback: specs like moe-alltoall, eplb, spec-decode,
slicegpt, knorm must not carry server_parameters that the current vLLM
runtime does not understand.  The run script (run-current-ascend-same-spec.sh)
converts each key to --key-name and passes it to vllm serve; unknown args
cause startup failure.

This test parses every specialty spec and asserts that each server_parameters
key (after _ -> - conversion) is a recognised vLLM CLI flag, with a small
allowlist of keys the run script strips before invocation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vllm_hust_benchmark.official_targets import build_registry

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keys that the run script removes before passing to vllm (see
# run-current-ascend-same-spec.sh line ~1127: del(.disable_log_requests)).
# These are safe to keep in the spec even though they are not vLLM args.
RUN_SCRIPT_STRIPPED_KEYS = frozenset({"disable_log_requests"})

# Standard vLLM server CLI args, extracted from
# python -m vllm.entrypoints.openai.api_server --help (v0.23.x).
# Only the flags relevant to the benchmark server launch are listed; the
# full set is too large to hard-code, so we validate against this curated
# allowlist plus a structural check (key matches --<a-z0-9-*>).
KNOWN_VLLM_SERVER_ARGS = frozenset(
    {
        # core launch
        "tensor_parallel_size",
        "enforce_eager",
        "trust_remote_code",
        "disable_log_stats",
        "host",
        "port",
        "model",
        "served_model_name",
        # memory / context
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "dtype",
        # caching
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "kv_cache_dtype",
        # quantization
        "quantization",
        # MoE / expert
        "enable_eplb",
        "eplb_config",
        # spec decode (vLLM >=0.6 uses --spec-model / --spec-tokens / --speculative-config)
        "spec_model",
        "spec_tokens",
        "spec_method",
        "speculative_config",
        # distributed
        "pipeline_parallel_size",
        "data_parallel_size",
        # kv transfer
        "kv_transfer_config",
        # vision
        "limit_mm_per_prompt",
        # compilation
        "enforce_compiled_graph",
        # additional config
        "additional_config",
    }
)

# Specs that must be explicitly covered by the param test (reviewer request).
REQUIRED_COVERAGE = frozenset(
    {
        "moe-alltoall",
        "eplb",
        "spec-decode",
        "slicegpt",
        "knorm",
    }
)


def _specialty_specs() -> list[tuple[str, dict]]:
    """Return (scenario_keyword, spec_dict) for all specialty targets."""
    registry = build_registry(REPO_ROOT)
    result = []
    for target in registry["targets"]:
        if target["intended_use"] != "specialty":
            continue
        spec_path = REPO_ROOT / target["source_spec"]["path"]
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        result.append((target["workload"]["name"], spec))
    return result


def test_specialty_specs_do_not_carry_unsupported_vllm_params() -> None:
    """Every server_parameters key must map to a real vLLM CLI flag."""
    unknown_found: list[str] = []
    for scenario, spec in _specialty_specs():
        sp = spec["server_parameters"]
        for key in sp:
            if key in RUN_SCRIPT_STRIPPED_KEYS:
                continue
            if key not in KNOWN_VLLM_SERVER_ARGS:
                unknown_found.append(f"{scenario}: {key}")
    assert not unknown_found, (
        "specialty specs carry server_parameters keys that are not recognised "
        "vLLM CLI args:\n  " + "\n  ".join(unknown_found)
    )


def test_specialty_specs_use_resolvable_model_identifiers() -> None:
    """Model identifiers must be real HuggingFace artifacts, not placeholders."""
    placeholder_patterns = [
        re.compile(r"-EAGLE$", re.IGNORECASE),
        re.compile(r"-slicegpt-", re.IGNORECASE),
        re.compile(r"MoE-A14B-Instruct$", re.IGNORECASE),
    ]
    bad_models: list[str] = []
    for scenario, spec in _specialty_specs():
        model = spec.get("model", "")
        for pattern in placeholder_patterns:
            if pattern.search(model):
                bad_models.append(f"{scenario}: {model}")
    assert not bad_models, (
        "specialty specs use placeholder model identifiers:\n  "
        + "\n  ".join(bad_models)
    )


def test_specialty_specs_are_non_executable_drafts() -> None:
    """All specialty targets must be status=provisional, not active."""
    registry = build_registry(REPO_ROOT)
    for target in registry["targets"]:
        if target["intended_use"] == "specialty":
            assert target["status"] == "provisional", (
                f"specialty target {target['target_id']} must be provisional, "
                f"got {target['status']!r}"
            )


@pytest.mark.parametrize(
    "scenario_keyword",
    sorted(REQUIRED_COVERAGE),
)
def test_required_coverage_specs_are_validated(scenario_keyword: str) -> None:
    """Explicitly cover MoE/EPLB, spec decode, SliceGPT/KNorm per review."""
    matched = [
        (scenario, spec)
        for scenario, spec in _specialty_specs()
        if scenario_keyword in scenario
    ]
    assert matched, f"no specialty spec found for keyword {scenario_keyword!r}"
    for scenario, spec in matched:
        sp = spec["server_parameters"]
        for key in sp:
            if key in RUN_SCRIPT_STRIPPED_KEYS:
                continue
            assert key in KNOWN_VLLM_SERVER_ARGS, (
                f"{scenario}: server_parameters key {key!r} is not a recognised "
                f"vLLM CLI arg"
            )


def test_json2args_conversion_produces_valid_flags() -> None:
    """Simulate the run script's json2args conversion and verify flags."""
    for scenario, spec in _specialty_specs():
        sp = spec["server_parameters"]
        for key, value in sp.items():
            if key in RUN_SCRIPT_STRIPPED_KEYS:
                continue
            # json2args converts _ to - and prepends --
            flag = "--" + key.replace("_", "-")
            # Flag must match vLLM's --<lowercase-with-dashes> convention
            assert re.match(r"^--[a-z][a-z0-9-]*$", flag), (
                f"{scenario}: key {key!r} produces invalid flag {flag!r}"
            )
            # Value must be serialisable (str, int, float, bool, dict, list)
            assert isinstance(value, (str, int, float, bool, dict, list)), (
                f"{scenario}: key {key!r} has unsupported value type {type(value).__name__}"
            )
