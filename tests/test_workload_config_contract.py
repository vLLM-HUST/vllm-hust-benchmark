from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from vllm_hust_benchmark.workload_config_contract import (
    REQUIRED_EFFECTIVE_PARAMETERS,
    WORKLOAD_CONFIG_CONTRACT_VERSION,
    requires_workload_config_contract,
    validate_explicit_workload_config,
)


def official_random_online_entry() -> dict:
    return {
        "engine": "vllm-hust",
        "workload": {
            "name": "random-online",
            "input_length": 1024,
            "output_length": 256,
            "batch_size": None,
            "concurrent_requests": None,
            "dataset": "random",
        },
        "metadata": {
            "submitted_at": "2026-07-25T00:00:00Z",
            "workload_config_contract": WORKLOAD_CONFIG_CONTRACT_VERSION,
        },
        "same_spec": {
            "spec_id": (
                "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
            ),
            "scenario": "random-online",
            "resolved_server_parameters": {
                "gpu_memory_utilization": 0.6,
            },
            "resolved_client_parameters": {
                "dataset_name": "random",
                "random_input_len": 1024,
                "random_output_len": 256,
                "num_prompts": 200,
                "request_rate": 1,
                "no_stream": False,
            },
        },
    }


def test_complete_effective_config_passes_contract() -> None:
    entry = official_random_online_entry()

    assert requires_workload_config_contract(entry)
    assert validate_explicit_workload_config(entry) == []


def test_contract_rejects_missing_defaults_and_workload_fields() -> None:
    entry = official_random_online_entry()
    del entry["workload"]["batch_size"]
    del entry["same_spec"]["resolved_server_parameters"]["gpu_memory_utilization"]
    del entry["same_spec"]["resolved_client_parameters"]["no_stream"]

    errors = validate_explicit_workload_config(entry)

    assert "workload.batch_size must be explicitly recorded" in errors
    assert any("gpu_memory_utilization" in error for error in errors)
    assert any("no_stream" in error for error in errors)


def test_contract_rejects_prompt_count_conflated_with_concurrency() -> None:
    entry = official_random_online_entry()
    entry["workload"]["concurrent_requests"] = 200

    errors = validate_explicit_workload_config(entry)

    assert any("must be null unless" in error for error in errors)


def test_contract_rejects_workload_values_that_disagree_with_client() -> None:
    entry = official_random_online_entry()
    entry["workload"]["input_length"] = 6144

    errors = validate_explicit_workload_config(entry)

    assert any(
        "does not match effective client value 1024" in error for error in errors
    )


def test_contract_accepts_real_explicit_max_concurrency() -> None:
    entry = deepcopy(official_random_online_entry())
    entry["same_spec"]["resolved_client_parameters"]["max_concurrency"] = 32
    entry["workload"]["concurrent_requests"] = 32

    assert validate_explicit_workload_config(entry) == []


def test_legacy_official_entry_is_grandfathered_by_activation_time() -> None:
    entry = official_random_online_entry()
    entry["metadata"]["submitted_at"] = "2026-07-23T23:59:59Z"
    del entry["metadata"]["workload_config_contract"]

    assert not requires_workload_config_contract(entry)


def test_official_single_chip_specs_define_required_effective_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    specs = []
    for path in (repo_root / "docs" / "official-baselines").glob(
        "official-ascend-jan-2026-v0180-*.json"
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("chip_count") == 1:
            specs.append(payload)

    by_scenario = {str(spec.get("scenario") or ""): spec for spec in specs}
    for scenario, scopes in REQUIRED_EFFECTIVE_PARAMETERS.items():
        spec = by_scenario[scenario]
        for key in scopes.get("server", ()):
            assert key in spec["server_parameters"], f"{scenario}: missing server {key}"
        for key in scopes.get("client", ()):
            assert key in spec["client_parameters"], f"{scenario}: missing client {key}"
