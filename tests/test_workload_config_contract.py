from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


from vllm_hust_benchmark.aggregate_results import VALID_AGG_METHODS
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
                "max_model_len": 32768,
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
    del entry["same_spec"]["resolved_server_parameters"]["max_model_len"]
    del entry["same_spec"]["resolved_client_parameters"]["no_stream"]

    errors = validate_explicit_workload_config(entry)

    assert "workload.batch_size must be explicitly recorded" in errors
    assert any("gpu_memory_utilization" in error for error in errors)
    assert any("max_model_len" in error for error in errors)
    assert any("no_stream" in error for error in errors)


def test_contract_rejects_wrong_official_single_chip_defaults() -> None:
    entry = official_random_online_entry()
    entry["same_spec"]["resolved_server_parameters"]["gpu_memory_utilization"] = "0.90"
    entry["same_spec"]["resolved_server_parameters"]["max_model_len"] = 65536

    errors = validate_explicit_workload_config(entry)

    assert any("gpu_memory_utilization must be 0.6" in error for error in errors)
    assert any("max_model_len must be 32768" in error for error in errors)


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


def test_snapshot_entries_never_conflate_num_prompts_with_concurrency() -> None:
    """PR#70 regression: snapshot entries must keep prompt count and concurrency apart.

    For every published snapshot entry:
      * ``workload.concurrent_requests`` must be ``null`` OR equal to
        ``same_spec.resolved_client_parameters.max_concurrency`` when that
        field exists and is non-null.
      * ``num_prompts`` must never appear in the ``workload`` object (it only
        belongs in ``same_spec.resolved_client_parameters``).
    """
    snapshots_dir = (
        Path(__file__).resolve().parents[1] / "leaderboard-data" / "snapshots"
    )
    snapshot_paths = [
        snapshots_dir / "leaderboard_single.json",
        snapshots_dir / "leaderboard_multi.json",
    ]

    violations: list[str] = []
    for snapshot_path in snapshot_paths:
        entries = json.loads(snapshot_path.read_text(encoding="utf-8"))
        for entry in entries:
            entry_id = str(entry.get("entry_id") or "<unknown>")
            workload = entry.get("workload")
            if not isinstance(workload, dict):
                violations.append(f"{entry_id}: workload is not an object")
                continue

            if "num_prompts" in workload:
                violations.append(
                    f"{entry_id}: 'num_prompts' must not appear in workload "
                    "(only in same_spec.resolved_client_parameters)"
                )

            actual_concurrency = workload.get("concurrent_requests")
            if actual_concurrency is None:
                continue

            same_spec = entry.get("same_spec")
            if not isinstance(same_spec, dict):
                continue
            client = same_spec.get("resolved_client_parameters")
            if not isinstance(client, dict):
                continue

            expected_concurrency = client.get("max_concurrency")
            if expected_concurrency is None:
                continue

            if actual_concurrency != expected_concurrency:
                violations.append(
                    f"{entry_id}: workload.concurrent_requests={actual_concurrency!r} "
                    f"!= resolved_client_parameters.max_concurrency="
                    f"{expected_concurrency!r}"
                )

    assert not violations, (
        "snapshot entries conflate prompt count with concurrency:\n  - "
        + "\n  - ".join(violations)
    )


def test_contract_rejects_max_as_aggregate_method() -> None:
    """``max`` must not be a valid aggregate method.

    Task 7 removed ``max`` from ``VALID_AGG_METHODS`` to prevent cherry-picking
    the highest values.
    """
    assert "max" not in VALID_AGG_METHODS
