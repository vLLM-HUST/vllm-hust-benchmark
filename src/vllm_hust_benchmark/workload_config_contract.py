from __future__ import annotations

from collections.abc import Mapping
from typing import Any


WORKLOAD_CONFIG_CONTRACT_VERSION = "explicit-effective/v1"
WORKLOAD_CONFIG_CONTRACT_REQUIRED_AFTER = "2026-07-24T00:00:00Z"
OFFICIAL_SPEC_PREFIX = "official-ascend-jan-2026-v0.18.0-"
VARIABLE_LENGTH_TRACE_SCENARIOS = {
    "burstgpt-production-replay",
    "tracelab-coding-agent-replay",
}

REQUIRED_EFFECTIVE_PARAMETERS: dict[str, dict[str, tuple[str, ...]]] = {
    "agent-research-online": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": (),
    },
    "instructcoder-online": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": ("no_stream",),
    },
    "prefix-repetition-online": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": ("no_stream",),
    },
    "random-latency": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": ("gpu_memory_utilization",),
    },
    "random-online": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": ("no_stream",),
    },
    "sharegpt-online": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": ("no_stream",),
    },
    "sharegpt-throughput": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": ("gpu_memory_utilization",),
    },
    "sonnet-throughput": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": ("gpu_memory_utilization",),
    },
    "visionarena-online": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": ("no_stream",),
    },
    "burstgpt-production-replay": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": (
            "trace_target_id",
            "trace_asset",
            "max_requests",
            "overflow_policy",
            "cohort_context_cap",
        ),
    },
    "tracelab-coding-agent-replay": {
        "server": ("gpu_memory_utilization", "max_model_len"),
        "client": (
            "trace_target_id",
            "trace_asset",
            "max_requests",
            "overflow_policy",
            "cohort_context_cap",
        ),
    },
}

OFFICIAL_SINGLE_CHIP_TEXT_DEFAULTS: dict[str, dict[str, Any]] = {
    "agent-research-online": {
        "server": {"gpu_memory_utilization": 0.6, "max_model_len": 32768},
    },
    "instructcoder-online": {
        "server": {"gpu_memory_utilization": 0.6, "max_model_len": 32768},
    },
    "prefix-repetition-online": {
        "server": {"gpu_memory_utilization": 0.6, "max_model_len": 32768},
    },
    "random-latency": {
        "server": {"gpu_memory_utilization": 0.6, "max_model_len": 32768},
    },
    "random-online": {
        "server": {"gpu_memory_utilization": 0.6, "max_model_len": 32768},
    },
    "sharegpt-online": {
        "server": {"gpu_memory_utilization": 0.6, "max_model_len": 32768},
    },
    "sharegpt-throughput": {
        "server": {"gpu_memory_utilization": 0.6, "max_model_len": 32768},
    },
    "sonnet-throughput": {
        "server": {"gpu_memory_utilization": 0.6, "max_model_len": 32768},
    },
    "visionarena-online": {
        "server": {"gpu_memory_utilization": 0.6, "max_model_len": 32768},
    },
    "burstgpt-production-replay": {
        "server": {"gpu_memory_utilization": 0.9, "max_model_len": 131072},
    },
    "tracelab-coding-agent-replay": {
        "server": {"gpu_memory_utilization": 0.9, "max_model_len": 131072},
    },
}


def is_official_workload_contract_entry(entry: Mapping[str, Any]) -> bool:
    same_spec = entry.get("same_spec")
    if not isinstance(same_spec, Mapping):
        return False
    return str(same_spec.get("spec_id") or "").startswith(OFFICIAL_SPEC_PREFIX)


def requires_workload_config_contract(entry: Mapping[str, Any]) -> bool:
    if not is_official_workload_contract_entry(entry):
        return False
    metadata = entry.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    submitted_at = str(metadata.get("submitted_at") or "").strip()
    return submitted_at >= WORKLOAD_CONFIG_CONTRACT_REQUIRED_AFTER


def _positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _numeric_equal(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return left == right


def _expected_workload_lengths(
    client: Mapping[str, Any],
) -> tuple[int | None, int | None]:
    dataset_name = str(client.get("dataset_name") or "")
    if dataset_name == "random":
        return (
            _positive_int(client.get("random_input_len")),
            _positive_int(client.get("random_output_len")),
        )
    if dataset_name == "prefix_repetition":
        prefix = _positive_int(client.get("prefix_repetition_prefix_len"))
        suffix = _positive_int(client.get("prefix_repetition_suffix_len"))
        output = _positive_int(client.get("prefix_repetition_output_len"))
        return (
            prefix + suffix if prefix is not None and suffix is not None else None,
            output,
        )
    return (
        _positive_int(client.get("input_len")),
        _positive_int(client.get("output_len")),
    )


def validate_explicit_workload_config(entry: Mapping[str, Any]) -> list[str]:
    if not is_official_workload_contract_entry(entry):
        return []

    errors: list[str] = []
    metadata = entry.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    contract_version = str(metadata.get("workload_config_contract") or "")
    if contract_version != WORKLOAD_CONFIG_CONTRACT_VERSION:
        errors.append(
            "metadata.workload_config_contract must be "
            f"{WORKLOAD_CONFIG_CONTRACT_VERSION!r}"
        )
    if not str(metadata.get("submitted_at") or "").strip():
        errors.append("metadata.submitted_at must be explicitly recorded")

    workload = entry.get("workload")
    if not isinstance(workload, Mapping):
        return [*errors, "workload must be an object"]
    for key in (
        "name",
        "input_length",
        "output_length",
        "batch_size",
        "concurrent_requests",
        "dataset",
    ):
        if key not in workload:
            errors.append(f"workload.{key} must be explicitly recorded")

    same_spec = entry.get("same_spec")
    same_spec = same_spec if isinstance(same_spec, Mapping) else {}
    server = same_spec.get("resolved_server_parameters")
    client = same_spec.get("resolved_client_parameters")
    server = server if isinstance(server, Mapping) else {}
    client = client if isinstance(client, Mapping) else {}
    scenario = str(same_spec.get("scenario") or workload.get("name") or "")
    variable_length_trace = scenario in VARIABLE_LENGTH_TRACE_SCENARIOS
    input_length = _positive_int(workload.get("input_length"))
    output_length = _positive_int(workload.get("output_length"))
    if variable_length_trace:
        if workload.get("input_length") is not None:
            errors.append("workload.input_length must be null for variable-length trace replay")
        if workload.get("output_length") is not None:
            errors.append("workload.output_length must be null for variable-length trace replay")
        for field in ("input_token_distribution", "output_token_distribution"):
            distribution = workload.get(field)
            if not isinstance(distribution, Mapping):
                errors.append(f"workload.{field} must record the trace cohort distribution")
                continue
            if _positive_int(distribution.get("count")) is None:
                errors.append(f"workload.{field}.count must be a positive integer")
            for quantile in ("min", "p50", "p95", "p99", "max", "total"):
                if _positive_int(distribution.get(quantile)) is None:
                    errors.append(
                        f"workload.{field}.{quantile} must be a positive integer"
                    )
    else:
        if input_length is None:
            errors.append("workload.input_length must be a positive integer")
        if output_length is None:
            errors.append("workload.output_length must be a positive integer")

    required = REQUIRED_EFFECTIVE_PARAMETERS.get(scenario, {})
    for key in required.get("server", ()):
        if key not in server:
            errors.append(
                f"same_spec.resolved_server_parameters.{key} must be explicitly recorded"
            )
    for key in required.get("client", ()):
        if key not in client:
            errors.append(
                f"same_spec.resolved_client_parameters.{key} must be explicitly recorded"
            )

    official_defaults = OFFICIAL_SINGLE_CHIP_TEXT_DEFAULTS.get(scenario, {})
    for key, expected in official_defaults.get("server", {}).items():
        actual = server.get(key)
        if not _numeric_equal(actual, expected):
            errors.append(
                f"same_spec.resolved_server_parameters.{key} must be "
                f"{expected!r} for the official single-chip text default, "
                f"got {actual!r}"
            )
    if variable_length_trace and not _numeric_equal(
        client.get("cohort_context_cap"), server.get("max_model_len")
    ):
        errors.append(
            "same_spec.resolved_client_parameters.cohort_context_cap must match "
            "same_spec.resolved_server_parameters.max_model_len"
        )

    expected_input, expected_output = _expected_workload_lengths(client)
    if expected_input is not None and input_length != expected_input:
        errors.append(
            f"workload.input_length {input_length!r} does not match "
            f"effective client value {expected_input}"
        )
    if expected_output is not None and output_length != expected_output:
        errors.append(
            f"workload.output_length {output_length!r} does not match "
            f"effective client value {expected_output}"
        )

    expected_batch = _positive_int(client.get("batch_size"))
    actual_batch = _positive_int(workload.get("batch_size"))
    if expected_batch is not None and actual_batch != expected_batch:
        errors.append(
            f"workload.batch_size {actual_batch!r} does not match "
            f"effective client value {expected_batch}"
        )

    expected_concurrency = _positive_int(
        client.get("max_concurrency") or client.get("concurrent_requests")
    )
    actual_concurrency = _positive_int(workload.get("concurrent_requests"))
    if expected_concurrency is None and actual_concurrency is not None:
        errors.append(
            "workload.concurrent_requests must be null unless the effective "
            "client config records max_concurrency"
        )
    elif (
        expected_concurrency is not None and actual_concurrency != expected_concurrency
    ):
        errors.append(
            f"workload.concurrent_requests {actual_concurrency!r} does not match "
            f"effective client value {expected_concurrency}"
        )

    return errors
