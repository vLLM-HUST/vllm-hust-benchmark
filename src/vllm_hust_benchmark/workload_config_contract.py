from __future__ import annotations

from collections.abc import Mapping
from typing import Any


WORKLOAD_CONFIG_CONTRACT_VERSION = "explicit-effective/v1"
WORKLOAD_CONFIG_CONTRACT_REQUIRED_AFTER = "2026-07-24T00:00:00Z"
OFFICIAL_SPEC_PREFIX = "official-ascend-jan-2026-v0.18.0-"

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
}

OFFICIAL_VISION_DEFAULTS: dict[str, dict[str, Any]] = {
    "visionarena-online": {
        "server": {"gpu_memory_utilization": 0.6, "max_model_len": 30720},
    },
}

TARGET_ID_REQUIRED_AFTER = "2026-07-29T00:00:00Z"


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


def _official_defaults_for_scenario(scenario: str) -> dict[str, Any]:
    """Return the official defaults for a scenario.

    Vision scenarios use ``OFFICIAL_VISION_DEFAULTS``; text scenarios fall back
    to ``OFFICIAL_SINGLE_CHIP_TEXT_DEFAULTS``.
    """
    if scenario in OFFICIAL_VISION_DEFAULTS:
        return OFFICIAL_VISION_DEFAULTS[scenario]
    return OFFICIAL_SINGLE_CHIP_TEXT_DEFAULTS.get(scenario, {})


def _validate_target_metadata(metadata: Mapping[str, Any]) -> list[str]:
    """Validate ``metadata.target_id`` / ``metadata.target_version``.

    Only invoked when the entry is official and ``submitted_at`` is at or after
    ``TARGET_ID_REQUIRED_AFTER``. The registry is loaded lazily to avoid a
    circular import.
    """
    errors: list[str] = []
    target_id = str(metadata.get("target_id") or "").strip()
    target_version = str(metadata.get("target_version") or "").strip()

    if not target_id:
        errors.append("metadata.target_id must be explicitly recorded")
        return errors

    from vllm_hust_benchmark.fixed_target_registry import (  # noqa: E402
        get_active_profiles,
        load_fixed_target_registry,
    )

    try:
        registry = load_fixed_target_registry()
    except ValueError as exc:
        errors.append(
            f"metadata.target_id validation failed to load registry: {exc}"
        )
        return errors

    active = get_active_profiles(registry)
    matching = [profile for profile in active if profile.target_id == target_id]
    if not matching:
        errors.append(
            f"metadata.target_id {target_id!r} does not match any active "
            f"profile in the fixed-target registry"
        )
        return errors

    if not target_version:
        errors.append("metadata.target_version must be explicitly recorded")
        return errors

    valid_versions = {profile.target_version for profile in matching}
    if target_version not in valid_versions:
        errors.append(
            f"metadata.target_version {target_version!r} does not match the "
            f"registry target_version for target_id {target_id!r} "
            f"(expected one of {sorted(valid_versions)})"
        )
    return errors


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
    submitted_at = str(metadata.get("submitted_at") or "").strip()
    if not submitted_at:
        errors.append("metadata.submitted_at must be explicitly recorded")
    elif submitted_at >= TARGET_ID_REQUIRED_AFTER:
        errors.extend(_validate_target_metadata(metadata))

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

    input_length = _positive_int(workload.get("input_length"))
    output_length = _positive_int(workload.get("output_length"))
    if input_length is None:
        errors.append("workload.input_length must be a positive integer")
    if output_length is None:
        errors.append("workload.output_length must be a positive integer")

    same_spec = entry.get("same_spec")
    same_spec = same_spec if isinstance(same_spec, Mapping) else {}
    server = same_spec.get("resolved_server_parameters")
    client = same_spec.get("resolved_client_parameters")
    server = server if isinstance(server, Mapping) else {}
    client = client if isinstance(client, Mapping) else {}
    scenario = str(same_spec.get("scenario") or workload.get("name") or "")

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

    official_defaults = _official_defaults_for_scenario(scenario)
    default_label = (
        "official vision default"
        if scenario in OFFICIAL_VISION_DEFAULTS
        else "official single-chip text default"
    )
    for key, expected in official_defaults.get("server", {}).items():
        actual = server.get(key)
        if not _numeric_equal(actual, expected):
            errors.append(
                f"same_spec.resolved_server_parameters.{key} must be "
                f"{expected!r} for the {default_label}, "
                f"got {actual!r}"
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
