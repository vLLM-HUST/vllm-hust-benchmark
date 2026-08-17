from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EFFECTIVE_CACHE_CONTRACT_VERSION = "p2-explicit-cache/v1"
PREFIX_SCENARIO = "prefix-repetition-online"


def expected_effective_cache_contract(scenario: str) -> dict[str, dict[str, Any]]:
    """Return the P2 cache contract without changing any historical target."""
    if scenario == PREFIX_SCENARIO:
        return {
            "server": {"enable_prefix_caching": True},
            "environment": {"VLLM_KNORM_ENABLED": "0"},
        }
    return {
        "server": {"no_enable_prefix_caching": True},
        "environment": {},
    }


def validate_effective_cache_contract(
    scenario: str,
    server_parameters: Mapping[str, Any],
    environment: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate an explicitly versioned P2 cache contract."""
    environment = environment or {}
    expected = expected_effective_cache_contract(scenario)
    errors: list[str] = []
    for key, value in expected["server"].items():
        if server_parameters.get(key) is not value:
            errors.append(f"server.{key} must be {value!r}")
    for key, value in expected["environment"].items():
        if str(environment.get(key, "")) != value:
            errors.append(f"environment.{key} must be {value!r}")
    if (
        scenario != PREFIX_SCENARIO
        and server_parameters.get("enable_prefix_caching") is True
    ):
        errors.append("non-prefix workloads must not enable prefix caching")
    return errors
