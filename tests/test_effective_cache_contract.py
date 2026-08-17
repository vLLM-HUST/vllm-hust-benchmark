from vllm_hust_benchmark.effective_cache_contract import (
    EFFECTIVE_CACHE_CONTRACT_VERSION,
    expected_effective_cache_contract,
    validate_effective_cache_contract,
)


def test_contract_is_separately_versioned_from_historical_targets() -> None:
    assert EFFECTIVE_CACHE_CONTRACT_VERSION == "p2-explicit-cache/v1"


def test_non_prefix_contract_requires_explicit_cache_disable() -> None:
    assert expected_effective_cache_contract("random-online") == {
        "server": {"no_enable_prefix_caching": True},
        "environment": {},
    }
    assert (
        validate_effective_cache_contract(
            "random-online", {"no_enable_prefix_caching": True}
        )
        == []
    )
    assert validate_effective_cache_contract("random-online", {}) == [
        "server.no_enable_prefix_caching must be True"
    ]


def test_prefix_contract_requires_cache_and_safe_knorm() -> None:
    assert (
        validate_effective_cache_contract(
            "prefix-repetition-online",
            {"enable_prefix_caching": True},
            {"VLLM_KNORM_ENABLED": "0"},
        )
        == []
    )
    errors = validate_effective_cache_contract(
        "prefix-repetition-online", {"enable_prefix_caching": True}
    )
    assert errors == ["environment.VLLM_KNORM_ENABLED must be '0'"]


def test_non_prefix_contract_rejects_conflicting_enable() -> None:
    errors = validate_effective_cache_contract(
        "sharegpt-throughput",
        {"no_enable_prefix_caching": True, "enable_prefix_caching": True},
    )
    assert errors == ["non-prefix workloads must not enable prefix caching"]
