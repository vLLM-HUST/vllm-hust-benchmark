from vllm_hust_benchmark.registry import filter_scenarios, get_scenario, load_official_scenarios


def test_load_official_scenarios() -> None:
    scenarios = load_official_scenarios()
    assert scenarios
    assert any(item.name == "sharegpt-online" for item in scenarios)


def test_get_scenario() -> None:
    scenario = get_scenario("visionarena-online")
    assert scenario.benchmark_type == "serve"
    assert "multimodal" in scenario.tags


def test_filter_scenarios() -> None:
    scenarios = filter_scenarios(benchmark_type="throughput")
    assert scenarios
    assert all(item.benchmark_type == "throughput" for item in scenarios)


def test_scenario_has_leaderboard_mapping() -> None:
    scenario = get_scenario("sharegpt-online")
    assert scenario.leaderboard["workload_name"] == "sharegpt-online"