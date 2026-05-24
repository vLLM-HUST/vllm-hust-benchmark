import json
from pathlib import Path

from vllm_hust_benchmark.registry import filter_scenarios, get_scenario, load_official_scenarios
from vllm_hust_benchmark.same_spec import build_same_spec_payload


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


def test_official_baseline_specs_cover_all_official_scenarios() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    specs_dir = repo_root / "docs" / "official-baselines"
    scenario_names = {item.name for item in load_official_scenarios()}
    covered_scenarios: dict[str, str] = {}

    for spec_path in sorted(specs_dir.glob("official-ascend-*.json")):
        if spec_path.name.endswith("constraints.stub.json"):
            continue

        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        scenario_name = payload["scenario"]
        assert scenario_name in scenario_names
        assert scenario_name not in covered_scenarios
        build_same_spec_payload(payload)
        covered_scenarios[scenario_name] = spec_path.name

    assert set(covered_scenarios) == scenario_names
