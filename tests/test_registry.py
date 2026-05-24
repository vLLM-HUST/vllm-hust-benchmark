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
*** Add File: /home/jeffrey.guest/workspace/vllm/vllm-hust-benchmark/docs/official-baselines/official-ascend-jan-2026-v0110-sharegpt-online-qwen25-14b-910b3.json
{
    "id": "official-ascend-jan-2026-v0.11.0-sharegpt-online-qwen25-14b-910b3",
    "label": "Official Ascend Jan 2026 ShareGPT online baseline for vllm-hust goal tracking",
    "baseline_target": {
        "id": "official-ascend-jan-2026-v0.11.0",
        "label": "Official Ascend Jan 2026",
        "engine": "vllm",
        "engine_version": "0.11.0",
        "github_repository": "vllm-project/vllm-ascend",
        "vllm_ref": "v0.11.0",
        "vllm_ascend_ref": "v0.11.0"
    },
    "scenario": "sharegpt-online",
    "model": "Qwen/Qwen2.5-14B-Instruct",
    "model_parameters": "14B",
    "model_precision": "FP16",
    "hardware_vendor": "Huawei",
    "hardware_chip_model": "910B3",
    "chip_count": 1,
    "node_count": 1,
    "server_parameters": {
        "tensor_parallel_size": 1,
        "enforce_eager": "",
        "trust_remote_code": "",
        "disable_log_stats": "",
        "disable_log_requests": "",
        "host": "0.0.0.0",
        "port": 8000
    },
    "client_parameters": {
        "backend": "vllm",
        "endpoint": "/v1/completions",
        "dataset_name": "sharegpt",
        "dataset_path": "ShareGPT_V3_unfiltered_cleaned_split.json",
        "num_prompts": 200,
        "request_rate": 1,
        "host": "127.0.0.1",
        "port": 8000
    },
    "export": {
        "engine": "vllm",
        "engine_version": "0.11.0",
        "submitter": "official-ascend-baseline",
        "baseline_engine": "vllm",
        "github_repository": "vllm-project/vllm-ascend",
        "github_ref": "v0.11.0",
        "git_commit": "2f1aed98ccdb0fcbe1ff4fd0abab225bfd8d0367",
        "data_source": "reference-vllm-ascend-benchmark"
    }
}
