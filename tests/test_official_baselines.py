import json
from pathlib import Path

from vllm_hust_benchmark.official_baselines import get_canonical_submission_dir
from vllm_hust_benchmark.official_baselines import (
    attest_canonical_submission,
    get_primary_metric_name_for_benchmark_type,
)
from vllm_hust_benchmark.official_baselines import has_canonical_run
from vllm_hust_benchmark.official_baselines import select_canonical_candidate
from vllm_hust_benchmark.same_spec import build_same_spec_payload


REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec() -> dict:
    return {
        "id": "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
        "scenario": "random-online",
    }


def test_get_canonical_submission_dir_uses_spec_id(tmp_path: Path) -> None:
    canonical_dir = get_canonical_submission_dir(_spec(), submissions_root=tmp_path)
    assert canonical_dir == tmp_path / _spec()["id"]


def test_perfgate_ascend_smoke_spec_is_available_for_ci() -> None:
    spec_file = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-qwen25-3b-910b2.json"
    )
    spec = json.loads(spec_file.read_text(encoding="utf-8"))

    assert spec["id"] == "perfgate-ascend-qwen25-3b-910b2"
    assert spec["scenario"] == "random-online"
    assert spec["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert spec["model_parameters"] == "3B"
    assert spec["model_precision"] == "BF16"
    assert spec["hardware_chip_model"] == "910B2"
    assert spec["server_parameters"]["max_model_len"] == 256
    assert spec["client_parameters"]["input_len"] == 64
    assert spec["client_parameters"]["output_len"] == 16


def test_perfgate_sharegpt_online_spec_is_available_for_ci() -> None:
    spec_file = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-sharegpt-online-qwen25-3b-910b2.json"
    )
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    same_spec = build_same_spec_payload(spec)

    assert spec["id"] == "perfgate-ascend-sharegpt-online-qwen25-3b-910b2"
    assert spec["scenario"] == "sharegpt-online"
    assert spec["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert spec["model_parameters"] == "3B"
    assert spec["model_precision"] == "BF16"
    assert spec["hardware_chip_model"] == "910B2"
    assert "max_model_len" not in spec["server_parameters"]
    assert spec["client_parameters"]["dataset_name"] == "sharegpt"
    assert (
        spec["client_parameters"]["dataset_path"]
        == "ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    assert spec["client_parameters"]["num_prompts"] == 8
    assert same_spec["scenario"] == "sharegpt-online"
    assert same_spec["resolved_client_parameters"]["dataset_name"] == "sharegpt"


def test_perfgate_prefix_repetition_online_spec_is_available_for_ci() -> None:
    spec_file = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-prefix-repetition-online-qwen25-3b-910b2.json"
    )
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    same_spec = build_same_spec_payload(spec)

    assert spec["id"] == "perfgate-ascend-prefix-repetition-online-qwen25-3b-910b2"
    assert spec["scenario"] == "prefix-repetition-online"
    assert spec["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert spec["model_parameters"] == "3B"
    assert spec["model_precision"] == "BF16"
    assert spec["hardware_chip_model"] == "910B2"
    assert spec["server_parameters"]["enable_prefix_caching"] == ""
    assert spec["server_parameters"]["max_model_len"] == 1280
    assert spec["client_parameters"]["dataset_name"] == "prefix_repetition"
    assert spec["client_parameters"]["num_prompts"] == 8
    assert spec["client_parameters"]["input_len"] == 1024
    assert spec["client_parameters"]["output_len"] == 64
    assert same_spec["scenario"] == "prefix-repetition-online"
    assert (
        same_spec["resolved_client_parameters"]["prefix_repetition_prefix_len"] == 768
    )
    assert (
        same_spec["resolved_client_parameters"]["prefix_repetition_suffix_len"] == 256
    )
    assert same_spec["resolved_client_parameters"]["prefix_repetition_output_len"] == 64


def test_perfgate_random_latency_spec_is_available_for_ci() -> None:
    spec_file = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-random-latency-qwen25-3b-910b2.json"
    )
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    same_spec = build_same_spec_payload(spec)

    assert spec["id"] == "perfgate-ascend-random-latency-qwen25-3b-910b2"
    assert spec["scenario"] == "random-latency"
    assert spec["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert spec["model_parameters"] == "3B"
    assert spec["model_precision"] == "BF16"
    assert spec["hardware_chip_model"] == "910B2"
    assert spec["server_parameters"]["max_model_len"] == 1280
    assert spec["server_parameters"]["max_num_seqs"] == 1
    assert spec["client_parameters"]["input_len"] == 1024
    assert spec["client_parameters"]["output_len"] == 128
    assert spec["client_parameters"]["batch_size"] == 1
    assert spec["client_parameters"]["num_iters_warmup"] == 1
    assert spec["client_parameters"]["num_iters"] == 3
    assert same_spec["scenario"] == "random-latency"
    assert same_spec["resolved_client_parameters"]["input_len"] == 1024
    assert same_spec["resolved_client_parameters"]["output_len"] == 128
    assert same_spec["resolved_client_parameters"]["batch_size"] == 1


def test_perfgate_sharegpt_throughput_spec_is_available_for_ci() -> None:
    spec_file = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-sharegpt-throughput-qwen25-3b-910b2.json"
    )
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    same_spec = build_same_spec_payload(spec)

    assert spec["id"] == "perfgate-ascend-sharegpt-throughput-qwen25-3b-910b2"
    assert spec["scenario"] == "sharegpt-throughput"
    assert spec["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert spec["model_parameters"] == "3B"
    assert spec["model_precision"] == "BF16"
    assert spec["hardware_chip_model"] == "910B2"
    assert spec["server_parameters"]["dtype"] == "bfloat16"
    assert spec["server_parameters"]["max_num_seqs"] == 1
    assert spec["client_parameters"]["dataset_name"] == "sharegpt"
    assert (
        spec["client_parameters"]["dataset_path"]
        == "ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    assert spec["client_parameters"]["num_prompts"] == 8
    assert spec["client_parameters"]["num_warmups"] == 0
    assert same_spec["scenario"] == "sharegpt-throughput"
    assert same_spec["resolved_client_parameters"]["dataset_name"] == "sharegpt"
    assert same_spec["resolved_client_parameters"]["num_prompts"] == 8


def test_perfgate_sonnet_throughput_spec_is_available_for_ci() -> None:
    spec_file = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-sonnet-throughput-qwen25-3b-910b2.json"
    )
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    same_spec = build_same_spec_payload(spec)

    assert spec["id"] == "perfgate-ascend-sonnet-throughput-qwen25-3b-910b2"
    assert spec["scenario"] == "sonnet-throughput"
    assert spec["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert spec["model_parameters"] == "3B"
    assert spec["model_precision"] == "BF16"
    assert spec["hardware_chip_model"] == "910B2"
    assert spec["server_parameters"]["dtype"] == "bfloat16"
    assert spec["client_parameters"]["dataset_name"] == "sonnet"
    assert spec["client_parameters"]["dataset_path"] == "benchmarks/sonnet.txt"
    assert spec["client_parameters"]["num_prompts"] == 8
    assert same_spec["scenario"] == "sonnet-throughput"
    assert same_spec["resolved_client_parameters"]["dataset_name"] == "sonnet"


def test_perfgate_instructcoder_online_spec_is_available_for_ci() -> None:
    spec_file = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-instructcoder-online-qwen25-coder-3b-910b2.json"
    )
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    same_spec = build_same_spec_payload(spec)

    assert spec["id"] == "perfgate-ascend-instructcoder-online-qwen25-coder-3b-910b2"
    assert spec["scenario"] == "instructcoder-online"
    assert spec["model"] == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert spec["model_parameters"] == "3B"
    assert spec["model_precision"] == "BF16"
    assert spec["hardware_chip_model"] == "910B2"
    assert spec["server_parameters"]["dtype"] == "bfloat16"
    assert spec["client_parameters"]["dataset_name"] == "hf"
    assert spec["client_parameters"]["dataset_path"] == "likaixin/InstructCoder"
    assert spec["client_parameters"]["num_prompts"] == 8
    assert spec["client_parameters"]["no_stream"] is True
    assert same_spec["scenario"] == "instructcoder-online"
    assert same_spec["resolved_client_parameters"]["dataset_name"] == "hf"
    assert same_spec["resolved_client_parameters"]["num_prompts"] == 8


def test_perfgate_agent_research_online_spec_is_available_for_ci() -> None:
    spec_file = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-agent-research-online-qwen25-3b-910b2.json"
    )
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    same_spec = build_same_spec_payload(spec)

    assert spec["id"] == "perfgate-ascend-agent-research-online-qwen25-3b-910b2"
    assert spec["scenario"] == "agent-research-online"
    assert spec["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert spec["model_parameters"] == "3B"
    assert spec["model_precision"] == "BF16"
    assert spec["hardware_chip_model"] == "910B2"
    assert spec["server_parameters"]["dtype"] == "bfloat16"
    assert spec["client_parameters"]["backend"] == "openai-chat"
    assert spec["client_parameters"]["dataset_name"] == "custom"
    assert (
        spec["client_parameters"]["dataset_path"]
        == "scripts/traces/evoscientist-workload-custom.jsonl"
    )
    assert spec["client_parameters"]["num_prompts"] == 8
    assert same_spec["scenario"] == "agent-research-online"
    assert same_spec["resolved_client_parameters"]["dataset_name"] == "custom"
    assert same_spec["resolved_client_parameters"]["num_prompts"] == 8


def test_perfgate_visionarena_online_spec_is_available_for_ci() -> None:
    spec_file = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "perfgate-ascend-visionarena-online-qwen25-vl-3b-910b2.json"
    )
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    same_spec = build_same_spec_payload(spec)

    assert spec["id"] == "perfgate-ascend-visionarena-online-qwen25-vl-3b-910b2"
    assert spec["scenario"] == "visionarena-online"
    assert spec["model"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert spec["model_parameters"] == "3B"
    assert spec["model_precision"] == "BF16"
    assert spec["hardware_chip_model"] == "910B2"
    assert spec["server_parameters"]["dtype"] == "bfloat16"
    assert spec["server_parameters"]["limit_mm_per_prompt"] == {"image": 1}
    assert spec["client_parameters"]["backend"] == "openai-chat"
    assert spec["client_parameters"]["endpoint"] == "/v1/chat/completions"
    assert spec["client_parameters"]["dataset_name"] == "hf"
    assert spec["client_parameters"]["dataset_path"] == "lmarena-ai/VisionArena-Chat"
    assert spec["client_parameters"]["hf_split"] == "train"
    assert spec["client_parameters"]["num_prompts"] == 8
    assert same_spec["scenario"] == "visionarena-online"
    assert same_spec["resolved_server_parameters"]["limit_mm_per_prompt"] == {
        "image": 1
    }
    assert same_spec["resolved_client_parameters"]["dataset_name"] == "hf"
    assert same_spec["resolved_client_parameters"]["num_prompts"] == 8


def test_has_canonical_run_requires_matching_spec_id_and_submitter(
    tmp_path: Path,
) -> None:
    canonical_dir = tmp_path / _spec()["id"]
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "run_leaderboard.json").write_text(
        json.dumps(
            {
                "metadata": {"submitter": "official-ascend-baseline"},
                "same_spec": {"spec_id": _spec()["id"]},
            }
        ),
        encoding="utf-8",
    )
    (canonical_dir / "leaderboard_manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"leaderboard_artifact": "run_leaderboard.json"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert has_canonical_run(_spec(), submissions_root=tmp_path)


def test_has_canonical_run_rejects_mismatched_submitter(tmp_path: Path) -> None:
    canonical_dir = tmp_path / _spec()["id"]
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "run_leaderboard.json").write_text(
        json.dumps(
            {
                "metadata": {"submitter": "someone-else"},
                "same_spec": {"spec_id": _spec()["id"]},
            }
        ),
        encoding="utf-8",
    )
    (canonical_dir / "leaderboard_manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"leaderboard_artifact": "run_leaderboard.json"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert not has_canonical_run(_spec(), submissions_root=tmp_path)


def _write_result_artifact(
    result_dir: Path,
    *,
    ttft_ms: float | None,
    throughput_tps: float | None,
    error_rate: float = 0.0,
) -> None:
    submission_dir = result_dir / "submission"
    submission_dir.mkdir(parents=True)
    (submission_dir / "run_leaderboard.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "ttft_ms": ttft_ms,
                    "throughput_tps": throughput_tps,
                    "error_rate": error_rate,
                }
            }
        ),
        encoding="utf-8",
    )


def test_get_primary_metric_name_for_benchmark_type() -> None:
    assert get_primary_metric_name_for_benchmark_type("serve") == "ttft_ms"
    assert get_primary_metric_name_for_benchmark_type("latency") == "ttft_ms"
    assert get_primary_metric_name_for_benchmark_type("throughput") == "throughput_tps"


def test_select_canonical_candidate_prefers_median_ttft(tmp_path: Path) -> None:
    repeat_a = tmp_path / "repeat-a"
    repeat_b = tmp_path / "repeat-b"
    repeat_c = tmp_path / "repeat-c"
    _write_result_artifact(repeat_a, ttft_ms=120.0, throughput_tps=200.0)
    _write_result_artifact(repeat_b, ttft_ms=100.0, throughput_tps=220.0)
    _write_result_artifact(repeat_c, ttft_ms=140.0, throughput_tps=180.0)

    payload = select_canonical_candidate(
        [repeat_a, repeat_b, repeat_c], benchmark_type="serve"
    )

    assert payload["primary_metric_name"] == "ttft_ms"
    assert payload["median_value"] == 120.0
    assert Path(payload["selected_result_dir"]) == repeat_a.resolve()


def test_select_canonical_candidate_uses_throughput_metric(tmp_path: Path) -> None:
    repeat_a = tmp_path / "repeat-a"
    repeat_b = tmp_path / "repeat-b"
    repeat_c = tmp_path / "repeat-c"
    _write_result_artifact(repeat_a, ttft_ms=0.0, throughput_tps=190.0)
    _write_result_artifact(repeat_b, ttft_ms=0.0, throughput_tps=210.0)
    _write_result_artifact(repeat_c, ttft_ms=0.0, throughput_tps=230.0)

    payload = select_canonical_candidate(
        [repeat_a, repeat_b, repeat_c], benchmark_type="throughput"
    )

    assert payload["primary_metric_name"] == "throughput_tps"
    assert payload["median_value"] == 210.0
    assert Path(payload["selected_result_dir"]) == repeat_b.resolve()


def test_select_canonical_candidate_prefers_lower_error_rate(tmp_path: Path) -> None:
    repeat_a = tmp_path / "repeat-a"
    repeat_b = tmp_path / "repeat-b"
    _write_result_artifact(
        repeat_a, ttft_ms=100.0, throughput_tps=220.0, error_rate=0.1
    )
    _write_result_artifact(
        repeat_b, ttft_ms=110.0, throughput_tps=215.0, error_rate=0.0
    )

    payload = select_canonical_candidate([repeat_a, repeat_b], benchmark_type="serve")

    assert Path(payload["selected_result_dir"]) == repeat_b.resolve()


def _write_attestable_repeat(
    result_dir: Path, *, spec_id: str, identity: str, ttft_ms: float
) -> None:
    submission_dir = result_dir / "submission"
    submission_dir.mkdir(parents=True)
    payload = {
        "metrics": {
            "ttft_ms": ttft_ms,
            "throughput_tps": 200.0,
            "error_rate": 0.0,
            "peak_mem_mb": 40960,
        },
        "environment": {
            "pytorch_version": "2.10.0",
            "cann_version": "9.0.0",
            "driver_version": "26.0.rc1",
        },
        "metadata": {
            "submitted_at": "2026-08-03T00:00:00Z",
            "idempotency_key": identity,
            "reproducible_cmd": "bash scripts/run-current-ascend-same-spec.sh spec.json",
            "workload_config_contract": "explicit-effective/v1",
            "runtime_provenance": {
                "engine": {"commit": "a" * 40},
                "plugin": {"commit": "b" * 40},
            },
        },
        "same_spec": {
            "spec_id": spec_id,
            "resolved_spec_hash": "c" * 64,
            "resolved_server_parameters": {"port": 8000},
            "resolved_client_parameters": {
                "port": 8000,
                "random_input_len": 1024,
                "random_output_len": 256,
            },
        },
    }
    (submission_dir / "run_leaderboard.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (result_dir / "raw_benchmark_result.json").write_text(
        json.dumps({"completed": 200, "failed": 0}), encoding="utf-8"
    )


def test_attest_canonical_submission_binds_three_repeat_evidence(
    tmp_path: Path,
) -> None:
    spec_id = "official-test-target"
    repeats = [tmp_path / f"repeat-{index}" for index in range(1, 4)]
    for index, repeat in enumerate(repeats, start=1):
        _write_attestable_repeat(
            repeat, spec_id=spec_id, identity=f"run-{index}", ttft_ms=100 + index
        )
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    source = repeats[1] / "submission" / "run_leaderboard.json"
    (canonical_dir / "run_leaderboard.json").write_bytes(source.read_bytes())
    registry_path = tmp_path / "official_targets.json"
    registry_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_id": spec_id,
                        "target_version": "1.2.3",
                        "status": "active",
                        "server_parameters": {"port": 8000},
                        "workload": {
                            "client_parameters": {
                                "port": 8000,
                                "input_len": 1024,
                                "output_len": 256,
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = attest_canonical_submission(
        canonical_dir,
        spec={"id": spec_id},
        result_dirs=repeats,
        selected_result_dir=repeats[1],
        primary_metric_name="ttft_ms",
        registry_path=registry_path,
    )

    metadata = payload["metadata"]
    assert metadata["verified"] is True
    assert metadata["target_id"] == spec_id
    assert metadata["target_version"] == "1.2.3"
    attestation = metadata["verification_attestation"]
    assert attestation["successful_repeats"] == 3
    assert attestation["independent_service_processes"] == 3
    assert attestation["selected_repeat_index"] == 2
    assert len(attestation["repeat_evidence"]) == 3
    assert len({item["idempotency_key"] for item in attestation["repeat_evidence"]}) == 3


def test_attest_canonical_submission_rejects_duplicate_repeat_identity(
    tmp_path: Path,
) -> None:
    spec_id = "official-test-target"
    repeats = [tmp_path / f"repeat-{index}" for index in range(1, 4)]
    for repeat in repeats:
        _write_attestable_repeat(
            repeat, spec_id=spec_id, identity="duplicate", ttft_ms=100
        )
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    source = repeats[0] / "submission" / "run_leaderboard.json"
    (canonical_dir / "run_leaderboard.json").write_bytes(source.read_bytes())
    registry_path = tmp_path / "official_targets.json"
    registry_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_id": spec_id,
                        "target_version": "1.0.0",
                        "status": "active",
                        "server_parameters": {"port": 8000},
                        "workload": {
                            "client_parameters": {
                                "port": 8000,
                                "input_len": 1024,
                                "output_len": 256,
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        attest_canonical_submission(
            canonical_dir,
            spec={"id": spec_id},
            result_dirs=repeats,
            selected_result_dir=repeats[0],
            primary_metric_name="ttft_ms",
            registry_path=registry_path,
        )
    except ValueError as exc:
        assert "duplicated" in str(exc)
    else:
        raise AssertionError("duplicate repeat identities must fail closed")


def test_public_official_baseline_specs_are_v0180_910b2_fp16() -> None:
    spec_dir = REPO_ROOT / "docs" / "official-baselines"
    spec_paths = [
        path
        for path in spec_dir.glob("*.json")
        if path.name != "official-ascend-constraints.stub.json"
        and not path.name.startswith("perfgate-")
    ]

    assert spec_paths
    for path in spec_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec_id = str(payload.get("id") or "")
        assert "v0180" in path.name or "v0.18.0" in spec_id
        assert payload.get("hardware_chip_model") == "910B2"
        assert payload.get("model_precision") == "FP16"
