from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from vllm_hust_benchmark.official_targets import (
    PUBLIC_CODE_MODEL,
    PUBLIC_TEXT_MODEL,
    PUBLIC_TRACE_MODEL,
    PUBLIC_TRACE_MODEL_REVISION,
    PUBLIC_TRACE_RUNTIME_IMAGE,
    PUBLIC_TRACE_RUNTIME_IMAGE_DIGEST,
    PUBLIC_TRACE_RUNTIME_PACKAGES,
    PUBLIC_TRACE_ADDITIONAL_CONFIG,
    PUBLIC_TRACE_COMPILATION_CONFIG,
    PUBLIC_TRACE_VLLM_ASCEND_COMMIT,
    PUBLIC_TRACE_VLLM_COMMIT,
    PUBLIC_VISION_MODEL,
    SIMLLM_RUNTIME_IMAGE,
    SIMLLM_RUNTIME_PACKAGES,
    SIMLLM_VLLM_ASCEND_HUST_COMMIT,
    SIMLLM_VLLM_HUST_COMMIT,
    SIMLLM_WORKLOAD_IDS,
    build_registry,
    generated_outputs,
    render_active_targets,
    _validate_simllm_target,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generated_outputs_are_current() -> None:
    for path, expected in generated_outputs(REPO_ROOT).items():
        assert path.read_text(encoding="utf-8") == expected


def test_registry_matches_schema() -> None:
    registry = build_registry(REPO_ROOT)
    schema = json.loads(
        (REPO_ROOT / "schemas" / "official_target_registry_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(registry)


def test_public_target_matrix_is_exact() -> None:
    registry = build_registry(REPO_ROOT)
    active = [target for target in registry["targets"] if target["status"] == "active"]
    public = [
        target for target in active if target["intended_use"] == "public-leaderboard"
    ]
    assert len(public) == 11
    assert {target["model"]["id"] for target in public} == {
        PUBLIC_TEXT_MODEL,
        PUBLIC_CODE_MODEL,
        PUBLIC_TRACE_MODEL,
        PUBLIC_VISION_MODEL,
    }
    core_targets = [
        target for target in public if target["profile"] != "production-trace"
    ]
    trace_targets = [
        target for target in public if target["profile"] == "production-trace"
    ]

    assert {target["workload"]["name"] for target in trace_targets} == {
        "burstgpt-production-replay",
        "tracelab-coding-agent-replay",
    }
    assert len(core_targets) == 9
    for target in core_targets:
        assert target["target_version"] == "1.3.0"
        assert target["hardware"]["chip_model"] == "910B2"
        assert target["hardware"]["chip_count"] == 1
        assert target["model"]["precision"] == "FP16"
        assert target["server_parameters"]["tensor_parallel_size"] == 1
        assert target["server_parameters"]["gpu_memory_utilization"] == 0.6
        assert target["server_parameters"]["max_model_len"] == 32768
        assert target["model"]["revision"]
        assert target["server_parameters"]["revision"] == target["model"]["revision"]
        assert target["workload"]["data_identity"]["kind"]

    for target in trace_targets:
        expected_target_version = (
            "1.6.2"
            if target["workload"]["name"] == "tracelab-coding-agent-replay"
            else "1.5.1"
        )
        assert target["target_version"] == expected_target_version
        assert target["hardware"]["chip_model"] == "910B2"
        assert target["hardware"]["chip_count"] == 2
        assert target["model"]["id"] == PUBLIC_TRACE_MODEL
        assert target["model"]["parameters"] == "32B"
        assert target["model"]["precision"] == "BF16"
        assert target["model"]["revision"] == PUBLIC_TRACE_MODEL_REVISION
        assert target["workload"]["data_identity"]["kind"] == "release-asset"
        assert target["server_parameters"]["tensor_parallel_size"] == 2
        assert target["server_parameters"]["gpu_memory_utilization"] == 0.92
        assert target["server_parameters"]["max_model_len"] == 131072
        if target["workload"]["name"] == "tracelab-coding-agent-replay":
            assert target["workload"]["client_parameters"]["timeout_s"] == 21600
        assert (
            target["server_parameters"]["additional_config"]
            == PUBLIC_TRACE_ADDITIONAL_CONFIG
        )
        assert (
            target["server_parameters"]["compilation_config"]
            == PUBLIC_TRACE_COMPILATION_CONFIG
        )
        runtime = target["baseline_runtime"]
        assert runtime["core_commit"] == PUBLIC_TRACE_VLLM_COMMIT
        assert runtime["backend_commit"] == PUBLIC_TRACE_VLLM_ASCEND_COMMIT
        assert runtime["runtime_image"] == PUBLIC_TRACE_RUNTIME_IMAGE
        assert runtime["runtime_image_digest"] == PUBLIC_TRACE_RUNTIME_IMAGE_DIGEST
        assert runtime["runtime_packages"] == PUBLIC_TRACE_RUNTIME_PACKAGES
        assert runtime["runtime_environment"] == {"VLLM_BATCH_INVARIANT": "1"}
        assert "runtime_image_recipe" not in runtime


def test_simllm_ab_targets_are_exact_and_do_not_promote_local_results() -> None:
    registry = build_registry(REPO_ROOT)
    targets = [
        target
        for target in registry["targets"]
        if target["workload"]["name"] in SIMLLM_WORKLOAD_IDS
    ]
    assert {target["workload"]["name"] for target in targets} == SIMLLM_WORKLOAD_IDS
    assert {target["profile"] for target in targets} == {"simllm-warm-cache"}
    assert {target["status"] for target in targets} == {"active"}
    assert {target["intended_use"] for target in targets} == {"specialty"}
    for target in targets:
        assert target["target_version"] == "1.6.2"
        assert target["server_parameters"]["prefix_caching_hash_algo"] == "sha256"
        assert target["model"] == {
            "id": PUBLIC_TEXT_MODEL,
            "parameters": "14B",
            "precision": "FP16",
        }
        assert target["hardware"]["chip_model"] == "910B2"
        assert target["hardware"]["chip_count"] == 1
        runtime = target["baseline_runtime"]
        assert runtime["core_commit"] == SIMLLM_VLLM_HUST_COMMIT
        assert runtime["backend_commit"] == SIMLLM_VLLM_ASCEND_HUST_COMMIT
        assert runtime["runtime_image"] == SIMLLM_RUNTIME_IMAGE
        assert runtime["runtime_packages"] == SIMLLM_RUNTIME_PACKAGES
        protocol = target["workload"]["protocol"]
        assert protocol["baseline_variant"] == "simllm-disabled"
        assert protocol["candidate_variant"] == "simllm-enabled-warm-cache"
        assert protocol["baseline_engine"] == "vllm-hust"
        assert protocol["candidate_engine"] == "vllm-hust-simllm"
        assert protocol["local_reference_results_are_official"] is False
        assert protocol["candidate_warmup"]["restart_before_measurement"] is False
        assert target["workload"]["base_scenario"] == "random-online"
        assert not {
            "baseline_variant",
            "candidate_variant",
            "baseline_engine",
            "candidate_engine",
            "candidate_warmup",
        }.intersection(target["workload"]["client_parameters"])

    by_workload = {target["workload"]["name"]: target for target in targets}
    random_client = by_workload["simllm-random-online-warm-cache"]["workload"][
        "client_parameters"
    ]
    assert random_client["num_prompts"] == 200
    assert random_client["input_len"] == 1024
    assert random_client["output_len"] == 256
    assert random_client["request_rate"] == 1
    assert by_workload["simllm-random-online-warm-cache"]["workload"]["protocol"][
        "simllm_config"
    ] == {
        "cosine_threshold": 0.8,
        "lsh_num_bits": 64,
        "lsh_batch_threshold": 32,
        "kv_cache_size": 200,
        "sandwich_bottom": 3,
        "sandwich_top": 3,
        "unmatched_store_mode": "top",
    }

    saturated = by_workload["simllm-saturated-throughput-warm-cache"]
    saturated_client = saturated["workload"]["client_parameters"]
    assert saturated_client["num_prompts"] == 32
    assert saturated_client["input_len"] == 4096
    assert saturated_client["output_len"] == 32
    assert saturated_client["request_rate"] == "inf"
    assert saturated_client["max_concurrency"] == 16
    assert saturated["server_parameters"]["max_num_batched_tokens"] == 4096
    assert saturated["workload"]["protocol"]["simllm_config"] == {
        "cosine_threshold": 0.8,
        "lsh_num_bits": 64,
        "lsh_batch_threshold": 32,
        "kv_cache_size": 32,
        "sandwich_bottom": 3,
        "sandwich_top": 3,
        "unmatched_store_mode": "top",
    }


def test_simllm_target_rejects_promoting_local_reference_results(
    tmp_path: Path,
) -> None:
    source = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "official-simllm-random-online-warm-cache-qwen25-14b-1chip-910b2.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["ab_protocol"]["local_reference_results_are_official"] = True
    with pytest.raises(ValueError, match="local_reference_results_are_official=False"):
        _validate_simllm_target(payload, tmp_path / source.name)


def test_simllm_target_rejects_config_drift(tmp_path: Path) -> None:
    source = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "official-simllm-saturated-throughput-warm-cache-qwen25-14b-1chip-910b2.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["ab_protocol"]["simllm_config"]["kv_cache_size"] = 200
    with pytest.raises(ValueError, match="must pin simllm_config"):
        _validate_simllm_target(payload, tmp_path / source.name)


def test_perfgate_never_appears_as_active_public_target() -> None:
    registry = build_registry(REPO_ROOT)
    perfgate = [
        target for target in registry["targets"] if target["intended_use"] == "perfgate"
    ]
    assert perfgate
    assert {target["status"] for target in perfgate} == {"provisional"}
    assert all(target["model"]["parameters"] == "3B" for target in perfgate)


def test_source_spec_hashes_match() -> None:
    registry = build_registry(REPO_ROOT)
    for target in registry["targets"]:
        path = REPO_ROOT / target["source_spec"]["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert target["source_spec"]["sha256"] == digest


def test_target_ids_are_unique() -> None:
    registry = build_registry(REPO_ROOT)
    target_ids = [target["target_id"] for target in registry["targets"]]
    assert len(target_ids) == len(set(target_ids))


def test_active_target_output_excludes_provisional_profiles() -> None:
    output = render_active_targets(build_registry(REPO_ROOT))
    assert "Qwen/Qwen2.5-14B-Instruct" in output
    assert "Qwen/Qwen2.5-3B-Instruct" not in output


def test_public_target_drift_fails_closed(tmp_path: Path) -> None:
    source = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json"
    )
    spec_root = tmp_path / "docs" / "official-baselines"
    spec_root.mkdir(parents=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["server_parameters"]["gpu_memory_utilization"] = 0.9
    (spec_root / source.name).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="gpu_memory_utilization=0.6"):
        build_registry(tmp_path)
