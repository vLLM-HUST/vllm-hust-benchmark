from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vllm_hust_benchmark.snapshot_target_binding import (
    OfficialTargetRegistry,
    bind_entry_to_official_target,
    bind_snapshot_set,
)


def _target(*, specialty: bool = False) -> dict:
    return {
        "target_id": "official-target",
        "target_version": "7",
        "profile": "specialty-text" if specialty else "core-text",
        "status": "provisional" if specialty else "active",
        "intended_use": "specialty" if specialty else "public-leaderboard",
        "baseline_runtime": {"engine": "vllm", "engine_version": "0.18.0"},
        "model": {
            "id": "Qwen/Qwen2.5-14B-Instruct",
            "parameters": "14B",
            "precision": "FP16",
        },
        "hardware": {
            "vendor": "Huawei",
            "chip_model": "910B2",
            "chip_count": 1,
            "node_count": 1,
        },
        "server_parameters": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.6,
            "max_model_len": 32768,
        },
        "workload": {
            "name": "random-online",
            "client_parameters": {
                "dataset_name": "random",
                "random_input_len": 1024,
                "random_output_len": 128,
                "num_prompts": 200,
            },
        },
    }


def _entry() -> dict:
    return {
        "entry_id": "entry-1",
        "engine": "vllm",
        "engine_version": "0.18.0",
        "model": {
            "repo_id": "Qwen/Qwen2.5-14B-Instruct",
            "parameters": "14B",
            "precision": "FP16",
        },
        "hardware": {"vendor": "Huawei", "chip_model": "910B2", "chip_count": 1},
        "workload": {"name": "random-online"},
        "metadata": {"verified": None},
        "same_spec": {
            "spec_id": "official-target",
            "scenario": "random-online",
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "model_parameters": "14B",
            "model_precision": "FP16",
            "model_quantization": "",
            "hardware_vendor": "Huawei",
            "hardware_chip_model": "910B2",
            "chip_count": 1,
            "node_count": 1,
            "resolved_server_parameters": {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.6,
                "max_model_len": 32768,
            },
            "resolved_client_parameters": {
                "dataset_name": "random",
                "random_input_len": 1024,
                "random_output_len": 128,
                "num_prompts": 200,
            },
        },
    }


def _registry(target: dict) -> OfficialTargetRegistry:
    return OfficialTargetRegistry(
        version="7", sha256="a" * 64, targets={target["target_id"]: target}
    )


def test_exact_active_public_contract_derives_admission_metadata() -> None:
    entry = _entry()
    target = _target()
    verified, errors = bind_entry_to_official_target(entry, _registry(target))
    assert verified is True
    assert errors == []
    assert entry["metadata"] == {
        "verified": True,
        "target_id": "official-target",
        "target_version": "7",
        "profile_id": "core-text",
        "target_registry_sha256": "a" * 64,
    }


def test_engine_mismatch_is_retained_but_never_marked_verified() -> None:
    entry = _entry()
    entry["engine"] = "vllm-hust"
    verified, errors = bind_entry_to_official_target(entry, _registry(_target()))
    assert verified is False
    assert "engine does not match target baseline runtime" in errors
    assert entry["metadata"]["verified"] is False
    assert entry["metadata"]["official_admission_status"] == "historical-unverified"
    assert entry["metadata"]["official_admission_reason"].startswith(
        "valid historical result; strict baseline target admission not completed:"
    )
    assert "target_id" not in entry["metadata"]


def test_server_drift_is_not_coerced_or_admitted() -> None:
    entry = _entry()
    entry["same_spec"]["resolved_server_parameters"]["gpu_memory_utilization"] = "0.6"
    verified, errors = bind_entry_to_official_target(entry, _registry(_target()))
    assert verified is False
    assert any("gpu_memory_utilization mismatch" in error for error in errors)
    assert entry["metadata"]["verified"] is False


def test_specialty_target_is_contract_checked_but_not_publicly_verified() -> None:
    entry = _entry()
    target = _target(specialty=True)
    verified, errors = bind_entry_to_official_target(entry, _registry(target))
    assert verified is False
    assert errors == [
        (
            "registry target is not active public-leaderboard: "
            "status='provisional' intended_use='specialty'"
        )
    ]
    assert entry["metadata"]["verified"] is False


def test_snapshot_binding_writes_report_and_is_idempotent(tmp_path: Path) -> None:
    target = _target()
    registry = _registry(target)
    snapshot = tmp_path / "leaderboard_single.json"
    snapshot.write_text(json.dumps([_entry()]) + "\n", encoding="utf-8")
    (tmp_path / "leaderboard_multi.json").write_text("[]\n", encoding="utf-8")

    first = bind_snapshot_set(tmp_path, registry)
    first_bytes = snapshot.read_bytes()
    second = bind_snapshot_set(tmp_path, registry)

    assert first == second
    assert first["verified"] == 1
    assert first["historical_unverified"] == 0
    assert first["reason_counts"] == {}
    assert snapshot.read_bytes() == first_bytes
    report = json.loads(
        (tmp_path / "target_binding_report.json").read_text(encoding="utf-8")
    )
    assert report == first
    assert hashlib.sha256(first_bytes).hexdigest()
