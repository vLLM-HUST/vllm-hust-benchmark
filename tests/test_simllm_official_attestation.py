from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import vllm_hust_benchmark.simllm_official_attestation as attestation


TARGET_ID = "official-simllm-saturated-throughput-warm-cache-qwen25-14b-1chip-910b2"


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target() -> dict:
    return {
        "target_id": TARGET_ID,
        "target_version": "1.6.0",
        "status": "active",
        "profile": "simllm-warm-cache",
        "baseline_runtime": {
            "engine": "vllm-hust",
            "core_commit": "a" * 40,
            "backend_commit": "b" * 40,
            "runtime_image": "image@sha256:" + "c" * 64,
            "runtime_image_digest": "sha256:" + "c" * 64,
            "runtime_packages": {"vllm": "0.21.0+empty"},
        },
        "workload": {
            "name": "simllm-saturated-throughput-warm-cache",
            "client_parameters": {"num_prompts": 32},
            "protocol": {
                "baseline_engine": "vllm-hust",
                "candidate_engine": "vllm-hust-simllm",
                "minimum_independent_repetitions": 3,
                "maximum_primary_metric_cv_percent": 5,
                "simllm_config": {"kv_cache_size": 32},
            },
        },
    }


def _make_arm(
    repeat: Path,
    name: str,
    *,
    engine: str,
    throughput: float,
    enabled: bool,
) -> None:
    arm = repeat / name
    raw = {
        "completed": 32,
        "failed": 0,
        "request_throughput": throughput,
    }
    _write(arm / "raw_benchmark_result.json", raw)
    _write(
        arm / "submission" / "run_leaderboard.json",
        {
            "entry_id": f"{repeat.name}-{name}",
            "engine": engine,
            "workload": {"name": "random-online"},
            "metadata": {},
        },
    )
    _write(arm / "submission" / "leaderboard_manifest.json", {"entries": []})
    _write(arm / "prompt_cohort_evidence.json", {"cohort_sha256": "cohort"})
    _write(arm / "warmup_pass_1.json", {"completed": 32})
    (arm / "server.stdout.log").write_text("clean\n", encoding="utf-8")
    for state in ("device_state_before.txt", "device_state_after.txt"):
        (arm / state).write_text("No process in device.\n", encoding="utf-8")
    evidence = {
        "engine": engine,
        "simllm_enabled": enabled,
        "warmup_performed": enabled,
        "core_commit": "a" * 40,
        "backend_commit": "b" * 40,
        "runtime": {
            "image": "image@sha256:" + "c" * 64,
            "image_digest": "sha256:" + "c" * 64,
            "packages": {"vllm": "0.21.0+empty"},
        },
        "failed": 0,
        "patch_applied": enabled,
        "rewrite_events": 2 if enabled else 0,
        "rewritten_requests": 32 if enabled else 0,
        "simllm_config": {"kv_cache_size": 32},
        "hashes": {"raw_result_sha256": _sha(arm / "raw_benchmark_result.json")},
    }
    _write(arm / "arm_evidence.json", evidence)


def _make_campaign(root: Path, candidate_values: list[float]) -> None:
    for index, candidate in enumerate(candidate_values, start=1):
        repeat = root / f"repeat-{index:02d}"
        _make_arm(
            repeat,
            "baseline-disabled",
            engine="vllm-hust",
            throughput=1.0 + index * 0.001,
            enabled=False,
        )
        _make_arm(
            repeat,
            "enabled-warm-cache",
            engine="vllm-hust-simllm",
            throughput=candidate,
            enabled=True,
        )
        _write(
            repeat / "paired_protocol_evidence.json",
            {
                "schema_version": "simllm-official-paired-protocol/v1",
                "spec_id": TARGET_ID,
                "exact_measured_setting_match": True,
                "zero_failed_requests": True,
                "resolved_spec_hash": "spec-hash",
                "prompt_cohort_sha256": "cohort",
            },
        )


def test_attests_three_stable_paired_repeats(tmp_path: Path, monkeypatch) -> None:
    results = tmp_path / "results"
    _make_campaign(results, [4.0, 4.02, 3.99])
    monkeypatch.setattr(attestation, "_target", lambda *_: (_target(), "d" * 64))
    monkeypatch.setattr(attestation, "_validate_exact_target", lambda *_: None)

    output = tmp_path / "output"
    result = attestation.attest_simllm_campaign(
        tmp_path,
        results,
        output,
        target_id=TARGET_ID,
        verified_by="test",
        verified_at="2026-08-03T00:00:00Z",
    )

    assert result["successful_repeats"] == 3
    assert result["candidate_statistics"]["cv_percent"] < 5
    assert result["median_improvement_percent"] > 290
    candidate = json.loads(
        (output / "simllm-enabled-warm-cache" / "run_leaderboard.json").read_text()
    )
    assert candidate["workload"]["name"] == "simllm-saturated-throughput-warm-cache"
    assert candidate["metadata"]["verified"] is True


def test_rejects_unstable_primary_metric(tmp_path: Path, monkeypatch) -> None:
    results = tmp_path / "results"
    _make_campaign(results, [2.0, 4.0, 8.0])
    monkeypatch.setattr(attestation, "_target", lambda *_: (_target(), "d" * 64))
    monkeypatch.setattr(attestation, "_validate_exact_target", lambda *_: None)

    with pytest.raises(ValueError, match="CV exceeds publication gate"):
        attestation.attest_simllm_campaign(
            tmp_path,
            results,
            tmp_path / "output",
            target_id=TARGET_ID,
            verified_by="test",
        )
