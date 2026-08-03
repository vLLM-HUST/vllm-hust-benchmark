from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.official_baseline_attestation import (
    attest_completed_baseline,
)

TRACE_IMAGE = "quay.io/ascend/vllm-ascend@sha256:" + "b" * 64
TRACE_DIGEST = "sha256:" + "b" * 64
TRACE_PACKAGES = {
    "transformers": "5.5.4",
    "huggingface-hub": "1.21.0",
    "click": "8.4.1",
    "vllm": "0.22.1+empty",
    "vllm-ascend": "0.22.1rc1",
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
}
TRACE_ENVIRONMENT = {"VLLM_BATCH_INVARIANT": "1"}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    repo = tmp_path / "repo"
    spec = {"id": "target-1"}
    spec_path = repo / "docs" / "spec.json"
    _write(spec_path, spec)
    target = {
        "target_id": "target-1",
        "target_version": "1.0.0",
        "profile": "core-text",
        "status": "active",
        "intended_use": "public-leaderboard",
        "baseline_runtime": {
            "engine": "vllm",
            "engine_version": "0.18.0",
            "git_commit": "plugin-sha",
        },
        "hardware": {
            "vendor": "Huawei",
            "chip_model": "910B2",
            "chip_count": 1,
            "node_count": 1,
        },
        "model": {"id": "Qwen/model", "parameters": "14B", "precision": "FP16"},
        "server_parameters": {"max_model_len": 32768},
        "workload": {
            "name": "sonnet-throughput",
            "client_parameters": {"num_prompts": 200},
        },
        "source_spec": {
            "path": "docs/spec.json",
            "sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        },
    }
    registry_path = repo / "leaderboard-data" / "official-targets.json"
    _write(registry_path, {"targets": [target]})
    (registry_path.parent / "official-targets.sha256").write_text(
        hashlib.sha256(registry_path.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    entry = {
        "entry_id": "entry-1",
        "engine": "vllm",
        "engine_version": "0.18.0",
        "hardware": {"vendor": "Huawei", "chip_model": "910B2", "chip_count": 1},
        "model": {"repo_id": "Qwen/model", "parameters": "14B", "precision": "FP16"},
        "metrics": {"throughput_tps": 100.0, "error_rate": 0},
        "same_spec": {
            "spec_id": "target-1",
            "model": "Qwen/model",
            "model_parameters": "14B",
            "model_precision": "FP16",
            "hardware_vendor": "Huawei",
            "hardware_chip_model": "910B2",
            "chip_count": 1,
            "node_count": 1,
            "resolved_server_parameters": {"max_model_len": 32768},
            "resolved_client_parameters": {"num_prompts": 200},
        },
        "metadata": {
            "idempotency_key": "key-1",
            "runtime_provenance": {
                "engine": {"commit": "core-sha"},
                "plugin": {"commit": "plugin-sha"},
            },
        },
    }
    staged = repo / "staged" / "target-1"
    _write(staged / "run_leaderboard.json", entry)
    _write(
        staged / "leaderboard_manifest.json",
        {
            "entries": [
                {
                    "idempotency_key": "key-1",
                    "leaderboard_artifact": "run_leaderboard.json",
                }
            ]
        },
    )
    results = repo / "results" / "target-1"
    for number in range(1, 4):
        repeat = results / f"repeat-{number:02d}"
        _write(repeat / "raw_benchmark_result.json", {"failed": 0})
        _write(repeat / "submission" / "run_leaderboard.json", entry)
        (repeat / "runner.log").write_text("ok\n", encoding="utf-8")
    return repo, staged, results, entry, target


def _trace_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo, staged, results, entry, target = _fixture(tmp_path)
    target["profile"] = "production-trace"
    target["baseline_runtime"].update(
        {
            "core_commit": "core-sha",
            "backend_commit": "plugin-sha",
            "runtime_packages": TRACE_PACKAGES,
            "runtime_image": TRACE_IMAGE,
            "runtime_image_digest": TRACE_DIGEST,
            "runtime_environment": TRACE_ENVIRONMENT,
        }
    )
    target["workload"] = {
        "name": "burstgpt-production-replay",
        "client_parameters": {"max_requests": 2},
    }
    entry["same_spec"]["resolved_client_parameters"] = {"max_requests": 2}
    registry_path = repo / "leaderboard-data" / "official-targets.json"
    _write(registry_path, {"targets": [target]})
    (registry_path.parent / "official-targets.sha256").write_text(
        hashlib.sha256(registry_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    _write(staged / "run_leaderboard.json", entry)

    signature = "cohort-signature"
    model_digest = "a" * 64
    for number in range(1, 4):
        repeat = results / f"repeat-{number:02d}"
        repeat_entry = json.loads(json.dumps(entry))
        repeat_entry["entry_id"] = f"entry-{number}"
        repeat_entry["metadata"]["idempotency_key"] = f"key-{number}"
        raw = {
            "completed": 2,
            "failed": 0,
            "repeat": number,
            "cohort_setting_signature": signature,
        }
        plan = {"cohort_setting_signature": signature}
        detail_lines = [
            json.dumps(
                {
                    "type": "metadata",
                    "plan": {"cohort_setting_signature": signature},
                }
            ),
            json.dumps({"request_id": f"{number}-1"}),
            json.dumps({"request_id": f"{number}-2"}),
        ]
        _write(repeat / "raw_benchmark_result.json", raw)
        _write(repeat / "trace_replay_plan.json", plan)
        _write(
            repeat / "model_artifact_provenance.json",
            {"model_artifact_digest": model_digest},
        )
        _write(
            repeat / "runtime_package_provenance.json",
            {
                "runtime_packages": TRACE_PACKAGES,
                "runtime_image": TRACE_IMAGE,
                "runtime_image_digest": TRACE_DIGEST,
                "runtime_environment": TRACE_ENVIRONMENT,
            },
        )
        (repeat / "trace_replay_results.jsonl").write_text(
            "\n".join(detail_lines) + "\n", encoding="utf-8"
        )
        (repeat / "server.stdout.log").write_text("ready\n", encoding="utf-8")
        (repeat / "runner.log").write_text(f"repeat {number}\n", encoding="utf-8")
        _write(repeat / "submission" / "run_leaderboard.json", repeat_entry)
        _write(
            repeat / "startup_evidence.json",
            {
                "startup_instance_id": f"startup-{number}",
                "run_id": f"run-{number}",
                "engine_source_commit": "core-sha",
                "plugin_source_commit": "plugin-sha",
                "model_artifact_digest": model_digest,
                "cohort_setting_signature": signature,
                "runtime_packages": TRACE_PACKAGES,
                "runtime_image": TRACE_IMAGE,
                "runtime_image_digest": TRACE_DIGEST,
                "runtime_environment": TRACE_ENVIRONMENT,
                "finished_at": f"2026-08-02T00:00:0{number}Z",
                "result_hashes": {
                    "raw_sha256": hashlib.sha256(
                        (repeat / "raw_benchmark_result.json").read_bytes()
                    ).hexdigest(),
                    "detail_sha256": hashlib.sha256(
                        (repeat / "trace_replay_results.jsonl").read_bytes()
                    ).hexdigest(),
                },
            },
        )
    return repo, staged, results


def test_attests_three_exact_zero_error_repeats(tmp_path: Path) -> None:
    repo, staged, results, _, target = _fixture(tmp_path)
    output = repo / "submissions" / "target-1"
    attested = attest_completed_baseline(
        repo,
        staged,
        results,
        output,
        verified_by="test-review",
        verified_at="2026-08-02T00:00:00Z",
    )
    assert attested["metadata"]["verified"] is True
    assert attested["metadata"]["target_version"] == target["target_version"]
    assert attested["metadata"]["profile_id"] == target["profile"]
    suite = json.loads((output / "repeat_suite.json").read_text())
    assert suite["successful_repeats"] == 3
    assert suite["selected_repeat"] == "repeat-01"


def test_rejects_fewer_than_three_repeats(tmp_path: Path) -> None:
    repo, staged, results, _, _ = _fixture(tmp_path)
    for path in (results / "repeat-03").glob("**/*"):
        if path.is_file():
            path.unlink()
    with pytest.raises(ValueError, match="insufficient successful repeats"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_attests_three_independent_production_trace_repeats(tmp_path: Path) -> None:
    repo, staged, results = _trace_fixture(tmp_path)
    attested = attest_completed_baseline(
        repo, staged, results, repo / "out", verified_by="test-review"
    )
    assert attested["metadata"]["verified"] is True
    suite = json.loads((repo / "out" / "repeat_suite.json").read_text())
    assert suite["successful_repeats"] == 3
    assert {repeat["startup_instance_id"] for repeat in suite["repeats"]} == {
        "startup-1",
        "startup-2",
        "startup-3",
    }


def test_rejects_cloned_production_trace_repeat(tmp_path: Path) -> None:
    repo, staged, results = _trace_fixture(tmp_path)
    repeat_2 = results / "repeat-02"
    repeat_1 = results / "repeat-01"
    (repeat_2 / "startup_evidence.json").write_bytes(
        (repeat_1 / "startup_evidence.json").read_bytes()
    )
    with pytest.raises(ValueError, match="startup raw result hash mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_incomplete_production_trace_evidence(tmp_path: Path) -> None:
    repo, staged, results = _trace_fixture(tmp_path)
    (results / "repeat-03" / "trace_replay_plan.json").unlink()
    with pytest.raises(ValueError, match="evidence is missing"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_production_trace_runtime_image_digest_mismatch(
    tmp_path: Path,
) -> None:
    repo, staged, results = _trace_fixture(tmp_path)
    provenance_path = results / "repeat-02" / "runtime_package_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime_image_digest"] = "sha256:" + "c" * 64
    _write(provenance_path, provenance)

    with pytest.raises(ValueError, match="runtime image provenance mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )


def test_rejects_production_trace_runtime_environment_mismatch(
    tmp_path: Path,
) -> None:
    repo, staged, results = _trace_fixture(tmp_path)
    provenance_path = results / "repeat-02" / "runtime_package_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime_environment"] = {"VLLM_BATCH_INVARIANT": "0"}
    _write(provenance_path, provenance)

    with pytest.raises(ValueError, match="runtime environment provenance mismatch"):
        attest_completed_baseline(
            repo, staged, results, repo / "out", verified_by="test-review"
        )
