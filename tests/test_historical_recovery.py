from __future__ import annotations

import json
from pathlib import Path

from vllm_hust_benchmark.historical_recovery import build_recovery
from vllm_hust_benchmark.same_spec import compute_resolved_spec_hash

ENGINE_COMMIT = "a" * 40
PLUGIN_COMMIT = "b" * 40
SPEC_ID = "target-random-online"


def _same_spec() -> dict:
    payload = {
        "schema_version": "benchmark-same-spec/v1",
        "spec_id": SPEC_ID,
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
            "model": "/runtime/model",
            "tensor_parallel_size": 1,
        },
        "resolved_client_parameters": {
            "model": "/runtime/model",
            "dataset_name": "random",
            "num_prompts": 200,
        },
    }
    payload["resolved_spec_hash"] = compute_resolved_spec_hash(payload)
    return payload


def _entry(*, throughput: float = 100.0, plugin_commit: str = PLUGIN_COMMIT) -> dict:
    return {
        "entry_id": f"entry-{throughput}",
        "engine": "vllm-hust",
        "config_type": "single_gpu",
        "workload": {"name": "random-online"},
        "metrics": {
            "throughput_tps": throughput,
            "ttft_ms": 10.0,
            "tbt_ms": 2.0,
            "peak_mem_mb": 0,
            "error_rate": 0.0,
        },
        "metadata": {
            "verified": None,
            "submitted_at": "2026-07-01T00:00:00Z",
            "git_commit": ENGINE_COMMIT,
            "runtime_provenance": {
                "engine": {"commit": ENGINE_COMMIT},
                "plugin": {"commit": plugin_commit},
            },
        },
        "same_spec": _same_spec(),
    }


def _write_inputs(tmp_path: Path, entries: list[tuple[str, dict]]) -> tuple[Path, Path]:
    for name, entry in entries:
        directory = tmp_path / "submissions" / name
        directory.mkdir(parents=True)
        (directory / "run_leaderboard.json").write_text(
            json.dumps(entry), encoding="utf-8"
        )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "registry_version": "1.0.0",
                "targets": [{"target_id": SPEC_ID, "target_version": "1.0.0"}],
            }
        ),
        encoding="utf-8",
    )
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps(
            {
                "aliases": {PLUGIN_COMMIT[:12]: PLUGIN_COMMIT},
            }
        ),
        encoding="utf-8",
    )
    return registry, aliases


def test_recovers_inferable_fields_without_requiring_verified(tmp_path: Path) -> None:
    entry = _entry(plugin_commit=PLUGIN_COMMIT[:12])
    entry["same_spec"]["resolved_spec_hash"] = "0" * 64
    registry, aliases = _write_inputs(tmp_path, [("run", entry)])

    recovered, report = build_recovery(
        repo_root=tmp_path,
        registry_path=registry,
        revision_aliases_path=aliases,
    )

    assert len(recovered) == 1
    result = recovered[0]
    assert result["metadata"]["verified"] is None
    assert result["metadata"]["target_id"] == SPEC_ID
    assert result["metadata"]["target_version"] == "1.0.0"
    assert result["metadata"]["runtime_provenance"]["plugin"]["commit"] == PLUGIN_COMMIT
    assert result["same_spec"]["resolved_spec_hash"] == compute_resolved_spec_hash(
        result["same_spec"]
    )
    assert report["summary"]["required_experiments"] == 0


def test_deduplication_prefers_evidence_not_best_metric(tmp_path: Path) -> None:
    older_high = _entry(throughput=999.0)
    newer_lower = _entry(throughput=100.0)
    newer_lower["metadata"]["submitted_at"] = "2026-07-02T00:00:00Z"
    registry, aliases = _write_inputs(
        tmp_path, [("older-high", older_high), ("newer-lower", newer_lower)]
    )

    recovered, report = build_recovery(
        repo_root=tmp_path,
        registry_path=registry,
        revision_aliases_path=aliases,
    )

    assert [entry["metrics"]["throughput_tps"] for entry in recovered] == [100.0]
    assert report["summary"]["superseded_entries"] == 1
    assert report["policy"]["deduplication_uses_metrics"] is False


def test_only_non_inferable_measurement_is_scheduled_for_rerun(
    tmp_path: Path,
) -> None:
    entry = _entry()
    entry["metrics"]["error_rate"] = None
    registry, aliases = _write_inputs(tmp_path, [("invalid", entry)])

    recovered, report = build_recovery(
        repo_root=tmp_path,
        registry_path=registry,
        revision_aliases_path=aliases,
    )

    assert recovered == []
    assert report["summary"]["required_experiments"] == 1
    assert report["required_experiments"][0]["missing_or_invalid"] == [
        "invalid-error-rate"
    ]


def test_derives_zero_error_rate_from_atomic_offline_latency_success(
    tmp_path: Path,
) -> None:
    entry = _entry()
    entry["workload"]["name"] = "random-latency"
    entry["same_spec"]["scenario"] = "random-latency"
    entry["same_spec"]["resolved_client_parameters"] = {
        "model": "/runtime/model",
        "input_len": 1024,
        "output_len": 128,
        "batch_size": 8,
        "num_iters_warmup": 10,
        "num_iters": 30,
    }
    entry["same_spec"]["resolved_spec_hash"] = compute_resolved_spec_hash(
        entry["same_spec"]
    )
    entry["metadata"]["data_source"] = "real-online-historical-pr-backfill"
    entry["metadata"]["idempotency_key"] = "original-artifact-key"
    entry["metrics"]["error_rate"] = None
    registry, aliases = _write_inputs(tmp_path, [("atomic-latency", entry)])

    recovered, report = build_recovery(
        repo_root=tmp_path,
        registry_path=registry,
        revision_aliases_path=aliases,
    )

    assert recovered[0]["metrics"]["error_rate"] == 0.0
    recovery = recovered[0]["historical_recovery"]
    assert "metrics.error_rate" in recovery["inferred_fields"]
    derivation = recovery["measurement_derivations"][0]
    assert derivation["rule_id"] == "atomic-offline-latency-success/v1"
    assert derivation["evidence"]["num_iters_warmup"] == 10
    assert derivation["evidence"]["num_iters"] == 30
    assert len(derivation["evidence"]["original_artifact_identity"]["sha256"]) == 64
    assert report["summary"]["required_experiments"] == 0
    assert report["summary"]["satisfied_experiments"] == 1
    assert report["satisfied_experiments"][0]["evidence_kind"] == "derived-success"


def test_real_rerun_satisfies_matching_invalid_measurement(tmp_path: Path) -> None:
    invalid = _entry()
    invalid["workload"]["name"] = "random-latency"
    invalid["same_spec"]["scenario"] = "random-latency"
    invalid["same_spec"]["resolved_client_parameters"].update(
        {"num_iters_warmup": 10, "num_iters": 30}
    )
    invalid["same_spec"]["resolved_spec_hash"] = compute_resolved_spec_hash(
        invalid["same_spec"]
    )
    invalid["metadata"]["data_source"] = "real-online-historical-pr-backfill"
    invalid["metrics"]["error_rate"] = None
    rerun = _entry(throughput=101.0)
    rerun["workload"]["name"] = "random-latency"
    rerun["same_spec"]["scenario"] = "random-latency"
    rerun["metadata"]["submitted_at"] = "2026-08-16T00:00:00Z"
    rerun["same_spec"]["resolved_server_parameters"]["max_model_len"] = 32768
    rerun["same_spec"]["resolved_spec_hash"] = compute_resolved_spec_hash(
        rerun["same_spec"]
    )
    registry, aliases = _write_inputs(
        tmp_path, [("historical-invalid", invalid), ("real-rerun", rerun)]
    )

    recovered, report = build_recovery(
        repo_root=tmp_path,
        registry_path=registry,
        revision_aliases_path=aliases,
    )

    assert len(recovered) == 1
    assert recovered[0]["metrics"]["error_rate"] == 0.0
    assert report["summary"]["required_experiments"] == 0
    assert report["summary"]["satisfied_experiments"] == 1
    assert report["satisfied_experiments"][0]["source_path"].endswith(
        "historical-invalid/run_leaderboard.json"
    )
    assert report["satisfied_experiments"][0]["replacement_source_path"].endswith(
        "real-rerun/run_leaderboard.json"
    )
    assert report["satisfied_experiments"][0]["evidence_kind"] == "fresh-rerun"
    assert (
        len(report["satisfied_experiments"][0]["original_artifact_identity"]["sha256"])
        == 64
    )
