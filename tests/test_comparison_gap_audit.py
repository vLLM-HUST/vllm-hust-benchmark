from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft7Validator

from vllm_hust_benchmark.comparison_gap_audit import build_comparison_gap_audit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _target(target_id: str, workload: str) -> dict[str, object]:
    return {
        "target_id": target_id,
        "target_version": "1.0.0",
        "status": "active",
        "intended_use": "public-leaderboard",
        "hardware": {"chip_count": 1},
        "model": {"id": "Qwen/model", "precision": "FP16"},
        "server_parameters": {"gpu_memory_utilization": 0.6},
        "workload": {
            "name": workload,
            "client_parameters": {"request_rate": 1},
        },
        "source_spec": {"path": f"docs/{target_id}.json"},
    }


def _entry(
    target_id: str,
    workload: str,
    engine: str,
    spec_hash: str,
    *,
    verified: bool = True,
) -> dict[str, object]:
    core_commit = "core-head" if engine == "vllm-hust" else "baseline-core"
    plugin_commit = "plugin-head" if engine == "vllm-hust" else "baseline-plugin"
    return {
        "entry_id": f"{target_id}-{engine}",
        "engine": engine,
        "workload": {"name": workload},
        "hardware": {"chip_count": 1},
        "model": {"name": "Qwen/model", "precision": "FP16"},
        "same_spec": {
            "spec_id": target_id,
            "model": "Qwen/model",
            "model_precision": "FP16",
            "resolved_server_parameters": {"gpu_memory_utilization": 0.6},
            "resolved_client_parameters": {"request_rate": 1},
            "resolved_spec_hash": spec_hash,
        },
        "metrics": {"error_rate": 0, "peak_mem_mb": 1024},
        "environment": {
            "cann_version": "9.0",
            "driver_version": "26.0",
            "pytorch_version": "2.8",
        },
        "metadata": {
            "submitted_at": "2026-08-03T00:00:00Z",
            "verified": verified,
            "target_id": target_id,
            "target_version": "1.0.0",
            "workload_config_contract": "explicit-effective/v1",
            "reproducible_cmd": "run benchmark",
            "runtime_provenance": {
                "engine": {"commit": core_commit},
                "plugin": {"commit": plugin_commit},
            },
            "verification_attestation": {"successful_repeats": 3},
        },
    }


def test_audit_includes_new_trace_workloads_and_requires_strict_pairs(
    tmp_path: Path,
) -> None:
    ready = _target("ready", "random-online")
    missing = _target("missing", "sharegpt-online")
    trace = _target("trace", "tracelab-coding-agent-replay")
    burst = _target("burst", "burstgpt-production-replay")
    trace["profile"] = "production-trace"
    burst["profile"] = "production-trace"
    _write_json(
        tmp_path / "leaderboard-data/official-targets.json",
        {
            "registry_version": "1.0.0",
            "targets": [ready, missing, trace, burst],
        },
    )
    entries = [
        _entry("ready", "random-online", "vllm", "same-hash"),
        _entry("ready", "random-online", "vllm-hust", "same-hash"),
        _entry("missing", "sharegpt-online", "vllm", "baseline-hash"),
        _entry(
            "missing",
            "sharegpt-online",
            "vllm-hust",
            "current-hash",
            verified=False,
        ),
    ]
    _write_json(
        tmp_path / "leaderboard-data/snapshots/leaderboard_single.json", entries
    )
    _write_json(tmp_path / "leaderboard-data/snapshots/leaderboard_multi.json", [])

    report = build_comparison_gap_audit(
        tmp_path,
        generated_at="2026-08-03T00:00:00Z",
        current_core_head="core-head",
        current_plugin_head="plugin-head",
    )

    assert report["policy"]["excluded_workloads"] == []
    assert report["summary"] == {
        "target_count": 4,
        "ready_pair_count": 1,
        "rerun_target_count": 3,
        "rerun_job_count": 5,
    }
    missing_current = next(
        item
        for item in report["rerun_queue"]
        if item["target_id"] == "missing" and item["engine"] == "vllm-hust"
    )
    assert "verified-attestation-missing" in missing_current["reasons"]


def test_current_checkout_must_match_both_heads(tmp_path: Path) -> None:
    target = _target("target", "random-online")
    _write_json(
        tmp_path / "leaderboard-data/official-targets.json",
        {"registry_version": "1.0.0", "targets": [target]},
    )
    entries = [
        _entry("target", "random-online", "vllm", "same-hash"),
        _entry("target", "random-online", "vllm-hust", "same-hash"),
    ]
    _write_json(
        tmp_path / "leaderboard-data/snapshots/leaderboard_single.json", entries
    )
    _write_json(tmp_path / "leaderboard-data/snapshots/leaderboard_multi.json", [])

    report = build_comparison_gap_audit(
        tmp_path,
        current_core_head="new-core",
        current_plugin_head="new-plugin",
    )
    reasons = report["records"][0]["current"]["reasons"]
    assert "current-core-head-stale" in reasons
    assert "current-plugin-head-stale" in reasons
    assert report["summary"]["ready_pair_count"] == 0


def test_current_workspace_report_matches_schema_and_includes_trace_targets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report = build_comparison_gap_audit(repo_root)
    schema = json.loads(
        (repo_root / "schemas/leaderboard_comparison_gap_audit_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft7Validator(schema).validate(report)
    workloads = {record["workload"] for record in report["records"]}
    assert "tracelab-coding-agent-replay" in workloads
    assert "burstgpt-production-replay" in workloads
