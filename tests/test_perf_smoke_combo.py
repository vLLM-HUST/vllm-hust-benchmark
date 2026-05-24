from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vllm_hust_benchmark.combo_manifest import (
    apply_benchmark_config_overrides,
    build_combo_manifest,
)
from vllm_hust_benchmark.compare_combo_baseline import build_compare_summary
from vllm_hust_benchmark.integration import RepoLayout
from vllm_hust_benchmark.run_ascend_perf_smoke import (
    build_perf_smoke_result,
    run_ascend_perf_smoke,
)


def test_build_combo_manifest_for_vllm_hust_trigger() -> None:
    manifest = build_combo_manifest(
        trigger_repo="vllm-hust",
        trigger_ref="refs/pull/123/head",
        trigger_commit="deadbeef",
        peer_ref="main",
        peer_commit="cafebabe",
    )

    assert manifest["benchmark_profile"] == "ascend-l1-qwen25-3b-dummy-v1"
    assert manifest["source_combo"]["trigger_repo"] == "vllm-hust"
    assert manifest["source_combo"]["vllm_hust"]["ref"] == "refs/pull/123/head"
    assert manifest["source_combo"]["vllm_hust"]["commit"] == "deadbeef"
    assert manifest["source_combo"]["vllm_ascend_hust"]["ref"] == "main"
    assert manifest["source_combo"]["vllm_ascend_hust"]["commit"] == "cafebabe"
    assert manifest["benchmark_config_fingerprint"].startswith("sha256:")
    assert manifest["source_combo_fingerprint"].startswith("sha256:")


def test_apply_benchmark_config_overrides_updates_nested_keys() -> None:
    updated = apply_benchmark_config_overrides(
        {"server": {"max_model_len": 2048, "max_num_seqs": 8}},
        {"server.max-model-len": 4096, "server.max_num_seqs": 16},
    )

    assert updated == {"server": {"max_model_len": 4096, "max_num_seqs": 16}}


def test_build_perf_smoke_result_normalizes_raw_payload(tmp_path) -> None:
    manifest = build_combo_manifest(
        trigger_repo="vllm-hust",
        trigger_ref="head-sha",
        trigger_commit="head-sha",
        peer_ref="peer-sha",
        peer_commit="peer-sha",
    )
    raw_result_file = tmp_path / "raw_benchmark.json"
    raw_result_file.write_text(
        json.dumps(
            {
                "mean_ttft_ms": 12.5,
                "median_ttft_ms": 11.0,
                "output_throughput": 345.0,
                "total_token_throughput": 567.0,
                "request_throughput": 3.5,
                "failed": 2,
                "num_prompts": 20,
            }
        ),
        encoding="utf-8",
    )
    server_log = tmp_path / "server.log"
    server_log.write_text("ready\n", encoding="utf-8")

    result = build_perf_smoke_result(
        manifest=manifest,
        materialized={
            "vllm_hust": {"commit": "head-sha"},
            "vllm_ascend_hust": {"commit": "peer-sha"},
        },
        raw_result_file=raw_result_file,
        server_log=server_log,
    )

    assert result["source_combo"]["vllm_hust"]["commit"] == "head-sha"
    assert result["source_combo"]["vllm_ascend_hust"]["commit"] == "peer-sha"
    assert result["source_combo_fingerprint"].startswith("sha256:")
    assert result["serve"]["output_throughput_tps"] == 345.0
    assert result["throughput"]["tokens_per_second"] == 567.0
    assert result["serve"]["error_rate"] == 0.1
    assert result["latency"]["mean_ms"] == 12.5


def test_run_perf_smoke_creates_missing_runtime_parent(
    tmp_path, monkeypatch
) -> None:
    manifest = build_combo_manifest(
        trigger_repo="vllm-hust",
        trigger_ref="head-sha",
        trigger_commit="head-sha",
        peer_ref="peer-sha",
        peer_commit="peer-sha",
    )
    manifest_path = tmp_path / "head.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "missing" / "results" / "head" / "perf_smoke_result.json"
    result_root = output_path.parent

    materialized_root = tmp_path / "materialized"
    trigger_repo_path = materialized_root / "vllm-hust"
    peer_repo_path = materialized_root / "vllm-ascend-hust"
    trigger_repo_path.mkdir(parents=True)
    peer_repo_path.mkdir(parents=True)

    observed_runtime_root: dict[str, Path] = {}

    def fake_materialize_combo_manifest(*, layout, manifest, runtime_root):
        observed_runtime_root["path"] = runtime_root
        return {
            "vllm_hust": {"path": str(trigger_repo_path), "commit": "head-sha"},
            "vllm_ascend_hust": {"path": str(peer_repo_path), "commit": "peer-sha"},
        }

    def fake_subprocess_run(command, cwd=None, check=False, env=None, **kwargs):
        if env is None:
            return subprocess.CompletedProcess(command, 0)
        assert Path(cwd) == trigger_repo_path
        raw_result_file = Path(env["RAW_RESULT_FILE"])
        server_log = Path(env["SERVER_LOG"])
        raw_result_file.write_text(
            json.dumps({
                "mean_ttft_ms": 12.5,
                "median_ttft_ms": 11.0,
                "output_throughput": 345.0,
                "total_token_throughput": 567.0,
                "request_throughput": 3.5,
                "failed": 0,
                "num_prompts": 20,
            }),
            encoding="utf-8",
        )
        server_log.write_text("ready\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "vllm_hust_benchmark.run_ascend_perf_smoke.materialize_combo_manifest",
        fake_materialize_combo_manifest,
    )
    monkeypatch.setattr(
        "vllm_hust_benchmark.run_ascend_perf_smoke.subprocess.run",
        fake_subprocess_run,
    )

    layout = RepoLayout(
        workspace_root=tmp_path,
        benchmark_repo=tmp_path / "benchmark",
        vllm_hust_repo=tmp_path / "vllm-hust",
        vllm_ascend_hust_repo=tmp_path / "vllm-ascend-hust",
        website_repo=tmp_path / "website",
    )

    exit_code = run_ascend_perf_smoke(
        layout=layout,
        manifest_path=manifest_path,
        output_path=output_path,
        result_root=result_root,
        execute=True,
    )

    assert exit_code == 0
    assert result_root.parent.exists()
    assert observed_runtime_root["path"].exists()
    assert output_path.is_file()


def test_compare_summary_reports_same_peer_regression_and_missing_latest() -> None:
    current = {
        "schema_version": "ascend-l1-perf-smoke/v1",
        "benchmark_config_fingerprint": "sha256:cfg",
        "source_combo_fingerprint": "sha256:head",
        "runner_class": "linux-aarch64-a2b3-0",
        "soc_version": "ascend910b3",
        "source_combo": {"trigger_repo": "vllm-hust"},
        "serve": {
            "output_throughput_tps": 90.0,
            "mean_ttft_ms": 10.0,
            "error_rate": 0.0,
        },
        "throughput": {"tokens_per_second": 95.0},
        "latency": {"mean_ms": 10.0},
    }
    same_peer_ancestor = {
        **current,
        "source_combo_fingerprint": "sha256:ancestor",
        "serve": {
            "output_throughput_tps": 100.0,
            "mean_ttft_ms": 9.0,
            "error_rate": 0.0,
        },
        "throughput": {"tokens_per_second": 100.0},
        "latency": {"mean_ms": 9.0},
    }

    summary = build_compare_summary(
        current=current,
        same_peer_ancestor=same_peer_ancestor,
        latest_protected_combo=None,
        branch_freshness={"status": "fresh"},
    )

    assert summary["would_fail"] is True
    assert summary["same_peer_ancestor"]["status"] == "available"
    assert "serve.output_throughput_tps" in summary["same_peer_ancestor"]["failures"]
    assert "throughput.tokens_per_second" not in summary["same_peer_ancestor"]["failures"]
    assert summary["latest_protected_combo"]["status"] == "missing"