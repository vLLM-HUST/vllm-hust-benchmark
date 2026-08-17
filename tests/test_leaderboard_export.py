from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_hust_benchmark.leaderboard_export import export_leaderboard_artifacts
from vllm_hust_benchmark.same_spec import build_same_spec_payload


REPO_ROOT = Path(__file__).resolve().parents[1]


def _metrics_file(tmp_path: Path) -> Path:
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(
        json.dumps(
            {
                "metrics": {
                    "ttft_ms": 10.0,
                    "throughput_tps": 100.0,
                    "peak_mem_mb": 1024.0,
                    "error_rate": 0.0,
                },
                "constraints_metrics": {
                    "single_chip_effective_utilization_pct": 91.0,
                    "typical_throughput_ratio_vs_baseline": 2.0,
                    "typical_ttft_reduction_pct_vs_baseline": 20.0,
                    "typical_tpot_reduction_pct_vs_baseline": 20.0,
                    "long_context_length": 32768,
                    "long_context_throughput_stable": True,
                    "long_context_ttft_p95_ms": 45.0,
                    "long_context_ttft_p99_ms": 58.0,
                    "long_context_tpot_p95_ms": 12.0,
                    "long_context_tpot_p99_ms": 16.0,
                    "long_context_ttft_p95_stable": True,
                    "long_context_ttft_p99_stable": True,
                    "long_context_tpot_p95_stable": True,
                    "long_context_tpot_p99_stable": True,
                    "unit_token_cost_reduction_pct": 31.0,
                    "multi_tenant_high_utilization": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return metrics_file


def _scenario() -> SimpleNamespace:
    return SimpleNamespace(
        name="sharegpt-online",
        benchmark_type="serve",
        leaderboard={
            "workload_name": "sharegpt-online",
            "representative_business_scenario": "general-serving",
            "default_config_type": "single_gpu",
        },
        defaults={
            "input_len": 1024,
            "output_len": 256,
            "dataset_name": "sharegpt",
        },
    )


def _write_same_spec_file(tmp_path: Path, spec_path: Path) -> Path:
    """Build a real same_spec payload from an official spec file."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    same_spec = build_same_spec_payload(spec)
    same_spec_file = tmp_path / "same_spec.json"
    same_spec_file.write_text(
        json.dumps(same_spec, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return same_spec_file


def _common_export_kwargs(
    tmp_path: Path,
    *,
    same_spec_file: Path | None,
    spec_path: Path | None,
) -> dict:
    return dict(
        scenario=_scenario(),
        metrics_file=_metrics_file(tmp_path),
        benchmark_result_file=None,
        constraints_file=None,
        same_spec_file=same_spec_file,
        output_dir=tmp_path / "out",
        artifact_name="run_leaderboard.json",
        run_id="test-run",
        engine="vllm",
        engine_version="0.18.0",
        model_name="Qwen/Qwen2.5-14B-Instruct",
        model_parameters="14B",
        model_precision="FP16",
        hardware_vendor="Huawei",
        hardware_chip_model="910B2",
        chip_count=1,
        node_count=1,
        submitter="official-ascend-baseline",
        baseline_engine="vllm",
        domestic_chip_class="Ascend-class",
        representative_model_band="14B",
        data_source="reference-vllm-ascend-benchmark",
        input_length=1024,
        output_length=256,
        batch_size=None,
        concurrent_requests=None,
        protocol_version="0.1.0",
        backend_version="0.1.0",
        core_version="0.1.0",
        peak_mem_mb=1024.0,
        git_commit="e18643f8a4d5bd9990727654318ad069ea0b56e2",  # pragma: allowlist secret
        github_user="official",
        github_commit_url=None,
        github_repository="vllm-project/vllm-ascend",
        github_ref="v0.18.0",
        github_event_name=None,
        github_pr_number=None,
        github_pr_url=None,
        runtime_python="python",
        engine_source_repository="vllm-project/vllm",
        engine_source_ref="v0.18.0",
        engine_source_commit="e18643f8a4d5bd9990727654318ad069ea0b56e2",  # pragma: allowlist secret
        plugin_source_engine="vllm-ascend",
        plugin_source_repository="vllm-project/vllm-ascend",
        plugin_source_ref="v0.18.0",
        plugin_source_commit="e18643f8a4d5bd9990727654318ad069ea0b56e2",  # pragma: allowlist secret
        spec_path=spec_path,
    )


def test_official_entry_records_target_id_and_target_version(tmp_path: Path) -> None:
    spec_path = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json"
    )
    same_spec_file = _write_same_spec_file(tmp_path, spec_path)

    artifact_path, _ = export_leaderboard_artifacts(
        **_common_export_kwargs(
            tmp_path, same_spec_file=same_spec_file, spec_path=spec_path
        )
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = artifact["metadata"]
    assert metadata["target_id"] == "official-ascend-jan-2026-v0.18.0"
    assert metadata["target_version"] == "Official Ascend Jan 2026"
    assert (
        metadata["target_contract_id"]
        == "official-ascend-jan-2026-v0.18.0-sharegpt-online-qwen25-14b-910b2"
    )
    assert metadata["target_contract_version"] == "1.3.5"
    assert metadata["workload_config_contract"] == "explicit-effective/v1"


def test_official_entry_fails_closed_when_spec_path_missing(tmp_path: Path) -> None:
    spec_path = (
        REPO_ROOT
        / "docs"
        / "official-baselines"
        / "official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json"
    )
    same_spec_file = _write_same_spec_file(tmp_path, spec_path)

    with pytest.raises(ValueError, match="spec_path is required"):
        export_leaderboard_artifacts(
            **_common_export_kwargs(
                tmp_path, same_spec_file=same_spec_file, spec_path=None
            )
        )


def test_non_official_entry_omits_target_id(tmp_path: Path) -> None:
    """Non-official entries (no same_spec.spec_id with official prefix) must
    not have target_id/target_version in metadata."""
    artifact_path, _ = export_leaderboard_artifacts(
        scenario=_scenario(),
        metrics_file=_metrics_file(tmp_path),
        benchmark_result_file=None,
        constraints_file=None,
        same_spec_file=None,
        output_dir=tmp_path / "out",
        artifact_name="run_leaderboard.json",
        run_id="test-run",
        engine="vllm-hust",
        engine_version="0.18.0",
        model_name="Qwen/Qwen2.5-14B-Instruct",
        model_parameters="14B",
        model_precision="FP16",
        hardware_vendor="Huawei",
        hardware_chip_model="910B2",
        chip_count=1,
        node_count=1,
        submitter="vllm-hust-team",
        baseline_engine="vllm",
        domestic_chip_class="Ascend-class",
        representative_model_band="14B",
        data_source="vllm-hust-benchmark",
        input_length=1024,
        output_length=256,
        batch_size=None,
        concurrent_requests=None,
        protocol_version="0.1.0",
        backend_version="0.1.0",
        core_version="0.1.0",
        peak_mem_mb=1024.0,
        git_commit="e18643f8a4d5bd9990727654318ad069ea0b56e2",  # pragma: allowlist secret
        github_user="developer",
        github_commit_url=None,
        github_repository="vLLM-HUST/vllm-hust",
        github_ref="main",
        github_event_name=None,
        github_pr_number=None,
        github_pr_url=None,
        runtime_python="python",
        engine_source_repository="vLLM-HUST/vllm-hust",
        engine_source_ref="main",
        engine_source_commit="e18643f8a4d5bd9990727654318ad069ea0b56e2",  # pragma: allowlist secret
        plugin_source_engine="vllm-ascend-hust",
        plugin_source_repository="vLLM-HUST/vllm-ascend-hust",
        plugin_source_ref="main",
        plugin_source_commit="abcdef1234567890",  # pragma: allowlist secret
        spec_path=None,
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = artifact["metadata"]
    assert "target_id" not in metadata
    assert "target_version" not in metadata
    assert "workload_config_contract" not in metadata
