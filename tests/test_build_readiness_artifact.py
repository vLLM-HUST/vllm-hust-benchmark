"""End-to-end tests for scripts/build_readiness_artifact.py.

Per PR #154 review round 2: three issues were fixed —
1. burst_config must reflect the REAL burst measurement (size >= 1,
   duration_s > 0) from client_result.json, not placeholder zeros.
2. The plugin (vllm-ascend-hust) commit must be resolved independently
   from the benchmark repo commit (VLLM_HUST_PLUGIN_COMMIT env var).
3. cold_readiness_median in matrix_summary.json must come from
   startup_metrics.cold_readiness_s, not output_throughput_tps.

These tests construct a minimal cell directory with synthetic
server.log, client_result.json, metrics.txt, startup_ts.txt and
probe_result.json, then invoke build_readiness_artifact.py as a
subprocess to verify the produced artifact has correct burst_config,
burst_recovery_s, and plugin provenance.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_readiness_artifact.py"

COMMIT_40 = "a" * 40
COMMIT_40_B = "b" * 40
COMMIT_40_C = "c" * 40


def _write_cell(
    tmp_path: Path,
    *,
    profile: str = "burst",
    report_type: str = "burst",
    completed: int = 50,
    failed: int = 0,
    duration: float = 42.5,
    request_rate: float = 10.0,
    probe_recovery_s: float = 1.5,
    with_probe: bool = True,
) -> Path:
    """Create a minimal cell directory with synthetic inputs."""
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir(parents=True, exist_ok=True)

    # server.log with a readiness line so extract_startup_metrics works.
    server_log = cell_dir / "server.log"
    server_log.write_text(
        "INFO 08-07 10:00:00 [__init__.py] Available KV cache memory: 29.0 GiB\n"
        "INFO 08-07 10:00:30 [api_server.py] Application startup complete.\n",
        encoding="utf-8",
    )

    # startup_ts.txt: line 1 = process start (UTC ISO), line 2 = readiness (UTC ISO).
    # extract_startup_metrics parses these with strptime("%Y-%m-%dT%H:%M:%SZ");
    # cold_readiness_s = (readiness - process_start) = 30s here.
    startup_ts = cell_dir / "startup_ts.txt"
    startup_ts.write_text(
        "2026-08-07T10:00:00Z\n2026-08-07T10:00:30Z\n", encoding="utf-8"
    )

    # client_result.json with actual burst measurement fields.
    client_result = cell_dir / "client_result.json"
    client_result.write_text(
        json.dumps(
            {
                "duration": duration,
                "completed": completed,
                "failed": failed,
                "output_throughput": 42.5,
                "mean_ttft_ms": 200.0,
                "p50_ttft_ms": 180.0,
                "p95_ttft_ms": 350.0,
                "p99_ttft_ms": 500.0,
                "mean_tpot_ms": 20.0,
                "p50_tpot_ms": 18.0,
                "p95_tpot_ms": 30.0,
                "p99_tpot_ms": 45.0,
                "mean_itl_ms": 15.0,
                "p50_itl_ms": 14.0,
                "p95_itl_ms": 25.0,
                "p99_itl_ms": 40.0,
                "prefix_cache_hit_rate": 0.6,
            }
        ),
        encoding="utf-8",
    )

    # metrics.txt (minimal Prometheus format).
    metrics = cell_dir / "metrics.txt"
    metrics.write_text(
        "# HELP vllm:requests_running Number of requests running.\n"
        'vllm:requests_running{model="test"} 0\n',
        encoding="utf-8",
    )

    # probe_result.json (only for burst profiles).
    if with_probe:
        probe = cell_dir / "probe_result.json"
        probe.write_text(
            json.dumps({"recovery_ttft_s": probe_recovery_s}), encoding="utf-8"
        )

    return cell_dir


def _env_for_build() -> dict[str, str]:
    """Set provenance env vars required by build_readiness_artifact.py."""
    env = os.environ.copy()
    env["VLLM_HUST_ENGINE_COMMIT"] = COMMIT_40
    env["VLLM_HUST_BENCHMARK_COMMIT"] = COMMIT_40_B
    env["VLLM_HUST_PLUGIN_COMMIT"] = COMMIT_40_C
    env["VLLM_HUST_ENGINE_VERSION"] = "v0.23.1-dev"
    env["VLLM_HUST_CANN_VERSION"] = "8.0.0"
    env["VLLM_HUST_DRIVER_VERSION"] = "23.0.0"
    env["VLLM_HUST_PYTHON_VERSION"] = "3.11.0"
    env["VLLM_HUST_OS_INFO"] = "Linux 5.10"
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def _run_builder(
    cell_dir: Path, *, profile: str, report_type: str, request_rate: float = 10.0
) -> dict:
    """Run build_readiness_artifact.py and return the parsed artifact."""
    env = _env_for_build()
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--cell-dir",
            str(cell_dir),
            "--workload",
            "random-online",
            "--load-profile",
            profile,
            "--rep-index",
            "1",
            "--rep-total",
            "3",
            "--cold-start",
            "--report-type",
            report_type,
            "--request-rate",
            str(request_rate),
            "--num-prompts",
            "50",
            "--input-len",
            "1024",
            "--output-len",
            "256",
            "--dataset",
            "random",
            "--served-model-name",
            "Qwen2.5-14B-Instruct",
            "--canonical-model-name",
            "Qwen/Qwen2.5-14B-Instruct",
            "--tensor-parallel-size",
            "1",
            "--gpu-memory-utilization",
            "0.6",
            "--max-model-len",
            "32768",
            "--port",
            "8011",
            "--cleared-paths",
            "/tmp/cache",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"builder failed (rc={result.returncode}):\n{result.stderr}"
    )
    artifact_path = cell_dir / "readiness_slo_artifact_rep1.json"
    return json.loads(artifact_path.read_text(encoding="utf-8"))


class TestBurstConfigFromRealMeasurement:
    """PR #154 review round 2 issue 1: burst_config must use real values."""

    def test_burst_config_has_real_size_and_duration(self, tmp_path: Path) -> None:
        cell_dir = _write_cell(tmp_path, completed=50, duration=42.5)
        artifact = _run_builder(cell_dir, profile="burst", report_type="burst")

        burst_config = artifact["load_profile"]["burst_config"]
        assert burst_config is not None
        # size must be completed + failed, not 0.
        assert burst_config["size"] == 50
        # duration_s must be > 0 (from client_result.json 'duration').
        assert burst_config["duration_s"] == 42.5
        assert burst_config["duration_s"] > 0
        # interval_s = 1 / request_rate.
        assert burst_config["interval_s"] == pytest.approx(0.1)
        # mean_arrival_rate = request_rate.
        assert burst_config["mean_arrival_rate"] == 10.0

    def test_burst_recovery_s_from_probe_result(self, tmp_path: Path) -> None:
        cell_dir = _write_cell(tmp_path, probe_recovery_s=1.5)
        artifact = _run_builder(cell_dir, profile="burst", report_type="burst")

        # burst_recovery_s must come from probe_result.json, not be 0.0.
        assert artifact["slo_metrics"]["burst_recovery_s"] == 1.5

    def test_burst_profile_rejects_zero_completed(self, tmp_path: Path) -> None:
        """If client_result.json has completed=0 and failed=0, builder must fail."""
        cell_dir = _write_cell(tmp_path, completed=0, failed=0, duration=42.5)
        env = _env_for_build()
        result = subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--cell-dir",
                str(cell_dir),
                "--workload",
                "random-online",
                "--load-profile",
                "burst",
                "--rep-index",
                "1",
                "--rep-total",
                "3",
                "--cold-start",
                "--report-type",
                "burst",
                "--request-rate",
                "10.0",
                "--num-prompts",
                "50",
                "--input-len",
                "1024",
                "--output-len",
                "256",
                "--dataset",
                "random",
                "--served-model-name",
                "Qwen2.5-14B-Instruct",
                "--canonical-model-name",
                "Qwen/Qwen2.5-14B-Instruct",
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                "0.6",
                "--max-model-len",
                "32768",
                "--port",
                "8011",
                "--cleared-paths",
                "/tmp/cache",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode != 0
        assert "size=0" in result.stderr or "at least 1" in result.stderr

    def test_burst_profile_rejects_zero_duration(self, tmp_path: Path) -> None:
        """If client_result.json has duration=0, builder must fail."""
        cell_dir = _write_cell(tmp_path, completed=50, duration=0.0)
        env = _env_for_build()
        result = subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--cell-dir",
                str(cell_dir),
                "--workload",
                "random-online",
                "--load-profile",
                "burst",
                "--rep-index",
                "1",
                "--rep-total",
                "3",
                "--cold-start",
                "--report-type",
                "burst",
                "--request-rate",
                "10.0",
                "--num-prompts",
                "50",
                "--input-len",
                "1024",
                "--output-len",
                "256",
                "--dataset",
                "random",
                "--served-model-name",
                "Qwen2.5-14B-Instruct",
                "--canonical-model-name",
                "Qwen/Qwen2.5-14B-Instruct",
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                "0.6",
                "--max-model-len",
                "32768",
                "--port",
                "8011",
                "--cleared-paths",
                "/tmp/cache",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode != 0
        assert "duration" in result.stderr

    def test_burst_profile_rejects_missing_probe_result(self, tmp_path: Path) -> None:
        """Without probe_result.json, burst profile must fail closed."""
        cell_dir = _write_cell(tmp_path, completed=50, duration=42.5, with_probe=False)
        env = _env_for_build()
        result = subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--cell-dir",
                str(cell_dir),
                "--workload",
                "random-online",
                "--load-profile",
                "burst",
                "--rep-index",
                "1",
                "--rep-total",
                "3",
                "--cold-start",
                "--report-type",
                "burst",
                "--request-rate",
                "10.0",
                "--num-prompts",
                "50",
                "--input-len",
                "1024",
                "--output-len",
                "256",
                "--dataset",
                "random",
                "--served-model-name",
                "Qwen2.5-14B-Instruct",
                "--canonical-model-name",
                "Qwen/Qwen2.5-14B-Instruct",
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                "0.6",
                "--max-model-len",
                "32768",
                "--port",
                "8011",
                "--cleared-paths",
                "/tmp/cache",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode != 0
        assert "probe_result.json" in result.stderr


class TestPluginCommitBinding:
    """PR #154 review round 2 issue 2: plugin commit must be independent."""

    def test_plugin_commit_is_independent_from_benchmark(self, tmp_path: Path) -> None:
        cell_dir = _write_cell(tmp_path)
        artifact = _run_builder(cell_dir, profile="burst", report_type="burst")

        plugin = artifact["metadata"]["runtime_provenance"]["plugin"]
        # plugin commit must be VLLM_HUST_PLUGIN_COMMIT (COMMIT_40_C),
        # NOT VLLM_HUST_BENCHMARK_COMMIT (COMMIT_40_B).
        assert plugin["commit"] == COMMIT_40_C
        assert plugin["commit"] != COMMIT_40_B

    def test_missing_plugin_commit_env_fails(self, tmp_path: Path) -> None:
        cell_dir = _write_cell(tmp_path)
        env = _env_for_build()
        del env["VLLM_HUST_PLUGIN_COMMIT"]
        result = subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--cell-dir",
                str(cell_dir),
                "--workload",
                "random-online",
                "--load-profile",
                "burst",
                "--rep-index",
                "1",
                "--rep-total",
                "3",
                "--cold-start",
                "--report-type",
                "burst",
                "--request-rate",
                "10.0",
                "--num-prompts",
                "50",
                "--input-len",
                "1024",
                "--output-len",
                "256",
                "--dataset",
                "random",
                "--served-model-name",
                "Qwen2.5-14B-Instruct",
                "--canonical-model-name",
                "Qwen/Qwen2.5-14B-Instruct",
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                "0.6",
                "--max-model-len",
                "32768",
                "--port",
                "8011",
                "--cleared-paths",
                "/tmp/cache",
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode != 0
        assert "VLLM_HUST_PLUGIN_COMMIT" in result.stderr


class TestSteadyProfileStillWorks:
    """Ensure steady profiles (no burst) still build correctly after fixes."""

    def test_steady_profile_builds_without_probe(self, tmp_path: Path) -> None:
        cell_dir = _write_cell(tmp_path, with_probe=False)
        artifact = _run_builder(
            cell_dir,
            profile="steady-1rps",
            report_type="fixed-qps",
            request_rate=1.0,
        )
        # Steady profiles have burst_config=None and burst_recovery_s=None.
        assert artifact["load_profile"]["burst_config"] is None
        assert artifact["slo_metrics"]["burst_recovery_s"] is None
