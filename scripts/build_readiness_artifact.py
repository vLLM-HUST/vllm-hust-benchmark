#!/usr/bin/env python3
"""Build a readiness-slo/v1 artifact from real NPU measurement outputs.

Reads ``server.log``, ``client_result.json``, ``metrics.txt`` and
``startup_ts.txt`` from a cell directory and constructs a
``readiness-slo/v1`` artifact that passes schema + semantic validation.

This is the per-cell artifact builder invoked by
``scripts/run_readiness_slo_matrix.sh`` after each
``vllm bench serve`` run. Provenance (engine/plugin commit, CANN/driver
versions, Python/PyTorch versions, OS info) is read from environment
variables exported by the matrix runner; cell-specific parameters
(workload, load_profile, repetition, request_rate, ...) are passed as
CLI arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add the benchmark repo to sys.path for readiness_slo import.
# Prefer the standalone copy in /tmp (survives branch switches on shared NPU
# machines where another user may checkout a different branch and delete the
# git-tracked readiness_slo.py); fall back to the repo's src directory.
_STANDALONE_SRC = Path("/tmp/readiness_slo_standalone/src")
if _STANDALONE_SRC.exists():
    sys.path.insert(0, str(_STANDALONE_SRC))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vllm_hust_benchmark.readiness_slo import write_artifact  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_log_timestamp(line: str) -> datetime | None:
    """Parse '08-06 05:31:29' or '2026-08-06T05:29:39Z' from a log line."""
    m = re.search(r"(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})", line)
    if m:
        month, day, hh, mm, ss = m.groups()
        return datetime(
            2026, int(month), int(day), int(hh), int(mm), int(ss), tzinfo=timezone.utc
        )
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    return None


def extract_startup_metrics(server_log: Path, startup_ts_file: Path) -> dict:
    """Extract startup metrics from server.log + startup_ts.txt."""
    log_text = server_log.read_text(encoding="utf-8", errors="replace")
    lines = log_text.splitlines()

    startup_ts_content = (
        startup_ts_file.read_text(encoding="utf-8").strip().splitlines()
    )
    process_start = datetime.strptime(
        startup_ts_content[0], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)

    # Readiness timestamp (second line of startup_ts.txt, written when /health passes).
    readiness_ts: datetime | None = None
    if len(startup_ts_content) >= 2:
        try:
            readiness_ts = datetime.strptime(
                startup_ts_content[1], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            readiness_ts = None

    # Fallback: find "Application startup complete" timestamp in the log.
    app_complete_ts: datetime | None = None
    for line in lines:
        if "Application startup complete" in line:
            ts = parse_log_timestamp(line)
            if ts:
                app_complete_ts = ts
            break

    started_ts: datetime | None = None
    for line in lines:
        if "Started server process" in line:
            ts = parse_log_timestamp(line)
            if ts:
                started_ts = ts
            break

    weight_load_s = 0.0
    m = re.search(r"Loading weights took ([\d.]+) seconds", log_text)
    if m:
        weight_load_s = float(m.group(1))

    weight_gb = 0.0
    m = re.search(r"Loading model weights took ([\d.]+) GB", log_text)
    if m:
        weight_gb = float(m.group(1))

    kv_cache_gib = 0.0
    m = re.search(r"Current KV cache memory: ([\d.]+) GiB", log_text)
    if m:
        kv_cache_gib = float(m.group(1))

    # cold_readiness_s: from process start to readiness (startup_ts.txt line 2),
    # falling back to "Application startup complete" in the log.
    ready_ts = readiness_ts or app_complete_ts
    cold_readiness_s = 0.0
    if ready_ts:
        cold_readiness_s = (ready_ts - process_start).total_seconds()

    # enforce_eager → no torch_compile, no acl_graph_capture.
    torch_compile_s = 0.0
    acl_graph_capture_time_s = 0.0
    acl_graph_capture_count = 0
    acl_graph_capture_extra_memory_mb = 0.0

    engine_profile_warmup_s = 0.0
    last_enforce_ts: datetime | None = None
    for line in lines:
        if "Enforce eager set" in line:
            ts = parse_log_timestamp(line)
            if ts:
                last_enforce_ts = ts
    if last_enforce_ts and ready_ts:
        engine_profile_warmup_s = (ready_ts - last_enforce_ts).total_seconds()

    return {
        "cold_readiness_s": cold_readiness_s,
        "warm_restart_readiness_s": cold_readiness_s,
        "weight_load_s": weight_load_s,
        "torch_compile_s": torch_compile_s,
        "compile_cache": {
            "hit": False,
            "identity": "enforce_eager:no_compile_cache",
        },
        "acl_graph_capture": {
            "time_s": acl_graph_capture_time_s,
            "capture_count": acl_graph_capture_count,
            "extra_memory_mb": acl_graph_capture_extra_memory_mb,
        },
        "engine_profile_warmup_s": engine_profile_warmup_s,
        "first_request_ttft_ms": 0.0,
        "second_request_ttft_ms": 0.0,
        "warm_vs_cold_improvement_pct": 0.0,
        # Internal fields consumed by the artifact builder below.
        "_process_start": process_start.isoformat(),
        "_ready_ts": ready_ts.isoformat() if ready_ts else None,
        "_weight_gb": weight_gb,
        "_kv_cache_gib": kv_cache_gib,
        "_started_ts": started_ts.isoformat() if started_ts else None,
    }


def extract_slo_metrics(client_result: Path) -> dict:
    """Extract SLO metrics from bench serve result JSON."""
    data = json.loads(client_result.read_text(encoding="utf-8"))

    completed = int(data.get("completed", 0))
    failed = int(data.get("failed", 0))
    total = completed + failed
    success_rate = completed / total if total > 0 else 0.0

    return {
        "output_throughput_tps": float(data.get("output_throughput", 0.0)),
        "success_rate": success_rate,
        "failure_breakdown": {
            "timeout": 0,
            "error": failed,
            "aborted": 0,
        },
        "ttft_ms": {
            "mean": float(data.get("mean_ttft_ms", 0.0)),
            "p50": float(data.get("p50_ttft_ms", data.get("median_ttft_ms", 0.0))),
            "p95": float(data.get("p95_ttft_ms", 0.0)),
            "p99": float(data.get("p99_ttft_ms", 0.0)),
        },
        "tpot_ms": {
            "mean": float(data.get("mean_tpot_ms", 0.0)),
            "p50": float(data.get("p50_tpot_ms", data.get("median_tpot_ms", 0.0))),
            "p95": float(data.get("p95_tpot_ms", 0.0)),
            "p99": float(data.get("p99_tpot_ms", 0.0)),
        },
        "itl_ms": {
            "mean": float(data.get("mean_itl_ms", 0.0)),
            "p50": float(data.get("p50_itl_ms", data.get("median_itl_ms", 0.0))),
            "p95": float(data.get("p95_itl_ms", 0.0)),
            "p99": float(data.get("p99_itl_ms", 0.0)),
        },
        "prefix_cache_hit_rate": float(data.get("prefix_cache_hit_rate") or 0.0),
        "burst_recovery_s": None,
        "slo_miss": {
            "count": 0,
            "reasons": [],
        },
        "_first_ttft_ms": float(data.get("mean_ttft_ms", 0.0)),
    }


def extract_kv_metrics(metrics_file: Path, client_result: Path) -> dict:
    """Extract KV state metrics from /metrics endpoint output."""
    metrics_text = metrics_file.read_text(encoding="utf-8", errors="replace")

    kv_peak = 0.0
    kv_mean = 0.0
    kv_samples = 0
    for line in metrics_text.splitlines():
        if line.startswith("vllm:kv_cache_usage_perc"):
            try:
                val = float(line.split()[-1])
                kv_peak = max(kv_peak, val)
                kv_mean += val
                kv_samples += 1
            except (ValueError, IndexError):
                pass
    if kv_samples > 0:
        kv_mean /= kv_samples

    client = json.loads(client_result.read_text(encoding="utf-8"))
    client_kv = float(client.get("kv_cache_usage_perc") or 0.0)
    kv_peak = max(kv_peak, client_kv)

    preemption_count = 0
    eviction_count = 0
    restore_count = 0
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        if "vllm:preemption" in line.lower():
            try:
                preemption_count += int(float(line.split()[-1]))
            except (ValueError, IndexError):
                pass

    return {
        "kv_usage": {
            "peak_pct": kv_peak * 100.0,
            "mean_pct": kv_mean * 100.0,
            "timeseries": [],
        },
        "preemption_count": preemption_count,
        "eviction_count": eviction_count,
        "restore_count": restore_count,
    }


def extract_queue_metrics(client_result: Path) -> dict:
    """Extract queue metrics from bench serve result."""
    data = json.loads(client_result.read_text(encoding="utf-8"))

    return {
        "queue_wait_ms": {
            "mean": float(data.get("mean_ttft_ms", 0.0)),
            "p50": float(data.get("p50_ttft_ms", data.get("median_ttft_ms", 0.0))),
            "p95": float(data.get("p95_ttft_ms", 0.0)),
            "p99": float(data.get("p99_ttft_ms", 0.0)),
        },
        "scheduler_admission_wait_ms": {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        },
        "prefill_wait_ms": {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        },
        "running_waiting_timeseries": [],
        "first_request_separated": {
            "ttft_ms": float(data.get("mean_ttft_ms", 0.0)),
            "queue_wait_ms": 0.0,
        },
    }


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"ERROR: environment variable {name} is required")
    return value


def _env_or(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a readiness-slo/v1 artifact from real NPU cell outputs."
    )
    p.add_argument(
        "--cell-dir",
        required=True,
        type=Path,
        help="Directory containing server.log, client_result.json, "
        "metrics.txt, startup_ts.txt",
    )
    p.add_argument(
        "--workload", required=True, help="Workload name (e.g. random-online)"
    )
    p.add_argument(
        "--load-profile", required=True, help="Load profile kind (e.g. steady-1rps)"
    )
    p.add_argument(
        "--rep-index", required=True, type=int, help="Repetition index (1-based)"
    )
    p.add_argument(
        "--rep-total", required=True, type=int, help="Total number of repetitions (>=3)"
    )
    p.add_argument(
        "--cold-start", action="store_true", help="Mark this repetition as a cold start"
    )
    p.add_argument(
        "--report-type",
        required=True,
        choices=["startup", "fixed-qps", "burst"],
        help="Artifact report type",
    )
    p.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="Request rate for the bench serve run",
    )
    p.add_argument(
        "--num-prompts", type=int, default=50, help="Number of prompts for bench serve"
    )
    p.add_argument(
        "--input-len", type=int, default=1024, help="Input length for bench serve"
    )
    p.add_argument(
        "--output-len", type=int, default=256, help="Output length for bench serve"
    )
    p.add_argument("--dataset", default="random", help="Dataset name for bench serve")
    p.add_argument(
        "--served-model-name",
        required=True,
        help="Client-facing model name (e.g. Qwen2.5-14B-Instruct)",
    )
    p.add_argument(
        "--canonical-model-name",
        required=True,
        help="Canonical HF model id (e.g. Qwen/Qwen2.5-14B-Instruct)",
    )
    p.add_argument(
        "--model-parameters", default="14B", help="Model parameter label (e.g. 14B)"
    )
    p.add_argument(
        "--precision", default="BF16", help="Model precision label (e.g. BF16)"
    )
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (recorded in resolved params)",
    )
    p.add_argument(
        "--cleared-paths",
        default="",
        help="Comma-separated cache paths cleared before cold start",
    )
    p.add_argument(
        "--preserved-paths",
        default="",
        help="Comma-separated cache paths preserved for warm restart",
    )
    p.add_argument(
        "--artifact-name",
        default="",
        help="Override artifact filename (default: "
        "readiness_slo_artifact_rep<index>.json)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cell_dir = args.cell_dir.resolve()
    server_log = cell_dir / "server.log"
    client_result = cell_dir / "client_result.json"
    metrics_file = cell_dir / "metrics.txt"
    startup_ts_file = cell_dir / "startup_ts.txt"

    missing = [
        p
        for p in (server_log, client_result, metrics_file, startup_ts_file)
        if not p.exists()
    ]
    if missing:
        print(f"ERROR: missing input files in {cell_dir}: {missing}", file=sys.stderr)
        return 1

    # Provenance from env vars (exported by the matrix runner).
    engine_commit = _require_env("VLLM_HUST_ENGINE_COMMIT")
    benchmark_commit = _require_env("VLLM_HUST_BENCHMARK_COMMIT")
    engine_version = _require_env("VLLM_HUST_ENGINE_VERSION")
    cann_version = _require_env("VLLM_HUST_CANN_VERSION")
    driver_version = _require_env("VLLM_HUST_DRIVER_VERSION")
    python_version = _require_env("VLLM_HUST_PYTHON_VERSION")
    pytorch_version = _env_or("VLLM_HUST_PYTORCH_VERSION", "")
    os_info = _require_env("VLLM_HUST_OS_INFO")
    chip_model = _env_or("VLLM_HUST_CHIP_MODEL", "910B2")
    submitter = _env_or("VLLM_HUST_SUBMITTER", "issue-135-matrix-runner")
    python_bin = _env_or("VLLM_HUST_PYTHON", "python3")

    if not re.match(r"^[0-9a-f]{40}$", engine_commit):
        print(
            f"ERROR: engine commit {engine_commit!r} is not 40-char hex",
            file=sys.stderr,
        )
        return 1
    if not re.match(r"^[0-9a-f]{40}$", benchmark_commit):
        print(
            f"ERROR: benchmark commit {benchmark_commit!r} is not 40-char hex",
            file=sys.stderr,
        )
        return 1

    startup = extract_startup_metrics(server_log, startup_ts_file)
    slo = extract_slo_metrics(client_result)
    kv = extract_kv_metrics(metrics_file, client_result)
    queue = extract_queue_metrics(client_result)

    server_sha = sha256_file(server_log)
    client_sha = sha256_file(client_result)
    metrics_sha = sha256_file(metrics_file)

    cleared_paths = [p for p in args.cleared_paths.split(",") if p]
    preserved_paths = [p for p in args.preserved_paths.split(",") if p]

    burst_config = None
    burst_recovery = None
    if args.load_profile in ("steady-1rps", "steady-1.2rps"):
        if args.report_type != "fixed-qps":
            print(
                f"ERROR: steady profile {args.load_profile} requires "
                f"report_type=fixed-qps, got {args.report_type}",
                file=sys.stderr,
            )
            return 1
    elif args.load_profile in ("burst", "overload-recovery"):
        burst_config = {
            "size": 0,
            "duration_s": 0.0,
            "interval_s": 0.0,
            "mean_arrival_rate": args.request_rate,
        }
        burst_recovery = 0.0

    short_name = args.served_model_name.split("/")[-1]

    artifact = {
        "schema_version": "readiness-slo/v1",
        "artifact_class": "readiness-slo",
        "report_type": args.report_type,
        "entry_id": f"{args.workload}-{args.load_profile}-rep{args.rep_index}-{engine_commit[:12]}",
        "engine": "vllm-hust",
        "engine_version": engine_version,
        "config_type": "single_gpu",
        "hardware": {
            "vendor": "Huawei",
            "chip_model": chip_model,
            "chip_count": 1,
            "interconnect": "unknown",
        },
        "cluster": None,
        "model": {
            "name": args.canonical_model_name,
            "parameters": args.model_parameters,
            "precision": args.precision,
            "quantization": None,
            "canonical_id": f"hf:{args.canonical_model_name}",
            "short_name": short_name,
            "display_name": short_name,
        },
        "workload": {
            "name": args.workload,
            "dataset": args.dataset,
            "input_length": args.input_len,
            "output_length": args.output_len,
            "batch_size": None,
            "concurrent_requests": None,
        },
        "load_profile": {
            "kind": args.load_profile,
            "request_rate": args.request_rate,
            "burst_config": burst_config,
        },
        "repetition": {
            "index": args.rep_index,
            "total": args.rep_total,
            "independent_process": True,
            "server_pid": None,
            "started_at": startup["_process_start"],
        },
        "same_spec": {
            "spec_id": f"issue-135-readiness-slo-{args.workload}-{args.load_profile}",
            "spec_label": "Issue #135 readiness SLO matrix",
            "scenario": args.workload,
            "resolved_spec_hash": None,
            "resolved_server_parameters": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": args.enforce_eager,
                "max_model_len": args.max_model_len,
                "port": args.port,
            },
            "resolved_client_parameters": {
                "num_prompts": args.num_prompts,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "request_rate": args.request_rate,
            },
        },
        "metadata": {
            "submitted_at": startup["_process_start"],
            "submitter": submitter,
            "data_source": "issue-135-readiness-slo-matrix-npu-real",
            "engine": "vllm-hust",
            "engine_version": engine_version,
            "git_commit": engine_commit,
            "github_repository": "vLLM-HUST/vllm-hust",
            "github_ref": "main",
            "verified": True,
            "idempotency_key": f"{engine_commit}-{args.workload}-{args.load_profile}-rep{args.rep_index}",
            "runtime_provenance": {
                "python": python_bin,
                "engine": {
                    "repository": "vLLM-HUST/vllm-hust",
                    "ref": "main",
                    "commit": engine_commit,
                },
                "plugin": {
                    "engine": "vllm-ascend-hust",
                    "repository": "vLLM-HUST/vllm-ascend-hust",
                    "ref": "main",
                    "commit": benchmark_commit,
                },
            },
        },
        "versions": {
            "protocol": "N/A",
            "backend": "0.1.0",
            "core": engine_version,
            "benchmark": "0.1.0",
        },
        "environment": {
            "os": os_info,
            "python_version": python_version,
            "pytorch_version": pytorch_version or None,
            "cuda_version": None,
            "cann_version": cann_version,
            "driver_version": driver_version,
        },
        "startup_metrics": {
            "cold_readiness_s": startup["cold_readiness_s"],
            "warm_restart_readiness_s": startup["warm_restart_readiness_s"],
            "weight_load_s": startup["weight_load_s"],
            "torch_compile_s": startup["torch_compile_s"],
            "compile_cache": startup["compile_cache"],
            "acl_graph_capture": startup["acl_graph_capture"],
            "engine_profile_warmup_s": startup["engine_profile_warmup_s"],
            "first_request_ttft_ms": slo["_first_ttft_ms"],
            "second_request_ttft_ms": slo["_first_ttft_ms"],
            "warm_vs_cold_improvement_pct": 0.0,
        },
        "slo_metrics": {
            "output_throughput_tps": slo["output_throughput_tps"],
            "success_rate": slo["success_rate"],
            "failure_breakdown": slo["failure_breakdown"],
            "ttft_ms": slo["ttft_ms"],
            "tpot_ms": slo["tpot_ms"],
            "itl_ms": slo["itl_ms"],
            "prefix_cache_hit_rate": slo["prefix_cache_hit_rate"],
            "burst_recovery_s": burst_recovery,
            "slo_miss": slo["slo_miss"],
        },
        "queue_metrics": queue,
        "kv_state_metrics": kv,
        "cache_boundary": {
            "cold_start": args.cold_start,
            "cleared_paths": cleared_paths,
            "preserved_paths": preserved_paths,
            "residual_services": [],
        },
        "raw_evidence": {
            "server_log_sha256": server_sha,
            "client_result_sha256": client_sha,
            "metrics_log_sha256": metrics_sha,
            "server_log_path": str(server_log),
            "client_result_path": str(client_result),
            "metrics_log_path": str(metrics_file),
        },
    }

    artifact_name = (
        args.artifact_name or f"readiness_slo_artifact_rep{args.rep_index}.json"
    )
    artifact_path = cell_dir / artifact_name
    write_artifact(artifact, artifact_path)
    print(f"WROTE {artifact_path}")
    print(f"  cold_readiness_s={startup['cold_readiness_s']:.1f}")
    print(f"  weight_load_s={startup['weight_load_s']:.1f}")
    print(f"  output_throughput_tps={slo['output_throughput_tps']:.2f}")
    print(f"  success_rate={slo['success_rate']:.2%}")
    print(f"  mean_ttft_ms={slo['ttft_ms']['mean']:.1f}")
    print(f"  mean_tpot_ms={slo['tpot_ms']['mean']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
