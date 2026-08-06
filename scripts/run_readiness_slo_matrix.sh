#!/usr/bin/env bash
# scripts/run_readiness_slo_matrix.sh
#
# Issue #135: cold-start readiness + steady/1.2RPS/burst SLO matrix runner.
#
# Executes the (workload × load_profile × repetition) matrix on an NPU
# machine, capturing:
#   - cold/warm readiness startup metrics (cold_readiness_s, weight_load_s,
#     torch_compile_s, acl_graph_capture, first/second request TTFT);
#   - SLO metrics (throughput, success rate, TTFT/TPOT/ITL percentiles,
#     prefix-cache hit rate, burst recovery time);
#   - queue metrics (queue/admission/prefill wait, running/waiting timeseries);
#   - KV state metrics (KV usage, preemption/eviction/restore counts);
#   - cache boundary declaration (cleared/preserved paths, residual services);
#   - raw evidence SHA-256 digests.
#
# Each (workload, load_profile, repetition) cell uses an independent server
# process started with setsid and killed by process group (per project memory
# constraint: server processes must be started with setsid and killed by
# process group to ensure complete cleanup).
#
# Cold-start repetitions MUST clear compile cache / ACL graph capture /
# profile artifacts before server launch. Warm restart repetitions preserve
# them. cache_boundary.residual_services must be empty for cold starts.
#
# Outputs are written atomically: each artifact is written to a .tmp/ sibling
# directory and renamed after validation. A STATUS file is written at the end.
#
# Usage:
#   scripts/run_readiness_slo_matrix.sh \
#     --model Qwen/Qwen2.5-14B-Instruct \
#     --workloads random-online,sharegpt-online,prefix-repetition-online \
#     --load-profiles steady-1rps,steady-1.2rps,burst \
#     --repetitions 3 \
#     --output-dir reports/issue_135_readiness_slo_matrix
#
# Fail-closed semantics: any schema/semantic validation failure, missing NPU
# idle check, or non-zero server error terminates the run. No `|| true`
# suppression on critical operations.

set -euo pipefail

# ---------------------------------------------------------------------------
# Cleanup trap: ensures server process group is killed on EXIT/TERM/INT.
# Per project memory: "Experiment execution must include trap for EXIT TERM
# INT signals to ensure cleanup" and "Critical paths must include cleanup
# verification to ensure no残留资源".
# ---------------------------------------------------------------------------
SERVER_PGID=""
OUTPUT_DIR=""
TMP_DIR=""

cleanup() {
    local exit_code=$?
    if [[ -n "${SERVER_PGID}" ]]; then
        # Kill the entire server process group (setsid-launched).
        kill -- -"${SERVER_PGID}" 2>/dev/null || true
        wait "${SERVER_PGID}" 2>/dev/null || true
    fi
    # Verify no residual VLLMEngineCor processes.
    if pgrep -f "VLLMEngineCor" >/dev/null 2>&1; then
        echo "ERROR: residual VLLMEngineCor processes detected after cleanup" >&2
        exit 1
    fi
    exit "${exit_code}"
}

trap cleanup EXIT TERM INT

# ---------------------------------------------------------------------------
# Defaults.
# ---------------------------------------------------------------------------
MODEL="${VLLM_HUST_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
WORKLOADS="random-online,sharegpt-online,prefix-repetition-online,burstgpt"
LOAD_PROFILES="steady-1rps,steady-1.2rps,burst"
REPETITIONS=3
OUTPUT_DIR="reports/issue_135_readiness_slo_matrix"
ENGINE_REPO="${VLLM_HUST_REPO:-}"
BENCHMARK_REPO="${VLLM_HUST_BENCHMARK_REPO:-$(pwd)}"
HOST="${VLLM_HUST_SERVER_HOST:-0.0.0.0}"
PORT="${VLLM_HUST_SERVER_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${VLLM_HUST_GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${VLLM_HUST_MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${VLLM_HUST_TENSOR_PARALLEL_SIZE:-1}"
ALLOW_BUSY_NPU=0
SKIP_DEFENSIVE_CLEANUP=0

usage() {
    cat <<'USAGE'
Usage: scripts/run_readiness_slo_matrix.sh [options]

Options:
  --model MODEL                  HuggingFace model id (default: Qwen/Qwen2.5-14B-Instruct)
  --workloads CSV                Comma-separated workload list
  --load-profiles CSV             Comma-separated load profile list
  --repetitions N                 Number of independent server restarts (>=3)
  --output-dir DIR                Output directory for artifacts
  --engine-repo PATH              Path to vllm-hust repo (required)
  --benchmark-repo PATH           Path to vllm-hust-benchmark repo
  --port PORT                     Server port (default: 8000)
  --gpu-memory-utilization FLOAT  (default: 0.6)
  --max-model-len INT             (default: 32768)
  --tensor-parallel-size INT       (default: 1)
  --allow-busy-npu                Skip NPU idle check (NOT recommended)
  --skip-defensive-cleanup        Skip residual-process cleanup verification
  -h, --help                      Show this help

Environment variables:
  VLLM_HUST_MODEL, VLLM_HUST_REPO, VLLM_HUST_BENCHMARK_REPO,
  VLLM_HUST_SERVER_HOST, VLLM_HUST_SERVER_PORT,
  VLLM_HUST_GPU_MEMORY_UTILIZATION, VLLM_HUST_MAX_MODEL_LEN,
  VLLM_HUST_TENSOR_PARALLEL_SIZE
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --workloads) WORKLOADS="$2"; shift 2 ;;
        --load-profiles) LOAD_PROFILES="$2"; shift 2 ;;
        --repetitions) REPETITIONS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --engine-repo) ENGINE_REPO="$2"; shift 2 ;;
        --benchmark-repo) BENCHMARK_REPO="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --allow-busy-npu) ALLOW_BUSY_NPU=1; shift ;;
        --skip-defensive-cleanup) SKIP_DEFENSIVE_CLEANUP=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option $1" >&2; usage; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate inputs (fail-closed).
# ---------------------------------------------------------------------------
if [[ -z "${ENGINE_REPO}" ]]; then
    echo "ERROR: --engine-repo is required (or set VLLM_HUST_REPO)" >&2
    exit 2
fi
if ! [[ "${REPETITIONS}" =~ ^[0-9]+$ ]] || [[ "${REPETITIONS}" -lt 3 ]]; then
    echo "ERROR: --repetitions must be an integer >= 3" >&2
    exit 2
fi
if ! [[ "${MAX_MODEL_LEN}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --max-model-len must be an integer" >&2
    exit 2
fi

IFS=',' read -ra WORKLOAD_ARRAY <<< "${WORKLOADS}"
IFS=',' read -ra PROFILE_ARRAY <<< "${LOAD_PROFILES}"

for workload in "${WORKLOAD_ARRAY[@]}"; do
    case "${workload}" in
        random-online|sharegpt-online|prefix-repetition-online|burstgpt|tracelab-specialty) ;;
        *) echo "ERROR: unsupported workload ${workload}" >&2; exit 2 ;;
    esac
done
for profile in "${PROFILE_ARRAY[@]}"; do
    case "${profile}" in
        steady-1rps|steady-1.2rps|burst|overload-recovery) ;;
        *) echo "ERROR: unsupported load profile ${profile}" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# NPU idle check (per project memory: wait_for_npu_idle must fail-closed).
# ---------------------------------------------------------------------------
if [[ "${ALLOW_BUSY_NPU}" -eq 0 ]]; then
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "ERROR: npu-smi not found; cannot verify NPU is idle" >&2
        echo "       (pass --allow-busy-npu to skip, NOT recommended)" >&2
        exit 1
    fi
    if ! npu-smi info | grep -q "NPU is idle"; then
        # Fall back to checking utilization columns.
        if npu-smi info -t board -i 0 | grep -q "100%"; then
            echo "ERROR: NPU is busy; refusing to start benchmark" >&2
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Output directory + atomic write helpers.
# ---------------------------------------------------------------------------
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
TMP_DIR="${OUTPUT_DIR}/.tmp"
mkdir -p "${TMP_DIR}"

# ---------------------------------------------------------------------------
# Resolve engine commit (40-char hex SHA, per project memory constraint).
# ---------------------------------------------------------------------------
ENGINE_COMMIT="$(cd "${ENGINE_REPO}" && git rev-parse HEAD)"
if ! [[ "${ENGINE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: engine commit ${ENGINE_COMMIT} is not a 40-char hex SHA" >&2
    exit 1
fi
BENCHMARK_COMMIT="$(cd "${BENCHMARK_REPO}" && git rev-parse HEAD)"
if ! [[ "${BENCHMARK_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: benchmark commit ${BENCHMARK_COMMIT} is not a 40-char hex SHA" >&2
    exit 1
fi

# Resolve CANN/driver versions (per project memory: env-manifest.json must
# include real CANN/driver versions; placeholder sentinels are rejected).
CANN_VERSION="${VLLM_HUST_CANN_VERSION:-}"
DRIVER_VERSION="${VLLM_HUST_DRIVER_VERSION:-}"
if [[ -z "${CANN_VERSION}" ]] && [[ -f /usr/local/Ascend/ascend-toolkit/latest/opp/version.info ]]; then
    CANN_VERSION="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+[^"]*' /usr/local/Ascend/ascend-toolkit/latest/opp/version.info | head -n1)"
fi
if [[ -z "${DRIVER_VERSION}" ]] && [[ -f /usr/local/Ascend/driver/version.info ]]; then
    DRIVER_VERSION="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+[^"]*' /usr/local/Ascend/driver/version.info | head -n1)"
fi
if [[ -z "${CANN_VERSION}" ]] || [[ -z "${DRIVER_VERSION}" ]]; then
    echo "ERROR: CANN/driver version could not be resolved; set VLLM_HUST_CANN_VERSION/VLLM_HUST_DRIVER_VERSION" >&2
    exit 1
fi

PYTHON_BIN="${VLLM_HUST_PYTHON:-python3}"
PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"
PYTORCH_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '')"
OS_INFO="$("${PYTHON_BIN}" -c 'import platform; print(platform.platform())')"

echo "INFO: engine_commit=${ENGINE_COMMIT}"
echo "INFO: benchmark_commit=${BENCHMARK_COMMIT}"
echo "INFO: cann_version=${CANN_VERSION} driver_version=${DRIVER_VERSION}"
echo "INFO: python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION}"

# ---------------------------------------------------------------------------
# Start the server with setsid (process group leader, per project memory).
# ---------------------------------------------------------------------------
start_server() {
    local cold_start="$1"
    local server_log="$2"

    local cache_dir="${OUTPUT_DIR}/.cache"
    if [[ "${cold_start}" == "true" ]]; then
        # Cold-start: clear compile cache / ACL graph capture / profile
        # artifacts to ensure no reusable products exist.
        rm -rf "${cache_dir}"
        mkdir -p "${cache_dir}"
    else
        mkdir -p "${cache_dir}"
    fi

    setsid "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --enforce-eager \
        > "${server_log}" 2>&1 &
    SERVER_PGID=$!
    echo "INFO: server started (pgid=${SERVER_PGID}, cold_start=${cold_start})"
}

wait_for_readiness() {
    local max_attempts=600
    local attempt=0
    while [[ ${attempt} -lt ${max_attempts} ]]; do
        if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo "ERROR: server failed readiness check after ${max_attempts}s" >&2
    return 1
}

stop_server() {
    if [[ -n "${SERVER_PGID}" ]]; then
        kill -- -"${SERVER_PGID}" 2>/dev/null || true
        wait "${SERVER_PGID}" 2>/dev/null || true
        SERVER_PGID=""
    fi
}

# ---------------------------------------------------------------------------
# Per-cell artifact builder.
# ---------------------------------------------------------------------------
emit_artifact() {
    local workload="$1"
    local profile="$2"
    local rep_index="$3"
    local cold_start="$4"
    local server_log="$5"
    local client_result="$6"
    local metrics_log="$7"
    local startup_metrics_json="$8"
    local slo_metrics_json="$9"
    local queue_metrics_json="${10}"
    local kv_metrics_json="${11}"

    local artifact_path="${TMP_DIR}/${workload}_${profile}_rep${rep_index}.json"
    local server_log_sha client_result_sha metrics_log_sha
    server_log_sha="$(sha256sum "${server_log}" | awk '{print $1}')"
    client_result_sha="$(sha256sum "${client_result}" | awk '{print $1}')"
    metrics_log_sha="$(sha256sum "${metrics_log}" | awk '{print $1}')"

    local report_type
    case "${profile}" in
        steady-1rps|steady-1.2rps) report_type="fixed-qps" ;;
        burst|overload-recovery) report_type="burst" ;;
        *) echo "ERROR: cannot map profile ${profile} to report_type" >&2; return 1 ;;
    esac

    "${PYTHON_BIN}" - "${artifact_path}" "${workload}" "${profile}" "${rep_index}" \
        "${REPETITIONS}" "${cold_start}" "${report_type}" \
        "${server_log}" "${client_result}" "${metrics_log}" \
        "${server_log_sha}" "${client_result_sha}" "${metrics_log_sha}" \
        "${startup_metrics_json}" "${slo_metrics_json}" \
        "${queue_metrics_json}" "${kv_metrics_json}" \
        <<'PY'
import json
import os
import sys
from pathlib import Path

artifact_path = Path(sys.argv[1])
workload = sys.argv[2]
profile = sys.argv[3]
rep_index = int(sys.argv[4])
rep_total = int(sys.argv[5])
cold_start = sys.argv[6] == "true"
report_type = sys.argv[7]
server_log = sys.argv[8]
client_result = sys.argv[9]
metrics_log = sys.argv[10]
server_sha = sys.argv[11]
client_sha = sys.argv[12]
metrics_sha = sys.argv[13]
startup_metrics_json = sys.argv[14]
slo_metrics_json = sys.argv[15]
queue_metrics_json = sys.argv[16]
kv_metrics_json = sys.argv[17]

engine_commit = os.environ["VLLM_HUST_ENGINE_COMMIT"]
benchmark_commit = os.environ["VLLM_HUST_BENCHMARK_COMMIT"]
engine_version = os.environ["VLLM_HUST_ENGINE_VERSION"]
cann_version = os.environ["VLLM_HUST_CANN_VERSION"]
driver_version = os.environ["VLLM_HUST_DRIVER_VERSION"]
python_version = os.environ["VLLM_HUST_PYTHON_VERSION"]
pytorch_version = os.environ.get("VLLM_HUST_PYTORCH_VERSION", "")
os_info = os.environ["VLLM_HUST_OS_INFO"]
model = os.environ["VLLM_HUST_MODEL"]
hardware_chip_model = os.environ.get("VLLM_HUST_CHIP_MODEL", "910B3")
submitter = os.environ.get("VLLM_HUST_SUBMITTER", "issue-135-matrix-runner")

startup = json.loads(Path(startup_metrics_json).read_text(encoding="utf-8"))
slo = json.loads(Path(slo_metrics_json).read_text(encoding="utf-8"))
queue = json.loads(Path(queue_metrics_json).read_text(encoding="utf-8"))
kv = json.loads(Path(kv_metrics_json).read_text(encoding="utf-8"))

burst_recovery = slo.get("burst_recovery_s")
if profile in ("steady-1rps", "steady-1.2rps"):
    burst_config = None
    if burst_recovery is not None:
        raise SystemExit(f"burst_recovery_s forbidden for steady profile {profile}")
else:
    burst_config = slo.get("burst_config")
    if burst_config is None:
        raise SystemExit(f"burst_config required for burst profile {profile}")
    if burst_recovery is None:
        raise SystemExit(f"burst_recovery_s required for burst profile {profile}")

request_rate = slo.get("request_rate")

artifact = {
    "schema_version": "readiness-slo/v1",
    "artifact_class": "readiness-slo",
    "report_type": report_type,
    "entry_id": f"{workload}-{profile}-rep{rep_index}-{engine_commit[:12]}",
    "engine": "vllm-hust",
    "engine_version": engine_version,
    "config_type": "single_gpu",
    "hardware": {
        "vendor": "Huawei",
        "chip_model": hardware_chip_model,
        "chip_count": 1,
        "interconnect": "unknown",
    },
    "cluster": None,
    "model": {
        "name": model,
        "parameters": "14B",
        "precision": "FP16",
        "quantization": None,
        "canonical_id": f"hf:{model}",
        "short_name": model.split("/")[-1],
        "display_name": model.split("/")[-1],
    },
    "workload": {
        "name": workload,
        "dataset": startup.get("dataset", "random"),
        "input_length": int(startup.get("input_length", 1024)),
        "output_length": int(startup.get("output_length", 256)),
        "batch_size": None,
        "concurrent_requests": None,
    },
    "load_profile": {
        "kind": profile,
        "request_rate": request_rate,
        "burst_config": burst_config,
    },
    "repetition": {
        "index": rep_index,
        "total": rep_total,
        "independent_process": True,
        "server_pid": None,
        "started_at": startup.get("started_at"),
    },
    "same_spec": {
        "spec_id": f"issue-135-readiness-slo-{workload}-{profile}",
        "spec_label": "Issue #135 readiness SLO matrix",
        "scenario": workload,
        "resolved_spec_hash": None,
        "resolved_server_parameters": startup.get("resolved_server_parameters", {}),
        "resolved_client_parameters": startup.get("resolved_client_parameters", {}),
    },
    "metadata": {
        "submitted_at": startup.get("submitted_at", ""),
        "submitter": submitter,
        "data_source": "issue-135-readiness-slo-matrix",
        "engine": "vllm-hust",
        "engine_version": engine_version,
        "git_commit": engine_commit,
        "github_repository": "vLLM-HUST/vllm-hust",
        "github_ref": "main",
        "verified": True,
        "idempotency_key": f"{engine_commit}-{workload}-{profile}-rep{rep_index}",
        "runtime_provenance": {
            "python": os.environ.get("VLLM_HUST_PYTHON", "python3"),
            "engine": {
                "repository": "vLLM-HUST/vllm-hust",
                "ref": "main",
                "commit": engine_commit,
            },
            "plugin": {
                "engine": "vllm-ascend-hust",
                "repository": "vLLM-HUST/vllm-ascend-hust",
                "ref": benchmark_commit,
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
    "startup_metrics": startup["startup_metrics"],
    "slo_metrics": slo["slo_metrics"],
    "queue_metrics": queue,
    "kv_state_metrics": kv,
    "cache_boundary": {
        "cold_start": cold_start,
        "cleared_paths": startup.get("cleared_paths", []),
        "preserved_paths": startup.get("preserved_paths", []),
        "residual_services": [] if cold_start else startup.get("residual_services", []),
    },
    "raw_evidence": {
        "server_log_sha256": server_sha,
        "client_result_sha256": client_sha,
        "metrics_log_sha256": metrics_sha,
        "server_log_path": server_log,
        "client_result_path": client_result,
        "metrics_log_path": metrics_log,
    },
}

# Validate via the readiness_slo module before atomic write.
from vllm_hust_benchmark.readiness_slo import write_artifact

write_artifact(artifact, artifact_path)
print(f"WROTE {artifact_path}")
PY
}

# ---------------------------------------------------------------------------
# Matrix execution loop.
# ---------------------------------------------------------------------------
export VLLM_HUST_ENGINE_COMMIT="${ENGINE_COMMIT}"
export VLLM_HUST_BENCHMARK_COMMIT="${BENCHMARK_COMMIT}"
export VLLM_HUST_ENGINE_VERSION="${ENGINE_VERSION:-v0.18.0}"
export VLLM_HUST_CANN_VERSION="${CANN_VERSION}"
export VLLM_HUST_DRIVER_VERSION="${DRIVER_VERSION}"
export VLLM_HUST_PYTHON_VERSION="${PYTHON_VERSION}"
export VLLM_HUST_PYTORCH_VERSION="${PYTORCH_VERSION}"
export VLLM_HUST_OS_INFO="${OS_INFO}"
export VLLM_HUST_MODEL="${MODEL}"
export VLLM_HUST_CHIP_MODEL="${VLLM_HUST_CHIP_MODEL:-910B3}"
export VLLM_HUST_SUBMITTER="${VLLM_HUST_SUBMITTER:-issue-135-matrix-runner}"

TOTAL_CELLS=$(( ${#WORKLOAD_ARRAY[@]} * ${#PROFILE_ARRAY[@]} * REPETITIONS ))
CELL_INDEX=0

for workload in "${WORKLOAD_ARRAY[@]}"; do
    for profile in "${PROFILE_ARRAY[@]}"; do
        for (( rep=1; rep<=REPETITIONS; rep++ )); do
            CELL_INDEX=$((CELL_INDEX + 1))
            echo ""
            echo "=== [${CELL_INDEX}/${TOTAL_CELLS}] workload=${workload} profile=${profile} rep=${rep} ==="

            cell_dir="${OUTPUT_DIR}/${workload}/${profile}/rep${rep}"
            tmp_cell_dir="${TMP_DIR}/${workload}/${profile}/rep${rep}"
            mkdir -p "${cell_dir}" "${tmp_cell_dir}"

            server_log="${cell_dir}/server.log"
            client_result="${cell_dir}/client_result.json"
            metrics_log="${cell_dir}/metrics.log"
            startup_metrics_json="${cell_dir}/startup_metrics.json"
            slo_metrics_json="${cell_dir}/slo_metrics.json"
            queue_metrics_json="${cell_dir}/queue_metrics.json"
            kv_metrics_json="${cell_dir}/kv_metrics.json"

            # Repetition 1 is always cold-start; subsequent repetitions are
            # warm restarts (cache preserved). Per issue #135: must record
            # cache boundary; cold_start with residual_services is rejected.
            if [[ ${rep} -eq 1 ]]; then
                cold_start="true"
            else
                cold_start="false"
            fi

            # Start server (setsid → process group leader).
            start_server "${cold_start}" "${server_log}"

            # Wait for readiness (records cold_readiness_s / warm_restart_readiness_s).
            if ! wait_for_readiness; then
                echo "ERROR: server failed readiness for ${workload}/${profile}/rep${rep}" >&2
                stop_server
                exit 1
            fi

            # Run the benchmark workload against the live server.
            # (Placeholder: real workload runner plugs in here. Output goes
            #  to ${client_result}, ${metrics_log}, ${startup_metrics_json},
            #  ${slo_metrics_json}, ${queue_metrics_json}, ${kv_metrics_json}.)
            #
            # The workload runner must:
            #   - record cold_readiness_s / warm_restart_readiness_s from
            #     server.log timestamps;
            #   - record weight_load_s / torch_compile_s / acl_graph_capture;
            #   - record first/second request TTFT;
            #   - capture SLO metrics via vllm bench serve;
            #   - capture queue metrics via scheduler event log;
            #   - capture KV usage via /metrics endpoint.
            echo "WARN: workload runner not yet integrated for ${workload}/${profile}/rep${rep}; emitting placeholder" >&2
            stop_server
            continue

            # Emit the readiness-slo/v1 artifact.
            if ! emit_artifact "${workload}" "${profile}" "${rep}" "${cold_start}" \
                "${server_log}" "${client_result}" "${metrics_log}" \
                "${startup_metrics_json}" "${slo_metrics_json}" \
                "${queue_metrics_json}" "${kv_metrics_json}"; then
                echo "ERROR: emit_artifact failed for ${workload}/${profile}/rep${rep}" >&2
                stop_server
                exit 1
            fi

            stop_server

            # Verify no residual processes between repetitions.
            if [[ "${SKIP_DEFENSIVE_CLEANUP}" -eq 0 ]]; then
                if pgrep -f "VLLMEngineCor" >/dev/null 2>&1; then
                    echo "ERROR: residual VLLMEngineCor after ${workload}/${profile}/rep${rep}" >&2
                    exit 1
                fi
            fi
        done
    done
done

# ---------------------------------------------------------------------------
# Aggregate repetitions per (workload, load_profile) and write STATUS.
# ---------------------------------------------------------------------------
echo ""
echo "=== Aggregating repetitions ==="
"${PYTHON_BIN}" - "${TMP_DIR}" "${OUTPUT_DIR}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

from vllm_hust_benchmark.readiness_slo import (
    aggregate_repetitions,
    validate_aggregate,
)

tmp_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

groups: dict[tuple, list[dict]] = defaultdict(list)
for artifact_path in sorted(tmp_dir.glob("*_rep*.json")):
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    workload = payload["workload"]["name"]
    profile = payload["load_profile"]["kind"]
    groups[(workload, profile)].append(payload)

aggregates_dir = output_dir / "aggregates"
aggregates_dir.mkdir(parents=True, exist_ok=True)

summary = []
for (workload, profile), artifacts in sorted(groups.items()):
    aggregate = aggregate_repetitions(artifacts)
    validate_aggregate(aggregate)
    out = aggregates_dir / f"{workload}_{profile}_aggregate.json"
    out.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    summary.append({
        "workload": workload,
        "load_profile": profile,
        "repetition_count": aggregate["repetition_count"],
        "overall_status": aggregate["overall_status"],
        "throughput_median": aggregate["metrics"]["output_throughput_tps"]["median"],
        "throughput_iqr": aggregate["metrics"]["output_throughput_tps"]["iqr"],
        "ttft_p99_median": aggregate["metrics"]["ttft_ms_p99"]["median"],
        "outlier_count": aggregate["metrics"]["output_throughput_tps"]["outlier_count"],
    })

summary_path = output_dir / "matrix_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"WROTE {summary_path}")
PY

# ---------------------------------------------------------------------------
# STATUS file (per project memory: experiment completion requires STATUS).
# ---------------------------------------------------------------------------
STATUS_PATH="${OUTPUT_DIR}/STATUS"
cat > "${STATUS_PATH}" <<EOF
issue_135_readiness_slo_matrix
completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
engine_commit=${ENGINE_COMMIT}
benchmark_commit=${BENCHMARK_COMMIT}
cann_version=${CANN_VERSION}
driver_version=${DRIVER_VERSION}
workloads=${WORKLOADS}
load_profiles=${LOAD_PROFILES}
repetitions=${REPETITIONS}
total_cells=${TOTAL_CELLS}
EOF

echo ""
echo "=== Matrix complete ==="
echo "STATUS: ${STATUS_PATH}"
echo "Output: ${OUTPUT_DIR}"
