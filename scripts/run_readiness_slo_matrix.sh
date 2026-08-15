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
#     --model-path /data/shared_datasets/strict-models/Qwen2.5-14B-Instruct \
#     --served-model-name Qwen2.5-14B-Instruct \
#     --canonical-model-name Qwen/Qwen2.5-14B-Instruct \
#     --engine-repo /root/vllm/vllm-hust \
#     --benchmark-repo /root/vllm/vllm-hust-benchmark \
#     --workloads random-online \
#     --load-profiles steady-1rps \
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
    if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
        echo "ERROR: residual vllm api_server processes detected after cleanup" >&2
        exit 1
    fi
    exit "${exit_code}"
}

trap cleanup EXIT TERM INT

# ---------------------------------------------------------------------------
# Defaults.
# ---------------------------------------------------------------------------
MODEL_PATH="${VLLM_HUST_MODEL_PATH:-/data/shared_datasets/strict-models/Qwen2.5-14B-Instruct}"
SERVED_MODEL_NAME="${VLLM_HUST_SERVED_MODEL_NAME:-Qwen2.5-14B-Instruct}"
CANONICAL_MODEL_NAME="${VLLM_HUST_CANONICAL_MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
WORKLOADS="random-online,sharegpt-online,prefix-repetition-online,burstgpt"
LOAD_PROFILES="steady-1rps,steady-1.2rps,burst"
REPETITIONS=3
OUTPUT_DIR="reports/issue_135_readiness_slo_matrix"
ENGINE_REPO="${VLLM_HUST_REPO:-}"
BENCHMARK_REPO="${VLLM_HUST_BENCHMARK_REPO:-$(pwd)}"
HOST="${VLLM_HUST_SERVER_HOST:-0.0.0.0}"
PORT="${VLLM_HUST_SERVER_PORT:-8011}"
GPU_MEMORY_UTILIZATION="${VLLM_HUST_GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${VLLM_HUST_MAX_MODEL_LEN:-32768}"
TENSOR_PARALLEL_SIZE="${VLLM_HUST_TENSOR_PARALLEL_SIZE:-1}"
NUM_PROMPTS="${VLLM_HUST_NUM_PROMPTS:-50}"
INPUT_LEN="${VLLM_HUST_INPUT_LEN:-1024}"
OUTPUT_LEN="${VLLM_HUST_OUTPUT_LEN:-256}"
PYTHON_BIN="${VLLM_HUST_PYTHON:-/root/miniconda3/envs/vllm-hust-dev/bin/python}"
CLI_COMPAT="${VLLM_HUST_CLI_COMPAT:-${BENCHMARK_REPO}/scripts/run_vllm_cli_compat.py}"
VLLM_ASCEND_HUST_REPO="${VLLM_HUST_ASCEND_REPO:-/root/vllm/vllm-ascend-hust}"
HF_HOME="${VLLM_HUST_HF_HOME:-/data/shared_datasets/vllm-hust-benchmark/huggingface}"
ALLOW_BUSY_NPU=0
SKIP_DEFENSIVE_CLEANUP=0

usage() {
    cat <<'USAGE'
Usage: scripts/run_readiness_slo_matrix.sh [options]

Options:
  --model-path PATH              Local model path for server --model
  --served-model-name NAME       Client-facing model name (vllm --served-model-name)
  --canonical-model-name NAME    Canonical HF model id (e.g. Qwen/Qwen2.5-14B-Instruct)
  --workloads CSV                Comma-separated workload list
  --load-profiles CSV            Comma-separated load profile list
  --repetitions N                Number of independent server restarts (>=3)
  --output-dir DIR               Output directory for artifacts
  --engine-repo PATH             Path to vllm-hust repo (required)
  --benchmark-repo PATH          Path to vllm-hust-benchmark repo
  --port PORT                    Server port (default: 8011)
  --gpu-memory-utilization FLOAT (default: 0.6)
  --max-model-len INT            (default: 32768)
  --tensor-parallel-size INT     (default: 1)
  --num-prompts INT              Bench serve num-prompts (default: 50)
  --input-len INT                Bench serve input-len (default: 1024)
  --output-len INT               Bench serve output-len (default: 256)
  --python PATH                  Python binary path
  --cli-compat PATH              run_vllm_cli_compat.py path
  --allow-busy-npu               Skip NPU idle check (NOT recommended)
  --skip-defensive-cleanup      Skip residual-process cleanup verification
  -h, --help                     Show this help

Environment variables:
  VLLM_HUST_MODEL_PATH, VLLM_HUST_SERVED_MODEL_NAME,
  VLLM_HUST_CANONICAL_MODEL_NAME, VLLM_HUST_REPO,
  VLLM_HUST_BENCHMARK_REPO, VLLM_HUST_ASCEND_REPO,
  VLLM_HUST_SERVER_HOST, VLLM_HUST_SERVER_PORT,
  VLLM_HUST_GPU_MEMORY_UTILIZATION, VLLM_HUST_MAX_MODEL_LEN,
  VLLM_HUST_TENSOR_PARALLEL_SIZE, VLLM_HUST_NUM_PROMPTS,
  VLLM_HUST_INPUT_LEN, VLLM_HUST_OUTPUT_LEN,
  VLLM_HUST_PYTHON, VLLM_HUST_CLI_COMPAT, VLLM_HUST_HF_HOME
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --canonical-model-name) CANONICAL_MODEL_NAME="$2"; shift 2 ;;
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
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --cli-compat) CLI_COMPAT="$2"; shift 2 ;;
        --allow-busy-npu) ALLOW_BUSY_NPU=1; shift ;;
        --skip-defensive-cleanup) SKIP_DEFENSIVE_CLEANUP=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option $1" >&2; usage; exit 2 ;;
    esac
done

# CLI_COMPAT is defaulted from BENCHMARK_REPO above, but BENCHMARK_REPO is only
# finalized by --benchmark-repo during arg parsing. If the user passed
# --benchmark-repo (or VLLM_HUST_BENCHMARK_REPO env) without an explicit
# --cli-compat / VLLM_HUST_CLI_COMPAT, recompute CLI_COMPAT so it points at the
# resolved benchmark repo instead of the pre-parse default (e.g. $(pwd)).
if [[ -z "${VLLM_HUST_CLI_COMPAT:-}" ]]; then
    CLI_COMPAT="${BENCHMARK_REPO}/scripts/run_vllm_cli_compat.py"
fi

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
if [[ ! -f "${CLI_COMPAT}" ]]; then
    echo "ERROR: CLI compat script not found: ${CLI_COMPAT}" >&2
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
# Resolve workload → bench serve dataset + load profile → request rate.
# ---------------------------------------------------------------------------
workload_dataset() {
    case "$1" in
        random-online) echo "random" ;;
        sharegpt-online) echo "sharegpt" ;;
        prefix-repetition-online) echo "random" ;;  # TODO: dedicated dataset
        burstgpt) echo "burstgpt" ;;
        tracelab-specialty) echo "random" ;;
        *) echo "random" ;;
    esac
}

profile_request_rate() {
    case "$1" in
        steady-1rps) echo "1.0" ;;
        steady-1.2rps) echo "1.2" ;;
        burst) echo "10" ;;       # burst: high arrival rate
        overload-recovery) echo "20" ;;
        *) echo "1.0" ;;
    esac
}

profile_report_type() {
    case "$1" in
        steady-1rps|steady-1.2rps) echo "fixed-qps" ;;
        burst|overload-recovery) echo "burst" ;;
        *) echo "fixed-qps" ;;
    esac
}

# ---------------------------------------------------------------------------
# NPU idle check (per project memory: wait_for_npu_idle must fail-closed).
# ---------------------------------------------------------------------------
if [[ "${ALLOW_BUSY_NPU}" -eq 0 ]]; then
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "ERROR: npu-smi not found; cannot verify NPU is idle" >&2
        echo "       (pass --allow-busy-npu to skip, NOT recommended)" >&2
        exit 1
    fi
    # Check AICore utilization; any non-zero means NPU is busy.
    if npu-smi info 2>/dev/null | grep -E "AICore\(%\)" -A1 | grep -qE "[1-9][0-9]?%"; then
        echo "ERROR: NPU AICore utilization is non-zero; refusing to start benchmark" >&2
        npu-smi info >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Ascend NPU environment setup (CANN toolkit + ATB + PYTHONPATH).
# These scripts use unbound vars; run under set +u.
# ---------------------------------------------------------------------------
echo "INFO: setting up Ascend NPU environment"
export ZSH_VERSION=""
set +u
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
fi
set -u
# Prepend vllm repos but preserve PYTHONPATH from Ascend set_env.sh (pyACL).
export PYTHONPATH="${VLLM_ASCEND_HUST_REPO}:${ENGINE_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME}"
# Bypass proxy for local server health checks and bench serve requests.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

# ---------------------------------------------------------------------------
# Output directory + atomic write helpers.
# ---------------------------------------------------------------------------
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
TMP_DIR="${OUTPUT_DIR}/.tmp"
mkdir -p "${TMP_DIR}"

# ---------------------------------------------------------------------------
# Resolve engine/benchmark commit (40-char hex SHA, per project memory).
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
# Per PR #154 review round 2: resolve the plugin (vllm-ascend-hust) commit
# independently from the benchmark repo so provenance reflects the actual
# plugin code that ran, not the benchmark repo's HEAD.
if [[ ! -d "${VLLM_ASCEND_HUST_REPO}" ]]; then
    echo "ERROR: VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO} is not a directory" >&2
    exit 1
fi
PLUGIN_COMMIT="$(cd "${VLLM_ASCEND_HUST_REPO}" && git rev-parse HEAD)"
if ! [[ "${PLUGIN_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: plugin commit ${PLUGIN_COMMIT} is not a 40-char hex SHA" >&2
    exit 1
fi

# Resolve CANN/driver versions (per project memory: env-manifest.json must
# include real CANN/driver versions; placeholder sentinels are rejected).
CANN_VERSION="${VLLM_HUST_CANN_VERSION:-}"
DRIVER_VERSION="${VLLM_HUST_DRIVER_VERSION:-}"
if [[ -z "${CANN_VERSION}" ]] && [[ -f /usr/local/Ascend/ascend-toolkit/latest/opp/version.info ]]; then
    CANN_VERSION="$(grep -oE 'Version=[0-9]+\.[0-9]+\.[0-9]+' /usr/local/Ascend/ascend-toolkit/latest/opp/version.info | head -n1 | cut -d= -f2)"
fi
if [[ -z "${DRIVER_VERSION}" ]] && [[ -f /usr/local/Ascend/driver/version.info ]]; then
    DRIVER_VERSION="$(grep -oE 'Version=[0-9]+\.[0-9]+\.[a-z0-9]+' /usr/local/Ascend/driver/version.info | head -n1 | cut -d= -f2)"
fi
if [[ -z "${CANN_VERSION}" ]] || [[ -z "${DRIVER_VERSION}" ]]; then
    echo "ERROR: CANN/driver version could not be resolved; set VLLM_HUST_CANN_VERSION/VLLM_HUST_DRIVER_VERSION" >&2
    exit 1
fi

# Resolve engine version from the repo.
ENGINE_VERSION="${VLLM_HUST_ENGINE_VERSION:-}"
if [[ -z "${ENGINE_VERSION}" ]] && [[ -f "${ENGINE_REPO}/vllm/version.py" ]]; then
    ENGINE_VERSION="$("${PYTHON_BIN}" -c "import sys; sys.path.insert(0, '${ENGINE_REPO}'); from vllm.version import __version__ as v; print(v)" 2>/dev/null || echo "")"
fi
if [[ -z "${ENGINE_VERSION}" ]]; then
    ENGINE_VERSION="v0.23.1-dev"
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"
PYTORCH_VERSION="$("${PYTHON_BIN}" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '')"
OS_INFO="$("${PYTHON_BIN}" -c 'import platform; print(platform.platform())')"
CHIP_MODEL="${VLLM_HUST_CHIP_MODEL:-910B2}"

echo "INFO: engine_commit=${ENGINE_COMMIT}"
echo "INFO: benchmark_commit=${BENCHMARK_COMMIT}"
echo "INFO: plugin_commit=${PLUGIN_COMMIT}"
echo "INFO: cann_version=${CANN_VERSION} driver_version=${DRIVER_VERSION}"
echo "INFO: python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION}"
echo "INFO: model_path=${MODEL_PATH} served_model_name=${SERVED_MODEL_NAME}"

# Export provenance env vars for build_readiness_artifact.py.
export VLLM_HUST_ENGINE_COMMIT="${ENGINE_COMMIT}"
export VLLM_HUST_BENCHMARK_COMMIT="${BENCHMARK_COMMIT}"
export VLLM_HUST_PLUGIN_COMMIT="${PLUGIN_COMMIT}"
export VLLM_HUST_ENGINE_VERSION="${ENGINE_VERSION}"
export VLLM_HUST_CANN_VERSION="${CANN_VERSION}"
export VLLM_HUST_DRIVER_VERSION="${DRIVER_VERSION}"
export VLLM_HUST_PYTHON_VERSION="${PYTHON_VERSION}"
export VLLM_HUST_PYTORCH_VERSION="${PYTORCH_VERSION}"
export VLLM_HUST_OS_INFO="${OS_INFO}"
export VLLM_HUST_CHIP_MODEL="${CHIP_MODEL}"
export VLLM_HUST_SUBMITTER="${VLLM_HUST_SUBMITTER:-issue-135-npu-runner}"
export VLLM_HUST_PYTHON="${PYTHON_BIN}"

# ---------------------------------------------------------------------------
# Start the server with setsid (process group leader, per project memory).
# Mirrors the proven _smoke_readiness_slo.sh invocation.
# ---------------------------------------------------------------------------
CACHE_ROOT="${OUTPUT_DIR}/.cache"

start_server() {
    local cold_start="$1"
    local server_log="$2"
    local startup_ts_file="$3"

    if [[ "${cold_start}" == "true" ]]; then
        # Cold-start: clear compile cache to ensure no reusable products exist.
        rm -rf "${CACHE_ROOT}"
        mkdir -p "${CACHE_ROOT}"
    else
        mkdir -p "${CACHE_ROOT}"
    fi

    # Record process start timestamp (UTC ISO) BEFORE launching the server.
    date -u +%Y-%m-%dT%H:%M:%SZ > "${startup_ts_file}"

    export VLLM_CACHE_ROOT="${CACHE_ROOT}"

    setsid "${PYTHON_BIN}" -u -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --enforce-eager \
        --trust-remote-code \
        --no-enable-log-requests \
        --host "${HOST}" \
        --port "${PORT}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        > "${server_log}" 2>&1 &
    SERVER_PGID=$!
    echo "INFO: server started (pgid=${SERVER_PGID}, cold_start=${cold_start})"
}

wait_for_readiness() {
    local startup_ts_file="$1"
    local max_attempts=600
    local attempt=0
    while [[ ${attempt} -lt ${max_attempts} ]]; do
        if curl --noproxy "*" -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            # Record readiness timestamp (UTC ISO) as second line.
            date -u +%Y-%m-%dT%H:%M:%SZ >> "${startup_ts_file}"
            echo "INFO: server ready after ${attempt}s"
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
        # Send SIGTERM to the process group first (graceful shutdown).
        kill -TERM -- -"${SERVER_PGID}" 2>/dev/null || true
        # Wait up to 10s for graceful shutdown.
        local term_wait=0
        while [[ ${term_wait} -lt 10 ]]; do
            if ! kill -0 -- -"${SERVER_PGID}" 2>/dev/null; then
                break
            fi
            sleep 1
            term_wait=$((term_wait + 1))
        done
        # If still alive, SIGKILL the process group (vllm may ignore SIGTERM).
        if kill -0 -- -"${SERVER_PGID}" 2>/dev/null; then
            echo "WARN: server did not exit on SIGTERM after 10s; sending SIGKILL" >&2
            kill -KILL -- -"${SERVER_PGID}" 2>/dev/null || true
        fi
        wait "${SERVER_PGID}" 2>/dev/null || true
        SERVER_PGID=""
    fi
    # Wait for all vllm api_server processes to exit (HBM release).
    local wait_attempts=0
    while [[ ${wait_attempts} -lt 30 ]]; do
        if ! pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
            break
        fi
        sleep 1
        wait_attempts=$((wait_attempts + 1))
    done
    # Precise residual cleanup. vllm's multiprocessing spawn can re-launch the
    # `-m vllm.entrypoints.openai.api_server` launcher (and its engine children)
    # into a NEW process group, so `kill -- -PGID` above may not reach them.
    # The smoke run for issue #135 observed exactly this: a launcher process
    # survived the process-group kill and kept HBM allocated on the chip.
    # Clean up any remaining vllm server process bound to OUR ${PORT}. The
    # pattern is anchored on `api_server`, NOT a bare `vllm`, so this does NOT
    # match this runner script itself (its cmdline contains a `vllm` path and
    # `--port ${PORT}` but no `api_server`). Scoping on `--port ${PORT}` keeps
    # this precise to this server and avoids killing other tenants' servers on
    # other ports (per project memory: use precise cleanup, never broad pkill).
    local residual_pids
    residual_pids="$(pgrep -f "api_server.*--port ${PORT}" 2>/dev/null || true)"
    if [[ -n "${residual_pids}" ]]; then
        echo "WARN: killing residual vllm processes on port ${PORT}: ${residual_pids}" >&2
        # shellcheck disable=SC2086
        for pid in ${residual_pids}; do
            kill -9 "${pid}" 2>/dev/null || true
        done
        # Give the killed processes a moment to release HBM.
        sleep 3
    fi
}

# ---------------------------------------------------------------------------
# Run bench serve workload against the live server + capture /metrics.
# Mirrors the proven _smoke_readiness_slo.sh bench serve invocation.
# ---------------------------------------------------------------------------
run_workload() {
    local workload="$1"
    local profile="$2"
    local client_result="$3"
    local metrics_file="$4"

    local dataset request_rate dataset_path
    dataset="$(workload_dataset "${workload}")"
    request_rate="$(profile_request_rate "${profile}")"
    # Resolve the dataset file for datasets that require an explicit path
    # (sharegpt-online / burstgpt). random-online uses vllm script defaults.
    case "${dataset}" in
        sharegpt) dataset_path="/data/shared_datasets/strict-inputs/ShareGPT_V3_unfiltered_cleaned_split.json" ;;
        burstgpt) dataset_path="/data/shared_datasets/strict-inputs/BurstGPT_3.csv" ;;
        *) dataset_path="" ;;
    esac

    echo "INFO: running bench serve (dataset=${dataset}, num_prompts=${NUM_PROMPTS}, rate=${request_rate})"

    NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" \
    "${PYTHON_BIN}" "${CLI_COMPAT}" bench serve \
        --save-result \
        --result-dir "$(dirname "${client_result}")" \
        --result-filename "$(basename "${client_result}")" \
        --backend vllm \
        --endpoint /v1/completions \
        --base-url "http://127.0.0.1:${PORT}" \
        --dataset-name "${dataset}" \
        ${dataset_path:+--dataset-path "${dataset_path}"} \
        --model "${SERVED_MODEL_NAME}" \
        --tokenizer "${MODEL_PATH}" \
        --num-prompts "${NUM_PROMPTS}" \
        --input-len "${INPUT_LEN}" \
        --output-len "${OUTPUT_LEN}" \
        --request-rate "${request_rate}" \
        --metric-percentiles '50,95,99'

    echo "INFO: fetching /metrics"
    curl --noproxy "*" -sf "http://127.0.0.1:${PORT}/metrics" > "${metrics_file}" \
        || echo "WARN: /metrics unavailable" >&2
}

# ---------------------------------------------------------------------------
# Run a recovery probe: send 1 request after the burst phase and record its
# TTFT (seconds) as burst_recovery_s in probe_result.json.
#
# Per PR #154 review round 2: the burst_recovery_s field must be a real
# measurement. The probe request's TTFT reflects how quickly the server
# returns to normal serving after the burst backlog drains. A low probe TTFT
# means the system recovered quickly; a high probe TTFT means the burst
# caused lasting queueing/scheduling pressure.
# ---------------------------------------------------------------------------
run_recovery_probe() {
    local probe_result="$1"
    local probe_dir
    probe_dir="$(dirname "${probe_result}")"
    local probe_client="${probe_dir}/probe_client_result.json"

    # serve.py asserts current_request_rate > 0.0, so rate 0 is rejected.
    # With a single probe prompt the first request is still dispatched at
    # t=0, so rate=1 preserves the 'immediate single request' semantics
    # while passing the assert.
    echo "INFO: recovery probe: sending 1 request (rate=1, immediate)"

    NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" \
    "${PYTHON_BIN}" "${CLI_COMPAT}" bench serve \
        --save-result \
        --result-dir "${probe_dir}" \
        --result-filename "probe_client_result.json" \
        --backend vllm \
        --endpoint /v1/completions \
        --base-url "http://127.0.0.1:${PORT}" \
        --dataset-name random \
        --model "${SERVED_MODEL_NAME}" \
        --tokenizer "${MODEL_PATH}" \
        --num-prompts 1 \
        --input-len "${INPUT_LEN}" \
        --output-len "${OUTPUT_LEN}" \
        --request-rate 1 \
        --metric-percentiles '50,95,99'

    if [[ ! -f "${probe_client}" ]]; then
        echo "ERROR: recovery probe did not produce ${probe_client}" >&2
        return 1
    fi

    # Extract probe TTFT (ms) and write recovery_ttft_s (seconds) to
    # probe_result.json. build_readiness_artifact.py reads this file.
    "${PYTHON_BIN}" - "${probe_client}" "${probe_result}" <<'PYEOF'
import json, sys
probe_client, probe_result = sys.argv[1], sys.argv[2]
data = json.load(open(probe_client))
ttft_ms = float(data.get("mean_ttft_ms", 0.0))
json.dump({"recovery_ttft_s": ttft_ms / 1000.0}, open(probe_result, "w"))
print(f"INFO: recovery probe ttft={ttft_ms:.2f}ms -> recovery_ttft_s={ttft_ms/1000.0:.4f}s")
PYEOF

    if [[ ! -f "${probe_result}" ]]; then
        echo "ERROR: failed to write ${probe_result}" >&2
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Matrix execution loop.
# ---------------------------------------------------------------------------
TOTAL_CELLS=$(( ${#WORKLOAD_ARRAY[@]} * ${#PROFILE_ARRAY[@]} * REPETITIONS ))
CELL_INDEX=0

for workload in "${WORKLOAD_ARRAY[@]}"; do
    for profile in "${PROFILE_ARRAY[@]}"; do
        report_type="$(profile_report_type "${profile}")"
        request_rate="$(profile_request_rate "${profile}")"
        dataset="$(workload_dataset "${workload}")"

        for (( rep=1; rep<=REPETITIONS; rep++ )); do
            CELL_INDEX=$((CELL_INDEX + 1))
            echo ""
            echo "=== [${CELL_INDEX}/${TOTAL_CELLS}] workload=${workload} profile=${profile} rep=${rep} ==="

            cell_dir="${OUTPUT_DIR}/${workload}/${profile}/rep${rep}"
            mkdir -p "${cell_dir}"

            server_log="${cell_dir}/server.log"
            client_result="${cell_dir}/client_result.json"
            metrics_file="${cell_dir}/metrics.txt"
            startup_ts_file="${cell_dir}/startup_ts.txt"

            # All repetitions are cold starts (independent process restarts).
            # Each rep clears the cache, starts a fresh server process, runs
            # the workload, and stops the server before the next rep.
            cold_start="true"

            # Start server (setsid → process group leader).
            start_server "${cold_start}" "${server_log}" "${startup_ts_file}"

            # Wait for readiness (records cold_readiness_s via startup_ts.txt).
            if ! wait_for_readiness "${startup_ts_file}"; then
                echo "ERROR: server failed readiness for ${workload}/${profile}/rep${rep}" >&2
                tail -30 "${server_log}" >&2
                stop_server
                exit 1
            fi

            # Run the benchmark workload against the live server.
            if ! run_workload "${workload}" "${profile}" "${client_result}" "${metrics_file}"; then
                echo "ERROR: bench serve failed for ${workload}/${profile}/rep${rep}" >&2
                stop_server
                exit 1
            fi

            # Per PR #154 review round 2: burst/overload-recovery profiles
            # must measure real burst recovery. After the burst phase, send
            # a single probe request and record its TTFT (seconds) as
            # burst_recovery_s in probe_result.json. The probe reflects how
            # quickly the system returns to normal serving after the burst
            # backlog drains.
            if [[ "${profile}" == "burst" || "${profile}" == "overload-recovery" ]]; then
                echo "INFO: running recovery probe (1 request) for ${profile}"
                if ! run_recovery_probe "${cell_dir}/probe_result.json"; then
                    echo "ERROR: recovery probe failed for ${workload}/${profile}/rep${rep}" >&2
                    stop_server
                    exit 1
                fi
            fi

            # Stop the server before building the artifact (frees NPU for
            # the next rep and ensures independent process restarts).
            stop_server

            # Build the readiness-slo/v1 artifact from raw cell outputs.
            echo "INFO: building readiness-slo/v1 artifact"
            if ! "${PYTHON_BIN}" "${BENCHMARK_REPO}/scripts/build_readiness_artifact.py" \
                --cell-dir "${cell_dir}" \
                --workload "${workload}" \
                --load-profile "${profile}" \
                --rep-index "${rep}" \
                --rep-total "${REPETITIONS}" \
                --cold-start \
                --report-type "${report_type}" \
                --request-rate "${request_rate}" \
                --num-prompts "${NUM_PROMPTS}" \
                --input-len "${INPUT_LEN}" \
                --output-len "${OUTPUT_LEN}" \
                --dataset "${dataset}" \
                --served-model-name "${SERVED_MODEL_NAME}" \
                --canonical-model-name "${CANONICAL_MODEL_NAME}" \
                --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
                --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
                --max-model-len "${MAX_MODEL_LEN}" \
                --port "${PORT}" \
                --cleared-paths "${CACHE_ROOT}"; then
                echo "ERROR: artifact build failed for ${workload}/${profile}/rep${rep}" >&2
                exit 1
            fi

            # Verify no residual processes between repetitions.
            if [[ "${SKIP_DEFENSIVE_CLEANUP}" -eq 0 ]]; then
                if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
                    echo "ERROR: residual vllm api_server after ${workload}/${profile}/rep${rep}" >&2
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
"${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Use standalone readiness_slo if available (survives branch switches on shared NPU machines).
_standalone = Path("/tmp/readiness_slo_standalone/src")
if _standalone.exists():
    sys.path.insert(0, str(_standalone))

from vllm_hust_benchmark.readiness_slo import (
    aggregate_repetitions,
    validate_aggregate,
)

output_dir = Path(sys.argv[1])

# Collect artifacts from cell directories: <output>/<workload>/<profile>/rep<N>/
groups: dict[tuple, list[dict]] = defaultdict(list)
for artifact_path in sorted(output_dir.glob("*/*/rep*/readiness_slo_artifact_rep*.json")):
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    workload = payload["workload"]["name"]
    profile = payload["load_profile"]["kind"]
    groups[(workload, profile)].append(payload)

if not groups:
    print("WARN: no artifacts found for aggregation", file=sys.stderr)
    sys.exit(0)

aggregates_dir = output_dir / "aggregates"
aggregates_dir.mkdir(parents=True, exist_ok=True)

summary = []
for (workload, profile), artifacts in sorted(groups.items()):
    aggregate = aggregate_repetitions(artifacts)
    validate_aggregate(aggregate)
    out = aggregates_dir / f"{workload}_{profile}_aggregate.json"
    out.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {out}")
    # Per PR #154 review round 2: cold_readiness_median must come from
    # startup_metrics.cold_readiness_s (seconds), not output_throughput_tps
    # (tps). The previous code reused throughput median, which had the wrong
    # field name and the wrong unit.
    cold_readiness_values = [
        float(a["startup_metrics"]["cold_readiness_s"])
        for a in artifacts
        if "startup_metrics" in a and "cold_readiness_s" in a["startup_metrics"]
    ]
    cold_readiness_median = (
        statistics.median(cold_readiness_values) if cold_readiness_values else 0.0
    )
    summary.append({
        "workload": workload,
        "load_profile": profile,
        "repetition_count": aggregate["repetition_count"],
        "overall_status": aggregate["overall_status"],
        "throughput_median": aggregate["metrics"]["output_throughput_tps"]["median"],
        "throughput_iqr": aggregate["metrics"]["output_throughput_tps"]["iqr"],
        "ttft_p99_median": aggregate["metrics"]["ttft_ms_p99"]["median"],
        "cold_readiness_median": cold_readiness_median,
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
plugin_commit=${PLUGIN_COMMIT}
cann_version=${CANN_VERSION}
driver_version=${DRIVER_VERSION}
model_path=${MODEL_PATH}
served_model_name=${SERVED_MODEL_NAME}
workloads=${WORKLOADS}
load_profiles=${LOAD_PROFILES}
repetitions=${REPETITIONS}
total_cells=${TOTAL_CELLS}
EOF

echo ""
echo "=== Matrix complete ==="
echo "STATUS: ${STATUS_PATH}"
echo "Output: ${OUTPUT_DIR}"
