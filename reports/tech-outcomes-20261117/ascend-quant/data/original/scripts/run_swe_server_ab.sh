#!/usr/bin/env bash
# ==============================================================================
# W4A8 vs W8A8 A/B —— 推理服务端启动脚本 (Ascend 910B3 / vllm-hust 运行时)
#
#   本脚本只做一件事: 在指定卡上把某个臂的服务起起来并等就绪, 不跑任何客户端。
#   客户端由 run_swe_ab_sweep.sh 单独调用。
#
#   ★ 与 run_swe_server.sh (已发布曲线的启动脚本) 的差异, 逐条说明:
#     1. 新增 w4a8 分支 -> /data/gcy/Qwen3.5-35B-A3B-W4A8, 并且 **给 w4a8 也加
#        --quantization ascend**。原脚本只在 w8a8|int8 时加这个参数, 不加的话
#        vLLM 会按普通权重去加载 int4 检查点, 必然失败。
#     2. 环境前置改为 source env_activate.sh, 并把 CANN/ATB 默认指向 9.1.0。
#        原脚本默认 /usr/local/Ascend (系统 CANN 8.5), 而 W8A8/W4A8 的 quant 融合
#        pass 需要 CANN 9.1 的 custom_transformer 算子包 (aclnnAddRmsNormBias)。
#        这与已发布 TP1 点的运行环境 (见 server-metadata-tp1.json
#        environment_overrides 块) 逐条一致。
#     3. PERF_CACHE_ROOT 按 LABEL 分目录 -> 各臂的 torch.compile / ACL 图缓存
#        互不污染。跨臂复用缓存会让"首次冷编译"的代价不同, 污染 A/B 比较。
#     4. 默认 TP=1 / 卡 7 / 端口 8200 / GPU_MEM_UTIL=0.95。
#        卡 0(8100)、卡 2+3(8101/8103) 上已有别的服务在跑, 本脚本一律不碰。
#     5. 日志/pid 文件名加 "ab_" 前缀, 不覆盖 run_swe_server.sh 的产物。
#
#   用法:
#       LABEL=w4a8 PORT=8200 ASCEND_RT_VISIBLE_DEVICES=7 \
#         EXTRA_SERVER_ARGS='--kv-cache-memory-bytes 34953463839' \
#         ./run_swe_server_ab.sh
#       LABEL=w8a8 EXTRA_SERVER_ARGS='--kv-cache-memory-bytes 20242376704' \
#         ./run_swe_server_ab.sh
#    停止: kill -- -$(cat logs/swe_server_ab_<label>.pid)
#
#   可选环境变量: LABEL MODEL_PATH SERVED_NAME TP PORT MAX_MODEL_LEN GPU_MEM_UTIL
#                 ASCEND_RT_VISIBLE_DEVICES EXTRA_SERVER_ARGS LOG_DIR PERF_CACHE_ROOT
#                 PERF_ENV CANN_SET_ENV ATB_SET_ENV VENDOR_OPP PERF_FIX_KERNELS_SO
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LABEL="${LABEL:-w4a8}"
case "${LABEL}" in
    w4a8)       DEFAULT_MODEL="/data/gcy/Qwen3.5-35B-A3B-W4A8" ;;
    w8a8|int8)  DEFAULT_MODEL="/data/jxd/Qwen3.5-35B-A3B-W8A8" ;;
    bf16|fp16)  DEFAULT_MODEL="/data/jxd/Qwen3.5-35B-A3B" ;;
    *)          DEFAULT_MODEL="/data/jxd/${LABEL}" ;;
esac
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL}}"
SERVED_NAME="${SERVED_NAME:-qwen35-a3b-${LABEL}}"

TP="${TP:-1}"
PORT="${PORT:-8200}"
# 与 swe-prefix-reuse 的 --server-max-context 必须一致 (写进 server-metadata)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
HOST="127.0.0.1"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

# ---- 与已发布 TP1 点逐条对齐的环境前置 --------------------------------------
# 取值来源: /proc/<卡0上那个 W8A8 TP1 服务 pid>/environ 实测抄本
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-7}"
export CANN_SET_ENV="${CANN_SET_ENV:-/data/jxd/Ascend-9.1.0/ascend-toolkit/set_env.sh}"
export ATB_SET_ENV="${ATB_SET_ENV:-/data/jxd/Ascend-9.1.0/nnal/atb/set_env.sh}"
# 每臂独立的编译/图缓存, 避免跨臂复用导致"首次冷编译"代价不同
export PERF_CACHE_ROOT="${PERF_CACHE_ROOT:-/data/jxd/cache/cann91-${LABEL}-ab}"
# 挂 custom_transformer 算子包 (W8A8/W4A8 的 norm_quant 融合 pass 需要)
export VENDOR_OPP="${VENDOR_OPP:-1}"
# 树内 libvllm_ascend_kernels.so 当前已是 CANN8.5 兼容产物 (sha256 eb5c3a24...,
# 与已发布点运行时加载的同一个文件), 因此不替换, 与记录一致。
export PERF_FIX_KERNELS_SO="${PERF_FIX_KERNELS_SO:-0}"
# torch>=2.13 默认开启 AOT autograd 缓存; 本机走 fusion_pass_compile 分支时
# 该分支不返回 OutputCode, 会被缓存 bundle 断言拒绝 -> 关掉
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-0}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_activate.sh"

VLLM_BIN="${VLLM_BIN:-${PERF_ENV:-/data/jxd/envs/vllm-hust-v1-cann91}/bin/vllm}"
[[ -x "${VLLM_BIN}" ]] || { echo "vllm 入口不存在: ${VLLM_BIN}" >&2; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "模型目录不存在: ${MODEL_PATH}" >&2; exit 1; }

if curl -sf -o /dev/null "http://${HOST}:${PORT}/health" 2>/dev/null; then
    echo "端口 ${PORT} 上已有服务在运行, 请先停止或换 PORT" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SERVER_LOG="${LOG_DIR}/swe_server_ab_${LABEL}_${TS}.log"
PID_FILE="${LOG_DIR}/swe_server_ab_${LABEL}.pid"

SERVER_ARGS=(
  serve "${MODEL_PATH}"
  --served-model-name "${SERVED_NAME}"
  --tensor-parallel-size "${TP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
)
# msmodelslim ascendv1 产物 (quant_model_description.json) 必须显式指定量化后端。
# ★ w4a8 也在这里 —— 这是相对 run_swe_server.sh 的关键修正。
case "${LABEL}" in
    w4a8|w8a8|int8) SERVER_ARGS+=(--quantization ascend) ;;
esac

# 可选: 追加额外 serve 参数 (空格分隔的单个 token 列表)
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARR=( ${EXTRA_SERVER_ARGS} )
    SERVER_ARGS+=( "${EXTRA_ARGS_ARR[@]}" )
fi

{
    echo "===================== run_swe_server_ab.sh ====================="
    echo "utc             : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "label           : ${LABEL}"
    echo "model_path      : ${MODEL_PATH}"
    echo "served_name     : ${SERVED_NAME}"
    echo "cards           : ${ASCEND_RT_VISIBLE_DEVICES} (tp=${TP})"
    echo "port            : ${PORT}"
    echo "max_model_len   : ${MAX_MODEL_LEN}"
    echo "gpu_mem_util    : ${GPU_MEM_UTIL}"
    echo "cann            : ${CANN_SET_ENV}"
    echo "atb             : ${ATB_SET_ENV}"
    echo "cache_root      : ${PERF_CACHE_ROOT}"
    echo "vendor_opp      : ${VENDOR_OPP}"
    echo "fix_kernels_so  : ${PERF_FIX_KERNELS_SO}"
    echo "extra_args      : ${EXTRA_SERVER_ARGS:-<none>}"
    echo "command         : ${VLLM_BIN} ${SERVER_ARGS[*]}"
    echo "==============================================================="
} | tee -a "${SERVER_LOG}"

setsid "${VLLM_BIN}" "${SERVER_ARGS[@]}" >> "${SERVER_LOG}" 2>&1 < /dev/null &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
echo "[swe] server pid : ${SERVER_PID} (pid file ${PID_FILE})"
echo "[swe] server log : ${SERVER_LOG}"
echo "[swe] stop       : kill -- -${SERVER_PID}"

for i in $(seq 1 240); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[swe] 服务进程提前退出, 见 ${SERVER_LOG}" >&2
        tail -60 "${SERVER_LOG}" >&2
        exit 1
    fi
    if curl -sf -o /dev/null "http://${HOST}:${PORT}/health" 2>/dev/null; then
        echo "[swe] SERVER_READY after ${i}0s -> http://${HOST}:${PORT}/v1/completions"
        break
    fi
    sleep 10
done
curl -sf -o /dev/null "http://${HOST}:${PORT}/health" || {
    echo "[swe] 服务未在 2400s 内就绪" >&2
    tail -60 "${SERVER_LOG}" >&2
    exit 1
}

echo "[swe] 服务保持运行"
wait "${SERVER_PID}" 2>/dev/null || true
