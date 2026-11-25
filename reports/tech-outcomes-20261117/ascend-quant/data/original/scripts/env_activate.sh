#!/usr/bin/env bash
# ==============================================================================
# 运行时环境一键就绪 (source 即用): /data/jxd/envs/vllm-hust-v1-cann91
#
#   source env_activate.sh                   # 激活运行时 + CANN/ATB + 缓存重定向 + 算子库顶替
#   VENDOR_OPP=1 source env_activate.sh      # 额外挂 custom_transformer 自定义算子包
#                                            #   (W8A8 的 aclnnAddRmsNormBias 需要, BF16 不需要)
#   NO_ACTIVATE=1 source env_activate.sh     # 只导出变量, 不激活/不改 PATH 顺序
#
#   退出: deactivate
#   说明: 本脚本按 run_perf_bench.sh / run_swe_server.sh 的前置逐条复刻, 幂等可重复 source;
#         它不是 `set -e` 脚本, 也不会改变调用者 shell 的选项。
# ==============================================================================

PERF_ENV="${PERF_ENV:-/data/jxd/envs/vllm-hust-v1-cann91}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
ATB_SET_ENV="${ATB_SET_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"

# set_env.sh 会引用未定义变量 (如 ZSH_VERSION), 与 set -u 冲突 -> 期间临时关闭, 末尾恢复
case "$-" in *u*) _env_had_u=1 ;; *) _env_had_u=0 ;; esac
set +u

# ---- 1) 激活运行时 ------------------------------------------------------------
# 本 env 是 conda 前缀环境 (conda create -p), 没有 bin/activate -> 用 conda activate;
# 若 shell 未初始化 conda, 退回把 <env>/bin 前置到 PATH (等价效果, 只是少了 CONDA_PREFIX)。
if [ "${NO_ACTIVATE:-0}" != "1" ]; then
    if [ -f "${PERF_ENV}/bin/activate" ]; then
        # shellcheck disable=SC1090
        source "${PERF_ENV}/bin/activate"
    elif command -v conda >/dev/null 2>&1; then
        conda activate "${PERF_ENV}" 2>/dev/null || true
    elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /root/miniconda3/etc/profile.d/conda.sh
        conda activate "${PERF_ENV}" 2>/dev/null || true
    fi
fi
case ":${PATH}:" in
    *":${PERF_ENV}/bin:"*) ;;
    *) PATH="${PERF_ENV}/bin:${PATH}"; export PATH ;;
esac

# ---- 2) CANN / ATB 环境 -------------------------------------------------------
for _env_f in "${CANN_SET_ENV}" "${ATB_SET_ENV}"; do
    if [ -f "${_env_f}" ]; then
        # shellcheck disable=SC1090
        source "${_env_f}" || true
    else
        echo "[env] 警告: 缺少 ${_env_f}" >&2
    fi
done
unset _env_f
export ASCEND_ATB_CXX_ABI="${ASCEND_ATB_CXX_ABI:-1}"

# ---- 3) 必备开关 -------------------------------------------------------------
# COMPILE_CUSTOM_KERNELS=0 (本机交互 shell 默认) 会导致 torch.ops._C_ascend 只剩 dummy op,
# 图编译阶段 npu_gemma_rms_norm 抛 AttributeError。未显式设置时强制置 1。
export COMPILE_CUSTOM_KERNELS="${COMPILE_CUSTOM_KERNELS:-1}"
[ "${COMPILE_CUSTOM_KERNELS}" = "1" ] || export COMPILE_CUSTOM_KERNELS=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:False}"
export PYTHONUNBUFFERED=1

# ---- 4) 缓存重定向 (根分区 / 只剩 ~45G) --------------------------------------
PERF_CACHE_ROOT="${PERF_CACHE_ROOT:-/data/jxd/cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${PERF_CACHE_ROOT}/vllm}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PERF_CACHE_ROOT}/torch_ext}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PERF_CACHE_ROOT}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PERF_CACHE_ROOT}/triton}"

# ---- 5) CANN8.5 兼容: 顶替依赖 CANN9 符号的 libvllm_ascend_kernels.so ----------
# 源码树里 2026-09-17 的预编译 so 依赖 CANN9 才有的 rtFunctionGetMetaInfo*,
# 本机 CANN 8.5.0 的 libruntime.so 不导出 -> dlopen 失败 -> enable_custom_op()=False。
# 用 support/ 下 2026-06-03 构建的 CANN8.5 兼容版顶替 (幂等, PERF_FIX_KERNELS_SO=0 可关)。
VLLM_ASCEND_PKG_DIR="${VLLM_ASCEND_PKG_DIR:-/data/jxd/src/vllm-ascend-hust-v1/vllm_ascend}"
KERNELS_SO="${VLLM_ASCEND_PKG_DIR}/libvllm_ascend_kernels.so"
COMPAT_KERNELS_SO="${PERF_KERNELS_COMPAT_SO:-${SRC_DIR}/support/libvllm_ascend_kernels.cann850.so}"
if [ "${PERF_FIX_KERNELS_SO:-1}" = "1" ] && [ -f "${KERNELS_SO}" ] && [ -f "${COMPAT_KERNELS_SO}" ]; then
    if nm -D --undefined-only "${KERNELS_SO}" 2>/dev/null | grep -q rtFunctionGetMetaInfoSize; then
        cp -a "${KERNELS_SO}" "${KERNELS_SO}.cann9-orig.so"
        cp -a "${COMPAT_KERNELS_SO}" "${KERNELS_SO}"
        echo "[env] 算子库: 已用 CANN8.5 兼容版顶替 (原产物备份 *.cann9-orig.so)"
    fi
fi

# ---- 6) 复用预编译 npu_utils.so (Triton Ascend driver 现场编译在本机必失败) ----
if [ "${PERF_FIX_TRITON_CACHE:-1}" = "1" ] && [ -d "${TRITON_CACHE_DIR}" ]; then
    _env_home_cache="${HOME:-/root}/.triton/cache"
    if [ -d "${_env_home_cache}" ] && [ "${_env_home_cache}" != "${TRITON_CACHE_DIR}" ]; then
        _env_copied=0
        for _env_item in "${_env_home_cache}"/*/npu_utils.so; do
            [ -f "${_env_item}" ] || continue
            _env_rel="${_env_item#"${_env_home_cache}"/}"
            if [ ! -f "${TRITON_CACHE_DIR}/${_env_rel}" ]; then
                mkdir -p "${TRITON_CACHE_DIR}/$(dirname "${_env_rel}")"
                cp -a "${_env_item}" "${TRITON_CACHE_DIR}/${_env_rel}"
                _env_copied=$(( _env_copied + 1 ))
            fi
        done
        [ "${_env_copied}" -gt 0 ] && echo "[env] Triton: 复用 ${_env_copied} 个预编译 npu_utils.so"
        unset _env_item _env_rel
    fi
    unset _env_home_cache _env_copied
fi

# ---- 7) 可选: custom_transformer 自定义算子包 (W8A8 需要) ---------------------
# 提供 aclnnAddRmsNormBias 等算子; 官方做法见该目录 bin/set_env.bash。
if [ "${VENDOR_OPP:-0}" = "1" ]; then
    VENDOR_OPP_DIR="${VENDOR_OPP_DIR:-/data/jxd/src/vllm-ascend-hust-v1/vllm_ascend/_cann_ops_custom/vendors/custom_transformer}"
    if [ -d "${VENDOR_OPP_DIR}" ]; then
        case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
            *":${VENDOR_OPP_DIR}:"*) ;;
            *) export ASCEND_CUSTOM_OPP_PATH="${VENDOR_OPP_DIR}:${ASCEND_CUSTOM_OPP_PATH:-}" ;;
        esac
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${VENDOR_OPP_DIR}/op_api/lib:"*) ;;
            *) export LD_LIBRARY_PATH="${VENDOR_OPP_DIR}/op_api/lib:${LD_LIBRARY_PATH:-}" ;;
        esac
        echo "[env] 已挂 custom_transformer 算子包: ${VENDOR_OPP_DIR}"
    else
        echo "[env] 警告: VENDOR_OPP=1 但目录不存在: ${VENDOR_OPP_DIR}" >&2
    fi
fi

# ---- 8) 恢复 nounset, 打印要点 ----------------------------------------------
[ "${_env_had_u}" = "1" ] && set -u
unset _env_had_u
echo "[env] python : $(command -v python3 || echo '?')"
echo "[env] vllm   : $(command -v vllm || echo '?')"
echo "[env] CANN   : ${ASCEND_HOME_PATH:-未设置}"
echo "[env] caches : ${VLLM_CACHE_ROOT} / ${TRITON_CACHE_DIR}"
echo "[env] 提示   : 起服务用 ./run_swe_server.sh (已含同一套前置); 退出用 deactivate"
