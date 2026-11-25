#!/usr/bin/env bash
set -e
cd /root/swe-c4-20261114
slot="$1"
case "$slot" in
  0) export ASCEND_RT_VISIBLE_DEVICES=0,1 ;;
  1) export ASCEND_RT_VISIBLE_DEVICES=2,3 ;;
  *) exit 2 ;;
esac
export PATH="/root/hust-runtime/.venv/bin:$PATH"
export PYTHONHASHSEED=0
export PYTHONPATH="/root/swe-c4-20261114/host-overlay${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HCCL_IF_BASE_PORT=$((29000+200*slot))
port=$((18410+slot))
pub=$((18510+slot))
rep=$((18610+slot))
kv=$(printf '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://127.0.0.1:%d","replay_endpoint":"tcp://127.0.0.1:%d","buffer_steps":10000}' "$pub" "$rep")
set -x
exec /root/hust-runtime/.venv/bin/vllm serve /models/Qwen3.5-35B-A3B \
 --host 127.0.0.1 --port "$port" --served-model-name qwen35-swe \
 --dtype bfloat16 --tensor-parallel-size 2 --enable-expert-parallel \
 --max-num-seqs 16 --max-model-len 262144 --gpu-memory-utilization 0.65 \
 --enable-prompt-tokens-details --enable-prefix-caching --prefix-caching-hash-algo sha256 \
 --mamba-cache-mode align --block-size 128 --enforce-eager \
 --kv-events-config "$kv"
