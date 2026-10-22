#!/bin/bash
# §3 enablement run for fi-sampling (release.md §3 ladder), card 7, port 8441.
# fi-sampling is NOT a manifest carrier, so its enable switch is supplied via the
# ambient environment (the manager only injects the cascade + fia-demask flags).
set -x
cd /tmp
export HF_HUB_OFFLINE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_HUST_FI_SAMPLING=1
export VLLM_HUST_FI_SAMPLING_TRACE=1
exec vllm-hust-ext run -- vllm serve /data/shared_models/Qwen--Qwen2.5-14B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port 8441 \
  --generation-config vllm \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[32,64,128]}'
