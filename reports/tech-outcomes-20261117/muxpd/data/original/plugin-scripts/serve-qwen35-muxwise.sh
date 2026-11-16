#!/usr/bin/env bash
# Opt-in MuxWise-aligned preset; see docs/muxwise-alignment.md.
# Pure-D steps use the native async queue, P runs in isolated-overlap 4-layer
# slices with independent captured decode graphs. Measured on TP2.
set -eo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_USE_V2_MODEL_RUNNER=0
exec "$script_dir/serve-qwen35.sh" \
  --async-scheduling \
  --scheduler-cls vllm_muxpd.async_scheduler.AsyncDecoupledMuxPDScheduler \
  --worker-cls vllm_muxpd.async_worker.AsyncDecoupledMuxPDNPUWorker \
  --no-enforce-eager --no-numa-bind \
  --additional-config '{"enable_cpu_binding":false}' \
  --compilation-config \
  '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' \
  --batch-admission-policy-config \
  '{"execution":"decoupled","async_decode_pipeline":true,"stream_mode":"overlap","overlap_communication":"isolated","decode_graph":true,"layers_per_slice":4}' \
  "$@"
