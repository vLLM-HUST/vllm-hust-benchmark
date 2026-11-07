#!/usr/bin/env bash
set -o pipefail
cd /root/workspace/muxpd-eval
[ "$(git -C /root/workspace/vllm-ascend-MuxPD-hust rev-parse HEAD)" = 43fcaf4279640bcc086119b0671cb7fd4228836a ] || { echo WRONG_HEAD; exit 1; }
[ -z "$(git -C /root/workspace/vllm-ascend-MuxPD-hust status --porcelain -- src scripts)" ] || { echo DIRTY; exit 1; }
sha256sum -c --quiet perf36/PLAN.sha256 || { echo PLAN_CHANGED; exit 1; }
sha256sum -c --quiet plans/formal/SHA256SUMS || { echo PLANS_CHANGED; exit 1; }
unset HCCL_DETERMINISTIC VLLM_BATCH_INVARIANT
G16='{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}'
S=/root/workspace/vllm-ascend-MuxPD-hust/scripts/serve-qwen35-muxwise.sh
arm() { # name dev tp root
  for attempt in 1 2; do
    n=$1; [ $attempt = 2 ] && n=$1-relaunch
    tools/launch.sh $n $2 8300 $3 $S --served-model-name qwen3.5-35b --seed 0 --max-num-seqs 16 --compilation-config "$G16"
    if tools/wait_healthy.sh $n 8300 1800; then tools/run_passes.sh $4 mux43-s16 8300 3; tools/stop.sh $n; sleep 10; return 0; fi
    tools/stop.sh $n; sleep 10
  done; echo "$1 FAILED_TO_START"
}
arm mux43-tp2 0,1 2 perf36-tp2
arm mux43-tp4 0,1,2,3 4 perf36-tp4
echo PERF36_DONE
