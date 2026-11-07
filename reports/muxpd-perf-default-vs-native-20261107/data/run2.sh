#!/usr/bin/env bash
# PLAN-REVISION-1: finish TP2 MuxPD pass1 on the running server, then TP2 native, TP4 MuxPD, TP4 native (1 pass each).
set -o pipefail
cd /root/workspace/muxpd-eval
[ "$(git -C /root/workspace/vllm-ascend-MuxPD-hust rev-parse HEAD)" = 43fcaf4279640bcc086119b0671cb7fd4228836a ] || { echo WRONG_HEAD; exit 1; }
[ -z "$(git -C /root/workspace/vllm-ascend-MuxPD-hust status --porcelain -- src scripts)" ] || { echo DIRTY; exit 1; }
sha256sum -c --quiet perf36/REVISION-1.sha256 || { echo PLAN_CHANGED; exit 1; }
sha256sum -c --quiet plans/formal/SHA256SUMS || { echo PLANS_CHANGED; exit 1; }
unset HCCL_DETERMINISTIC VLLM_BATCH_INVARIANT
G16='{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}'
MUX=/root/workspace/vllm-ascend-MuxPD-hust/scripts/serve-qwen35-muxwise.sh
NAT=/root/workspace/muxpd-eval/tools/serve-native.sh
while kill -0 $(cat perf36/inflight.pid) 2>/dev/null; do sleep 10; done
echo "$(date +%T) in-flight contend-s1 finished"
tools/run_passes.sh perf36-tp2 mux43-s16 8300 1
tools/stop.sh mux43-tp2; sleep 15
arm() { # runname label dev tp root script args...
  local rn=$1 lb=$2 dev=$3 tp=$4 root=$5 s=$6; shift 6
  for attempt in 1 2; do
    n=$rn; [ $attempt = 2 ] && n=$rn-relaunch
    tools/launch.sh $n $dev 8300 $tp $s "$@"
    if tools/wait_healthy.sh $n 8300 1800; then tools/run_passes.sh $root $lb 8300 1; tools/stop.sh $n; sleep 15; return 0; fi
    tools/stop.sh $n; sleep 15
  done; echo "$rn FAILED_TO_START"
}
arm nat36-tp2 native-s16 0,1 2 perf36-tp2 $NAT --max-num-seqs 16 --max-num-batched-tokens 16384 --compilation-config "$G16"
arm mux43-tp4 mux43-s16 0,1,2,3 4 perf36-tp4 $MUX --served-model-name qwen3.5-35b --seed 0 --max-num-seqs 16 --compilation-config "$G16"
arm nat36-tp4 native-s16 0,1,2,3 4 perf36-tp4 $NAT --max-num-seqs 16 --max-num-batched-tokens 16384 --compilation-config "$G16"
echo PERF36_R1_DONE
bash /root/workspace/runtime/npu-task.sh start --duty 0.2 --size 2048; echo "$(date -u +%FT%TZ) keepalive restart rc=$?"
