#!/usr/bin/env bash
# wl38: TP2 MuxPD, TP2 native, TP4 MuxPD, TP4 native; 1 pass of 6 derived workloads. No PR.
set -o pipefail
E=/root/workspace/muxpd-eval
cd $E
[ "$(git -C /root/workspace/vllm-ascend-MuxPD-hust rev-parse HEAD)" = 43fcaf4279640bcc086119b0671cb7fd4228836a ] || { echo WRONG_HEAD; exit 1; }
[ -z "$(git -C /root/workspace/vllm-ascend-MuxPD-hust status --porcelain -- src scripts)" ] || { echo DIRTY; exit 1; }
(cd wl38 && sha256sum -c --quiet PLAN.sha256) || { echo PLAN_CHANGED; exit 1; }
(cd wl38/plans && sha256sum -c --quiet SHA256SUMS) || { echo PLANS_CHANGED; exit 1; }
unset HCCL_DETERMINISTIC VLLM_BATCH_INVARIANT
bash /root/workspace/runtime/npu-task.sh stop; sleep 5
echo "$(date -u +%FT%TZ) keepalive stopped"
G16='{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}'
MUX=/root/workspace/vllm-ascend-MuxPD-hust/scripts/serve-qwen35-muxwise.sh
NAT=$E/tools/serve-native.sh
PLANS="wl-docchat-r1.3 wl-2wiki-r1.6 wl-hotpot-r1.0 wl-govsum-r1.0 wl-code-r1.0 wl-chat-r1.5"
plans() { # root label
  local out=results/$1/$2/pass1; mkdir -p $out
  for p in $PLANS; do
    [ -s $out/$p.json ] && continue
    timeout 2400 /root/workspace/.venv/bin/python tools/loadgen.py run --plan wl38/plans/$p.json --port 8300 \
      --out $out/$p.json --label $2 > /dev/null || echo "$(date +%T) $1 $2 $p FAILED rc=$?"
    echo "$(date +%T) $1 $2 $p done"
  done
}
arm() { # runname label dev tp root script args...
  local rn=$1 lb=$2 dev=$3 tp=$4 root=$5 s=$6; shift 6
  for attempt in 1 2; do
    n=$rn; [ $attempt = 2 ] && n=$rn-relaunch
    tools/launch.sh $n $dev 8300 $tp $s "$@"
    if tools/wait_healthy.sh $n 8300 1800; then plans $root $lb; tools/stop.sh $n; sleep 15; return 0; fi
    tools/stop.sh $n; sleep 15
  done; echo "$rn FAILED_TO_START"
}
arm wl38-mux-tp2 mux43 0,1 2 wl38-tp2 $MUX --served-model-name qwen3.5-35b --seed 0 --max-num-seqs 16 --compilation-config "$G16"
arm wl38-nat-tp2 native 0,1 2 wl38-tp2 $NAT --max-num-seqs 16 --max-num-batched-tokens 16384 --compilation-config "$G16"
arm wl38-mux-tp4 mux43 0,1,2,3 4 wl38-tp4 $MUX --served-model-name qwen3.5-35b --seed 0 --max-num-seqs 16 --compilation-config "$G16"
arm wl38-nat-tp4 native 0,1,2,3 4 wl38-tp4 $NAT --max-num-seqs 16 --max-num-batched-tokens 16384 --compilation-config "$G16"
echo WL38_DONE
bash /root/workspace/runtime/npu-task.sh start --duty 0.2 --size 2048; echo "$(date -u +%FT%TZ) keepalive restart rc=$?"
