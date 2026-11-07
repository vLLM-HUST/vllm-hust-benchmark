#!/usr/bin/env bash
# QUALITY-V2-SYNC-TP4.md part B (frozen): final-main 43fcaf4 TP2 checks. B1 default x2 in parallel, then B2 ROW.
set -eo pipefail
cd /root/workspace/muxpd-eval
[ "$(git -C /root/workspace/vllm-ascend-MuxPD-hust rev-parse HEAD)" = 43fcaf4279640bcc086119b0671cb7fd4228836a ] || { echo WRONG_HEAD; exit 1; }
( tools/quality_arm.sh qf-c1 0,1 8300 candidate ) & ( tools/quality_arm.sh qf-c2 2,3 8302 candidate ) & wait
G='{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}'
tools/launch.sh qf-row 0,1 8300 2 /root/workspace/vllm-ascend-MuxPD-hust/scripts/serve-qwen35-muxwise-row.sh --served-model-name qwen3.5-35b --seed 0 --max-num-seqs 16 --compilation-config "$G"
tools/wait_healthy.sh qf-row 8300 1500
source /root/workspace/runtime/env.sh >/dev/null 2>&1
PYTHONPATH=/root/workspace/muxpd-eval/ext/hust-eval-client-mmlu-pro/src python -m hust_eval_mmlu_pro mmlu-pro run --plan /root/workspace/muxwise-runs/plans/fixed200 --base-url http://127.0.0.1:8300 --served-model qwen3.5-35b --output quality/qf-row --concurrency 16 > quality/qf-row.log 2>&1
tools/stop.sh qf-row
echo FINAL_MAIN_TP2_DONE
