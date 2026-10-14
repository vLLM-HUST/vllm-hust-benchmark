#!/bin/bash
set -eo pipefail
source /home/lumi-deepseek-baseline/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_IF_BASE_PORT=29620 HCCL_HOST_SOCKET_PORT_RANGE=29620-29651 HCCL_SOCKET_IFNAME='=bond0'
exec python -m vllm.entrypoints.cli.main serve /mnt/lumi-model-disk2/lumi-models/DeepSeek-V4-Flash-0731-w8a8 \
 --host 127.0.0.1 --port 29610 --served-model-name dsv4 \
 --max-model-len 32768 --max-num-batched-tokens 10240 --max-num-seqs 64 \
 --gpu-memory-utilization 0.9 --data-parallel-size 4 --tensor-parallel-size 4 \
 --enable-expert-parallel --tokenizer-mode deepseek_v4 \
 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
 --no-enable-prefix-caching \
 --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}' \
 --quantization ascend --block-size 32 \
 --speculative-config '{"method":"dspark","num_speculative_tokens":7,"enforce_eager":true}' \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
 --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_dsa_cp":true,"multistream_overlap_shared_expert":true}'
