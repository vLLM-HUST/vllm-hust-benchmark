#!/bin/bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/lumi-pd27/atb-unpacked/atb/set_env.sh --cxx_abi=1
export LD_LIBRARY_PATH=/home/lumi-pd27/rdma-runtime/root/usr/lib64:/home/lumi-pd27/cxx-runtime/lib:$LD_LIBRARY_PATH
unset LD_PRELOAD ASCEND_CUSTOM_OPP_PATH
export PYTHONPATH=/home/lumi-pd27/a3-wheel:$PYTHONPATH
export PATH=/home/lumi-pd27/serving-venv/bin:$PATH
export OMP_PROC_BIND=false OMP_NUM_THREADS=10 TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_BUFFSIZE=1024 HCCL_OP_EXPANSION_MODE=AIV
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=/home/lumi-deepseek-baseline/cache
export TRITON_CACHE_DIR=/home/lumi-deepseek-baseline/cache/triton
