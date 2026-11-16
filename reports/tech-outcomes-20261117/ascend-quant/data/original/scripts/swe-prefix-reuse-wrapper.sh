#!/usr/bin/env bash
export PYTHONPATH=/data/jxd/swe-prefix-reuse/src:/data/jxd/swe-prefix-reuse/.venv/lib/python3.11/site-packages
exec /data/jxd/envs/vllm-hust-v1-cann91/bin/python /data/jxd/swe-prefix-reuse/.venv/bin/swe-prefix-reuse "$@"
