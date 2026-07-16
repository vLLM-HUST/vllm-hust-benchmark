


## Purpose
可以一键式 生成，聚合，提交 某一指定版本，指定workload的 benchmark 数据。


## How to Run

```bash

cd /root/vllm/vllm-hust-benchmark

# 跑一个 commit 的 random-latency
/root/miniconda3/envs/vllm-hust-dev/bin/python scripts/backfill_single_gpu.py run \
  --commit 2206f1f7b7212801187bc001c5f6cb86b2289214 --only random-latency

# 聚合 + 推送到 leaderboard
/root/miniconda3/envs/vllm-hust-dev/bin/python scripts/backfill_single_gpu.py aggregate
/root/miniconda3/envs/vllm-hust-dev/bin/python scripts/backfill_single_gpu.py push -m "feat: backfill data"
```