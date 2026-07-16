


## Purpose
可以一键式 生成，聚合，提交 某一指定版本，指定workload的 benchmark 数据。


## How to Run

```bash

# 查看缺失的 cell
python scripts/backfill_single_gpu.py plan

# 跑指定 workload 和 commit
# workload:
#       sharegpt-throughput， random-latency，sonnet-throughput，
#       Online serving scenarios
#       random-online，sharegpt-online，prefix-repetition-online，instructcoder-online
python scripts/backfill_single_gpu.py run --only <workload> --commit <sha>

# 重建 snapshot
python scripts/backfill_single_gpu.py aggregate

# 验证所有提交
python scripts/backfill_single_gpu.py validate