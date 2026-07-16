


## Purpose
可以一键式 生成，聚合，提交 某一指定版本，指定workload的 benchmark 数据。


## How to Run

```bash
# setup
cd /root/vllm/vllm-hust-benchmark
source /usr/local/Ascend/ascend-toolkit/set_env.sh


# 查看缺失的 cell
python scripts/backfill_single_gpu.py plan

# 跑指定 workload 和 commit
# workload:
#       sharegpt-throughput， random-latency，sonnet-throughput，
#       Online serving scenarios
#       random-online，sharegpt-online，prefix-repetition-online，instructcoder-online
# random-latency, sharegpt-throughput  83cf83f
python3 scripts/backfill_single_gpu.py run --only random-latency --commit 83cf83f

# 重建 snapshot
python scripts/backfill_single_gpu.py aggregate

# 验证所有提交
python scripts/backfill_single_gpu.py validate





## check data
```bach
cd /root/vllm/vllm-hust-benchmark && python3 -c "
import json
with open('leaderboard-data/snapshots/leaderboard_single.json') as f:
    data = json.load(f)
print(f'Total entries: {len(data)}')
commit = '83cf83f'
count = 0
for entry in data:
    meta = entry.get('metadata', {})
    git_commit = meta.get('git_commit', '')
    if commit in git_commit:
        wl = entry.get('workload', {}).get('name', '?')
        submitter = meta.get('submitter', '?')
        eng = entry.get('engine_version', '?')
        print(f'  workload={wl:30s} submitter={submitter:25s} engine={eng}')
        count += 1
print(f'Total: {count}')
"
```