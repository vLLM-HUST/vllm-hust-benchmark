


## Purpose
可以一键式 生成，聚合，提交 某一指定版本，指定workload的 benchmark 数据。

## feat
>  vllm-hust跟vllm-ascend-hust要在同一根目录（最好是按照vllm-hust-dev-hub中的setup.

- ascend 插件 commit 默认使用 `vllm-ascend-hust` 最新 `origin/main`，不会命中 `leaderboard-exclusions.json` 的排除列表。
- 自定义 commit 支持
已支持：
    - --commit <sha> ：指定 vllm-hust 的自定义 commit ✅
        - 当 --commit 未指定时：使用 DEFAULT_CELLS （backfill 模式）✅
    - --ascend-commit <sha> ：指定 vllm-ascend-hust 的自定义 commit ✅
        - 当 --ascend-commit 未指定时：自动解析到 origin/main 最新 commit（_resolve_compatible_ascend_commit() ）✅
    - 当 --ascend-commit 指定而 --commit 未指定时，自动使用 vllm-hust 的 origin/main 最新 commit (_resolve_latest_hust_commit)✅

## 规范与验证
所有验证均已通过：
- validate 命令 ：所有 9 个 single-gpu-backfill submissions + 所有 historical submissions + snapshot 文件均 OK ✅
- pytest ：18/18 测试全部通过 ✅
- 黑名单检查 ： _remove_excluded_submissions() 在 aggregate 前自动清理被排除的提交 ✅
- error_rate 检查 ： _check_error_rate() 拒绝 error_rate=1.0 的结果 ✅
- 归一化 ： normalize_submission_artifact_file() 确保提交文件通过归一化测试 ✅


## How to Run

```bash
# setup
cd /root/vllm/vllm-hust-benchmark
source /usr/local/Ascend/ascend-toolkit/set_env.sh


# 查看缺失的 cell
python3 scripts/backfill_single_gpu.py plan

# 跑指定 workload 和 vllm-hust@commit
# workload:
#       sharegpt-throughput， random-latency，sonnet-throughput，
#       Online serving scenarios
#       random-online，sharegpt-online，prefix-repetition-online，instructcoder-online
# random-latency, sharegpt-throughput  83cf83f
python3 scripts/backfill_single_gpu.py run --only sharegpt-throughput --commit a46abb7ae

# 使用指定的 ascend 插件 commit（默认使用 vllm-ascend-hust 最新 origin/main）
python3 scripts/backfill_single_gpu.py run --only sharegpt-throughput --commit 83cf83f --ascend-commit 03a12f9

# 运行所有的missing的workload 和 commit(vllm-hust)
python3 scripts/backfill_single_gpu.py run
nohup python3 scripts/backfill_single_gpu.py run > backfill.log 2>&1 &
# Commit 过滤 ： --commit 指定时不过滤，自动发现时过滤非 main 分支

# 重建 snapshot
python3 scripts/backfill_single_gpu.py aggregate

# 验证所有提交
python3 scripts/backfill_single_gpu.py validate





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

## 手工补点的配置字段检查

手工补点与 CI/CD 使用同一份 `explicit-effective/v1` 配置契约。提交前必须检查
`workload` 的 input/output/batch/concurrency/dataset 字段，以及 `same_spec` 中
解析后的 server/client 参数均已写全；不适用的字段写 JSON `null`，不能直接省略。
`metadata.submitted_at` 必须保留真实生成时间，不能删除或倒填来绕过新记录校验。

特别注意：`num_prompts` 是总请求数，不是并发数，禁止将其写入
`workload.concurrent_requests`。只有真实使用了 `max_concurrency` 或
`concurrent_requests` 时才能填写并发字段。完整清单和校验命令见
`docs/HISTORICAL_PR_BACKFILL.md` 的 “Required effective-configuration metadata”。
