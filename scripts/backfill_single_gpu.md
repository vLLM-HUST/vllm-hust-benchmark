

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
- 自定义模型路径
已支持：
    - --model <path_or_name> ：指定模型名称或本地路径（默认 `Qwen/Qwen2.5-14B-Instruct`）✅
- 自定义配置
已支持：
    - --additional-config <json> ：传入 vllm `--additional-config` 参数，支持 vllm-ascend 特定配置（如 split_batch_config）✅
        - 对 latency/throughput/serve 三种场景均生效 ✅
    - --compilation-config <json> ：传入 vllm `--compilation-config` 参数（如 cudagraph 配置）✅
        - 对 latency/throughput/serve 三种场景均生效 ✅
    - --temperature <float> ：采样温度，serve 场景传 `--temperature`，latency/throughput 传 `--override-generation-config` ✅
- 不支持的 workload
    - `visionarena-online` 和 `agent-research-online` 使用 `openai-chat` backend + `/v1/chat/completions` 端点，需要不同的 CLI 调用方式，暂未集成

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

# 使用本地模型路径
python3 scripts/backfill_single_gpu.py run --only sharegpt-throughput --commit 83cf83f \
  --model /data/shared-models/Qwen2.5-14B-Instruct

# 使用 --additional-config 传入 vllm-ascend 特定配置（split_batch_config 等）
python3 scripts/backfill_single_gpu.py run --only sharegpt-throughput --commit 83cf83f \
  --additional-config '{"split_batch_config":{"enabled":true,"mode":"inplace_parallel","num_splits":2,"enable_parallel_streams":true,"enable_inplace_lazy_capture":true,"inplace_split_planner_policy":"largest_lower","inplace_offset_match_policy":"exact","inplace_parallel_replay_policy":"full_graph_parallel","inplace_offset_capture_sizes":[1,2,4,8,16,32,64],"parallel_capture_sizes":[1,2,4,8,16,32,64]}}'

# 使用 --compilation-config 传入 cudagraph 配置
python3 scripts/backfill_single_gpu.py run --only sharegpt-throughput --commit 83cf83f \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256]}'

# 使用 --temperature 传入温度
python3 scripts/backfill_single_gpu.py run --only sharegpt-throughput --commit 83cf83f \
  --temperature 0.0

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
