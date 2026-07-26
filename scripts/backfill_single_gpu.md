# backfill_single_gpu.py

一键生成、聚合、提交 single-GPU 910B2 benchmark 数据。

## 前置条件

- `vllm-hust` 和 `vllm-ascend-hust` 在同一父目录
- Ascend 环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- 数据集：`/data/shared_datasets/vllm-hust-benchmark/current-benchmark-datasets/ShareGPT_V3_unfiltered_cleaned_split.json`

## Python 解释器自动发现

`python3` 调用时若缺少 `vllm_hust_benchmark` 包，自动重执行到 `BACKFILL_PYTHON` → `~/miniconda3/envs/vllm-hust-dev/bin/python` → `sys.executable`。

## 子命令

| 子命令 | 用途 | 需要 NPU |
|--------|------|----------|
| `plan` | 列出缺失 cell，commit 列表来自 `leaderboard_single.json` | 否 |
| `fill` | **一键补全**所有 commit 的缺失 workload（plan + run 组合） | **是** |
| `status` | 查看 checkpoint 进度 | 否 |
| `validate` | 验证 submissions 和 snapshots | 否 |
| `aggregate` | 从 submissions/ 重建 snapshots | 否 |
| `run` | 执行 benchmark | **是** |
| `push` | stage + commit + push | 否 |
| `restore` | 恢复原始 HEAD | 否 |

## 快速开始

```bash
cd /root/vllm/vllm-hust-benchmark
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 查看缺失的 benchmark 数据
python3 scripts/backfill_single_gpu.py plan
python3 scripts/backfill_single_gpu.py plan --group

# 查看 checkpoint 进度
python3 scripts/backfill_single_gpu.py status

# 验证 submissions 和 snapshots
python3 scripts/backfill_single_gpu.py validate

# 一键补全所有缺失的 benchmark 数据（需要 NPU）
python3 scripts/backfill_single_gpu.py fill
```

## 参数处理规范

| 参数 | 是否可选 | 默认值 | 说明 |
|------|----------|--------|------|
| `--commit` | 可选 | `latest` | vllm-hust commit，未提供时解析为 `origin/main` 最新 commit |
| `--ascend-commit` | 可选 | `latest` | vllm-ascend-hust commit，未提供时解析为 `origin/main` 最新 commit |
| `--workload` | 可选 | 全部缺失 workload | 指定 workload 名称，未提供时自动补全所有缺失的 workload |

## run — 执行 benchmark

`--commit` 和 `--ascend-commit` 均可选，未提供时默认为 `latest`（解析为对应仓库的 `origin/main` 最新 commit）。`--workload` 可选：指定则单跑，省略则自动补全该 commit 所有缺失的 workload。

### 选项

| 选项 | 说明 |
|------|------|
| `--commit SHA` | vllm-hust commit（可选，默认 latest → origin/main） |
| `--ascend-commit SHA` | ascend 插件 commit（可选，默认 latest → origin/main） |
| `--workload NAME` | 指定 workload（可选，省略则补全所有缺失 workload） |
| `--force` | 重新运行已完成的 cell |
| `--fail-fast` | 遇到第一个失败停止 |
| `--npu-device N` | 指定 NPU 设备索引 |

### 示例

```bash
# 默认 latest（两者都自动解析为 origin/main），补全所有缺失 workload
python3 scripts/backfill_single_gpu.py run

# 指定 commit，ascend 自动解析为 latest origin/main
python3 scripts/backfill_single_gpu.py run \
    --commit 51621c35b --workload random-latency

# 补全指定 commit 所有缺失 workload（ascend 自动解析为 latest origin/main）
python3 scripts/backfill_single_gpu.py run --commit 51621c35b

# 指定 ascend commit，vllm-hust 自动解析为 latest origin/main
python3 scripts/backfill_single_gpu.py run \
    --ascend-commit 03a12f9 --workload random-latency

# 两个都指定
python3 scripts/backfill_single_gpu.py run \
    --commit 83cf83f --ascend-commit 03a12f9 --workload random-latency
```

### 支持的 workload

```
random-latency    sharegpt-throughput   sonnet-throughput
random-online     sharegpt-online       prefix-repetition-online
instructcoder-online
```

## plan — 查看缺失 cell

commit 列表从数据文件 `/root/vllm/vllm-hust-benchmark/leaderboard-data/snapshots/leaderboard_single.json` 中提取（`metadata.git_commit` 字段），确保与基准数据保持一致。`--group` 按 commit 分组。

```bash
python3 scripts/backfill_single_gpu.py plan
python3 scripts/backfill_single_gpu.py plan --group
```

输出说明：
- `skip` — 已存在
- `MISSING` — 需要运行
- `NOT-FOUND` — commit 在本地 repo 中不存在
- `[non-main]` — commit 不在 `origin/main` 分支上

### commit sha 获取机制

执行 `plan` 时，两个仓库的 commit sha 值均从 `leaderboard_single.json` 中包含的 commit sha 列表中获取，确保与基准数据保持一致。`latest` 则解析为对应仓库 `origin/main` 分支的最新 commit。

## fill — 一键补全所有缺失数据

遍历 `leaderboard_single.json` 中所有 commit，为每个 commit 自动解析 ascend 插件 commit 并运行所有缺失的 workload。相当于 `plan` 的发现能力 + `run` 的执行能力。

### 自动跳过策略

`fill` 会自动跳过以下三种情况，无需手动干预：

1. **已存在（skip）** — 该 commit 在该 workload 下已有数据，无需重复运行
2. **NOT-FOUND** — 该 commit 在本地 vllm-hust 仓库中不存在，无法 checkout
3. **non-main** — 该 commit 不在 `origin/main` 分支上，属于非主线分支提交

### 选项

| 选项 | 说明 |
|------|------|
| `--workload NAME` | 指定 workload（可选，省略则补全所有缺失 workload） |
| `--force` | 重新运行已完成的 cell |
| `--fail-fast` | 遇到第一个失败停止 |
| `--npu-device N` | 指定 NPU 设备索引 |

### 示例

```bash
# 一键补全所有 commit 的缺失 workload（自动跳过已存在、NOT-FOUND、non-main）
python3 scripts/backfill_single_gpu.py fill
nohup python3 scripts/backfill_single_gpu.py fill > backfill.log 2>&1 &

# 只补全 random-latency 在所有 commit 中的缺失
python3 scripts/backfill_single_gpu.py fill --workload random-latency

# 强制重新运行所有已完成的 cell
python3 scripts/backfill_single_gpu.py fill --force
```

## 其他子命令

```bash
python3 scripts/backfill_single_gpu.py aggregate
python3 scripts/backfill_single_gpu.py push --dry-run
python3 scripts/backfill_single_gpu.py push
python3 scripts/backfill_single_gpu.py restore
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BACKFILL_PYTHON` | `~/miniconda3/envs/vllm-hust-dev/bin/python` | Python 解释器路径 |
| `SAME_SPEC_GPU_MEMORY_UTILIZATION` | `0.6` | same-spec hash 的 gpu_memory_utilization |
| `SAME_SPEC_MAX_MODEL_LEN` | `30720` | same-spec hash 的 max_model_len |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像 |
| `VLLM_USE_V1` | `1` | 启用 V1 引擎 |


## 已知问题与修复

| 问题 | 修复 | 说明 |
|------|------|------|
| `fuser` 命令不存在 | `_kill_port_process()` 函数 | 此环境缺少 `psmisc` 包（`fuser` 命令），使用纯 Python 读取 `/proc/net/tcp` 替代。已在 `run_cell()`、`cmd_run()`、`cmd_fill()` 三处替换。 |

### `_kill_port_process()` 实现

通过读取 `/proc/net/tcp` 查找指定端口上的进程并发送 SIGKILL。

```python
def _kill_port_process(port: int) -> None:
    """Kill any process holding the given TCP port using /proc/net/tcp."""
    try:
        with open("/proc/net/tcp") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                local_addr = parts[1]  # e.g. 00000000:1F40
                if ":" not in local_addr:
                    continue
                hex_port = local_addr.split(":")[1]
                if int(hex_port, 16) == port:
                    pid = int(parts[9], 16)
                    if pid > 0:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except OSError:
                            pass
    except OSError:
        pass
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

## Plugin commit consistency guard

`backfill_single_gpu.py` 在 `cmd_run` 解析出 plugin commit 后，以及 `run_cell`
真正写 submission 之前，会调用 `assert_plugin_commit_consistent()` 拦截
「同一 vllm-hust commit 配上两个不同的 vllm-ascend-hust plugin commit」这一类
错误。错误模式见 `docs/HISTORICAL_PR_BACKFILL.md` 的 “Plugin commit alignment rule”
（a46abb7 案例就是这种 split）。

### Canonical 来源

- Canonical plugin commit = **snapshot 中已有的、针对该 `metadata.git_commit`
  最早提交的那条 leaderboard entry 的 `runtime_provenance.plugin.commit`**。
- 首次跑某个 vllm-hust commit 时 snapshot 中没有任何记录 → 没有可比对的
  canonical，guard 直接 pass，由首次跑的结果自然成为后续 canonical。
- Snapshot miss + time-align fallback 选中的 plugin commit 也会作为 entry 写入
  新 submission，进而成为下次的 canonical。

### `--force-mismatched-plugin-commit`

`run` 子命令专属 flag。仅当确有「需要把同一 vllm-hust commit 重新绑定到另一个
plugin commit」的极端场景（如 snapshot 数据污染、plugin 路径故意回退实验）才使用。
使用时三元组 `(hust_commit, canonical_plugin_commit, override_plugin_commit)`
会被追加写到 `.benchmarks/backfill-single-gpu/state.json` 的
`audit.plugin_override` 列表中，便于事后追溯。

```bash
python scripts/backfill_single_gpu.py run \
  --commit a46abb7ae \
  --ascend-commit 03a12f9bdd \
  --force-mismatched-plugin-commit
```

`fill` 子命令没有这个 flag，因为它是「按 snapshot 一致」为前提的全自动模式，
任何不一致都应当人工处理而不是自动 override。

### `plan` 输出中的警告

`plan` 永远只读、永不 fail，但当 snapshot canonical 与 chain 解析结果不一致
（即 `run` 会拒绝）时，会在对应 commit 块的 plugin 预览行下加一行 `⚠`：

```
[a46abb7ae] (6/7 present)
  → plugin 03a12f9bdd (via fallback-head)
  ⚠ plugin mismatch: snapshot canonical=f430530ad resolved=03a12f9bd; run would abort unless --force-mismatched-plugin-commit is set
```

看到 ⚠ 时优先确认是否真的需要 override，多数情况应当先修正环境使 chain 走
snapshot 命中（例如把 snapshot 更新到包含正确 plugin commit 的状态）。
