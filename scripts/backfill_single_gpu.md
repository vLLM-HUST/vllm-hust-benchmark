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
# python3 scripts/backfill_single_gpu.py fill
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

### 选项

| 选项 | 说明 |
|------|------|
| `--workload NAME` | 指定 workload（可选，省略则补全所有缺失 workload） |
| `--force` | 重新运行已完成的 cell |
| `--fail-fast` | 遇到第一个失败停止 |
| `--npu-device N` | 指定 NPU 设备索引 |

### 示例

```bash
# 一键补全所有 commit 的缺失 workload
python3 scripts/backfill_single_gpu.py fill

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