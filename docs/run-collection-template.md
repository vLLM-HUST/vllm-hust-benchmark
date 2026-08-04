# 运行与采集模板 (Run & Collection Templates)

> **关联任务**: T19 — 准备运行与采集模板\
> **状态**: Done\
> **前置**: T18 (固定真实运行基线)\
> **后置**: T20, T21, T23, T25, T27, T29\
> **最后更新**: 2026-07-25

______________________________________________________________________

## 目录

1. [设计原则](#1-%E8%AE%BE%E8%AE%A1%E5%8E%9F%E5%88%99)
1. [Artifact 目录约定](#2-artifact-%E7%9B%AE%E5%BD%95%E7%BA%A6%E5%AE%9A)
1. [采集文件清单](#3-%E9%87%87%E9%9B%86%E6%96%87%E4%BB%B6%E6%B8%85%E5%8D%95)
1. [运行模板](#4-%E8%BF%90%E8%A1%8C%E6%A8%A1%E6%9D%BF)
1. [单次运行脚本：run-single-repetition.sh](#5-%E5%8D%95%E6%AC%A1%E8%BF%90%E8%A1%8C%E8%84%9A%E6%9C%AC-run-single-repetitionsh)
1. [批次运行脚本：run-campaign-repetitions.sh](#6-%E6%89%B9%E6%AC%A1%E8%BF%90%E8%A1%8C%E8%84%9A%E6%9C%AC-run-campaign-repetitionsh)
1. [环境采集脚本：collect-run-artifact.sh](#7-%E7%8E%AF%E5%A2%83%E9%87%87%E9%9B%86%E8%84%9A%E6%9C%AC-collect-run-artifactsh)
1. [校验脚本：validate-run-artifact.sh](#8-%E6%A0%A1%E9%AA%8C%E8%84%9A%E6%9C%AC-validate-run-artifactsh)
1. [完整使用示例](#9-%E5%AE%8C%E6%95%B4%E4%BD%BF%E7%94%A8%E7%A4%BA%E4%BE%8B)
1. [验收清单](#10-%E9%AA%8C%E6%94%B6%E6%B8%85%E5%8D%95)

______________________________________________________________________

## 1. 设计原则

| 原则           | 说明                                                    |
| -------------- | ------------------------------------------------------- |
| **禁止覆盖**   | 任何已有 STATUS=OK 的 artifact 目录不得被新运行覆盖     |
| **独立目录**   | 每次独立服务进程写入独立 artifact 目录                  |
| **环境可追溯** | 每次运行附带完整的 OS/Python/依赖/CANN 环境快照         |
| **完整性校验** | 每次运行计算 artifact 目录内所有文件的 SHA256 校验和    |
| **失败标记**   | 运行失败时写入 STATUS=FAILED，producer 据此跳过无效数据 |
| **人工可读**   | STATUS 文件为纯文本，可直接 `cat` 查看运行状态          |

### 禁止覆盖机制

```
前置检查: artifact 目录是否存在且 STATUS=OK?
  ├── 是 → 拒绝运行，输出错误并退出 (exit 3)
  └── 否 → 允许运行（覆盖不完整/失败的 artifact）
```

### 与 T18 的关系

T18 定义了三个 campaign 的版本锁定和 spec 文件。T19 提供运行这些 spec 的**采集模板**——确保每次运行产出的 artifact 满足 producer
消费要求，并且重跑不会丢失旧数据。

______________________________________________________________________

## 2. Artifact 目录约定

### 命名格式

```
submissions/<campaign-prefix>-<workload-name>-<chip-count>chip-<timestamp>/
```

| 段                | 来源                       | 示例                  |
| ----------------- | -------------------------- | --------------------- |
| `campaign-prefix` | 人工指定                   | `full-stack-jul-2026` |
| `workload-name`   | spec 文件中的 `scenario`   | `random-online`       |
| `chip-count`      | spec 文件中的 `chip_count` | `2`                   |
| `timestamp`       | ISO 8601 紧凑格式          | `20260725T120000Z`    |

### 等效存储位置

所有最终 artifact 写入 `submissions/` 目录（与现有历史数据一致）。 运行时状态写入
`.benchmarks/<campaign-prefix>/<artifact-dir-name>/`， 完成后仅将最终产物复制到 `submissions/`。

### 输入目录结构（运行时状态）

```
.benchmarks/<campaign-prefix>/<artifact-dir-name>/
├── raw_benchmark_result.json      # 原始 benchmark 输出
├── resolved_same_spec.json        # 解析后的 same-spec
├── server.stdout.log              # 服务端 stdout（serve 类型）
├── offline_graph_proof.json       # graph mode 验证证据（offline 类型）
├── submission/
│   ├── run_leaderboard.json       # leaderboard artifact（主产物）
│   └── leaderboard_manifest.json  # artifact manifest
```

### 输出目录结构（最终 artifact）

```
submissions/<campaign-prefix>-<workload-name>-<chip-count>chip-<timestamp>/
├── run_leaderboard.json           # 主 artifact（由 export-leaderboard-artifact 产生）
├── leaderboard_manifest.json      # manifest，引用 run_leaderboard.json
├── env-manifest.json              # 环境快照（OS/Python/env vars/conda/CANN/git）
├── pip-packages.json              # pip 包列表（JSON 格式）
├── checksums.sha256               # 目录内所有文件的 SHA256
├── STATUS                         # "OK" 或 "FAILED: <reason>"
├── server.stdout.log              # 服务端 stdout
├── raw_benchmark_result.json      # 原始 benchmark 输出
├── resolved_same_spec.json        # resolved same-spec
└── offline_graph_proof.json       # graph mode 验证（如有）
```

______________________________________________________________________

## 3. 采集文件清单

### 3.1 `STATUS` — 运行状态

纯文本文件，内容为单行：

```
OK
```

或失败时：

```
FAILED: benchmark exit code 1
```

Producer 在消费 artifact 前必须检查 `STATUS` 内容。只有 `OK` 的 artifact 可进入趋势数据。

### 3.2 `env-manifest.json` — 环境快照

记录运行时的系统环境，包含：

| 字段               | 来源                          | 用途                                                                    |
| ------------------ | ----------------------------- | ----------------------------------------------------------------------- |
| `manifest_version` | 固定值 `run-env-manifest/v2`  | schema 版本标识                                                         |
| `collected_at`     | `date -u +%Y-%m-%dT%H:%M:%SZ` | 采集时间戳                                                              |
| `os`               | `uname -a`                    | 内核版本、架构                                                          |
| `python_version`   | `python3 --version`           | Python 版本                                                             |
| `hostname`         | `hostname`                    | 机器标识                                                                |
| `conda_env`        | `$CONDA_DEFAULT_ENV`          | conda 环境名                                                            |
| `ascend_toolkit`   | 文件检测                      | CANN 安装路径                                                           |
| `npu_smi`          | `npu-smi info -t board`       | NPU 硬件信息                                                            |
| `env_vars`         | 关键环境变量                  | PATH, LD_LIBRARY_PATH, PYTHONPATH, HF_HOME, VLLM_CACHE_ROOT, ASCEND\_\* |
| `git_info`         | git rev-parse + 显式环境变量  | core/backend 的 declared/observed SHA 与 benchmark SHA                  |
| `frozen_inputs`    | 显式环境变量 + runtime 探测   | image、model revision、CANN、torch-npu 与拓扑                           |
| `campaign`         | campaign runner               | campaign/comparison/role/load profile 与独立服务 repetition 序号        |
| `pip_packages`     | pip list（引用外部文件）      | 详见 pip-packages.json                                                  |

### 3.3 `checksums.sha256` — 文件校验和

标准 SHA256 校验和文件，每行格式：

```
<sha256hex>  <filename>
```

由 `sha256sum` 生成，可用 `sha256sum -c checksums.sha256` 校验。

注意：`checksums.sha256` 和 `STATUS` 自身不包含在校验和中（循环依赖）。

### 3.4 `pip-packages.json` — Python 依赖快照

`pip list --format=json` 的直接输出，便于 producer 解析。

______________________________________________________________________

## 4. 运行模板

T19 提供三个层级的运行模板，从底层到高层：

```
run-campaign-repetitions.sh    ← 3x 重复循环 + cooldown
        │
        └── run-single-repetition.sh    ← 单次运行包装 + 采集
                │
                ├── run-current-ascend-same-spec.sh    ← 现有 benchmark runner
                └── collect-run-artifact.sh            ← 后处理采集
```

### 调用关系

```
用户调用:
  run-campaign-repetitions.sh <spec> --campaign-prefix <prefix> --repetitions 3
    │
    ├── (重复 3 次)
    │   └── run-single-repetition.sh <spec> <prefix> <index>
    │         │
    │         ├── run-current-ascend-same-spec.sh <spec>
    │         │     └── python -m vllm_hust_benchmark.cli export-leaderboard-artifact ...
    │         │
    │         └── collect-run-artifact.sh <artifact-dir> [--mark-failed <reason>]
    │               ├── env-manifest.json
    │               ├── checksums.sha256
    │               └── STATUS
    │
    └── port cooldown between repetitions
```

______________________________________________________________________

## 5. 单次运行脚本：run-single-repetition.sh

### 用法

```bash
bash scripts/run-single-repetition.sh <spec-file> <campaign-prefix> <run-index>
```

### 参数

| 参数              | 说明               | 示例                                                                                    |
| ----------------- | ------------------ | --------------------------------------------------------------------------------------- |
| `spec-file`       | spec JSON 文件路径 | `docs/official-baselines/full-stack-jul-2026-random-online-qwen25-14b-2chip-910b2.json` |
| `campaign-prefix` | Campaign 标识前缀  | `full-stack-jul-2026`                                                                   |
| `run-index`       | 重复序号（1-3）    | `1`                                                                                     |

### 环境变量

透传给 `run-current-ascend-same-spec.sh` 的标准变量：

| 变量                        | 说明                    |
| --------------------------- | ----------------------- |
| `CURRENT_SUBMITTER`         | 提交者标识              |
| `CURRENT_DATA_SOURCE`       | 数据来源                |
| `CURRENT_GIT_COMMIT`        | vllm-hust commit        |
| `CURRENT_PLUGIN_GIT_COMMIT` | vllm-ascend-hust commit |
| `CURRENT_ENGINE_VERSION`    | 引擎版本号              |

### 流程

```
1. 解析 spec → workload name, chip count
2. 构造 artifact 目录名: <prefix>-<workload>-<chip>chip-<timestamp>
3. 前置检查: 目录是否已存在且 STATUS=OK? → 拒绝
4. 创建运行时目录 (.benchmarks/<prefix>/<dir>/)
5. 设置 RUN_ID, RESULT_DIR
6. 调用 run-current-ascend-same-spec.sh
7. 复制产物到 submissions/<dir>/
8. 调用 collect-run-artifact.sh (成功/失败分别处理)
```

______________________________________________________________________

## 6. 批次运行脚本：run-campaign-repetitions.sh

### 用法

```bash
bash scripts/run-campaign-repetitions.sh <spec-file> \
  [--campaign-prefix <prefix>] \
  [--repetitions N] \
  [--cooldown <seconds>]
```

### 参数

| 参数                | 默认值             | 说明                       |
| ------------------- | ------------------ | -------------------------- |
| `--campaign-prefix` | 从 spec 文件名推断 | Campaign 前缀              |
| `--repetitions`     | `3`                | 重复运行次数               |
| `--cooldown`        | `60`               | 每次重复间的冷却时间（秒） |

### 流程

```
1. 解析参数
2. 循环 REPETITIONS 次:
   a. 非首次运行: cooldown 等待 + 端口释放等待
   b. 调用 run-single-repetition.sh
   c. 记录成功/失败
3. 输出摘要
```

### 端口释放策略

`run-current-ascend-same-spec.sh` 在 server 关闭后自动清理端口。批次脚本额外等待：

- `COOLDOWN_SECONDS`（默认 60s）给 NPU 显存释放时间
- 最多 `MAX_PORT_WAIT_SECONDS`（默认 120s）等待端口监听消失

______________________________________________________________________

## 7. 环境采集脚本：collect-run-artifact.sh

### 用法

```bash
bash scripts/collect-run-artifact.sh <artifact-dir> [--mark-failed <reason>]
```

### 功能

- **环境快照**：捕获 OS、Python、pip 依赖、CANN 路径、NPU 信息、git commits
- **校验和**：对目录内所有文件计算 SHA256
- **状态标记**：正常完成写入 `OK`；失败时写入 `FAILED: <reason>`

### 输出文件

| 文件                | 始终生成           |
| ------------------- | ------------------ |
| `env-manifest.json` | ✅                 |
| `pip-packages.json` | ✅                 |
| `checksums.sha256`  | ✅                 |
| `STATUS`            | ✅（OK 或 FAILED） |

### 参数

| 参数                     | 说明                         |
| ------------------------ | ---------------------------- |
| `--mark-failed <reason>` | 标记为失败状态，记录失败原因 |

______________________________________________________________________

## 8. 校验脚本：validate-run-artifact.sh

### 用法

```bash
bash scripts/validate-run-artifact.sh <artifact-dir>
```

### 校验项

| #   | 检查项                                                    | 失败影响                 |
| --- | --------------------------------------------------------- | ------------------------ |
| 1   | STATUS 文件存在且内容为 "OK"                              | Producer 跳过此 artifact |
| 2   | run_leaderboard.json 存在且为合法 JSON                    | Producer 无法解析        |
| 3   | leaderboard_manifest.json 存在且引用 artifact             | 聚合失败                 |
| 4   | env-manifest.json 存在且包含必要字段                      | 环境不可追溯             |
| 5   | checksums.sha256 存在且所有校验和通过                     | 文件可能损坏             |
| 6   | run_leaderboard.json 通过 artifact contract normalization | schema 不兼容            |

### 返回值

| 退出码 | 含义         |
| ------ | ------------ |
| 0      | 所有检查通过 |
| 1+     | 发现错误数   |

______________________________________________________________________

## 9. 完整使用示例

### 9.1 Campaign A: full-stack-jul-2026/v1 (random-online, 2-chip, 3x)

```bash
cd /workspace/vllm-hust-benchmark

# 设置版本锁定（T18 manifest）
export CURRENT_GIT_COMMIT=5536d0873fb41c4925d0e6e9112a1ea70faeeb3a
export CURRENT_PLUGIN_GIT_COMMIT=b42a66b63b73ceda32fb8983edf7de3c69cce516
export CURRENT_ENGINE_VERSION=v0.23.1rc0-1327-g5536d0873f
export CURRENT_SUBMITTER=full-stack-jul-2026
export CURRENT_DATA_SOURCE=full-stack-jul-2026

# 3 次重复运行 random-online 2-chip
bash scripts/run-campaign-repetitions.sh \
  docs/official-baselines/full-stack-jul-2026-random-online-qwen25-14b-2chip-910b2.json \
  --campaign-prefix full-stack-jul-2026 \
  --repetitions 3 \
  --cooldown 60

# 验证所有 3 个 artifact
for dir in submissions/full-stack-jul-2026-random-online-2chip-*; do
  bash scripts/validate-run-artifact.sh "$dir"
done
```

### 9.2 Campaign A: full-matrix (all 5 workloads, 2-chip)

```bash
cd /workspace/vllm-hust-benchmark

export CURRENT_GIT_COMMIT=5536d0873fb41c4925d0e6e9112a1ea70faeeb3a
export CURRENT_PLUGIN_GIT_COMMIT=b42a66b63b73ceda32fb8983edf7de3c69cce516
export CURRENT_ENGINE_VERSION=v0.23.1rc0-1327-g5536d0873f
export CURRENT_SUBMITTER=full-stack-jul-2026
export CURRENT_DATA_SOURCE=full-stack-jul-2026

SPEC_DIR=docs/official-baselines

# 2-chip workloads
for spec in \
  full-stack-jul-2026-agent-research-online-qwen25-14b-2chip-910b2.json \
  full-stack-jul-2026-prefix-repetition-online-qwen25-14b-2chip-910b2.json \
  full-stack-jul-2026-random-online-qwen25-14b-2chip-910b2.json \
  full-stack-jul-2026-sharegpt-online-qwen25-14b-2chip-910b2.json \
  full-stack-jul-2026-sonnet-throughput-qwen25-14b-2chip-910b2.json; do

  bash scripts/run-campaign-repetitions.sh \
    "$SPEC_DIR/$spec" \
    --campaign-prefix full-stack-jul-2026 \
    --repetitions 3 \
    --cooldown 60
done
```

### 9.3 Campaign A: 4-chip (same as above, with 4chip specs)

```bash
# 仅 spec 文件名不同（4chip 替换 2chip）
export CURRENT_SUBMITTER=full-stack-jul-2026
export CURRENT_DATA_SOURCE=full-stack-jul-2026

for spec in \
  full-stack-jul-2026-agent-research-online-qwen25-14b-4chip-910b2.json \
  full-stack-jul-2026-prefix-repetition-online-qwen25-14b-4chip-910b2.json \
  full-stack-jul-2026-random-online-qwen25-14b-4chip-910b2.json \
  full-stack-jul-2026-sharegpt-online-qwen25-14b-4chip-910b2.json \
  full-stack-jul-2026-sonnet-throughput-qwen25-14b-4chip-910b2.json; do

  bash scripts/run-campaign-repetitions.sh \
    "$SPEC_DIR/$spec" \
    --campaign-prefix full-stack-jul-2026 \
    --repetitions 3 \
    --cooldown 60
done
```

### 9.4 Campaign B: targeted-pair (fullgraph head, 2-chip)

```bash
export CURRENT_GIT_COMMIT=5536d0873fb41c4925d0e6e9112a1ea70faeeb3a
export CURRENT_PLUGIN_GIT_COMMIT=b42a66b63b73ceda32fb8983edf7de3c69cce516
export CURRENT_ENGINE_VERSION=v0.23.1rc0-1327-g5536d0873f
export CURRENT_SUBMITTER=targeted-pair-jul-2026
export CURRENT_DATA_SOURCE=targeted-pair-jul-2026

# T23: fullgraph-split-online head
bash scripts/run-campaign-repetitions.sh \
  docs/official-baselines/targeted-pair-jul-2026-fullgraph-split-online-qwen25-14b-2chip-910b2.json \
  --campaign-prefix targeted-pair-jul-2026 \
  --repetitions 3

# T25: ngram-instructcoder-online base
bash scripts/run-campaign-repetitions.sh \
  docs/official-baselines/targeted-pair-jul-2026-ngram-instructcoder-online-qwen25-14b-2chip-910b2.json \
  --campaign-prefix targeted-pair-jul-2026 \
  --repetitions 3
```

### 9.5 Campaign C: upstream-ref (instructcoder-online, 1-chip)

```bash
# 使用 upstream v0.18.0 环境
export CURRENT_SUBMITTER=upstream-ref-jul-2026
export CURRENT_DATA_SOURCE=upstream-ref-jul-2026

bash scripts/run-campaign-repetitions.sh \
  docs/official-baselines/official-ascend-jan-2026-v0180-instructcoder-online-qwen25-coder-14b-910b2.json \
  --campaign-prefix upstream-ref-jul-2026 \
  --repetitions 3
```

### 9.6 手动单次运行 + 验证

```bash
# 单次运行
bash scripts/run-single-repetition.sh \
  docs/official-baselines/full-stack-jul-2026-random-online-qwen25-14b-2chip-910b2.json \
  full-stack-jul-2026 \
  1

# 验证
bash scripts/validate-run-artifact.sh \
  submissions/full-stack-jul-2026-random-online-2chip-20260725T120000Z/

# 查看状态
cat submissions/full-stack-jul-2026-random-online-2chip-20260725T120000Z/STATUS
```

______________________________________________________________________

## 10. 验收清单

| #   | 验收项                                       | 验证方法                                                                                |
| --- | -------------------------------------------- | --------------------------------------------------------------------------------------- |
| 1   | 重复运行产生独立目录，不覆盖已有 OK artifact | 运行 2 次，确认 `submissions/` 下有 2 个不同时间戳的目录；第 3 次对同一目录重跑时被拒绝 |
| 2   | 每次运行包含 env-manifest.json               | `validate-run-artifact.sh` 检查通过                                                     |
| 3   | 每次运行包含 checksums.sha256                | 同上                                                                                    |
| 4   | STATUS=OK 的 artifact 可通过全量校验         | `validate-run-artifact.sh` 返回 0                                                       |
| 5   | 失败运行产生 STATUS=FAILED                   | 人为制造失败场景，确认 STATUS 内容不是 OK                                               |
| 6   | 失败 artifact 不堵塞后续运行                 | 即使前一次失败，下次运行仍可正常创建新目录                                              |
| 7   | Artifact 可被 producer 消费                  | `run_leaderboard.json` 通过 contract normalization                                      |
| 8   | 批次运行产出 3 次有效 artifact               | `run-campaign-repetitions.sh --repetitions 3` 产出 3 个 STATUS=OK 目录                  |
| 9   | 重跑不覆盖旧数据                             | 对已有 STATUS=OK 的目录再次运行 `run-single-repetition.sh` 被拒绝（exit 3）             |
| 10  | 校验和在文件不变时一致                       | 对同一目录连续运行 2 次 `collect-run-artifact.sh`，`checksums.sha256` 结果相同          |

______________________________________________________________________

## 附录 A：文件索引

| 文件                          | 路径                                  | 描述               |
| ----------------------------- | ------------------------------------- | ------------------ |
| `run-single-repetition.sh`    | `scripts/run-single-repetition.sh`    | 单次重复包装       |
| `run-campaign-repetitions.sh` | `scripts/run-campaign-repetitions.sh` | 批次重复循环       |
| `collect-run-artifact.sh`     | `scripts/collect-run-artifact.sh`     | 后处理采集脚本     |
| `validate-run-artifact.sh`    | `scripts/validate-run-artifact.sh`    | Artifact 校验脚本  |
| 本文档                        | `docs/run-collection-template.md`     | 运行与采集模板文档 |

## 附录 B：与现有系统对比

| 维度          | 现有系统 (run-current-ascend-same-spec.sh)              | T19 增强                                                              |
| ------------- | ------------------------------------------------------- | --------------------------------------------------------------------- |
| Artifact 目录 | 固定 `.benchmarks/current-ascend-same-spec/submission/` | 每次运行独立目录 `submissions/<campaign>-<workload>-<chip>chip-<ts>/` |
| 覆盖保护      | 无（下次运行覆盖前次结果）                              | 严格禁止覆盖 STATUS=OK 的目录                                         |
| 环境快照      | 无                                                      | `env-manifest.json` + `pip-packages.json`                             |
| 校验和        | 无                                                      | `checksums.sha256`                                                    |
| 失败标记      | 无（失败时目录可能为空或不完整）                        | `STATUS` 文件明确标记 OK/FAILED                                       |
| 批量重复      | 手动循环                                                | `run-campaign-repetitions.sh` 自动管理                                |
| 冷却机制      | 无                                                      | 内置 cooldown + port 释放等待                                         |
