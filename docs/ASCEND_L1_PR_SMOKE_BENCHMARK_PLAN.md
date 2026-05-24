# L1 PR 门禁快速验证（Smoke Benchmark）落地方案

## 1. 目标

本文给出一版结合当前 workspace 和 upstream `vllm` 代码仓约束后的、可直接执行的 L1 PR 性能门禁方案，覆盖：

1. 触发条件
2. 模型选择
3. benchmark 参数
4. 基线定义
5. 失败判定
6. 具体落地步骤

目标不是做“完整性能评估”，而是给 PR 一个最快速、最稳定、最便宜的组合栈回归门禁。这里的被测对象默认不是单个仓，而是 `vllm-hust + vllm-ascend-hust` 的运行时组合。

## 2. 现状约束

### 2.1 当前 workspace 的真实边界

当前仓内已经有三类相近但不等价的机制：

1. `vllm-hust/.github/workflows/linux-ascend-inference-smoke.yml`
   这是功能健康检查，但当前是关闭状态，原因是 CI 效率。
2. `vllm-hust/.github/workflows/linux-ascend-inference-regression.yml`
   这是小规模真实推理回归校验，重点是功能和结果，不是 PR 性能门禁。
3. `vllm-hust/.github/workflows/ascend-benchmark-leaderboard.yml`
   这是 publication-oriented benchmark / leaderboard 流程，偏数据导出和发布，不适合作为 PR 级快速门禁。
4. `vllm-ascend-hust/.github/workflows/ascend-benchmark-leaderboard.yml`
   这个 workflow 已经体现了组合栈语义：它不只跑 `vllm-ascend-hust` 自身，还显式携带 `ASCEND_HUST_TARGET` 和 `VLLM_HUST_REF`，说明当前 workspace 已经有跨仓 benchmark wiring。

另外，当前 `run_ascend_benchmark_ci.sh` 默认使用真实小模型 `Qwen/Qwen2.5-0.5B-Instruct`，且参数极小：

- `BENCH_NUM_PROMPTS=8`
- `BENCH_RANDOM_INPUT_LEN=64`
- `BENCH_RANDOM_OUTPUT_LEN=16`
- `BENCH_MAX_CONCURRENCY=4`

这更像 CI preview / artifact preview，不适合作为稳定的性能门禁基线。

这也意味着，当前这份 L1 文档如果仍然把被测对象理解成单个 `vllm-hust` 仓，就会偏离实际关注对象。需要把门禁的语义从“单仓性能”切到“组合栈性能”。

### 2.2 upstream 的现有代表测试习惯

upstream `vllm` 的 `.buildkite/performance-benchmarks/tests/` 给了两个重要信号：

1. 代表性 dense text performance 测试默认使用 8B dense causal LM。
2. latency / serving / throughput 的 performance 测试广泛使用 `load_format=dummy`，即不加载真实权重，只测框架主路径性能。

这说明“dummy 权重 + dense text 模型 + 少量固定参数”的 L1 思路是成立的，但它测的是框架性能主路径，不是完整真实推理性能。

## 3. 方案范围

### 3.1 L1 要覆盖什么

L1 只覆盖以下能力：

1. `vllm-hust` dense decoder-only text serving 主路径
2. `vllm-hust` scheduler / continuous batching 主路径
3. `vllm-hust` KV cache / PagedAttention 主路径
4. `vllm-ascend-hust` 作为硬件插件接入 `vllm-hust` 的组合路径
5. request queue 和 online benchmark 的基础性能回归

### 3.2 L1 不覆盖什么

以下内容不属于 L1：

1. 多模态
2. prefix repetition 专项
3. structured output
4. tool calling
5. spec decode
6. 真实权重加载速度
7. 真实模型质量或结果一致性

这些应保留给 L2/L3 或现有功能 smoke / regression workflow。

### 3.3 被测单元定义

L1 默认被测对象不是单个仓，而是一份组合栈 manifest：

1. `vllm-hust` 的仓库引用与 commit
2. `vllm-ascend-hust` 的仓库引用与 commit
3. `pairing_strategy`
4. `benchmark_config`

默认配对策略建议如下：

1. 如果 PR 来自 `vllm-hust`：
   - head combo = `vllm-hust@PR_HEAD + vllm-ascend-hust@protected_main`
2. 如果 PR 来自 `vllm-ascend-hust`：
   - head combo = `vllm-ascend-hust@PR_HEAD + vllm-hust@protected_main`
3. 如果是双仓协同改动：
   - 使用 `workflow_dispatch` 或 `repository_dispatch` 显式传入两边的 ref

只有把这份组合栈 manifest 视为真正的被测单元，后续 baseline、ancestor 比较和 latest-main 比较才有正确语义。

## 4. 触发条件

### 4.1 workflow 触发事件

建议拆成“两层 wrapper + 一层共享逻辑”：

1. `vllm-hust` 仓的 PR 门禁 wrapper workflow：
   - `vllm-hust/.github/workflows/linux-ascend-perf-smoke.yml`
2. `vllm-ascend-hust` 仓的 PR 门禁 wrapper workflow：
   - `vllm-ascend-hust/.github/workflows/linux-ascend-perf-smoke.yml`
3. 共享 orchestration 逻辑：
   - 建议沉淀到 `vllm-hust-benchmark`

同理，main baseline publish 也建议在两个运行时仓各有一个薄 wrapper，由共享逻辑统一产出组合栈 baseline。

两个 PR wrapper workflow 的触发事件建议为：

1. `pull_request`:
   - `opened`
   - `reopened`
   - `synchronize`
   - `labeled`
2. `workflow_dispatch`

两个 main baseline publish wrapper workflow 的触发事件建议为：

1. `push` 到 `main`
2. `workflow_dispatch`

### 4.2 job 级触发条件

不要用 workflow 级 `paths-ignore` 直接跳过 docs PR。为了支持后续通过标签强制重跑，skip 判断应放在 job 内部。

建议的 job 级规则：

1. 先做一个轻量 selector job，判断本次事件来自哪个触发仓，并解析对应的 peer repo ref。
2. fork PR 默认跳过，只写 summary note。
3. docs-only、comment-only、纯 markdown 变更默认跳过。
4. 单仓 PR 满足以下任一条件时进入 freshness precheck：
   - 变更命中性能敏感路径
   - PR 带 `perf-benchmarks` 标签
   - 手动 `workflow_dispatch`
5. 双仓协同改动默认不走普通 `pull_request` 触发链路，而是走 `workflow_dispatch` / `repository_dispatch`，显式传入 `vllm_hust_ref` 和 `vllm_ascend_hust_ref`。
6. branch freshness precheck 在普通 GitHub-hosted runner 上执行。
7. 如果命中“强制 rebase”门槛，则直接以 `rebase_required` 结束，不再占用 Ascend runner。

### 4.3 建议纳入 changed-files 检测的路径

`vllm-hust` wrapper 建议命中以下路径时触发 L1：

1. `vllm/**`
2. `csrc/**`
3. `benchmarks/**`
4. `.buildkite/performance-benchmarks/**`
5. `.github/workflows/linux-ascend-*.yml`
6. `.github/workflows/scripts/**`
7. `setup.py`
8. `pyproject.toml`
9. `requirements/**`

`vllm-ascend-hust` wrapper 建议命中以下路径时触发 L1：

1. `vllm_ascend/**`
2. `csrc/**`
3. `benchmarks/**`
4. `.github/workflows/**`
5. `scripts/**`
6. `setup.py`
7. `pyproject.toml`
8. `requirements*.txt`

如果共享逻辑下沉到 `vllm-hust-benchmark`，则还应把以下路径接入人工重跑或 dispatch 触发链路：

1. `vllm-hust-benchmark/src/**`
2. `vllm-hust-benchmark/tests/**`

### 4.4 与现有标签体系的关系

当前仓现有 PR 流程更偏 `ready` / `verified` 标签控制。L1 不建议复用这两个标签，而是单独增加：

- `perf-benchmarks`

原因：

1. `ready` / `verified` 的语义偏“允许跑 CI”。
2. `perf-benchmarks` 的语义偏“强制执行性能门禁或手动重跑”。
3. 两者职责不同，不应混用。

## 5. 模型选择

### 5.1 最终建议

L1 默认模型建议使用：

- `Qwen/Qwen2.5-3B-Instruct`

加载方式：

- `--load-format dummy`

### 5.2 选择理由

这个选择是基于当前 workspace 和 upstream 的折中，而不是单纯追求最小参数量：

1. 当前 `vllm-hust` 的 Ascend smoke / benchmark 默认已经明显偏向 Qwen 家族。
2. 0.5B 虽然更小，但它更适合作为功能 smoke 模型，不适合作为性能门禁的代表模型。
3. 7B 虽然更接近 upstream 8B dense baseline，但对 L1 PR gate 来说偏重，不符合“最快速反馈”。
4. 3B 是“规模最小但仍不至于把 scheduler / KV cache / batching 行为压扁”的合理折中。

### 5.3 为什么不是 0.5B

不建议 `Qwen/Qwen2.5-0.5B-Instruct` 作为 L1 性能门禁模型，原因是：

1. 太小，容易低估真实 batching / cache / scheduler 开销。
2. 当前仓里它更像功能检查默认模型，而不是性能代表模型。
3. 用它做 merge gate，信噪比容易偏差。

### 5.4 为什么不是 7B

不建议 `Qwen/Qwen2.5-7B-Instruct` 作为 L1 默认模型，原因是：

1. 它并不是“规模最小”。
2. PR 级 gate 应优先控制时间和 runner 占用。
3. 7B 更适合 L2 或定期 calibration，而不是每次 PR 默认门禁。

### 5.5 与 upstream 的关系

upstream performance 代表集更偏 `Llama-3.1-8B-Instruct` 或同量级 8B dense 模型。为了保持与 upstream 的可比性，建议再保留一个非门禁、低频 calibration 任务：

- `Meta-Llama-3.1-8B-Instruct` + `--load-format dummy`

用途：

1. 作为 L2 / nightly 校准样本。
2. 用来观察 Qwen-3B L1 是否与 upstream 8B 走势一致。

## 6. 参数设计

### 6.1 通用 server 参数

建议统一为：

- `--dtype bfloat16`
- `--load-format dummy`
- `--tensor-parallel-size 1`
- `--max-model-len 2048`
- `--max-num-seqs 8`
- `--enforce-eager`

说明：

1. `dummy` 保证不下载和加载真实权重。
2. `max-model-len=2048` 能覆盖中等上下文长度下的 KV / cache / page 行为。
3. `max-num-seqs=8` 能让 `qps=4` 下的 batching 行为不被人为压死。

### 6.2 latency benchmark 参数

建议使用：

- 输入长度：`1024`
- 输出长度：`128`
- warmup：`5`
- measurement：`10`

说明：

1. `1024 -> 128` 比当前 CI preview 的 `64 -> 16` 更接近真实 dense online serving。
2. warmup / measurement 不直接照抄 upstream `5 + 15`，但保留 5 次 warmup 来减少首轮抖动。
3. `10` 次测量仍保持 L1 级别的速度。

### 6.3 throughput benchmark 参数

建议使用：

- dataset：`random`
- 输入长度：`1024`
- 输出长度：`128`
- `num_prompts=100`
- `random_batch_size=1`

说明：

1. 使用 `random` 避免 ShareGPT 数据依赖。
2. `num_prompts=100` 保证 throughput 不至于只测到启动和抖动。
3. 仍保持 PR 级别可接受耗时。

### 6.4 serve benchmark 参数

建议使用：

- scenario：`random-online`
- `num_prompts=100`
- `request_rate=4`
- `max_concurrency=4`
- endpoint：`/v1/completions`

说明：

1. `qps=4` 是当前仓内已有 online benchmark 语义的自然延续。
2. `random-online` 不需要 dataset path，减少 runner 外部依赖。
3. 这一组参数足以覆盖 continuous batching 的基本回归。

### 6.5 指标输出契约

L1 结果文件不应只输出“最少够做门禁的几个数字”，而应至少覆盖当前 benchmark / exporter 语义里最关键的在线性能指标，避免后续为了补分析字段再重改 schema。

建议 `perf_smoke_result.json` 至少输出以下字段：

1. `benchmark_profile`
2. `benchmark_config`
3. `benchmark_config_fingerprint`
4. `source_combo`
5. `source_combo_fingerprint`
6. `serve.mean_ttft_ms`
7. `serve.mean_tpot_ms` 或 `serve.mean_tbt_ms`
8. `serve.request_throughput_rps`
9. `serve.output_throughput_tps`
10. `serve.error_rate`
11. `serve.failed_requests`
12. `throughput.tokens_per_second`
13. `latency.mean_ms`
14. `latency.p50_ms`

其中：

1. `TTFT`、`Tokens/s` 必须输出。
2. `TBT/TPOT` 建议纳入正式结果契约，而不是留到后续补充。
3. `error_rate` 必须输出，因为 PR 级性能门禁不能把“失败请求导致的吞吐下降”误判成纯性能回归。
4. `benchmark_config` 和 `benchmark_config_fingerprint` 也必须输出，因为 latest-main baseline 的可比性判断不能靠临时解析 shell 命令或对原始命令行文本做 hash。
5. `source_combo` 和 `source_combo_fingerprint` 也必须输出，因为当前真正关注的是 `vllm-hust + vllm-ascend-hust` 的组合，而不是单个 commit。

建议输出结构如下：

```json
{
   "schema_version": "ascend-l1-perf-smoke/v1",
   "benchmark_profile": "ascend-l1-qwen25-3b-dummy-v1",
   "benchmark_config_fingerprint": "sha256:example",
   "source_combo_fingerprint": "sha256:combo-example",
   "source_combo": {
      "pairing_strategy": "single-repo-pr-with-peer-main",
      "trigger_repo": "vllm-hust",
      "vllm_hust": {
         "repo": "vLLM-HUST/vllm-hust",
         "ref": "pull/123/head",
         "commit": "abc123"
      },
      "vllm_ascend_hust": {
         "repo": "vLLM-HUST/vllm-ascend-hust",
         "ref": "main",
         "commit": "def456"
      }
   },
   "benchmark_config": {
      "model": "Qwen/Qwen2.5-3B-Instruct",
      "load_format": "dummy",
      "dtype": "bfloat16",
      "tensor_parallel_size": 1,
      "server": {
         "max_model_len": 2048,
         "max_num_seqs": 8,
         "enforce_eager": true
      },
      "latency": {
         "input_len": 1024,
         "output_len": 128,
         "warmup": 5,
         "measurement": 10
      },
      "throughput": {
         "dataset": "random",
         "input_len": 1024,
         "output_len": 128,
         "num_prompts": 100,
         "random_batch_size": 1
      },
      "serve": {
         "scenario": "random-online",
         "num_prompts": 100,
         "request_rate": 4,
         "max_concurrency": 4,
         "endpoint": "/v1/completions"
      }
   },
   "model": "Qwen/Qwen2.5-3B-Instruct",
   "load_format": "dummy",
   "dtype": "bfloat16",
   "tensor_parallel_size": 1,
   "runner_class": "linux-aarch64-a2b3-0",
   "soc_version": "ascend910b3",
   "serve": {
      "mean_ttft_ms": 0.0,
      "mean_tpot_ms": 0.0,
      "request_throughput_rps": 0.0,
      "output_throughput_tps": 0.0,
      "error_rate": 0.0,
      "failed_requests": 0
   },
   "throughput": {
      "tokens_per_second": 0.0
   },
   "latency": {
      "mean_ms": 0.0,
      "p50_ms": 0.0
   },
   "raw_artifacts": {
      "serve_result_json": "path-or-artifact-ref",
      "throughput_result_json": "path-or-artifact-ref",
      "latency_result_json": "path-or-artifact-ref",
      "server_log": "path-or-artifact-ref"
   }
}
```

`benchmark_config_fingerprint` 的生成建议固定为：

1. 先构造结构化 `benchmark_config` 对象，而不是直接对 shell command 文本做 hash。
2. 所有默认值必须显式写入对象，避免“命令行没写但运行时默认生效”导致同配置生成不同 fingerprint。
3. 对象字段顺序必须固定，推荐序列化为 canonical JSON 后再做 `sha256`。
4. artifact 中必须同时保留 `benchmark_config` 原文和 `benchmark_config_fingerprint`，保证可审计，不能只存 hash 值。

这里需要明确区分两类 fingerprint：

1. `benchmark_config_fingerprint`
   - 表示 benchmark 运行配置本身是否可比。
2. `source_combo_fingerprint`
   - 表示当前测到的是哪一个 `vllm-hust + vllm-ascend-hust` 代码组合。

二者不能混用：

1. `benchmark_config_fingerprint` 用于可比性判断。
2. `source_combo_fingerprint` 用于代码组合追踪、缓存和审计。

### 6.6 指标与门禁的关系

并不是所有输出指标都直接参与阻断判定。

建议区分两层：

1. `gate metrics`
    - `serve.output_throughput_tps`
    - `throughput.tokens_per_second`
    - `serve.mean_ttft_ms`
    - `latency.mean_ms`
    - `serve.error_rate`
2. `diagnostic metrics`
    - `serve.mean_tpot_ms` / `serve.mean_tbt_ms`
    - `serve.request_throughput_rps`
    - `serve.failed_requests`
    - `latency.p50_ms`

这样可以保证：

1. 门禁逻辑足够稳定，不被过多边缘指标扰动。
2. 失败后仍有足够的信息用于快速定位是 scheduler、decode、失败请求还是纯吞吐退化。

## 7. 基线定义

### 7.1 基线不能直接来自 L3

L1 不应直接对比“最新一次 L3 任务”，原因是：

1. L3 往往使用不同模型。
2. L3 可能使用真实权重。
3. L3 的 prompt 规模、场景和采样点可能不同。
4. 两者不满足同配置可比性。

因此，L1 必须有自己的同配置基线。

### 7.2 不建议用“官方极限”做 PR 门禁基线

L1 不应与“官方极限结果”或“某次最佳 showcase 结果”做阻断比较。

原因是：

1. 官方极限往往对应不同模型、不同权重、不同参数和不同调优状态。
2. 官方极限具有目标值语义，不具有稳定的 PR 回归检测语义。
3. PR 门禁最核心的目标是发现当前 PR 相对于其主线祖先是否发生回退，而不是判断当前实现是否追平世界最优。

因此更合适的分工是：

1. 官方极限：作为 aspirational reference，用于 L2/L3、nightly、季度回顾和优化路线图。
2. main 基线结果：作为 L1 PR gate 的 blocking baseline。

### 7.3 推荐的全自动方案：组合栈双比较

在双仓组合约束下，L1 主方案不应再被表述为“单仓 PR + 单份 main baseline”，而应采用“单仓 PR 双比较，双仓协同显式输入”的策略：

1. `push to main` in either repo 时运行 L1 smoke，并发布一份 protected combo result。
2. 如果触发仓是 `vllm-hust`：
   - protected combo = `vllm-hust@protected_main + vllm-ascend-hust@protected_main`
3. 如果触发仓是 `vllm-ascend-hust`：
   - protected combo = `vllm-ascend-hust@protected_main + vllm-hust@protected_main`
4. 单仓 PR 运行时，先把 peer repo 的 protected commit 固定下来。
5. head combo = `trigger_repo@PR_HEAD + peer_repo@fixed_protected_commit`
6. ancestor combo = `trigger_repo@merge-base/base SHA + peer_repo@fixed_protected_commit`
7. latest protected combo = 最近一次成功发布、且 `benchmark_config_fingerprint` 匹配的 protected combo result。
8. PR workflow 默认执行两条比较：
   - `delta_vs_same_peer_ancestor`
   - `delta_vs_latest_protected_combo`
9. 只有在 same-peer ancestor combo 不可得且 latest protected combo 也不可得时，才标记 `baseline_missing`，本次只上报、不阻断。
10. 双仓协同改动默认走 `workflow_dispatch` / `repository_dispatch`，显式传入两边的 ref；如果没有显式 base pair，第一阶段建议 report-only。

这样做的核心优化点是：

1. same-peer ancestor comparison 用来隔离触发仓本身的改动。
2. latest protected combo comparison 用来保证当前组合栈没有落后受保护主线前沿。

### 7.4 为什么组合门禁不能只存一个 `commit_sha`

一旦被测对象变成 `vllm-hust + vllm-ascend-hust` 组合，单个 `commit_sha` 就不再能表达真实基线。

例如：

1. PR 来自 `vllm-hust`。
2. 如果只记录 `vllm-hust` 的 `merge-base SHA`，却不固定 `vllm-ascend-hust` 的 peer commit，那么 ancestor 比较会混入 plugin 仓变化。
3. 反过来，PR 来自 `vllm-ascend-hust` 时，只记录 plugin 仓 `merge-base SHA` 也同样不够。

因此，组合门禁里需要同时区分两类信息：

1. `benchmark_config`
   - 表示“怎么测”。
2. `source_combo`
   - 表示“测的是哪一组代码组合”。

进一步地：

1. `benchmark_config_fingerprint` 用于判断两次结果是否可比。
2. `source_combo_fingerprint` 用于追踪本次结果对应的双仓组合。

如果没有这层区分，L1 可能会把“配置相同但代码组合不同”和“代码组合相同但配置不同”的情况混在一起，造成错误比较。

### 7.5 分支新鲜度与强制 rebase 规则

双基线 hard fail 仍需要配套的 branch freshness guard，否则长期分叉分支会让比较语义和 runner 成本都变差。

建议同时记录两个 freshness 指标：

1. `merge_base_age_days`
2. `base_branch_ahead_commits`

这两个指标都应基于触发仓计算，而不是对 peer repo 同时计算一套独立 freshness 规则。

推荐的初始门槛如下：

1. `rebase_recommended`：满足任一条件即提示
   - `merge_base_age_days > 3`
   - `base_branch_ahead_commits > 20`
2. `rebase_required`：满足任一条件即 hard fail
   - `merge_base_age_days > 7`
   - `base_branch_ahead_commits > 50`

执行策略建议如下：

1. branch freshness precheck 在普通 GitHub-hosted runner 上先执行。
2. selector job 必须先判定“本次是否需要跑 L1”，避免 docs-only / fork PR 被错误打成 `rebase_required`。
3. 命中 `rebase_required` 时，直接失败并提示开发者先 rebase，再重新触发 L1。
4. 仅命中 `rebase_recommended` 时，允许继续跑 benchmark，但 summary 必须显式提示。
5. 上述阈值作为初始值，建议在上线后根据 1 到 2 周样本再调优。

### 7.6 baseline source of truth

主 source of truth 建议是按 SHA 索引的 main benchmark 结果存储，例如：

1. artifact store
2. GH Pages / GitHub artifact index
3. HF dataset prefix

结果对象应至少包含：

1. `schema_version`
2. `trigger_repo`
3. `source_combo`
4. `source_combo_fingerprint`
5. `runner_class`
6. `soc_version`
7. `benchmark_profile`
8. `benchmark_config`
9. `benchmark_config_fingerprint`
10. `serve.mean_ttft_ms`
11. `serve.mean_tpot_ms` 或 `serve.mean_tbt_ms`
12. `serve.request_throughput_rps`
13. `serve.output_throughput_tps`
14. `serve.error_rate`
15. `throughput.tokens_per_second`
16. `latency.mean_ms`
17. `latency.p50_ms`
18. `generated_at`

`benchmark_config` 建议至少覆盖以下稳定字段组合：

1. `model` 维度：`model`、`load_format`、`dtype`、`tensor_parallel_size`
2. `server` 维度：`max_model_len`、`max_num_seqs`、`enforce_eager`
3. `latency` 维度：`input_len`、`output_len`、`warmup`、`measurement`
4. `throughput` 维度：`dataset`、`input_len`、`output_len`、`num_prompts`、`random_batch_size`
5. `serve` 维度：`scenario`、`num_prompts`、`request_rate`、`max_concurrency`、`endpoint`

也就是说，`benchmark_config_fingerprint` 不是“当前命令行长什么样”的 hash，而是“当前 L1 benchmark 逻辑配置是什么”的稳定标识。

其中，需要明确区分两类 baseline：

1. same-peer ancestor combo
   - 优先由当前 PR workflow 同步生成，不优先依赖历史 artifact。
2. latest protected combo
   - 由 main baseline publish workflow 预先发布。

latest protected combo 的查找与可比性判断必须至少匹配以下维度，否则视为不可比：

1. `schema_version`
2. `benchmark_config_fingerprint`
3. `runner_class`
4. `soc_version`
5. `source_combo.pairing_strategy`

如果候选 latest protected combo 存在，但 `benchmark_config_fingerprint` 不匹配，应标记为 `fingerprint_mismatch`，而不是勉强参与比较。

### 7.7 PR 查找 baseline 的顺序

PR workflow 中建议固定两条比较链路：

1. same-peer ancestor 链：
   - 如果触发仓是 `vllm-hust`：运行 `vllm-hust@merge-base + vllm-ascend-hust@fixed_protected_commit`
   - 如果触发仓是 `vllm-ascend-hust`：运行 `vllm-ascend-hust@merge-base + vllm-hust@fixed_protected_commit`
2. latest protected combo 链：
   - 查询最近成功发布的 protected combo result

其中：

1. same-peer ancestor comparison 用于隔离触发仓本身的性能变化。
2. latest protected combo comparison 用于防止“相对旧祖先有收益，但相对当前受保护组合栈已经无收益”的情况漏检。
3. latest protected combo 必须满足同配置 fingerprint 匹配，不能只按“最新成功结果”做宽松查找。
4. 双仓协同改动不建议沿用普通单仓 ancestor 语义；应显式传入双仓 candidate refs 和双仓 base refs。

### 7.8 首次运行或 baseline 缺失时的处理

如果 PR 找不到任何可用 baseline，不应阻断合并。

建议状态机如下：

1. same-peer ancestor combo 可得，latest protected combo 也可得：两条比较都执行，且都可 hard fail。
2. 只得到 same-peer ancestor combo：
   - 执行 `delta_vs_same_peer_ancestor`。
   - summary 标记 `latest_protected_combo_missing` 或 `latest_protected_combo_fingerprint_mismatch`。
3. 只得到 latest protected combo：
   - 执行 `delta_vs_latest_protected_combo`。
   - summary 标记 `same_peer_ancestor_missing`。
4. same-peer ancestor combo 不可得，latest protected combo 也不可得：
   - 记录当前 run。
   - 标记 `baseline_missing`。
   - workflow 成功结束。

也就是说，只有在完全没有比较对象时，首次运行的职责才是“建可用比较链路”，不是“强制判回归”。

## 8. 失败判定

### 8.1 硬失败条件

以下任一命中则直接阻止合并：

1. workflow 或 benchmark 命令执行失败。
2. 输出 JSON 缺失或无法解析。
3. server 无法启动或 readiness check 失败。
4. benchmark 结果存在明显无效值：
   - throughput `<= 0`
   - TTFT `<= 0`
   - error_rate `> 0`
   - required metric missing
5. branch freshness 命中 `rebase_required`：
   - `merge_base_age_days > 7`
   - 或 `base_branch_ahead_commits > 50`

### 8.2 性能回归阈值

当且仅当 PR 找到了至少一个可用 baseline 时，才执行 blocking 回归判定。

branch freshness 的判定顺序高于性能 delta 判定：

1. 先做 freshness precheck。
2. 命中 `rebase_required` 时，直接以 `rebase_required` 失败，不再解释为性能回归。
3. 仅命中 `rebase_recommended` 时，继续执行性能比较。

建议初始阈值如下：

1. 对 same-peer ancestor combo 比较：serve `output_throughput_tps` 低于 baseline 超过 `5%`：失败
2. 对 same-peer ancestor combo 比较：throughput `tokens_per_second` 低于 baseline 超过 `5%`：失败
3. 对 same-peer ancestor combo 比较：serve `mean_ttft_ms` 高于 baseline 超过 `8%`：失败
4. 对 same-peer ancestor combo 比较：latency `mean_ms` 高于 baseline 超过 `8%`：失败
5. 对 latest protected combo 比较：serve `output_throughput_tps` 低于 baseline 超过 `5%`：失败
6. 对 latest protected combo 比较：throughput `tokens_per_second` 低于 baseline 超过 `5%`：失败
7. 对 latest protected combo 比较：serve `mean_ttft_ms` 高于 baseline 超过 `8%`：失败
8. 对 latest protected combo 比较：latency `mean_ms` 高于 baseline 超过 `8%`：失败
9. `error_rate > 0`：失败

补充说明：

1. `mean_tpot_ms` / `mean_tbt_ms` 默认作为诊断指标，不直接阻断。
2. 如果后续发现 decode 路径回退频繁漏检，再考虑把 `TPOT/TBT` 提升为 gate metric。
3. 只要某条 baseline 存在，该条比较就具有 hard fail 语义；不是 warning-only 指标。

### 8.3 降噪策略

为减少自托管 Ascend runner 抖动带来的误报，建议：

1. 首次阈值触发时自动重跑一次同配置 benchmark。
2. 两次都超阈值才真正 fail。
3. 如果 runner preflight 失败，则标记为 infra failure，而不是 perf regression。
4. 如果仅找到了 latest protected combo 而未找到 same-peer ancestor combo，summary 中必须显式标注，避免误解为触发仓级精确比较。
5. `rebase_required` 不进入自动重跑，它代表分支新鲜度策略命中，而不是性能噪声。

### 8.4 PR 评论与 summary 内容

建议 workflow 输出以下信息：

1. `trigger_repo`
2. 当前 `source_combo`
3. `runner_class`
4. 模型、dtype、load_format
5. 当前 latency / throughput / TTFT / TPOT(TBT) / error_rate
6. same-peer ancestor combo latency / throughput / TTFT / error_rate
7. latest protected combo latency / throughput / TTFT / error_rate
8. `delta_vs_same_peer_ancestor` 和 `delta_vs_latest_protected_combo`
9. baseline 状态：
   - same_peer_ancestor_baseline
   - latest_protected_combo_baseline
   - same_peer_ancestor_missing
   - latest_protected_combo_missing
   - latest_protected_combo_fingerprint_mismatch
   - baseline_missing
10. freshness 状态：
   - fresh
   - rebase_recommended
   - rebase_required
11. `merge_base_age_days`
12. `base_branch_ahead_commits`
13. 失败原因分类：
   - infra failure
   - startup failure
   - benchmark invalid
   - rebase required
   - perf regression

### 8.5 官方极限结果如何使用

官方极限或 upstream 代表 benchmark 结果建议作为 summary 中的非阻断参考列，而不是 gating baseline。

建议展示为：

1. `delta_vs_same_peer_ancestor`
2. `delta_vs_latest_protected_combo`
3. `official_reference_delta`

其中：

1. `delta_vs_same_peer_ancestor` 用于检测在 peer repo 固定不变时，触发仓是否把组合栈做坏。
2. `delta_vs_latest_protected_combo` 用于检测当前组合栈是否已经落后受保护主线前沿。
3. 这两个 delta 都用于 pass / fail。
4. `official_reference_delta` 只用于说明当前距离目标还有多远。

## 9. 具体落地步骤

### 9.1 不建议直接复用现有 publication benchmark workflow

不建议直接把 L1 叠加到：

- `vllm-hust/.github/workflows/ascend-benchmark-leaderboard.yml`

原因：

1. 这个 workflow 的目标是 benchmark publication 和 leaderboard export。
2. 当前脚本 `run_ascend_benchmark_ci.sh` 默认走真实模型和 artifact/export 逻辑。
3. 它还带 same-spec、constraints、HF publish 等额外语义，不适合 PR 级快速 gate。

### 9.2 建议新增的文件

建议新增：

1. `vllm-hust/.github/workflows/linux-ascend-perf-smoke.yml`
2. `vllm-ascend-hust/.github/workflows/linux-ascend-perf-smoke.yml`
3. `vllm-hust/.github/workflows/linux-ascend-perf-smoke-baseline.yml`
4. `vllm-ascend-hust/.github/workflows/linux-ascend-perf-smoke-baseline.yml`
5. `vllm-hust-benchmark/src/vllm_hust_benchmark/combo_manifest.py`
6. `vllm-hust-benchmark/src/vllm_hust_benchmark/check_branch_freshness.py`
7. `vllm-hust-benchmark/src/vllm_hust_benchmark/run_ascend_perf_smoke.py`
8. `vllm-hust-benchmark/src/vllm_hust_benchmark/compare_combo_baseline.py`
9. `vllm-hust-benchmark/src/vllm_hust_benchmark/publish_combo_baseline.py`
10. `vllm-hust-benchmark/data/ascend-l1-smoke-official-reference.json`

### 9.3 check_branch_freshness.py 的职责

这个轻量 precheck 脚本建议在普通 GitHub-hosted runner 上执行，职责固定为：

1. 仅在 selector job 判定“需要跑 L1”后执行。
2. 计算 `merge-base SHA`。
3. 计算 `merge_base_age_days`。
4. 计算 `base_branch_ahead_commits`。
5. 输出 `branch_freshness.json`。
6. 命中 `rebase_required` 时直接返回非零 exit code。

### 9.4 run_ascend_perf_smoke.sh 的职责

这个共享执行逻辑建议直接做四件事：

1. 解析 `source_combo`，并在工作目录中 materialize `vllm-hust` 与 `vllm-ascend-hust` 两个 ref。
2. 按组合栈 manifest 安装或挂接这两个 repo，使 `vllm-hust + vllm-ascend-hust` 作为一个整体运行。
3. 启动 `vllm serve`，显式传入 `--load-format dummy`
4. 顺序执行 latency / throughput / serve 三个 benchmark，并统一整理输出为一个 `perf_smoke_result.json`

建议输出结构：

```json
{
   "schema_version": "ascend-l1-perf-smoke/v1",
   "benchmark_profile": "ascend-l1-qwen25-3b-dummy-v1",
   "benchmark_config_fingerprint": "sha256:example",
   "source_combo_fingerprint": "sha256:combo-example",
   "source_combo": {
      "pairing_strategy": "single-repo-pr-with-peer-main",
      "trigger_repo": "vllm-hust",
      "vllm_hust": {
         "repo": "vLLM-HUST/vllm-hust",
         "ref": "pull/123/head",
         "commit": "abc123"
      },
      "vllm_ascend_hust": {
         "repo": "vLLM-HUST/vllm-ascend-hust",
         "ref": "main",
         "commit": "def456"
      }
   },
   "benchmark_config": {
      "model": "Qwen/Qwen2.5-3B-Instruct",
      "load_format": "dummy",
      "dtype": "bfloat16",
      "tensor_parallel_size": 1,
      "server": {
         "max_model_len": 2048,
         "max_num_seqs": 8,
         "enforce_eager": true
      },
      "latency": {
         "input_len": 1024,
         "output_len": 128,
         "warmup": 5,
         "measurement": 10
      },
      "throughput": {
         "dataset": "random",
         "input_len": 1024,
         "output_len": 128,
         "num_prompts": 100,
         "random_batch_size": 1
      },
      "serve": {
         "scenario": "random-online",
         "num_prompts": 100,
         "request_rate": 4,
         "max_concurrency": 4,
         "endpoint": "/v1/completions"
      }
   },
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "load_format": "dummy",
  "dtype": "bfloat16",
   "tensor_parallel_size": 1,
  "runner_class": "linux-aarch64-a2b3-0",
   "soc_version": "ascend910b3",
  "latency": {
    "mean_ms": 0.0,
    "p50_ms": 0.0
  },
  "throughput": {
    "tokens_per_second": 0.0
  },
  "serve": {
      "mean_ttft_ms": 0.0,
      "mean_tpot_ms": 0.0,
      "request_throughput_rps": 0.0,
      "output_throughput_tps": 0.0,
      "error_rate": 0.0,
      "failed_requests": 0
   },
   "raw_artifacts": {
      "serve_result_json": "path-or-artifact-ref",
      "throughput_result_json": "path-or-artifact-ref",
      "latency_result_json": "path-or-artifact-ref",
      "server_log": "path-or-artifact-ref"
  }
}
```

### 9.5 compare_perf_smoke_baseline.py 的职责

职责建议固定为：

1. 读取当前 head combo run JSON、`benchmark_config` 和 `source_combo`
2. 读取 same-peer ancestor combo run JSON（如果当前模式是单仓 PR）
3. 查找 latest protected combo baseline JSON
4. 读取 `branch_freshness.json`
5. 可选读取 `official_reference` JSON
6. 仅在 `schema_version`、`benchmark_config_fingerprint`、`runner_class`、`soc_version` 同时匹配时使用 latest protected combo baseline
7. 如果 latest protected combo 候选存在但 fingerprint 不匹配，则输出 `fingerprint_mismatch` 原因并将其视为不可比
8. 分别计算 `delta_vs_same_peer_ancestor` 和 `delta_vs_latest_protected_combo`
9. 输出 markdown summary
10. 只要任一可用比较超阈值，就返回非零 exit code

补充职责：

1. 当两类 baseline 都找不到时，返回 `baseline_missing`，而不是 fail。
2. 当 same-peer ancestor combo 或 latest protected combo 缺失时，在 summary 中分别显式标记。
3. 不允许通过 hash 原始命令行文本来替代 `benchmark_config` 的结构化比较。

### 9.6 workflow 落地顺序

#### 第一步：只上报不拦截

先落地 workflow，但先不 required，不阻止合并：

1. 先验证两个 wrapper workflow、selector job、branch freshness precheck 和 Ascend runner 稳定性。
2. 先收集 1 到 2 周 main / PR 样本。
3. 先确认 `Qwen2.5-3B + dummy` 的噪声区间。
4. 先校准 `3/7 days` 与 `20/50 commits` 阈值是否需要微调。
5. 先确认 same-peer ancestor dual-run 的额外耗时是否可接受。

#### 第二步：生成 baseline

在 main 上跑 baseline workflow，生成按 SHA 可检索的 baseline 结果：

1. 每次 `push to main` in either repo 成功运行 L1 smoke。
2. 将结果连同 `source_combo` 一起发布到可检索存储。
3. 保证 PR 可以查询最新成功 protected combo 结果作为主线前沿基线。
4. 保证 latest protected combo 查询受 `benchmark_config_fingerprint` 约束。

#### 第三步：启用软门禁

PR 失败时仅打 comment，不设 required：

1. 收集误报率
2. 调整 5% / 8% 阈值
3. 验证双基线 hard fail 规则下 auto-retry 是否足够降噪
4. 验证 stale branch rebase 规则是否过严或过松

#### 第四步：启用硬门禁

当误报率可接受后，把 workflow 设为 required check。

## 10. 推荐实施顺序

### PR 1：新增脚本与 workflow 骨架

内容：

1. `vllm-hust-benchmark` 中新增 `combo_manifest` 与共享 benchmark CLI 骨架
2. `vllm-hust` 和 `vllm-ascend-hust` 各自新增 thin wrapper workflow
3. selector job
4. `check_branch_freshness.py`
5. summary 输出

目标：

1. 跑通 selector + freshness precheck + combo manifest + benchmark + compare 的 end-to-end
2. 能在 summary 中同时输出 same-peer ancestor 和 latest protected combo 比较
3. 不做 required gate

### PR 2：新增 main baseline publish workflow

内容：

1. 两个 repo 的 `linux-ascend-perf-smoke-baseline.yml` wrapper
2. protected combo baseline JSON 发布逻辑
3. `source_combo` 与 SHA 索引逻辑
4. `benchmark_config` 结构化序列化逻辑
5. `benchmark_config_fingerprint` 与 `source_combo_fingerprint` 生成逻辑

目标：

1. 形成按组合栈可检索的 protected combo baseline source of truth
2. 形成 latest protected combo 的稳定查询入口

### PR 3：加入 changed-files 和标签强制重跑

内容：

1. 两个 repo 各自的 docs-only skip / changed-files 规则
2. `perf-benchmarks` 标签强制运行
3. fork PR summary-only 逻辑
4. 双仓 `workflow_dispatch` / `repository_dispatch` 入口

目标：

1. 控制 runner 成本
2. 保留人工重跑和双仓协同入口

### PR 4：启用阈值和自动重跑

内容：

1. same-peer ancestor compare 逻辑
2. latest protected combo compare 逻辑
3. threshold breach auto-retry
4. stale branch rebase 规则
5. 失败原因分类

目标：

1. 让 same-peer ancestor 和 latest protected combo 都具备 hard fail 语义
2. 让失败真正表达“性能回归”，而不是 runner 抖动

### PR 5：把 workflow 设为 required

前提：

1. 至少 1 到 2 周稳定运行
2. 误报率可接受
3. baseline 已稳定更新

## 11. 最终建议

L1 的推荐最终形态如下：

1. L1 默认门禁对象是 `vllm-hust + vllm-ascend-hust` 组合栈，而不是单个仓。
2. `vllm-hust` 和 `vllm-ascend-hust` 各自保留一个 thin wrapper workflow，共享 benchmark orchestration 逻辑建议沉淀在 `vllm-hust-benchmark`。
3. 使用 `Qwen/Qwen2.5-3B-Instruct` + `--load-format dummy` 作为默认门禁模型。
4. 使用 `random-online` 风格参数，避免 ShareGPT 数据依赖。
5. `push to main` in either repo 时运行 L1 smoke，并发布 protected combo baseline。
6. 单仓 PR 默认做两条 hard fail 比较：
   - `delta_vs_same_peer_ancestor`
   - `delta_vs_latest_protected_combo`
7. 官方极限结果保留为 aspirational reference，只展示、不阻断。
8. throughput 回归阈值先设 `5%`，latency / TTFT / TPOT(TBT) 先设 `8%`，`error_rate > 0` 直接失败，并带一次自动重跑降噪。
9. 分支新鲜度默认采用双门槛：
   - `rebase_recommended`: `> 3 days` 或 `> 20 commits`
   - `rebase_required`: `> 7 days` 或 `> 50 commits`
10. `benchmark_config_fingerprint` 必须从规范化的 `benchmark_config` 对象生成，而不是对原始命令行文本做 hash。
11. `source_combo_fingerprint` 必须明确记录当前跑的是哪组 `vllm-hust + vllm-ascend-hust` 代码组合。
12. latest protected combo baseline 必须匹配同一 `benchmark_config_fingerprint`，否则不参与比较。

这版方案能同时满足：

1. 比当前 `0.5B` 功能 smoke 更有性能代表性。
2. 比直接上 `7B` 或真实权重更轻。
3. 与 upstream 的 dummy performance 思路保持一致。
4. 比 PR 内双跑 `base + head` 更快拿到结果。
5. 与当前 workspace 的 Ascend runner、workflow、benchmark 组织方式不冲突。

## 附录 A：PR 双跑 `base + head` 方案

PR 双跑方案不是主方案，但仍然是一个合理备选。

### A.1 方案定义

在同一个 PR workflow 中：

1. checkout `base` 或 `merge-base` 代码。
2. 跑一次同配置 L1 smoke，得到 `base_result.json`。
3. checkout `head` 代码。
4. 再跑一次同配置 L1 smoke，得到 `head_result.json`。
5. 直接比较两者。

### A.2 优点

1. `base` 和 `head` 在同一 workflow、同一 runner、同一时间窗口中执行。
2. 环境噪声最小。
3. 语义非常直接，实验纯度最高。

### A.3 缺点

1. PR 时长接近翻倍。
2. runner 占用更高。
3. 需要在一个 workflow 内两次 checkout、两次安装、两次 benchmark，工程复杂度更高。

### A.4 适用时机

更适合以下场景：

1. runner 资源充足。
2. L1 benchmark 已经被压缩到极短耗时。
3. 团队更看重实验纯度，而不是 PR 反馈速度。

### A.5 当前未采纳的原因

当前阶段没有把 PR 双跑 `base + head` 作为主方案，原因是：

1. 它会显著拉长 PR 反馈时间，而当前更优先的是尽快给开发者返回可用门禁结果。
2. Ascend self-hosted runner 资源相对宝贵，双跑会直接把单个 PR 的 runner 占用放大到接近两倍。
3. 这套方案要求在同一 workflow 中完成两次 checkout、两次环境准备和两次 benchmark，落地复杂度高于“main 预生成 baseline，PR 单跑 head”。
4. 当前文档中的主方案已经同时覆盖“same-peer ancestor 回归检测”“相对 latest protected combo 的前沿一致性”和“长期分叉分支强制 rebase”，同时避免每个 PR 都双倍消耗机器时长。
5. 在当前项目阶段，PR 吞吐、反馈速度和基础设施成本的权重高于“同一工作流内对照实验纯度”的额外收益。

对于当前项目阶段，主方案仍推荐“protected combo baseline + same-peer ancestor compare”的组合栈门禁路径。