# SimLLM 官方 Warm-Cache 吞吐测试需求

## 需求摘要

在 `vllm-hust-benchmark` 中为 SimLLM 定制一套官方测试方案和执行脚本，用于衡量 上 SimLLM 在 warm-cache 命中场景下对在线推理吞吐的影响。

现有本地吞吐脚本是本需求的参考实现依据，工程师应复用其已验证的测试语义、参数约束和错误门禁，但不得未经官方化改造就将其直接作为官方脚本提交。现有本地结果仅用于验证需求可行性，不得直接作为官方数据。工程师应基于仓库现有 official baseline、same-spec、artifact exporter 和 submission 机制完成正式实现，并通过代码评审后重新采集数据。

本需求的核心变化是：正式测量阶段不能再使用 `request_rate=1`。该设置会把客户端到达率限制在 1 req/s，无法体现服务端的最大吞吐能力。正式测量必须使用饱和请求流，并设置明确的最大并发数。

## 背景与问题

当前通用 `random-online` 官方规格使用：

- `request_rate=1`
- 1024 input tokens
- 256 output tokens

它适合在线延迟基线，但不适合评估 SimLLM 的 warm-cache 吞吐收益，原因如下：

1. 1 req/s 的客户端到达率可能低于 Baseline 和 SimLLM 的服务能力，最终测到的是请求发生器上限，而不是引擎吞吐上限。
2. SimLLM 主要减少可复用长 prompt 的 prefill 工作量。短输入、长输出会弱化其目标负载特征。
3. SimLLM 的 measured pass 必须复用 warmup pass 建立的缓存；重启服务或变更 prompt、seed、token budget 都会破坏可比性。
4. 只记录最终 req/s 不足以证明收益来自 SimLLM 命中，需要同时提供 cache hit 和 scheduler rewrite 证据。

## 参考脚本

本需求以仓库中的以下脚本作为工程参考：

```text
scripts/run_simllm_saturated_throughput_warm_cache.sh
```

参考脚本调用现有 warm-cache runner：

```text
scripts/run_simllm_random_online_warm_cache.sh
```

工程师在编写官方脚本前应阅读并验证这两个脚本。官方实现应继承以下已经过 Ascend device 5 实机验证的逻辑：

1. 从现有 official random-online baseline spec 派生 saturated same-spec，而不是维护一套与官方参数体系无关的命令行。
2. Baseline 和 SimLLM measured pass 使用相同 seed、prompt、模型、server 参数和 client 参数。
3. measured pass 设置 `request_rate=inf` 和有界 `max_concurrency`，解除 1 req/s 的客户端限速。
4. SimLLM 组先执行低速 warmup，且 warmup 与 measured pass 使用相同 seed 和 prompt 数；两者之间不重启服务。
5. 显式设置 `temperature=0`，避免模型 generation config 自动启用不适合该测试的采样路径。
6. 强制 `max_num_batched_tokens >= input_len`，防止 prompt 分块造成 warmup hash 与 measured hash 不一致。
7. SimLLM KV cache 容量默认与 prompt 数量一致，避免为未使用的 cache entries 占用 HBM。
8. 分别保存 Baseline 和 SimLLM 原始 benchmark JSON，并自动生成 JSON/Markdown 对比结果。
9. 检查两组 `completed == num_prompts`；结果不完整时非零退出，不允许使用成功请求子集计算收益。
10. 在退出和异常路径中清理服务进程树、监听端口与设备资源。

参考脚本的默认实测入口为：

```bash
cd /workspace/vllm-hust-benchmark
ASCEND_RT_VISIBLE_DEVICES=5 \
CURRENT_MODEL_PATH=/data/shared_models/Qwen2.5-14B-Instruct \
bash scripts/run_simllm_saturated_throughput_warm_cache.sh
```

上述命令仅用于复现和理解参考实现。官方脚本不得硬编码 device 5 或本地模型绝对路径，而应接入官方 runner 的设备分配、模型解析和运行环境记录机制。

参考脚本当前产生的主要文件包括：

```text
.benchmarks/simllm-saturated-throughput-warm-cache/
├── saturated-same-spec.json
├── baseline-disabled/raw_benchmark_result.json
├── enabled-warm-cache/raw_benchmark_result.json
├── throughput_comparison.json
└── throughput_comparison.md
```

官方实现应保留这些核心原始信息，但应按第 10 节扩展为多 repetition、结构化 SimLLM metrics、manifest 和 canonical submission 兼容的正式 artifact 布局。

## 3. 测试目标

### 3.1 主要目标

在同一硬件、模型、代码版本和请求集合下，对比以下两组：

- Baseline：SimLLM 关闭，不执行 SimLLM warmup。
- SimLLM warm cache：SimLLM 开启，先暖缓存，再在不重启服务的情况下执行正式测量。

主要指标为成功请求吞吐量 `request_throughput`，辅助指标包括 total token throughput、TTFT 和 TPOT。

### 3.2 非目标

- 本测试不用于证明模型输出质量等价；质量评估应作为独立测试需求。
- 不使用现有本地 `+240.78%` 结果作为验收阈值。
- 不允许只运行 SimLLM 组而缺少 same-spec Baseline。
- warmup 时间不得计入 measured benchmark duration。

## 4. 官方主测试规格

### 4.1 固定环境

| 项目 | 要求 |
| --- | --- |
| 硬件 | 单张 Huawei Ascend 910B2 |
| 模型 | `Qwen/Qwen2.5-14B-Instruct` |
| 精度 | FP16 |
| Tensor Parallel | 1 |
| 服务端 | vLLM-HUST + vLLM-Ascend-HUST，版本和 commit 必须写入 artifact |
| SimLLM | 使用被测 commit，不得在 Baseline 与 SimLLM 组之间切换其他代码 |
| 设备选择 | 由官方 runner 分配；必须记录实际 device id 和芯片型号，不能从文件名推断 |

### 4.2 固定请求规格

| 参数 | 值 |
| --- | ---: |
| scenario | `simllm-random-online-warm-cache-throughput` |
| endpoint | `/v1/completions` |
| dataset | deterministic random |
| seed | `0` |
| num_prompts | `32` |
| input_len | `4096` |
| output_len | `32` |
| temperature | `0` |
| measured request_rate | `inf` |
| measured max_concurrency | `16` |
| warmup request_rate | `1` |
| warmup passes | `1` |
| warmup seed | 与 measured seed 相同 |
| warmup num_prompts | 与 measured num_prompts 相同 |
| max_num_batched_tokens | 不小于 `4096`，主规格固定为 `4096` |
| SimLLM KV cache entries | 不小于 `32`，主规格固定为 `32` |

选择 4096-token 输入是为了让负载以 prefill 为主；选择 32 个请求和 16 并发，是当前实现已验证能够稳定完成 warm-cache 饱和测试的主规格。该规格不是通用扩展性结论，因此还必须执行第 8 节的稳定性测试。

## 5. A/B 执行流程

每次 repetition 必须执行完整、相互隔离的 A/B 流程。

### 5.1 Baseline

1. 确认目标 device 空闲并记录初始 HBM。
2. 启动 SimLLM disabled 服务。
3. 等待 `/health` ready，并执行一个不计入结果的健康检查请求。
4. 使用第 4.2 节的 measured 参数执行饱和压测。
5. 保存原始结果、服务日志、resolved same-spec 和运行元数据。
6. 停止完整服务进程树，并确认端口和 NPU 内存释放。

### 5.2 SimLLM warm cache

1. 使用相同代码、模型和 server 参数启动 SimLLM enabled 服务。
2. 等待服务 ready。
3. 使用与 measured pass 完全相同的 prompt 集合、顺序和 seed 执行一次低速 warmup。
4. warmup 完成后不得重启服务、清空缓存或改变请求集合。
5. 执行 `request_rate=inf`、`max_concurrency=16` 的 measured pass。
6. 保存 warmup 结果、measured 原始结果、cache hit/rewrite 统计、服务日志和运行元数据。
7. 停止完整服务进程树，并确认资源释放。

### 5.3 重复次数

- 正式采集至少执行 3 个有效 repetitions，建议 5 个。
- 每个 repetition 都必须重新启动 Baseline 和 SimLLM 服务，不能复用上一轮缓存。
- 官方汇总采用各 repetition 的中位数，同时报告最小值、最大值和变异系数。
- 若主指标变异系数大于 5%，不得直接发布单一官方结果；应继续运行或定位环境抖动。

## 6. 公平性与 same-spec 要求

Baseline 与 SimLLM measured pass 之间，以下字段必须一致：

- model、dtype、tensor parallel、block size 和服务端内存参数
- dataset、prompt token ids、seed 和请求顺序
- input/output token length
- temperature、endpoint、request rate 和 max concurrency
- max_num_batched_tokens
- vLLM-HUST 与 vLLM-Ascend-HUST commits
- CANN、torch、torch-npu 和 Python 环境

唯一允许的功能性差异是 SimLLM enable/config 环境变量，以及 SimLLM 组在 measured pass 前执行 warmup。

脚本必须生成 resolved same-spec 文件及稳定的 spec hash。汇总程序在发现 A/B setting signature 不一致时必须失败，不能只打印 warning。

## 7. 指标和证据

### 7.1 必须报告

- successful requests / failed requests
- benchmark duration
- request throughput，req/s，主指标
- input、output、total token throughput
- mean、median、P99 TTFT
- mean、median、P99 TPOT 或 ITL
- Peak concurrent requests
- 每轮和汇总后的提升百分比

提升计算方式：

```text
improvement_percent = (simllm_median - baseline_median) / baseline_median * 100
```

### 7.2 必须新增或导出的 SimLLM 证据

正式 artifact 必须能够证明 measured 请求确实命中并触发 KV 复用，至少包含：

- warmup cache entries committed
- measured cache lookup count
- measured cache hit count 和 hit ratio
- scheduler rewrite request count
- rewritten/skipped prefill token count
- eviction count
- SimLLM fallback/error count

若当前 runtime 尚未公开这些统计，工程师需要增加结构化 metrics 或在 runner 中导出机器可读统计。仅凭吞吐变快不能认定为有效 SimLLM 官方数据。

## 8. 稳定性与扩展性测试

主测试之外必须增加一组非发布型稳定性检查，用来暴露当前实现的容量边界：

- 256 prompts
- 2048 input tokens
- 32 output tokens
- `request_rate=inf`
- `max_concurrency=64`

此检查用于验证无 OOM、无 EngineCore fatal、无异步调度 placeholder assertion，并不要求吞吐提升。若失败，应将结果标记为已知限制并保留日志，不能用部分成功请求计算吞吐提升。

在扩展主规格前，工程师还应特别检查 SimLLM scheduler rewrite 与 vLLM async scheduling 的一致性。历史自测曾在大请求量下出现 `num_output_placeholders >= 0` 断言，因此官方脚本不能默默忽略 500 响应或仅统计成功子集。

## 9. 有效性门禁

满足以下全部条件，一轮测试才是有效 repetition：

1. Baseline 和 SimLLM 都完成全部请求，`completed == num_prompts`。
2. failed requests 为 0，HTTP 5xx 为 0。
3. 服务端不存在 OOM、fatal error、AssertionError 或 EngineDeadError。
4. SimLLM warmup 成功提交预期数量的缓存条目。
5. SimLLM measured pass 的 cache hit/rewrite 统计非零，并与测试预期相符。
6. A/B resolved same-spec 除 SimLLM 开关和 warmup 状态外完全一致。
7. 未发生设备复用、其他进程占用或测试期间的 NPU reset。
8. 原始结果和日志完整，能够从 artifact 复算汇总结果。

任一条件不满足时，runner 必须非零退出，并禁止写入 canonical official submissions。

## 10. 工程交付物

工程师需要提交：

1. 官方 same-spec JSON，例如：
   `docs/official-baselines/official-ascend-simllm-warm-cache-throughput-qwen25-14b-910b2.json`
2. 官方执行脚本，例如：
   `scripts/run-official-ascend-simllm-warm-cache-throughput.sh`
3. A/B orchestration、warmup 和进程清理实现。
4. 机器可读的 SimLLM cache hit/rewrite metrics。
5. 单元测试和 dry-run 测试，覆盖参数解析、same-spec 一致性、失败请求门禁和汇总计算。
6. 一份 artifact schema 文档或 JSON Schema。
7. 至少 3 次有效运行产生的原始 artifacts 和汇总报告。
8. 合法的 leaderboard submission/manifest；在评审确认前先以 report-only 或 candidate 状态保存，不直接进入公开 canonical 数据。
9. 一份“参考脚本到官方实现”的映射说明，逐项说明第 2.1 节的逻辑被复用、替换或调整的位置及理由。

建议输出目录：

```text
<result-dir>/
├── spec.json
├── repetitions/
│   ├── 01/
│   │   ├── baseline/
│   │   └── simllm-warm-cache/
│   ├── 02/
│   └── 03/
├── summary.json
├── summary.md
├── run_leaderboard.json
└── leaderboard_manifest.json
```

## 11. 已知陷阱

- `request_rate=1` 只能用于 warmup，不能用于 measured throughput。
- `max_num_batched_tokens < input_len` 会切分 prompt，使 warmup 与 measured 阶段的 hash/KV 覆盖不一致。
- 不显式设置 `temperature=0` 时，模型 generation config 可能启用 Ascend TopK/TopP 路径并触发算子错误。
- 过大的 SimLLM KV cache 会占满 HBM；缓存容量应与测试请求数匹配，并记录峰值 HBM。
- KV extraction 不应重新构造超大 padded hidden-state tensor；应复用 scheduler 阶段 embedding。
- benchmark client 的可选 Triton import warning 不等同于服务端失败，但必须在报告中区分 warning 与 fatal error。
- 清理阶段发送 SIGTERM/SIGKILL 的日志不能被误判为运行期 EngineDeadError；runner 应按时间阶段分类。

## 12. 验收标准

### 12.1 脚本验收

- 支持 dry-run，能够只解析并输出最终 A/B 参数。
- 支持官方 runner 自动选择设备，不在脚本中硬编码 device 5。
- 支持指定 repetitions、结果目录和模型本地路径。
- 任何请求失败、same-spec 漂移或服务端 fatal 均非零退出。
- 无论成功或失败，都能清理服务进程、端口和 NPU 资源。
- 脚本和 spec 通过仓库现有 lint、单测和 official baseline validation。

### 12.2 数据验收

- 至少 3 个有效 repetitions。
- 每轮 A/B 均为 100% 请求成功。
- 主指标变异系数不超过 5%。
- cache hit/rewrite 证据完整。
- 可从 raw artifact 独立复算 summary。
- 吞吐收益按实报告，不以复现本地 `+240.78%` 作为通过条件；若结果不增反降，也必须保留并解释。

## 13. 本地可行性结果，仅供工程设计参考

以下结果来自非官方本地脚本，只用于说明规格能够暴露 SimLLM 吞吐差异，不得进入 official submission：

| 指标 | Baseline | SimLLM warm cache | 差异 |
| --- | ---: | ---: | ---: |
| Requests/s | 1.4055 | 4.7897 | +240.78% |
| Total tokens/s | 5801.86 | 19771.68 | +240.78% |
| Mean TTFT | 3616.75 ms | 547.20 ms | -84.87% |
| 成功请求 | 32/32 | 32/32 | 均无失败 |

工程师应从官方实现重新采集数据，不应复制本地结果、结果目录或本地汇总文件。

## 14. Issue/工单标题建议

```text
[Benchmark][SimLLM] Add official saturated warm-cache throughput benchmark on Ascend 910B2
```

完成定义：官方脚本、same-spec、结构化命中证据、测试和至少 3 次候选运行数据全部提交并通过评审，且候选结果满足第 9 节有效性门禁。
