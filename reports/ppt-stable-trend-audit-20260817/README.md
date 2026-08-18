# 910B2 9×3 稳定检查点：PPT-ready 审计报告

审计日期：2026-08-17\
数据基线：`vLLM-HUST/vllm-hust-benchmark@f67475ec32692c9f65dbabe80a37c50905bf6443`\
适用范围：单机、单卡 Huawei Ascend 910B2；9 个固定 workload；三组 core/plugin 精确版本对。

## 一页结论（可直接用于 PPT）

**建议标题：**「9 类 workload 的历史版本健康线（Ascend 910B2）」

**可直接使用的结论：**

> 在三组 2026 年 7 月历史版本检查点上，9 类 workload 的同规格结果完整覆盖 27/27 个 cell；按预先声明的非显著回退带（throughput 1%、TTFT
> 10%、TBT 5%），全部通过稳定性检查。这说明系统在该段功能演进中整体保持稳定，没有发现超过阈值的性能回退。

必须同时保留以下限定：

- 三点是 **historical version health checkpoints（历史版本健康检查点）**，不是由其 commit subject 定义的能力里程碑，也不是 current
  latest；本图不代表 current latest。
- C1/C2 与 C3 使用不同 Ascend plugin pin，而且 C3 的 plugin 反而是 C1/C2 plugin 的祖先；因此 这不是严格单调前进的 full-stack
  发布序列。
- 27 个 cell 均保留独立三次 benchmark invocation，并按预声明 primary metric 取中位数。失败 invocation
  也保留为诊断证据，且不替代成功重复；PPT 主页面和公众网站仍只写 **“代表性实测”**，不展示质控过程词。
- 这张图支持“整体稳定/无显著回退”，**不支持“9 类性能持续提升”**。
- 严格同规格 vLLM baseline 仅覆盖 **1/9 workload（random-latency）**；不能用这 27 点替代 baseline 对比，也不能宣称相对 vLLM
  的加速比已经完成。

### 可用数字

| 项目 | 审计数字 | | ------------------------------ | --------------------------------------: | | 完整矩阵 |
9 workloads × 3 checkpoints = **27/27** | | 在线 benchmark cell | **18** | | 离线 benchmark cell | **9**
| | 独立三次 invocation 取中位数 | **27/27** | | 单次 invocation | **0/27** | | 证据等级 A / B / C | **27 / 0 /
0** | | `repeat_group` 非空 | **0/27** | | `canonical_aggregate` 非空 | **0/27** | | 严格三重复 vLLM baseline
| **1/9 workloads** | | 仍需补齐的 baseline invocation | **24（8 workloads × 3）** |

### 当前可用 / 临时可用 / 待补（内部审计表，不直接放 PPT 主页面）

| 内部状态 | 当前范围 | 对外可见文案 | 内部行动 | | ---------------------- |
\------------------------------------------- | ----------------------------- |
--------------------------------- | | 当前可用（正式三重复） | 9 workloads 在 C1/C2/C3，共 27 cells | “代表性实测” | 保留
repeat suite 与全部 raw | | 历史 baseline | 当前 CANN 缺少上游所需算子 | 不展示整体 vs-vLLM 结论 | 保留功能不可用失败证据 | | current
对照 | 9 workloads × C3/current 已完成 | 另见 P2 安全数字表 | 使用显式 cache 合同 |

内部执行矩阵和进度记录在 benchmark
[issue #214](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/214)。方法细节应放数据说明页或附录， PPT
可见主页面只保留“历史健康检查点 / 代表性实测”以及证据允许的结论。

证据等级定义：A = 原始 artifact + 独立三次 invocation 中位数；B = 单次 invocation，但有原始结果、日志、环境和 checksum；C = 只有单次 raw
result/manifest。本矩阵 27 个 cell 现均为 A 级。

## 不可使用的表述

以下文案均越过现有证据边界，不应出现在 PPT、网站或口头汇报中：

- “9 类 workload 性能持续提升”或“所有曲线持续上升”。
- “三个重要能力里程碑”或把 `fix beam search import paths`、`move KV tiering examples` 和 macOS L2 cache fix
  直接命名为国产卡推理能力突破。
- “current/latest 性能趋势”“覆盖当前主线”“已验证 #173/#220/#236、Ascend #153/#216 后的性能稳定性”。
- “每个点均为三次独立重复”“每个 workload 都做了 3-repeat median”。
- “已经完成与 vLLM 的严格同规格对比”或引用整体相对 vLLM 加速比。
- 把 `metadata.data_source=real-online-stable-trend-delta` 当作执行类型事实。9 个离线 cell 也带有该统一标签；执行类型必须以
  invocation 与 raw artifact 为准。
- 把默认 `performance_index` 当成统一性能优劣结论。它优先选择 throughput，会隐藏同一在线 workload 的 TTFT 变化。

另有一组仅限 **内部审计/数据说明页** 的术语：单次 invocation、独立三重复、中位数、补跑中、 正式统计复核。它们不是虚假表述，但不应出现在 PPT 主页面或公众网站的可见文案中。

## 版本检查点与能力里程碑

### 当前横轴的真实含义

| 点 | 精确 core/plugin | commit 时间与 subject | 能力覆盖 | 审计分类 | | --- | --------------------------- |
\------------------------------------------------------------------------- |
----------------------------------------------------------------------- | ------------------ | | C1
| `0657f3f2a6` / `03a12f9bdd` | core 2026-07-13, `fix beam search import paths`; plugin 是 Ascend PR
#151 | Core #42；不含 Core #124/#173/#220/#236；plugin 含 #151，不含 #153/#216 | 历史版本健康检查点 | | C2 |
`73187bc8ba` / `03a12f9bdd` | core 2026-07-16, `move KV tiering examples out of runtime` | 与 C1 相同 |
历史版本健康检查点 | | C3 | `1aa7cd10b7` / `03ae1d03db` | core 2026-07-19, macOS CPU extension L2 cache fix |
Core #42、#124；不含 #173/#220/#236；plugin 早于 #151/#153/#216 | 历史版本健康检查点 |

重要补充：`03ae1d03db` 是 `03a12f9bdd` 的祖先，所以按完整 core/plugin 组合看，C1→C2→C3 不是一条严格按时间推进的 full-stack release
lineage。网站横轴只能按 core 检查点排序，不能声称 是完整系统能力逐步累积。

### 能力 PR coverage map

| 能力 PR | 能力含义 | C1 | C2 | C3 | hardened current candidate | | ----------- |
------------------------------------------ | :-: | :-: | :-: | :------------------------: | | Core
#42 | unified_comm + GroupCoordinator | ✓ | ✓ | ✓ | ✓ | | Core #124 | KV cache tiering
residency/lifecycle | — | — | ✓ | ✓ | | Core #173 | prefix-aware routing integration | — | — | — | ✓
| | Core #220 | KV transfer instrumentation/layout capture | — | — | — | ✓ | | Core #236 |
KV-recovery observer sidecars | — | — | — | ✓ | | Ascend #151 | AddRmsNormBias runtime-capability
gate | ✓ | ✓ | — | ✓ | | Ascend #153 | mapped-host gather path | — | — | — | ✓ | | Ascend #216 |
KV-recovery first-compute observation | — | — | — | ✓ |

这里的 `hardened current candidate` 是审计时的 main pair：core `43341b177dbaa8c7f04662f71e885ee7dfe22704` +
plugin `0a46364814eedd3314f04eff3490c3ab422438bd`。GitHub ancestry 审计确认该 pair 覆盖表中所有 PR， 但 snapshot
历史 snapshot 对这个精确版本对是 **0/9**；后续 P2 已按版本化显式 cache 合同完成 C3/current 的 9×2 对照，结果不混入本历史矩阵。

仓内存在一个 post-Core-#173 的部分候选：core `e4ce33646f` + plugin `0f38988f47`，仅有
agent、random-online、sharegpt-online、sharegpt-throughput、sonnet-throughput **5/9** 个 workload，
且不是完整三重复矩阵；另一个 `5536d0873f` 只有 2 个重复 workload。它们不能替代当前 9×3 横轴。仓内没有对齐 Core #220/#236 或 Ascend
#153/#216 的精确同规格 9-workload 数据。

结论：**没有现成数据可以把当前三点无损替换成三个真正能力里程碑。** 若 PPT 必须讲能力里程碑， 应把能力 PR 时间线与本健康线分成两张图；不要重命名横轴冒充。

## 原始指标与预声明 primary metric

指标选择规则在看单条曲线涨跌之前统一声明：

- 在线/交互 workload：primary = `ttft_ms`（越低越好）；`tbt_ms` 与 throughput 为 guardrail。
- 显式 throughput workload：primary = `throughput_tps`（越高越好）。
- `random-latency`：primary = 一次 offline latency invocation 的平均 measured latency，导出为 `ttft_ms`（越低越好）。

该规则按 workload 类型确定，不按结果事后挑最好指标。

| Workload | Primary | C1 | C2 | C3 | C1→C3 raw change | C1→C3 benefit | | ------------------------
| \---------------- | -------: | -------: | -------: | ---------------: | ------------: | |
agent-research-online | `ttft_ms` | 3147.526 | 3109.841 | 3031.676 | -3.68% | +3.68% | |
instructcoder-online | `ttft_ms` | 114.893 | 114.994 | 114.144 | -0.65% | +0.65% | |
prefix-repetition-online | `ttft_ms` | 279.129 | 259.331 | 261.991 | -6.14% | +6.14% | |
random-latency | `ttft_ms` | 4909.350 | 4907.898 | 4946.433 | +0.76% | -0.76% | | random-online |
`ttft_ms` | 234.514 | 230.557 | 231.664 | -1.22% | +1.22% | | sharegpt-online | `ttft_ms` | 128.695
| 131.942 | 128.029 | -0.52% | +0.52% | | sharegpt-throughput | `throughput_tps` | 2054.498 |
2041.982 | 2042.348 | -0.59% | -0.59% | | sonnet-throughput | `throughput_tps` | 3654.776 | 3621.894
| 3634.784 | -0.55% | -0.55% | | visionarena-online | `ttft_ms` | 431.601 | 420.048 | 430.758 |
-0.20% | +0.20% |

默认 `performance_index` 使用 throughput 时，agent 的 throughput 从 205.522 降至 205.231 （-0.14%），会遮住 TTFT 的
11.10% 改善；prefix throughput 仅 +0.03%，也会遮住 TTFT 的 6.14% 改善。PPT 应直接用上表和
`stable_trend_raw_metrics.csv`，不要只截默认指数图。

同时要诚实展示反向但在噪声带内的值：sonnet throughput C1→C3 为 -0.55%，random-online TTFT 变慢 0.75%。这正是“无显著回退”，而不是“持续提升”。

## 重复性与原始证据矩阵

完整 27-cell 明细见 `stable_trend_evidence_matrix.csv` 和 `stable_trend_audit.json`。每个 cell 均列出：

- `source_path` 与 `raw_result_path`
- online/offline 执行类型及 real-online 证据
- 独立 benchmark invocation 数
- 每次 invocation 的请求/iteration 数
- warmup 数量与语义
- 是否为独立 3-repeat median
- `repeat_group` / `canonical_aggregate` 是否存在
- `inferred_fields`
- exact `resolved_spec_hash`
- 原始 throughput / TTFT / TBT / error rate

重复结构汇总：

| Workload | C1/C2/C3 独立 invocation 数 | 单次请求或 measured unit | Warmup | 独立三重复中位数 | |
\------------------------ | --------------------------- | -------------------------------- |
------------------------------ | ---------------- | | agent-research-online | 3 / 3 / 3 | 32
requests | serve invocation 未声明 warmup | 是 | | instructcoder-online | 3 / 3 / 3 | 2048 requests | 同上
| 是 | | prefix-repetition-online | 3 / 3 / 3 | 200 requests | 同上 | 是 | | random-latency | 3 / 3 / 3
| 30 measured iterations × batch 8 | 10 iterations | 是 | | random-online | 3 / 3 / 3 | 200 requests
| serve invocation 未声明 warmup | 是 | | sharegpt-online | 3 / 3 / 3 | 200 requests | 同上 | 是 | |
sharegpt-throughput | 3 / 3 / 3 | 200 requests | 0 | 是 | | sonnet-throughput | 3 / 3 / 3 | 每次 200
requests | 0 | 是 | | visionarena-online | 3 / 3 / 3 | 每次 1000 requests | serve invocation 未声明 warmup
| 是 |

27 个 snapshot entry 的 `repeat_group` 与 `canonical_aggregate` 均为空。三重复的可审计信息存放在 相邻
`repeat_suite.json`，不能从空字段反推。仅 agent 的三个 cell 有 inferred fields：补入 online CLI 默认 `no_stream=false`
并据此重算 exact spec hash；其余 24 个没有 inferred fields。

`metadata.data_source=real-online-stable-trend-delta` 是 campaign 级旧标签，不是执行类型。矩阵已按 artifact 修正为 18 个
online cell、9 个 offline cell；PPT 必须采用矩阵中的 `execution_mode`。

## Baseline 与 current 对照状态

历史 constraints 中的 `declared_baseline_engine=vllm` 与 `baseline_status=pending-baseline` 仍保留，不能解释为
baseline 已完成。原始 `baseline_gap.csv` 记录的是启动 issue #214 时的 artifact 缺口；随后的兼容性验证证明 upstream v0.18 和正式
v0.23 baseline 在当前 CANN 8.5.1 环境均缺少 `aclnnAddRmsNormBias`，无法启动。审计没有关闭 fusion，也没有用 HUST fork 冒充
upstream baseline。因此 PPT 仍不能展示整体 “vs vLLM” 柱状图或倍数。

Current 对照已经按 `p2-explicit-cache/v1` 完成：C3 与 hardened-current exact pair 在九个 workload 上各保留三次独立
invocation。非 prefix 场景显式关闭 prefix cache；prefix 场景显式 开启 cache 并设置安全的 Knorm 合同。九项 primary metric
均未达到可行动回退阈值；完整结果见 `reports/issue-214-p2-closure-20260817/`。

## 后续维护要求

1. 网站三点继续称 7 月 13/16/19 的历史健康检查点，不把 commit subject 改写成能力里程碑。
1. PPT 使用 `stable_trend_raw_metrics.csv` 与 workload-specific primary metric；默认指数图只作导航。
1. 后续 stable matrix producer 应把 repeat suite 提升为一等字段；当前 snapshot 的
   `repeat_group/canonical_aggregate` 为空不代表缺少三重复，必须沿 `repeat_suite_path` 审计。
1. upstream baseline 只有在兼容的 CANN/driver 环境中按同一官方合同重测后才能发布；不得关闭 fusion 或更改 benchmark 语义来制造数值。
1. 所有失败 invocation 均应保留在 diagnostic provenance 中，不得挑最好值或删除异常。

## 附件

- `stable_trend_evidence_matrix.csv`：27-cell 证据矩阵。
- `stable_trend_raw_metrics.csv`：27 个原始指标点与预声明 primary metric。
- `stable_trend_audit.json`：完整机器可读审计包。
- `capability_coverage.csv`：检查点/能力 PR coverage map。
- `baseline_gap.csv`：严格 vLLM baseline 可复用性与最小补测清单。
