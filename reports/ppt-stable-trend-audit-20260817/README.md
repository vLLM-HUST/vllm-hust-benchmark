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
  latest。
- C1/C2 与 C3 使用不同 Ascend plugin pin，而且 C3 的 plugin 反而是 C1/C2 plugin 的祖先；因此 这不是严格单调前进的 full-stack
  发布序列。
- 27 个 cell 中，只有 **6 个**是独立三次 benchmark invocation 后取中位数；其余 **21 个**为 单次 invocation。一次 invocation 内的
  200/1000/2048 个请求或 30 次 measured iteration 不能表述为“三次独立重复”。这是内部审计事实；PPT 主页面和公众网站只写
  **“代表性实测”**，不展示单次/多次/中位数/补跑中/统计复核等实验过程词。
- 这张图支持“整体稳定/无显著回退”，**不支持“9 类性能持续提升”**。
- 严格同规格 vLLM baseline 仅覆盖 **1/9 workload（random-latency）**；不能用这 27 点替代 baseline 对比，也不能宣称相对 vLLM
  的加速比已经完成。

### 可用数字

| 项目                           |                                审计数字 |
| ------------------------------ | --------------------------------------: |
| 完整矩阵                       | 9 workloads × 3 checkpoints = **27/27** |
| 在线 benchmark cell            |                                  **18** |
| 离线 benchmark cell            |                                   **9** |
| 独立三次 invocation 取中位数   |                                **6/27** |
| 单次 invocation                |                               **21/27** |
| 证据等级 A / B / C             |                          **6 / 20 / 1** |
| `repeat_group` 非空            |                                **0/27** |
| `canonical_aggregate` 非空     |                                **0/27** |
| 严格三重复 vLLM baseline       |                       **1/9 workloads** |
| 仍需补齐的 baseline invocation |               **24（8 workloads × 3）** |

### 当前可用 / 临时可用 / 待补（内部审计表，不直接放 PPT 主页面）

| 内部状态               | 当前范围                                    | 对外可见文案                  | 内部行动                          |
| ---------------------- | ------------------------------------------- | ----------------------------- | --------------------------------- |
| 当前可用（正式三重复） | Sonnet、VisionArena 在 C1/C2/C3，共 6 cells | “代表性实测”                  | 保留 repeat suite 与全部 raw      |
| 临时可用（单次）       | 其余 7 workloads 在 C1/C2/C3，共 21 cells   | “代表性实测”                  | P1：逐格验证旧 run，通常再补 2 次 |
| 待补 baseline          | P0 四个代表场景；全量仍缺 8 workloads × 3   | 主页面不展示整体 vs-vLLM 结论 | issue #214 P0 已启动              |
| 待补 current           | hardened-current exact pair，当前 0/9       | 主页面仍称“历史健康线”        | P0 先补 4×3；P2 再补 9×3          |

内部执行矩阵和进度记录在 benchmark
[issue #214](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/214)。方法细节应放数据说明页或附录， PPT
可见主页面只保留“历史健康检查点 / 代表性实测”以及证据允许的结论。

证据等级定义：A = 原始 artifact + 独立三次 invocation 中位数；B = 单次 invocation，但有原始 结果、日志、环境和 checksum；C = 只有单次 raw
result/manifest。C 级仅有 C3 random-latency。

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

| 点  | 精确 core/plugin            | commit 时间与 subject                                                     | 能力覆盖                                                                | 审计分类           |
| --- | --------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------ |
| C1  | `0657f3f2a6` / `03a12f9bdd` | core 2026-07-13, `fix beam search import paths`; plugin 是 Ascend PR #151 | Core #42；不含 Core #124/#173/#220/#236；plugin 含 #151，不含 #153/#216 | 历史版本健康检查点 |
| C2  | `73187bc8ba` / `03a12f9bdd` | core 2026-07-16, `move KV tiering examples out of runtime`                | 与 C1 相同                                                              | 历史版本健康检查点 |
| C3  | `1aa7cd10b7` / `03ae1d03db` | core 2026-07-19, macOS CPU extension L2 cache fix                         | Core #42、#124；不含 #173/#220/#236；plugin 早于 #151/#153/#216         | 历史版本健康检查点 |

重要补充：`03ae1d03db` 是 `03a12f9bdd` 的祖先，所以按完整 core/plugin 组合看，C1→C2→C3 不是一条严格按时间推进的 full-stack release
lineage。网站横轴只能按 core 检查点排序，不能声称 是完整系统能力逐步累积。

### 能力 PR coverage map

| 能力 PR     | 能力含义                                   | C1  | C2  | C3  | hardened current candidate |
| ----------- | ------------------------------------------ | :-: | :-: | :-: | :------------------------: |
| Core #42    | unified_comm + GroupCoordinator            |  ✓  |  ✓  |  ✓  |             ✓              |
| Core #124   | KV cache tiering residency/lifecycle       |  —  |  —  |  ✓  |             ✓              |
| Core #173   | prefix-aware routing integration           |  —  |  —  |  —  |             ✓              |
| Core #220   | KV transfer instrumentation/layout capture |  —  |  —  |  —  |             ✓              |
| Core #236   | KV-recovery observer sidecars              |  —  |  —  |  —  |             ✓              |
| Ascend #151 | AddRmsNormBias runtime-capability gate     |  ✓  |  ✓  |  —  |             ✓              |
| Ascend #153 | mapped-host gather path                    |  —  |  —  |  —  |             ✓              |
| Ascend #216 | KV-recovery first-compute observation      |  —  |  —  |  —  |             ✓              |

这里的 `hardened current candidate` 是审计时的 main pair：core `43341b177dbaa8c7f04662f71e885ee7dfe22704` +
plugin `0a46364814eedd3314f04eff3490c3ab422438bd`。GitHub ancestry 审计确认该 pair 覆盖表中所有 PR， 但 snapshot
对这个精确版本对是 **0/9**，尚未做同规格性能验证。

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

| Workload                 | Primary          |       C1 |       C2 |       C3 | C1→C3 raw change | C1→C3 benefit |
| ------------------------ | ---------------- | -------: | -------: | -------: | ---------------: | ------------: |
| agent-research-online    | `ttft_ms`        | 3265.631 | 3109.841 | 2903.262 |          -11.10% |       +11.10% |
| instructcoder-online     | `ttft_ms`        |  114.332 |  115.129 |  113.721 |           -0.53% |        +0.53% |
| prefix-repetition-online | `ttft_ms`        |  279.129 |  259.331 |  261.991 |           -6.14% |        +6.14% |
| random-latency           | `ttft_ms`        | 4909.350 | 4907.898 | 4887.086 |           -0.45% |        +0.45% |
| random-online            | `ttft_ms`        |  230.242 |  230.289 |  231.972 |           +0.75% |        -0.75% |
| sharegpt-online          | `ttft_ms`        |  128.695 |  128.419 |  126.988 |           -1.33% |        +1.33% |
| sharegpt-throughput      | `throughput_tps` | 2039.292 | 2041.982 | 2055.472 |           +0.79% |        +0.79% |
| sonnet-throughput        | `throughput_tps` | 3654.776 | 3621.894 | 3634.784 |           -0.55% |        -0.55% |
| visionarena-online       | `ttft_ms`        |  431.601 |  420.048 |  430.758 |           -0.20% |        +0.20% |

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

| Workload                 | C1/C2/C3 独立 invocation 数 | 单次请求或 measured unit         | Warmup                         | 独立三重复中位数 |
| ------------------------ | --------------------------- | -------------------------------- | ------------------------------ | ---------------- |
| agent-research-online    | 1 / 1 / 1                   | 32 requests                      | serve invocation 未声明 warmup | 否               |
| instructcoder-online     | 1 / 1 / 1                   | 2048 requests                    | 同上                           | 否               |
| prefix-repetition-online | 1 / 1 / 1                   | 200 requests                     | 同上                           | 否               |
| random-latency           | 1 / 1 / 1                   | 30 measured iterations × batch 8 | 10 iterations                  | 否               |
| random-online            | 1 / 1 / 1                   | 200 requests                     | serve invocation 未声明 warmup | 否               |
| sharegpt-online          | 1 / 1 / 1                   | 200 requests                     | 同上                           | 否               |
| sharegpt-throughput      | 1 / 1 / 1                   | 200 requests                     | 0                              | 否               |
| sonnet-throughput        | 3 / 3 / 3                   | 每次 200 requests                | 0                              | 是               |
| visionarena-online       | 3 / 3 / 3                   | 每次 1000 requests               | serve invocation 未声明 warmup | 是               |

27 个 snapshot entry 的 `repeat_group` 与 `canonical_aggregate` 均为空。三重复的可审计信息存放在 相邻
`repeat_suite.json`，不能从空字段反推。仅 agent 的三个 cell 有 inferred fields：补入 online CLI 默认 `no_stream=false`
并据此重算 exact spec hash；其余 24 个没有 inferred fields。

`metadata.data_source=real-online-stable-trend-delta` 是 campaign 级旧标签，不是执行类型。矩阵已按 artifact 修正为 18 个
online cell、9 个 offline cell；PPT 必须采用矩阵中的 `execution_mode`。

## Baseline 缺口

三点的 constraints 均仍是 `declared_baseline_engine=vllm`、`baseline_status=pending-baseline`，该状态
是正确的，不应删除或解释为 baseline 已完成。

严格 pin 为 upstream vLLM `bcf2be9612` + vLLM Ascend `e18643f8a4`（v0.18.0）。审计结果：

- **random-latency：可复用。** 有 exact-target 3-repeat attestation、3/3 成功、选定 repeat-03。
- agent、instructcoder、random-online、sharegpt-online、sharegpt-throughput、sonnet-throughput： 只有历史
  aggregate snapshot，没有 raw result、独立 repeat suite 与完整运行环境，不能作为严格 PPT baseline。
- prefix-repetition-online、visionarena-online：没有 admitted exact-spec baseline point。

最小 baseline 补测清单是其余 **8 workloads × 3 个独立 invocation = 24 次**；固定同一 target registry、模型、910B2
单卡、`gpu_memory_utilization=0.6`、`max_model_len=32768` 和输入身份。Vision 必须复用 content SHA
`2b41a850b78bc901caedf7e4d86ce52fc1804edc584f5f4da53a070df2a34b41`。每个 workload 按预声明 primary metric
取三次中位数，并保留每次 raw/server/bench/env/checksum。详见 `baseline_gap.csv`。

在这 24 次完成前，PPT 只能展示 HUST 版本内部健康线，不应出现整体 “vs vLLM” 柱状图或倍数。

## Current hardened checkpoint：必要性与成本

必要性为 **高**：当前 C3 截止 2026-07-19，覆盖不到 Core #173/#220/#236 与 Ascend #153/#216；若 PPT
标题包含“当前系统”“最新版本”或重点讲这些能力，必须补 current checkpoint。

审计时推荐的 exact pair：

- core `43341b177dbaa8c7f04662f71e885ee7dfe22704`
- plugin `0a46364814eedd3314f04eff3490c3ab422438bd`

先做禁 NPU import/startup smoke 与 exact-spec contract freeze，再运行：

1. **诊断级 9×1：** 9 次 invocation，纯 benchmark payload 按历史实测约 1.11 NPU-hours；加模型 加载、容器启动与一次失败重试，建议预算 2–3
   NPU-hours。在 7 张空闲卡并行时约 40–60 分钟。 它只能回答“能否运行、量级是否异常”，不能称 hardened PPT point。
1. **发布级 9×3（推荐）：** 27 次独立 invocation，纯 payload 约 3.34 NPU-hours；保守预算 6–9 NPU-hours，7 卡并行约 1.5–3
   小时。每个 workload 按本报告预声明 primary metric 取中位数。
1. **代表性 5×3：** agent、instructcoder、prefix-repetition、sonnet、VisionArena，共 15 次；覆盖 agent、代码、prefix
   cache、离线吞吐和多模态，纯 payload 约 2.78 NPU-hours，保守 4–6 NPU-hours，约 1–2 小时。它能支撑“关键场景 current
   spot-check”，不能升级为 9-workload current health line。

若本轮不补跑，PPT 和网站必须显式写：**“稳定趋势为 2026-07-13 至 2026-07-19 的历史健康线， 不代表 current latest。”**

## 最小补强行动（按优先级）

1. 立即把网站三点文案改为 7 月 13/16/19 “health checkpoint”，删除任何 capability-style 命名， 并增加“历史健康线、非 current
   latest”提示。
1. PPT 使用 `stable_trend_raw_metrics.csv` 与本报告的 workload-specific primary metric；默认指数图
   只能作为导航图，不作为性能结论图。
1. 若 PPT 要讲 current 能力，执行 current exact pair 的 9×3；时间不足则做 5×3 并明确 spot-check。
1. 单独补 8×3 upstream vLLM baseline；不要把 baseline 补测与 HUST 历史线重复性混为一项。
1. 后续 stable matrix producer 必须把 repeat suite 提升为一等字段，避免 snapshot 的
   `repeat_group/canonical_aggregate` 持续为空；统一 data_source 标签也应拆成 online/offline。

已建立 issue #214 执行矩阵：P0 先补 agent、prefix-repetition、sharegpt-throughput、 sonnet-throughput 的 strict
baseline/current 各 3 次；P1 把 21 个单次历史 cell 补齐；P2 完成 9-workload hardened-current 3-repeat 闭环。P0
已启动，内部进度只写 issue/审计附件，不进入公众 页面或 PPT 主页面。

## 附件

- `stable_trend_evidence_matrix.csv`：27-cell 证据矩阵。
- `stable_trend_raw_metrics.csv`：27 个原始指标点与预声明 primary metric。
- `stable_trend_audit.json`：完整机器可读审计包。
- `capability_coverage.csv`：检查点/能力 PR coverage map。
- `baseline_gap.csv`：严格 vLLM baseline 可复用性与最小补测清单。
