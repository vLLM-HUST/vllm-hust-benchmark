# B0 实测基线与相对性能公式

本说明将 B0 固定为实测比较基线，将相对吞吐效率记为 `η_B0`。B0 自身为 100%，B1 超过 100% 表示吞吐高于 B0。该指标衡量相对性能，数值没有 100% 上限。

B0 使用各测试项预先固定的版本和配置。当前课题四对照为 vLLM 0.18 + vLLM-Ascend 0.18； B1
使用相应运行记录中的实际配置。基线身份、配对关系和原始记录固定后不随优化结果改变。

## 1. 比较条件与实现自由

统一基础模型与源权重、指定数据集及实际请求、输入构造、采样与输出约束、质量和 SLO 要求、 硬件与完整资源边界、到达负载、缓存初始条件和计时边界。固定输出实验与自然结束实验分别报告。
相同数据集名称本身不证明请求和输入完全一致，应记录请求清单、输入身份及分词方式。

算子、融合、图模式、批处理策略、调度、推测解码和缓存管理可以不同；它们是引擎比较的内容。 允许的精度、量化或逻辑并行差异按测试大纲预先声明，并核对相应质量与资源条件。
不要求两个引擎采用相同内部执行图，也不以新实现更快为由重新选择 B0。

服务测量使用同一层级的端到端边界，包含该窗口内的排队、必要准备和完成等待。 初始化、编译、预热和完整生命周期分别说明；计数插桩轮与无插桩性能轮单列。
失败、超时、重试、长度终止及正确数与性能同时记录，不通过删去失败请求改善结果。

## 2. 符号

下标 `i=0` 表示 B0，`i=1` 表示 B1。

| 符号               | 定义                                   | 单位             |
| ------------------ | -------------------------------------- | ---------------- |
| `T_i`              | 预定测量窗口的完整耗时                 | 秒               |
| `O_i`              | 该窗口成功完成请求的输出 token 总数    | token            |
| `Q_i=O_i/T_i`      | 输出吞吐                               | token/秒         |
| `L_i`              | 同一统计规则下的 TTFT、TPOT 或请求延迟 | 毫秒或毫秒/token |
| `E_i`              | 对应窗口、完整资源范围的分摊费用       | 元               |
| `K_i=10^6 E_i/O_i` | 每百万输出 token 成本                  | 元/百万 token    |

所有比值要求分母为正且输入有限。无有效分母时报告缺失，不填零、不裁剪百分比。 `O_i` 的成功指响应完成；答错和长度终止的已完成输出按既定协议计入，任务质量另行评分。

## 3. η_B0、倍率与提升率

```text
S_Q = Q_1 / Q_0 = (O_1 × T_0) / (O_0 × T_1)
η_B0 = 100 × S_Q                          [%]
G_Q = 100 × (S_Q − 1) = η_B0 − 100       [%]
```

`η_B0=190%` 表示达到 B0 吞吐的 190%，即 1.9 倍、提升 90%；`η_B0=90%` 表示比 B0 低 10%。 目标提升 `g%` 对应
`S_Q≥1+g/100`、`η_B0≥100+g`。达到 B0 的百分比与提升百分比使用不同标签。

同一固定工作量、相同完成覆盖且 `O_0=O_1` 时：

```text
S_T = T_0 / T_1 = S_Q
η_B0 = 100 × T_0 / T_1
R_T = 100 × (1 − T_1/T_0) = 100 × (1 − 1/S_T)   [%]
```

完整回答的输出长度可以不同，此时保留 `S_T=T_0/T_1` 为批次完成时间倍率，吞吐使用 `S_Q`。 不能将时间倍率直接替代输出吞吐倍率。两者共同呈现，并附同轮质量与完成覆盖。

## 4. 时延与成本

```text
R_L = 100 × (1 − L_1/L_0)                         [%]
R_K = 100 × (1 − K_1/K_0)
    = 100 × [1 − (E_1 × O_0)/(E_0 × O_1)]        [%]
```

平均 TTFT、TPOT 按请求等权计算；P95/P99 分别与同分位数比较，不把逐请求吞吐的平均数当整体吞吐。 均值、分位数和测量窗口保持一致。

只有完整部署单位时间费用相同、费用与计时使用同一窗口，且没有未计入的额外资源费用时， `E_i=hT_i`，成本降幅才可简化为：

```text
R_K = 100 × (1 − Q_0/Q_1) = 100 × (1 − 1/S_Q)
```

固定租费按完整占用资源与时间分摊，不再乘利用率折扣；不同卡数、CPU、主存、存储或网络费用 按统一范围计入。公开报价折算标注为估算，实际账单另列。

## 5. 大纲第一项：有效算力与效率

大纲第一项以 `C` 和 `η` 为考核指标；`η_B0` 只作吞吐补充比较。最新完整回答算量见[逐项复算](../reports/full-answer-compute-20261101/README.md)。

V1.5 文档中的 `η` 在本说明及网站中写作 `η_ref`，其原公式保持为：

```text
C_i = F_i / (k_i × T_i × 10^12)       [TFLOPS/芯片]
η_ref,i = 100 × T_ref,i / T_i         [%]
```

`F_i` 是同轮必要有效算量，`k_i` 是全部参与芯片数；`T_ref,i` 由独立标定及依赖、资源约束推导。 `η_B0` 使用实测 B0 吞吐，`η_ref` 使用独立参考时间，两者各自计算。
原大纲的中期 50%、最终 90% 条件属于 `η_ref`，不转成 `η_B0` 的达标门槛。 新定义不追溯改写既有验收结论；正式验收依据相应版本的大纲和认可的测量方法。

归档的 `47.503348961177785` 秒来自旧 B1 的标定依赖图，不是 B0 实测或 B0 标定结果。 它不作为本说明 `η_B0` 的分子，也不作为跨实现的物理性能上限。

## 6. 已有完整回答配对的复算

使用 [原始汇总](../reports/full-answer-joint-20261030/joint-summary.json) 中完整的两个运行，
不新增模型调用、不修改原测量。MMLU-Pro 固定 200 请求，输出上限 8192 token，自然结束。

| 量                   |              B0 |             B1 |
| -------------------- | --------------: | -------------: |
| 成功请求             |             200 |            200 |
| 正确数               |             143 |            153 |
| 输出 tokens          |          706796 |         661040 |
| 完成时间（秒）       | 1425.4634465482 | 699.9029783104 |
| 输出吞吐（token/秒） |  495.8359344194 | 944.4737634862 |

```text
S_Q = (661040 / 699.9029783104) / (706796 / 1425.4634465482)
    = 1.9048110432
η_B0 = 190.4811043177%
G_Q = 90.4811043177%
R_K = 47.5013543426%    （同价租赁、同窗口估算）
```

这是一组已有配对，正确数分别为 143/200、153/200；不由单组结果推导置信区间或统计非劣性。 多次独立配对时先逐对计算 `S_Q,j`，再报告 `median(S_Q,j)` 及全部逐次值；
不能用两组吞吐中位数之比替代配对倍率中位数，也不将不同工况的最佳值拼成一次成绩。

复算命令（只读已有记录）：

```bash
python scripts/recompute_b0_baseline_metrics.py --check
```

结果见 [summary.json](../reports/b0-baseline-metrics-20261101/summary.json)。

## English summary

B0 is the fixed measured baseline. With output tokens `O_i`, elapsed seconds `T_i`, and throughput
`Q_i=O_i/T_i`, define `η_B0=100 Q_1/Q_0` and gain `G_Q=η_B0−100` percent. B0 is 100%; values above
100% are valid relative performance, not hardware utilization. A 90% gain means `η_B0=190%`.

Compare the same model/task, quality and output constraints, resource scope, workload and timing
boundary. Operators, batching and scheduling may differ. Only equal completed work and equal output
token counts permit `Q_1/Q_0=T_0/T_1`. For natural answers, report both ratios and same-run quality.
Latency reduction is `100(1−L_1/L_0)`. Cost per million tokens is `10^6 E_i/O_i`; its reduction is
`100[1−(E_1 O_0)/(E_0 O_1)]`, simplifying to `100(1−Q_0/Q_1)` only for equal complete deployment
rates and matching measurement windows. Preserve all repetitions and aggregate within-pair ratios.

The existing outline's independently calibrated efficiency is labeled `η_ref=100 T_ref/T`
separately. Its acceptance thresholds are unchanged and do not apply to `η_B0`. The archived 47.5033
s reference belongs to the old B1 graph, not B0. The measured full-answer pair yields
`η_B0=190.4811%`, throughput gain 90.4811%, and a same-rate rental cost reduction of 47.5014%.
