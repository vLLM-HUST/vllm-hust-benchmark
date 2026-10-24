# Qwen3.5-35B-A3B MTP2 / EAGLE3 Graph 优化报告

## 固定测试口径

- 数据集：GSM8K materialized，关闭 shuffle
- 请求数：200；最大并发：16；每请求输出：256 tokens
- `temperature=0`、`seed=0`、`request-rate=inf`
- Target：`/workspace/models/Qwen3.5-35B-A3B`
- EAGLE3：`/workspace/models/Qwen3.5-35B-A3B-Eagle3-Specforge`
- TP2 + EP2，`max-num-seqs=16`，`max-num-batched-tokens=8192`
- BF16，block size 128，prefix caching、chunked prefill、async scheduling 开启
- Graph：`FULL_AND_PIECEWISE`，Draft proposer 为 `FULL`

Graph target-only baseline 固定为一次正式 N200 测量：`558.1435 tok/s`、
`91.7148 s`。所有 gamma 和投机方法都与该值比较，不为每个试验重测或更换 baseline。

## 最终 Graph 结果

| 方案 | rollout gamma | Run 1 (tok/s) | Run 2 (tok/s) | 均值 (tok/s) | 相对固定 baseline | 目标 |
|---|---:|---:|---:|---:|---:|---:|
| Target-only | - | 558.14 | - | 558.14 | 1.000x | - |
| Frontier MTP2 head | 4 | 773.31 | 799.71 | **786.51** | **1.409x** | >=1.3x，通过 |
| EAGLE3 | 5 | 695.47 | 670.13 | **682.80** | **1.223x** | >=1.2x，通过 |

两种方案的 200 个请求均全部成功。MTP 两轮平均接受率为 `78.91%`、平均接受长度
为 `4.16`；EAGLE3 两轮平均接受率为 `51.30%`、平均接受长度为 `3.56`。

这里的 “MTP2” 指 Qwen3.5 checkpoint 内置的两层 MTP head；最佳 Graph rollout
一次生成 4 个候选 token，因此运行时 gamma 为 4，并不表示模型被换成 MTP4 head。

## 关键优化

### MTP2

1. 保持 target 与 proposer 的 Graph 路径，严格禁止静默回退到 Eager。
2. 将 rollout gamma 从 2 扫描到 4，gamma=4 在 B16 下摊薄 target verification 成本。
3. 增加 cohort refill，活跃请求降到 8 或以下时才补入新请求，避免持续重排 Mamba/KV 状态。
4. 保持 TP2 + EP2、异步调度和 APC，不启用会改变输出路径的 local argmax。

### EAGLE3

1. Draft KV 使用独立 128 block，target hybrid cache 保持 2048 page；compact KV group
   固定为 4，避免把 draft group 错归入 target Mamba groups。
2. 共享稳定 decode 批次的 GDN speculative metadata，并行更新 4 组 target Graph 状态。
3. Draft body 使用 W8A16，保留 logits 和验收语义。
4. cohort refill 阈值为 8，减少请求加入/退出导致的状态重建。
5. B16 下对每个活跃请求数 `1..16` 捕获精确验证尺寸，不再把 9--15 个请求的
   `6 x batch` token 填充到 96。该策略已固化在 launcher 的 `exact` 捕获策略中。

精确捕获前，EAGLE3 gamma=5 的 N200 为 `579.44 tok/s`；优化后两轮均值为
`682.80 tok/s`，提升约 `17.8%`。额外 Graph 内存约 `0.15 GiB`，总 Graph 内存
约 `0.79 GiB`。

## Gamma 筛选

EAGLE3 在相同 B16、N64 和精确捕获配置下：

| gamma | 吞吐 (tok/s) | 接受率 | 接受长度 |
|---:|---:|---:|---:|
| 4 | 692.64 | 61.50% | 3.46 |
| 5 | **706.09** | 52.97% | 3.65 |
| 6 | 677.38 | 45.67% | 3.74 |

gamma=5 是当前拐点。gamma=4 虽然接受率更高，但每轮提交 token 更少；gamma=6 的额外
Draft/verification 成本超过接受长度收益。因此正式配置固定 gamma=5，不通过降低上限掩盖
选择器问题。

## EAGLE3 Adaptive Gamma

当前 Adaptive 配置将候选范围扩展为 `gamma=1..6`，只在安全的 cohort 边界允许改变
宽度；稳定 cohort 内使用原生 `gamma=5` Graph 路径，避免逐 decode step 重跑选择和
重复修改 runner metadata。与同一测试窗口内的固定 `gamma=5` 对照相比：

| 方案 | 吞吐 (tok/s) | 成功请求 | 接受长度 | 相对固定 gamma |
|---|---:|---:|---:|---:|
| 固定 gamma=5 | 678.04 | 200/200 | 3.61 | 1.000x |
| Adaptive gamma=1..6 | **683.68** | 200/200 | 3.55 | **1.008x** |

候选集合在运行时确认为 `[1, 2, 3, 4, 5, 6]`。正式轮没有 Graph miss 或 Eager
回退，吞吐高于固定 gamma `0.83%`。配置容量为 6，但运行锚点为 5；投机统计向量固定按
容量 6 上报，解决了原先前端按 6 创建 Prometheus counter、调度器按 5 返回向量导致的
`IndexError`。

需要区分两件事：Adaptive 模式已经稳定超过同窗口固定 gamma 对照，但在这组单次连续
B16 burst 中，控制器判断 `gamma=5` 始终更优，因此没有发生 gamma 切换。该结果证明
Adaptive 的运行时开销已经被消除，并不证明本次提升来自动态切换；动态切换收益还需要
混合并发或工作负载阶段变化的独立测试验证。

## MTP2 Adaptive Gamma

MTP2 Adaptive 的配置上限同样为 6，但候选只包含完整 MTP2 组：`[2, 4, 6]`。运行锚点
为固定扫描的最优值 4；同窗口结果如下：

| 方案 | 吞吐 (tok/s) | 成功请求 | 接受长度 | 相对固定 gamma |
|---|---:|---:|---:|---:|
| MTP2 固定 gamma=4 | 777.73 | 200/200 | 4.17 | 1.000x |
| MTP2 Adaptive gamma=2/4/6 | **810.53** | 200/200 | 4.15 | **1.042x** |

Adaptive 吞吐高于同窗口固定 gamma `4.22%`，全程保持 `mtp_strict_graph=true`。第一次
N200 尝试暴露了尾批捕图缺口：B12、q5 需要 60 tokens，但稀疏 capture list 只能填充到
64，无法构造 uniform FULL descriptor。修复后只为合法 q3/q5/q7 宽度注册 B1..16 的
48 个精确 FULL decode 图，覆盖 45、60 等尾批尺寸，不通过关闭 strict guard 掩盖 Eager
回退。

该稳定 B16 burst 最终选择 `gamma=4`，没有发生 cohort 内切换。控制器仍持续维护
2/4/6 三个候选的在线收益；此次结果同样表示动态路径已经消除额外开销，实际切换收益需
在阶段性负载中单独评估。

## Graph 路径核验

- target 完成 mixed prefill/decode `PIECEWISE` 和 decode `FULL` 捕获；
- MTP2/EAGLE3 proposer 日志均为 `ACLGraphWrapper: runtime_mode=FULL`；
- 正式运行未出现 graph miss、fallback、force-eager 或 graph-break；
- EAGLE3 精确配置捕获 16 个 mixed graph 和 16 个 FULL decode graph。

因此原问题不是整条请求路径回退到 Eager，而是稀疏捕获在 cohort refill/drain 阶段产生了
大量 padding，且 speculative 状态维护没有获得 target-only 同等比例的 Graph 收益。

## 与 Eager 收益的差距

历史同口径 Eager gamma=2 结果仅作为参考：

| 方案 | Eager 吞吐 (tok/s) | Eager baseline (tok/s) | 相对 Eager baseline |
|---|---:|---:|---:|
| MTP2 gamma=2 | 140.70 | 61.32 | 2.294x |
| EAGLE3 gamma=2 | 111.61 | 61.32 | 1.820x |

最终 Graph 的绝对吞吐远高于 Eager，但相对加速比尚未达到上述 Eager 比值。若按固定 Graph
baseline 换算，需要 MTP 达到 `1280.63 tok/s`、EAGLE3 达到 `1015.82 tok/s`；当前分别
还差约 `62.8%` 和 `48.8%`。原因是 target-only 从 Graph 获得约 `9.10x` 的提升，而
Draft forward、验收、KV/Mamba 状态维护及同步没有获得同等倍率。gamma 4/5/6 扫描已经
排除仅靠继续调 gamma 达到该目标的可能，后续需要 fused proposer/acceptance kernel 或
target-Draft overlap，而不是继续增加捕获桶。

## 优化配置

- MTP2 head B16：`configs/qwen35-35b-a3b-frontier-mtp2-b16.toml`
- MTP2 Adaptive B16：`configs/qwen35-35b-a3b-frontier-mtp2-adaptive.toml`
- EAGLE3 B16：`configs/qwen35-35b-a3b-eagle3.toml`
- EAGLE3 Adaptive B16：`configs/qwen35-35b-a3b-eagle3-adaptive.toml`

前者保留原生 MTP2 head，但使用最佳 rollout gamma=4；原有
`qwen35-frontier-mtp2` 固定 gamma=2 兼容协议保持不变。

## 原始数据

- 固定 Graph baseline：
  `benchmark_results/qwen35_spec_compare_gsm8k/20260928-fixed-c16-n200/graph/baseline/gsm8k.json`
- MTP2 Graph 两轮：
  `benchmark_results/qwen35_frontier_mtp2_gsm8k/20260928-cohort8-n200-mtp4{,-r2}/gsm8k.json`
- EAGLE3 Graph 两轮：
  `benchmark_results/qwen35_eagle3_gsm8k/20260928-g5-exact-b1to16-w8a16-n200{,-r2}/gsm8k.json`
- EAGLE3 Adaptive 同窗口固定对照及三轮 Adaptive：
  `benchmark_results/qwen35_eagle3_gsm8k/20260929-adaptive-g4to5-n200/`
- EAGLE3 Adaptive gamma=1..6 同窗口对照：
  `benchmark_results/qwen35_eagle3_gsm8k/20260929-adaptive-g1to6-n200/`
- MTP2 Adaptive gamma=2/4/6 同窗口对照：
  `benchmark_results/qwen35_frontier_mtp2_gsm8k/20260929-adaptive-g2to6-n200/`
- Eager 参考：
  `benchmark_results/qwen35_spec_compare_gsm8k/20260928-fixed-c16-n200/eager/`

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
