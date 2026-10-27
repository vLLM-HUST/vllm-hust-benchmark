# Pipeline Microbatch：PP 下的请求分组与批次准入

研究日期：2026-10-26。源码固定于
`vLLM-HUST/vllm-hust-pipeline-microbatch@4fe263f159faf7687316a6a334ac064791754764`。本轮阅读
policy、planner、host contract、相关单测与仓库验收报告；未运行硬件实验，也未独立审计宿主 executor。

## 1. 它想解决什么

Pipeline Parallelism（PP）把模型层分到不同 stage。一个批次从前 stage 流向后 stage 时，如果没有别的独立工作可供前 stage 执行，就产生空闲。多个
microbatch 可以在不同 stage 重叠推进。若不同 microbatch 的工作量差异很大，或 stage 成本失衡，流水线仍会等待。

示意，非实测时间线：

```text
时隙          1          2          3
PP stage 0    A 前半层   B 前半层   后续工作
PP stage 1    空闲       A 后半层   B 后半层
```

一个请求的下一个 decode token 仍依赖本轮结果；流水线利用的是不同请求组之间的独立性，不消除这个依赖。

这里的 microbatch 是**请求分组**。不是把单个 prompt 切成 token chunks，不是切模型层的实现，也不是投机或近似 KV 替换。PP=1 时没有该插件面向的跨 stage
流水线，配置会拒绝启用。

## 2. 当前插件实际负责什么

活跃入口为 `PipelineMicrobatchPolicy`，使用 `vllm.batch-admission-policy` API 1.1，非 monkey patch：

- 宿主提供不可变请求工作快照和正在飞行的 batch IDs。
- 插件维护 `request_id → batch_id`，返回 batch ID 和候选 request IDs。
- 宿主保留 Request、KV、实际调度输出、future 队列与执行的所有权，检查准入结果。
- 当前策略轮转选择非空且不在 flight 中的组，避免同一组重复飞行。
- request 不再出现在快照时回收分组；已有请求不会因负载变化而主动重新分组。
- 完成/中止回调在插件里主要记统计，in-flight 事实由宿主上下文提供。按宿主合同，非法决策或异常会禁用策略并恢复内建调度；本轮未独立执行该宿主故障路径。

所以它不是从零实现 PP 计算/通信流水线，而是在宿主已有能力上提供稳定分组和 admission policy。宿主已有基线也不应被假定为完全串行。

[policy 源码](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch/blob/4fe263f159faf7687316a6a334ac064791754764/src/vllm_hust_pipeline_microbatch/policy.py)；[宿主合同](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch/blob/4fe263f159faf7687316a6a334ac064791754764/HOST_CONTRACT.md)。

## 3. 两种分组策略

### balanced

新请求优先放入请求数量最少的组；相同时，选累计 context length 最小的组；再以组 ID 打破平局。不是直接把所有组的预计计算耗时均分。

已有映射保持不动。因此上下文增长、短请求结束、长请求继续时，最初平衡不保证后续平衡。新请求会依据当前快照分配，但不是全局重排。

### calibrated

用户提供各 PP rank 的成本系数。对一个候选 microbatch，设请求数 B、上下文长度和 C、该 rank 层数 L：

```text
T_rank = p0*B*L + p1*B + p2*C*L + p3*C + p4*L + p5
T_batch = max_rank(T_rank)
```

活跃 policy 为每个候选组构造“加入新请求之后”的快照，选择最大 rank
预测成本最小的组。这是贪心分配，不是全局调度最优解。没有从在线完成耗时自动学习系数的控制环；本轮读取的完成回调只记次数。

注意：`planner.assign_lowest_cost` 是按加入前成本选组的纯函数，**活跃 policy 并不调用它**。解释运行机制应以 `_select_assignment`
为准，不能仅按 legacy helper 命名猜测。

成本特征没有显式区分本轮 prefill token 数、decode token 数、图 bucket、通信量或 MoE 路由；其表达能力与泛化边界需要实测，不能因模式叫 calibrated
就视为准确。模型、PP/TP 拓扑与 rank 覆盖有配置检查，但检查存在系数不等于证明系数来自可靠校准。

[planner](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch/blob/4fe263f159faf7687316a6a334ac064791754764/src/vllm_hust_pipeline_microbatch/planner.py)。

## 4. 为什么可能更慢？

以下是机制分析，不是我们独立定位出的性能根因：

- 小组降低单次计算 batch，可能牺牲矩阵运算效率和权重访存摊销。
- 多个批次增加调度、通信与执行开销；收益要抵过这些成本。
- 按请求数平衡不能代表按耗时平衡。
- 稳定分组不重排，可能让长请求集中在某组，尾部等待加重。
- 总请求量过小、stage 严重不均衡，单靠分组无法填满流水线。

不修改模型语义是它与 SimLLM 近似路径的重要区别，但执行顺序、batch 形状变化仍需数值正确性与生命周期验证，不能仅凭结构宣称逐 token 等价。

## 5. 已有证据：balanced 在该测试单元退化

作者验收记录固定于 Mod `7115254`、Core candidate `3e57b2f75c`、Ascend ABI candidate `1cf88e9e6`，不是本轮 HEAD
的新实测。模型 Qwen3.8-27B，Ascend 910B2，PP2+TP2，FULL_DECODE_ONLY graph，async scheduling。

| 负载                    | completion tok/s 变化 | P95 延迟变化 |
| ----------------------- | --------------------: | -----------: |
| 同质 C4                 |                -9.05% |       +7.43% |
| 同质 C8                 |                -4.37% |       +0.89% |
| 固定工作量、混合长度 C8 |               -26.33% |      +80.16% |

混合场景 P50 下降 11.29% 与 P95 上升 80.16% 同时存在，不能只拿中位数宣传改进。报告解释为工作分组改善了部分短任务但导致长请求尾部失衡；我们没有独立 trace 复核这一归因。

报告同时记录了功能、取消恢复、图回放和策略真实准入，并报告五个确定性答案匹配。因此合理结论是“该设定功能可用，但不推荐其 balanced 性能配置”，而不是“有 API 接入所以必然加速”或“所有
calibrated 配置都已失败”。历史 PP4+TP2 成本系数不能直接沿用。

文档还有措辞张力：HOST_CONTRACT 将 balanced/synthetic profile 限定为 smoke；验收文档确实发布了 balanced
的负向实测。我们保留其为已报告退化观察，不把 balanced 当成已校准推荐配置。

[验收报告](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch/blob/4fe263f159faf7687316a6a334ac064791754764/docs/evidence/sage-mate-20260905-qwen38-pp2tp2-graph.md)。

## 当前判断

它试图解决的是成立的 PP 调度问题，当前实现则是较窄的稳定分组＋轮转准入＋可选成本模型。性能价值取决于相对宿主原有调度，是否真正减少
bubble、同时保住单批效率与长尾；已读到的当前平台测试未证明这一点。

继续深入时首先需要宿主原有 batching/future 行为和逐 stage 时间线，区分“减少 bubble”“降低 batch 效率”“分组长尾”各自贡献。当前没有授权或开展硬件复现。

## 共同评审 comment（2026-10-26）

Fletcher：**“这玩意也没有用，现在 PP 不是主流技术。pass。”**

本轮决定：**PASS，不再投入深入研究或复现。** PP 方向不符合 Fletcher 当前关注的主流部署与投入优先级，且已读测试单元没有显示性能收益。这里保留的是选题与采用判断，不把“PP
不是主流”扩写为已完成市场调查的结论，也不据此断言 PP 在所有部署中无用。
