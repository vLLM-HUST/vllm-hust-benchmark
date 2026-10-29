# DLA：浦江五数据集适用性候选矩阵

候选日期：2026-10-20 MOD
工单：[vLLM-HUST/vllm-hust-dla#6](https://github.com/vLLM-HUST/vllm-hust-dla/issues/6)
总账：[vLLM-HUST/vllm-hust-benchmark#254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254)
机器可读矩阵：[`applicability.json`](applicability.json)

本报告是 **MOD 级候选适用性材料**，不含任何任务质量成绩、B0/B1 配对测量或优化声明。 逐项 status 供 benchmark #254 维护者评审；未获评审前不得据此提升任何全局
readiness。

## DLA 的作用面

DLA 是运行在 vLLM scheduler 进程内的**抢占 victim 选择策略** （`vllm.preemption-policy.v1`，载体
`dla.policy:DlaVictimPolicy`）。它只在一种时刻改变决策： **KV 块池已满、scheduler 必须踢掉一个正在运行的请求**。此时它选择"预测剩余输出最长"的请求作为
victim，保护接近完成的短请求，避免其整段 prompt 白算。没有预测时它弃权（返回 `None`）， 调度行为与上游逐位一致。

因此 DLA 的可测差异只存在于**并发 > 1 且发生抢占**的 serving 场景，其收益指标是 JCT 与抢占次数， 不是任务质量、也不是吞吐（本机制在池饱和时减少白算，吞吐持平或略降）。

## 激活链：当前三层前置中，前两层未解除

| 层          | 条件                                                                 | 当前实测状态（2026-10-20）                                                                                                                                                                                                   |
| ----------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. 宿主版本 | manifest 声明 `host.version_range: >=0.23.1.dev653,<0.24`            | 当前 `vllm-hust` 已在 **0.31.1** source line（维护者 2026-10-10 审计）。声明区间未放宽，ECPA 在装机阶段即 fail closed。                                                                                                      |
| 2. 预测协议 | 要求协议 `vllm.predicted-length ==1.0`，即宿主把预测值挂到抢占候选上 | 直接核查 `vllm/v1/core/sched/preemption.py` @ `a4c2404dbe`：`PreemptionCandidate` **不含** `predicted_length` 字段（该文件只有 `builtin_victim_id`）。`DlaVictimPolicy._any_prediction()` 因此恒为 `False`，策略**恒弃权**。 |
| 3. 负载     | 并发 ≥ 2 且 KV 池饱和，scheduler 实际发生抢占                        | 五项主数据集的评测协议全部是**单并发串行**（逐项理由见下表），不产生抢占。                                                                                                                                                   |

前两层是宿主侧缺口，由 core 侧改动与 manifest 版本区间负责；第三层是评测协议本身的性质， 不随宿主修复而改变。三层没有同时成立，因此**五项都不存在可执行的 DLA 激活路径**。

预测器适用性另有一处缺口：`MOD_METADATA.json` 的 `supported_workloads` 只记录
`Qwen2.5-7B-Instruct — length predictor (EGTP head) reproduced on ForeLen`， 没有在 V5.5 强制模型
`Qwen/Qwen3.5-35B-A3B` 上验证过的预测器。同一文件的 `evidence.qualification` 为 `unqualified`、`performance_claims`
为空。

## 五数据集矩阵

| 数据集             | DLA status      | 具体理由                                                                                                                                                                                              | 主指标（非 DLA 指标）              |
| ------------------ | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| MMLU-Pro           | `not-exercised` | 逐题单请求质量评测。已冻结快照的参考五样例提示为 898–2,910 tokens，评测按题串行提交，活跃请求数为 1，KV 池不会因并发饱和而抢占。历史 200-prompt 材料只是服务遥测，不是任务准确率。                    | 固定评分器下的任务准确率           |
| HLE-Verified       | `not-exercised` | 单请求串行质量评测；2,500 行来源含 668 题 Gold，其中 93 道图像题。无论子集如何选择，评测形态都是每次一个请求，不产生抢占。数据集侧的许可/评测范围/judge 合同缺口由 #254 单独跟踪，不是 DLA 的作用面。 | 固定 HLE-Verified 评分器下的准确率 |
| SWE-bench-Pro      | `not-exercised` | Agent 多轮，但单 Agent 单容器串行执行（已归档单轨迹为 79 次模型回复后提交）。任一时刻只有 1 个活跃请求，即使上下文达到 37,370 tokens 也不与第二个请求争抢 KV 池。                                     | 隔离重评下的任务解决率             |
| FrontierScience    | `not-exercised` | 逐题单请求、problem-only / thinking-disabled 渲染；Olympiad 100 题与 Research 60 题两赛道各自串行评测。不产生并发抢占。                                                                               | 分赛道准确率 / rubric 分           |
| Terminal-Bench 2.1 | `not-exercised` | 候选 Harbor 协议明确写死 **one trial per task, one concurrent trial**（100 turns、32K 上下文）。并发恒为 1，抢占不可能发生。                                                                          | Harbor 任务解决率（pass@1）        |

五项均未标记 `applicable`。没有任何一项存在可运行的 DLA 激活路径，因此本矩阵不包含 B0/B1 配对、不包含分数，也不申请任何优化结论。

## 不得由本报告推出的结论

- 不得把离线块池模拟（BurstGPT 1 万真实请求、vLLM LIFO 基线、3 seeds； JCT p50 −23%、preempt −28%、白算 token
  −56%）当作五项中任何一项的成绩或收益。 那是机制侧离线证据，不是真实服务测量，也不是任务质量。
- 不得把"策略已实现 / 单元测试通过 / 插件可被发现"等同于"已在真实抢占用例中生效"。
- 不得用 DLA 的 JCT 或抢占次数替代任务准确率；也不得反向用任务准确率证明 DLA 有效。
- 五项中的四项另有数据集侧合同缺口（许可、judge、沙箱、镜像），由 #254 跟踪； 本报告只回答"DLA 在这些数据集上能否被激活"。

## 若要在这些数据集上产生 DLA 可测差异

需要同时满足，并在同一冻结 manifest 下留证：

1. 宿主侧：放宽 manifest 的 host 版本区间至当前 source line，并把 `predicted_length` 暴露到 `PreemptionCandidate`（core
   侧改动尚未合并）。
1. 预测器侧：在 `Qwen/Qwen3.5-35B-A3B` 上验证长度预测器，记录模型 revision 与预测误差分布。
1. 负载侧：在保持任务质量评测协议不变的前提下，**另行**构造并发服务负载使 KV 池达到饱和， 测量 JCT、抢占次数与白算 token；任务质量与 serving 指标分开报告，不得混算。
1. 证据侧：模型/runtime/容器/数据/评分器身份固定，保存逐请求输出、server argv、原始产物与 SHA-256。

在此之前，DLA 对这五项的诚实状态是 `not-exercised`。

## Owner 与发布门

- MOD 负责人：@xiehanlong834-gif（`MOD_METADATA.json` maintainers）。
- 数据集权利、评测器与沙箱合同：benchmark #254 维护者协调。
- 本报告不关闭 DLA #6，也不提升 `dataset_program_v1.json` 的任何 readiness。 DLA 站点/清单状态在门槛闭合前保持 `unqualified`。

矩阵为候选材料，等待维护者逐项评审；任何一项 status 变化都必须附新的实测证据， 不得因评审便利而改为 `applicable`。
