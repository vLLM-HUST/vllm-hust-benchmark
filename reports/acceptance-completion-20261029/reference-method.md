# 同工况有效算力参考复核

## 方法

按 V1.5 第 3.4.1 节，使用独立标定的计算、访存、通信及必要调度，结合受测路径的实际阶段依赖与资源争用推导 `T参`；分别核对 `F、T、T参` 的工作量与边界。
`C=F/(k×T×10¹²)`，`η=T参/T×100%`。路径或精度不同的 B0/B1 分别核算参考时间，不能强行共用一个每芯片参考能力。

V1.01 中另加的“整链实现见证和三次实现运行中位数”不是 V1.5 的普遍准入条件，本次纠正。
独立标定模型仍需证明覆盖和适用性，不因缺少整链参考程序而自动作废，也不因模型可复算而自动视为验收通过。 旧共享实现经验参考 7.1085 TFLOPS/芯片及其达成率仍仅保留为历史，不恢复为当前结论。

## 已复算的 B1 模型场景

模型为 Qwen3.5-35B-A3B、BF16、Ascend 910B2×2、TP2、固定 MMLU-Pro 200 请求、282265 输入与 51200 输出 token。 绑定原 B1
自身必要算量 `1812905151088034 FLOPs`，不是把其他部署的吞吐乘以本轮 FLOPs/token。

| 场景             | 依赖图节点 | 参考时间（秒） | 导出参考能力（TFLOPS/芯片） | 对原计数轮的诊断比值 |
| ---------------- | ---------: | -------------: | --------------------------: | -------------------: |
| 预定主场景 seed0 |     359368 |  47.5033489612 |               19.0818667603 |       37.2302188966% |
| 敏感性 seed1     |     359368 |  47.5301478655 |               19.0711078389 |       37.2512222384% |
| 敏感性 seed2     |     359368 |  47.5571895196 |               19.0602637519 |       37.2724158326% |

原同轮 `T=127.59352582134306 秒`、`C=7.1042207644080255 TFLOPS/芯片` 保持不变。
参考包含资源前驱、配对集合通信的共同就绪、计算/访存重叠及必要调度；不是计算、访存和通信总时间的全局最大值。 本次回验公开包的 61 个文件字节，再重算三场景关键路径；942210 条标定查询用于
5394622 次叶节点引用，未补零或反推目标效率。

统计 draft 路由不冒充正式运行的逐专家真实路由。原独立适用性审核覆盖 3711 个不同真值点、20 个误差组，未支持查询/上下文为 0；各组最大 P95/P99 相对误差为
0.109293/0.144250。 这些是条件验证误差，三 seed 是敏感性，不是整链置信区间，也不能简单乘在整个参考时间上当误差界。

## 当前判定

本次取得的是可复算的 B1 参考模型及相对原计数轮的诊断值，**不是正式无插桩验收成绩**。 仍影响结论的条件是：B0
自身路径/工作量的参考绑定；适用的无插桩同步计时与算量对应；同输出约束下的质量适用性。 原 out256 回答评分为 B0 21/200、B1 19/200，不能以另轮 out8192 的
151/200、156/200 替代。 不估算扣除计数开销，不把该 B1 模型直接缩放给 B0；正式 η 和中期/最终通过状态仍为 null。

## 复算入口

- [本次逐项回验与关键路径结果](reference-reassessment-20261029.json)
- [原公开标定与依赖图包](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-r7/storage-reference-public-20261024-r5.tar.gz)，SHA256
  `78bb9be370824b9838e6355ac385b2e0ad9fa48c5c43813e33a97d4ff8d0f894`
- [原规范标定与方法索引](../vllm-hust-midterm-20261019/m01-recovery/storage-reference-closed-20261024/README.md)
- [本次验证程序](verify_reference_attempt.py.source)

## English summary

V1.5 permits independently calibrated, dependency/resource-aware reference-time modeling with
justified applicability. A separate executable whole-chain witness and a three-run implementation
median are not universal contractual requirements. B0 and B1 require separate reference bindings
when their necessary work or execution paths differ.

The archived B1 primary DAG was recomputed: `T_ref=47.5033489612 s`, derived
`C_ref=19.0818667603 TFLOPS/chip`, and `T_ref/T=37.2302188966%` against the retained host-counted
run. The two other seeds are sensitivity scenarios, not confidence bounds. Raw measurements are
unchanged. This is a model diagnostic, not an uninstrumented acceptance result. B0-specific binding,
matched uninstrumented timing and output-budget quality remain unresolved; formal efficiency and
stage verdicts remain null. The withdrawn 7.1085 reference is not reinstated.
