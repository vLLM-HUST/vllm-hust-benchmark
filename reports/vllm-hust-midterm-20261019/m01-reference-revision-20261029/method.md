# M01 同工况参考修订 / Workload-reference revision

状态：参考推导修订中；当前 C_ref、η、中期及最终判定均未确定。 Status: reference derivation under revision; current C_ref, η and
both stage verdicts are unresolved.

## 已有测量 / Retained measurements

Qwen3.5-35B-A3B、BF16、Ascend 910B2 ×2、TP2、MMLU-Pro固定200请求，
282265输入及51200输出token，性能输出预算256。B0固定vLLM/Ascend 0.18/0.18； B1为0.28.1/0.25.1rc2开发栈的BetterScale、FA
Prefill Microflow和纠正KV物化组合。 同轮计数开销全部保留，不称严格无插桩；质量200题out8192单列，不混用其耗时。

Compute is recalculated as `C = F / (k × T × 10^12)` from each arm's own recorded necessary work and
synchronized time. B0/B1 keep their actual runtime identities. The fixed 200-task quality results
(151/200 and 156/200, out8192) do not replace the out256 timing or establish full-corpus
noninferiority.

## 原参考为何不再用于当前判定 / Why the old denominator is withdrawn

三次共享实现的独立C200运行中位数7.108506511466548 TFLOPS/芯片，及H200验证， 证明该实现的重复运行与样本适用性，不证明已接近该工况可实现能力。
其54.0301%/99.9397%保留为旧方法算术，不用于当前50%中期或90%最终判定。 不同实现不是必要条件；关键是参考实现选择和能力强度有独立、可核验依据。

Independent repeats of the shared implementation establish repeatability and sample applicability,
not workload-capacity strength. The old attainment rates are historical arithmetic only. Sharing
code is allowed; it does not itself qualify or disqualify a reference.

## 新参考的推导与准入 / Derivation and admission

1. 冻结上述模型、精度、资源、输入/输出语义、负载、质量与服务约束。明确并发、 批处理、缓存初态、源权重和实际加载代码；不以同为200请求或相同token总数推定可比。
1. 先复用独立算子标定：按实际必要工作、阶段依赖、计算/访存/通信/CPU资源争用 构造参考时间模型。报告校准覆盖、模型误差、统计路由和实际路由差异；
   不能用各组件最快值简单相加，也不能以硬件标称峰值或全局max代替整链参考。
1. 对现有更高吞吐候选，核验同轮必要F、执行形状、设备同步T端点、源权重/加载代码、 同工况质量与资源身份，再与上述模型逐阶段比对。吞吐不能按另一次F/token换算C。
   当前候选的未闭合项是这些绑定，不再重复要求已有质量评分。
1. 只有能力选择依据和整链实现见证同时满足时，才用事先指定实现的三次有效独立 校准C的中位数作为共同C_ref；非重叠匹配样本用于验证适用性。
   不因η高低替换慢轮、调整模型或分母；所有尝试均保留。现有证据不足则C_ref为null。
1. 对两臂分别计算 `η_i = 100 × C_i / C_ref`、 `T_ref,i = F_i / (k_i × C_ref × 10^12)`。50%为中期门槛，90%为最终门槛。
   缺少参考时判定为未确定，不是失败、0%或通过。描述性C提升不依赖C_ref。

Freeze workload and resource identities; derive a stage/resource-aware model from independent
calibrations, then validate an executable reference against it. Reuse stronger existing candidates
only with their own work ledger, execution geometry, synchronized endpoints, loaded-weight/code
continuity and applicable quality. A candidate's throughput multiplied by another run's FLOPs/token
is inadmissible. Use a preselected implementation's three-run median only after both
capacity-selection evidence and a whole-chain witness qualify. No tuning to a target η, no discarded
slow valid runs, no zero-filled missing work or durations.

## 现有候选与当前缺口 / Existing alternatives and current gap

- 7.1085：共享实现经验参考，能力强度未建立，退出当前验收分母。
- 19.081866760274465：既有算子依赖图seed0模型值（T_ref=47.503348961177785秒）；
  其他两场景19.071107838945064、19.060263751919415。三场景是敏感性，非置信区间；
  统计draft路由不等于正式运行实际路由，尚无该调度的整链执行见证。不直接替换成当前C_ref。
- E16/KV与FA既有性能候选：已有质量和性能记录，缺少可直接准入的新参考绑定； 2026-10-27复核见benchmark PR
  #540的`reports/m01-reference-strength-20261027/summary.json`。 该PR尚未合并，其结论不是新参考值。

The operator-DAG estimate is an independently modeled scenario, not a demonstrated executable
whole-chain capacity. Its three schedules do not give a confidence interval. The existing stronger
throughput candidates cannot yet supply a qualified replacement denominator.

本次仅复算既有测量并纠正发布状态，没有新建模型服务、校准、质量或性能任务。 本文件是修订方法和证据准入要求，不是已完成新参考测量或新的实验执行清单。

This revision reuses existing measurements. No new model, calibration, quality or performance run
was started. This method is not an executable experiment plan or a completed new reference.
