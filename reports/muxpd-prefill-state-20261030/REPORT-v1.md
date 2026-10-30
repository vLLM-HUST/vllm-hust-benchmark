# MuxPD：目标题 prefill 输出与持久状态复核

本轮没有修复答案退化。针对新增退化题 MMLU-Pro 07338，原版与 prefix 候选在首次 prefill 结束时，所观测的层输出和全部目标持久状态相同。因此下一步转向后续 decode
的首次数值分歧， 不继续凭猜测修改 prefill 的 SSM 分块方式。

这是 [运行时 Issue23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23)
的定位进展。[运行时 PR27](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/pull/27) 仍为 Draft，答案不退化门仍 FAIL。

## 原有质量结论保持不变

完整 200 题的 native1/native2/raw/prefix 正确数分别为 165/165/166/167，length 数分别为 9/9/7/11。prefix 修复原有三题
03766、10672、01741，但新增 07338 退化：原版正确并在 4723 token 停止，raw 正确并在 5133 token 停止，prefix 达到 8192 token 上限。
167/200 的总分不能抵消这条新增退化。

本报告没有运行新的完整质量集或性能基准，不扩大此前特定场景的 Goodput 收益结论。 观测中保留的 208 token 前缀不构成完整答题验收。

## 本轮结果

| 观测边界                         | 跨角色比较数量 | 结果                           | 主分析与独立复核                                                            |
| -------------------------------- | -------------: | ------------------------------ | --------------------------------------------------------------------------- |
| 目标首次 P 的前四层选定张量      |         120 对 | 数值与 NPY 字节全部一致        | [主分析](first4-primary-v2.json.gz)、[独立复核](first4-independent-v1.json) |
| 目标首次 P 的 40 层最终 MoE 输出 |         160 对 | 数值与 NPY 字节全部一致        | [主分析](moe40-primary-v2.json.gz)、[独立复核](moe40-independent-v2.json)   |
| 目标 P 结束、首次 D 前的持久状态 |         320 对 | 数值与规范顺序状态字节全部一致 | [主分析](state-primary-v1.json.gz)、[独立复核](state-independent-v1.json)   |

三轮各自执行原生与 prefix 两个角色，保留原 batch7 的八题、顺序、seed、准入及目标 208 token 截止条件。每个角色的 80
个既定完整词表向量及八条完整保留前缀均与既有同角色对照一致； prefix 与 B-v2 plain 桥接也通过。这些是历史对照，不应称为本轮新做的成对无采集器运行。

这是后验选定 batch7 的新进程诊断，未重放前七批或其缓存历史；保留前缀对照是准入条件，不能据此声称完整复现原 full200 的执行上下文。

模型为 Qwen3.5-35B-A3B BF16，TP4/EP4，物理设备 3、4、5、6（进程内逻辑 0–3）。保持 q4、FULL_DECODE_ONLY
[1,2,4,8]、npugraphEx、static=false、HCCL_DETERMINISTIC=true、 BI 关闭及既定运行时。原生/P
的请求布局不同，分析按目标请求与实际映射重建数据，不能把隐藏层行偏移当作缓存索引。

## 状态观测如何确认

状态采集只包裹 `_model_forward` 和 `_bookkeeping_sync`。原完整 P forward 返回后，在原 producer stream 克隆目标保存的状态；普通
bookkeeping 后才等待事件并转到 CPU。 没有额外算子重放、数学路径修改、collective 或缓存替换。

每 rank 包含 80 个逻辑状态：30 份 BF16 卷积状态 `[3,2048]`、30 份 FP32 recurrent 状态 `[8,128,128]`、20 份 BF16 K/V
`[1340,1,256]`。实际对应 280 个状态片段和 23 个映射文件，状态数据 29,818,880 字节、映射数据 5,494 字节。独立复核直接重读两角色共 2240 个状态片段 NPY
和 184 个映射 NPY，验证请求归属、物理块映射、stream/device、源绑定、 两处 hook 恢复及历史输出对照。

320 对规范顺序状态的最大误差、RMS 误差均为零，没有首个不同状态。 该结论指状态数据，不声称请求 ID、地址、元数据或物理块编号相同。

| 角色   | 控制器终态（UTC，2026-10-30） | 生命周期结果                                       |
| ------ | ----------------------------- | -------------------------------------------------- |
| native | 11:18:12.217318               | exit 0；前后无所属子进程残留；四卡空闲；无强制清理 |
| prefix | 11:22:36.888899               | exit 0；前后无所属子进程残留；四卡空闲；无强制清理 |

先验证控制器终态，再读取子进程结果。证据见
[native 控制器](state-native-controller-v1.json)、[prefix 控制器](state-prefix-controller-v1.json)、
[native 运行记录](state-native-run-v1.json.gz)、[prefix 运行记录](state-prefix-run-v1.json.gz)。

状态采集前的 CPU 集成验证为 98 项通过，另有两项源代码字面一致检查通过；
[CPU 记录](state-cpu-validation-v1.json)、[独立预检](state-preflight-review-v1.json)和
[冻结清单](state-freeze-v1.json)均保留。CPU 通过只支持采集协议，不支持答案修复结论。

## 失败与预检修正没有被覆盖

前四层采集最初的 native 分析因分析器混淆决策条数和 80 份向量回执而失败。后续使用新版本外部 后处理器，未重跑 native 或覆盖冻结生产文件，prefix 的继续执行有显式控制器记录。

MoE-v1 的原生尝试在启动和 warmup 后遇到源绑定保护失败：实际 Ascend decoder forward 已被适配层替换，而最初保护错误地要求 upstream
绑定。该次未准入请求、未产出观测 NPY，随后 父进程超时并执行清理；prefix 未启动。[失败分析](moe40-v1-failed-analysis.json)、
[完整性复核](moe40-v1-failure-integrity.json)和[v2 血缘说明](moe40-failure-lineage-v2.json) 保留，v2
使用实际导入绑定进行新的协议验证。

状态采集在模型运行前纠正了四个保护条件：owner/decode 共用 requests 对象时去重，正确比较 AttentionType 枚举，校验
worker/source/current-stream 的实际设备，以及允许原生路径尚未 初始化的 `_CURRENT_STREAM=None`。实际原生记录确为 None；prefix 的
P/current/缓存 stream 均为 96。没有为通过检查而修改被测模型数学行为。

## 下一步：定位首次 decode 数值分歧

对已完成实验的只读[完整 logits 补充比较](existing-logits-cross-role-v1.json.gz)显示： 八题的首次 P logits 跨角色逐字节相同；目标 07338
在输出位置 200–207，自己的输入与前缀 相同但 logits 已不同；位置 208 的自身前缀已不同。原采集没有位置 2–199 的完整 logits， 所以不能把位置 200
当作首次数值分歧。

下一轮保持原普通采集窗口 `[1,200..208]` 与 80 份对照，另外仅在 rank0 采集目标题位置 2–199 的完整 logits。重叠窗口共用一次原 full-batch clone
和一次延后的 D2H，分析逐步核对 实际 token、计算位置、调度数量及同批上下文。先确定第一个数值分歧，再选该步的具体边界做干预。

实际隔离策略只为 P 创建 sibling TP/EP communicator；D/None 使用捕获的原 communicator。 因此不能把 prefix 的 D 误描述成独立 decode
通信组，并据此断定原因。

只有候选干预有效后，才落实到运行时代码，复测原退化题所在原批次和全新完整 200 题，逐题报告 修复、新增错误与新增 length。局部数值一致不是完成条件。

## 发布范围

[源码包](state-source-v1.tar.gz)包含冻结状态生产代码、预检、CPU/源绑定证据与独立复核脚本， 共 39 个成员；每个成员已在本地按清单校验。它不含原始 NPY
或全部历史依赖，运行仍依赖原服务器环境。

[状态发布映射](state-publication-map-v1.json)及[此前观测来源](local-source-map-v1.json) 给出原路径、字节数及 SHA256。压缩 JSON
解压后与原文件逐字节一致。原始设备张量仍在服务器， 本报告不声称完整张量包已公开，也不声称做过该包的公开下载复核。
