# MuxPD q40 原 batch7 完整答案：八题自然结束且全部正确

在先前 q40 短窗口诊断之后，本轮让原 batch7 的八条请求运行到自然结束或原8192上限。**八题均以 `stop` 自然结束，原 canonical scorer 判定8/8正确；07338
回答 H，输出3292 tokens。** 历史 full200 外部 prefix q4 中同题曾达到8192 `length`，按原规则为不完整且答错。

这是一轮新进程的后验八题诊断。历史 native/raw/prefix 的对应八题来自完整200题运行，前面已有七批请求历史；本轮未重放这些历史，也没有新增匹配的新进程 q4
完整答案对照。**局部退化已在这次 q40 诊断中消失，不能据此宣布完整200题、部署运行时、吞吐或 Goodput 验收通过。**
[PR #27](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/pull/27) 保持 Draft。

## 八题逐项结果

每格为“提取答案 / 实际 finish_reason / 输出 token 数”。`—` 表示原评分器不提取答案；`length` 即使文本包含金答案仍按不完整和错误计分。原件中的
`stop_reason` 均为 `null`，未据此推断具体 EOS 原因。

| case  | gold | native1 历史    | native2 历史    | raw 历史        | prefix q4 历史    | 本轮 prefix q40 |
| ----- | ---- | --------------- | --------------- | --------------- | ----------------- | --------------- |
| 07151 | H    | H / stop / 289  | H / stop / 289  | H / stop / 445  | H / stop / 445    | H / stop / 486  |
| 07213 | J    | J / stop / 302  | J / stop / 302  | J / stop / 237  | J / stop / 302    | J / stop / 302  |
| 07275 | E    | E / stop / 379  | E / stop / 379  | E / stop / 805  | E / stop / 447    | E / stop / 378  |
| 07338 | H    | H / stop / 4723 | H / stop / 4723 | H / stop / 5133 | — / length / 8192 | H / stop / 3292 |
| 07401 | A    | A / stop / 575  | A / stop / 575  | A / stop / 702  | A / stop / 682    | A / stop / 598  |
| 07461 | G    | G / stop / 468  | G / stop / 468  | G / stop / 481  | G / stop / 462    | G / stop / 468  |
| 07524 | I    | I / stop / 467  | I / stop / 467  | I / stop / 302  | I / stop / 417    | I / stop / 477  |
| 07585 | A    | A / stop / 532  | A / stop / 532  | A / stop / 544  | A / stop / 533    | A / stop / 533  |

两个 native、raw 对应八题均为8/8；历史 prefix 为7/8；本轮 q40 为8/8。在这八题中，两次 native 均正确的题没有新增错误，历史 prefix
的07338退化得到恢复。seed07151输出486 tokens，仍不能把其与 native 不同的生成路径当作答错。

## 输入、调度和来源约束

使用 Qwen3.5-35B-A3B BF16、TP4/EP4、物理卡3/4/5/6，沿用冻结200题计划字节，仅选择原
batch7，顺序为07151、07213、07275、07338、07401、07461、07524、07585。仍先让 seed 输出一个 token，再加入原七条
peer；温度0、seed42、`max_tokens=8192`、自然 EOS/stop 策略和原评分器不变，无目标208截断、无成功路径的活请求取消。

配置保留外部 prefix 布局、P的独立通信组、D沿用的原TP通信组、重叠执行、`FULL_DECODE_ONLY [1,2,4,8]`，并由 treatment sidecar 将原配置唯一覆盖为
`layers_per_slice: 4→40`。没有数值采集器，也没有新模型算子或运行时代码候选。原运行时源码、模型config和tokenizer身份、基础与解析后启动参数以及终态源码收据均经主/独立校验。

实际记录3293次 serving step，包含 seed P 和一次完整40层 peer P推进、七条完成ACK；每rank seed记录81次D0 bypass，peer记录81次P
collective。各rank D图 replay增量为3292，按真实非空D调度推导，保留启动warmup计数并核对运行期间无新增eager
fallback/图失败。随着请求自然结束，D批次缩小；没有把固定短窗口208或最长输出数当作唯一调度依据。正常最终空maintenance不计入serving收据。

所有完整答案均延续已资格通过的短 q40 plain 前缀：seed前209、其余七条前208完全相同。因此本轮连接了先前短轨迹和自然完整答案；没有重新采集或公开短窗口完整logits张量。

## 失败与修正记录

1. **v1启动环境失败保留。** root首次调用时遗漏
   `source /root/workspace/runtime/env.sh`，原运行环境一致性门在导入vLLM/构造模型前拒绝。退出码1，outputs=0、admissions=0，自有进程为空，四卡空闲；这不是错误答案或被替换的模型样本。冻结v1源码、控制器、子报告、日志及[失败谱系](failure-lineage-v1.json)全部保留。
1. **v2为重新声明的启动修复。**
   root显式加载原环境，并在控制器CPU预检中比较既有子进程环境，不自动修补环境。24个producer中21个字节不变；变化仅控制器环境检查、实验标签和矩阵路径。原模型driver、自然收集循环、采样和评分器不变；先前37项CPU验证保留，新版只重跑受影响14项环境/控制器和4项标签/引用检查。
1. **独立CPU后处理修订保留。** 独立脚本v2逐token重复查询不变的
   `get_vocab_size()`，该CPU核查进程被中断（exit130/KeyboardInterrupt）；当时没有输出结果。v3将同一词表大小缓存一次，数据、门限、评分器均不变，然后完成独立复核。旧脚本、修订脚本和[中断说明](cpu-postprocess-interruption-v1.json)均入包；这不是模型重跑或选择性挑选答案。

## 终态与独立复核

| 记录         | 结果或SHA256                                                       |
| ------------ | ------------------------------------------------------------------ |
| v1 UTC终态   | 2026-10-30 13:09:44.535319，exit1，模型未构造                      |
| v2 UTC终态   | 2026-10-30 13:22:28.015148，exit0，自有进程为空，四卡空闲          |
| v2控制器     | `aaa98bc025a8152a23efbe1cbd113018ed6b45fd2bd0d0d9de0571e29978f8f3` |
| v2完整子报告 | `67afca2a3189071289437be95bd8a8d6184a0f5a5dad189a963b3b8e2be4365d` |
| v2冻结清单   | `ef37960b4869885dc6ff488d0e598ac90e74f76820c6478695f20062ff9244ae` |
| 主分析       | `029831e8b47748516fecf70de37941bd4a52d87db471b836e09e177b386b9804` |
| 独立终态复核 | `7ec93a9ed3a8a24e263308c0d108fd15b79fa6990e93a411f1a125cb1b1f661e` |
| 原评分器     | `157b52ffdbae6d7f0f990bb27379c463df1807daeebf7298fd396968e96b17fd` |

主分析和独立脚本均在root确认终态后读取子报告/日志。独立复核重新核对原始JSON/调度JSONL、8份token到文本解码、原评分器结果、全部短前缀延续与调度证据；答案结果和采集成功分别报告。新八题实验的
`full_quality_gate_attempted=false`、`full_frozen_plan_passed=null`、`runtime_candidate=false`。历史完整200题严格门仍FAIL；两次历史native各有9个length案例，其原始完整性条件没有修改。

## 发布范围与下一步

本目录发布完整token/文本子报告、全部本轮scheduler
JSONL、进程日志的无损gzip副本、控制器和空闲设备记录、主/独立分析，以及76成员源码包（含成员清单）。[发布映射](publication-map-v1.json)记录每份原件和存储副本的SHA/字节数，包内清单记录全部成员SHA；本地已逐字节核对gzip还原和全部成员。历史完整报告依赖及既有短logits
NPY不在本包，**没有原始短张量公开上传或匿名HTTPS读回声明**。

下一步先将有效的显式q40配置纳入真实内置运行时，完成来源绑定和原posterior32批次，再用独立新进程执行完整200题。后续即使相对新增错题检查通过，也要与历史native不完整导致的原严格门FAIL分开；性能/Goodput需另做匹配场景评测，不能由本轮token数推断。

关联：[Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23)、[q40短窗口调度对照](../muxpd-q40-scheduling-20261030/REPORT-v1.md)、[首次P状态证据](../muxpd-prefill-state-20261030/REPORT-v1.md)。
