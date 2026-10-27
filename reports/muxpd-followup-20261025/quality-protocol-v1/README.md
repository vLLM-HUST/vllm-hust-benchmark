# 已公开的质量协议：独立补充指标，非验收结果

**原严格 v1/v2 的“所有回答均不得截断”条件已在原生校准中失败，保持失败。** 这里追加已冻结、已公开的协议及下载回验身份，不报告新的正确率、成对质量通过或性能收益，不据此关闭 Issue
#23 或合并 PR #10。

原 2048-token v1 与统一扩大到 8192-token 的 v2 都保留原件及严格评分条件。补充指标
`supplementary_budgeted_mcq_nonregression_v1`
是另一项固定预算下的有限检查，不能把原门禁重新命名为通过。它由原生截断校准启发，在任何候选质量输出出现前固定，属于**校准后、候选运行前确定的补充终点**，不是未经校准调整的主要预注册终点。

## 冻结时间和范围

[采纳时的校准回执](native-only-calibration-receipt.json)记录
`2026-10-25 23:25:18 UTC`：当时只观察到第一轮原生的64个返回结果，其中2个达到长度上限。该部分输出快照只用于说明协议制定时已知的信息，不是终态质量证据；完整快照留在
Release 包中，未复制到 Git。

[最终冻结回执](freeze-receipt.json)时间为 **2026-10-25 23:40:29.715330
UTC**，仍在任何候选质量输出之前。它绑定最终[补充协议](protocol.json)、评分器以及未变的 v2 驱动/输入身份：

| 身份                 | SHA-256                                                            |
| -------------------- | ------------------------------------------------------------------ |
| 补充协议语义摘要     | `99a826511c6471883ce03c03446a9706c06dec791176667e50a2edd3e693560c` |
| 补充协议文件         | `24827a9261f9f84ceff86c00fc45985a9a7e4fdae63bf52cb028bc2a1caefc93` |
| 独立补充评分器       | `1a28cd97265f8c3b50442b76df4ced70f7acdf9ba1b87771e7cda80c473f16f4` |
| 原严格评分器，未修改 | `157b52ffdbae6d7f0f990bb27379c463df1807daeebf7298fd396968e96b17fd` |
| v2 请求计划语义摘要  | `dd3e1f305475ec556dc824947f26d5fb5743242e82b5099155c95b8954bb2e65` |

范围仍是原200个选定 MMLU-Pro 题目、原提示词/token/gold/解析器、25批各8题、统一8192输出预算。硬件配置沿用公开 f7 插件、固定
Core/Ascend、TP4/EP4、BF16、`HCCL_DETERMINISTIC=true`、共同 `enable_npugraph_ex=false` /
`enable_static_kernel=false`；这不资格化后来的其他 profile 源码。

原始[严格计划](QUALITY-PLAN.md.source)、[v2预算修订](QUALITY-PLAN-V2.md.source)、[执行身份](execution-identity-v2.json)、[环境决定](environment-decision-v2.json)、[独立评审](adopted-review.md.source)和[补充说明原件](supplementary-README.md.source)保持原字节。

## 补充指标保留的限制

- 两个新进程原生对照和候选必须各返回完整200个终态 in-process API 响应，输入、源码、配置及环境一致，清理正常。缺失、错误或不完整收集不能通过。
- **单题评分没有放宽：所有 `length` 截断都判错，即使文本里能解析到正确字母；无效、越界、缺失和错误也判错，分母恒为200。**
  原生采用的规范提示词/gold/解析器不变，但此规则比允许解析部分 length 文本的 canonical aggregate 评分更保守，不能宣传为完整 MMLU-Pro 榜单成绩。
- 两次原生均答对的非空集合中，候选新增任何一道错误即否决；候选总正确数还必须不低于两次原生中的较低值。其他题的改善不能抵消该稳定正确集合中的退化。
- 25个候选批次还需实际 P 层推进与 D token 退休回执，按本批次已接纳请求、阶段及步骤绑定；group 另需同条件 raw 参考并报告成对变化。这里不声称物理 kernel 同时执行。
- 原严格结果与新补充结果分别保存。完整200题子集上的零新增错误也不证明总体等价；统计不显著不能当作等价证明。该指标不能单独完成数值/状态、取消 ACK/块复用或性能价值验收。

补充评分器在冻结前通过30个 CPU
用例，冻结后相同源码再次通过30项检查；[最终复跑日志](cpu-tests-final.log)与[静态检查](ruff-final.log)仅验证工具。完整源码和所有历史失败用例日志在包内；[原校验和清单](SHA256SUMS-supplementary-v1.source)针对包内完整
`quality/supplementary/` 目录，不是这个精简 Git 目录的文件集。评分器还绑定原执行路径和模型身份；下载回验不代表可在任意新路径直接执行，重新运行需另行明确路径映射和环境。

## 公开附件与独立回验

附件位于同一
[Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-followup-20261025)，未替换旧
v1/v2 附件：

| 公开资产                                                                                                                                                                   | 大小            | SHA-256                                                            |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------ |
| [muxpd-quality-protocol-v1.tar.gz](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-followup-20261025/muxpd-quality-protocol-v1.tar.gz)   | 1,385,189 bytes | `e306ed595a3207fe25be12c8405822323c561e26c88cc5d34bc485c808e59c6b` |
| [quality-protocol-manifest-v1.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-followup-20261025/quality-protocol-manifest-v1.json) | 25,270 bytes    | `8a23e966ec461893d8cd87dc0841f74241e5f8b9e89ed7a42d10c3f23958bc8b` |

[上传回执](upload-receipt.json)于23:44:09 UTC完成；[匿名公开下载回验](public-readback-receipt.json)于23:44:48
UTC确认两个资产字节摘要及包内 **87个成员、8,345,096 bytes**
全部一致。[逐成员清单](manifest.json)和[本地建包回执](capsule-receipt.json)可离线复核。Git
只保存本页和小型元数据，不保存压缩包、完整题目/token数据或原生部分输出快照。

[23:56:18 UTC 的独立公开 API 清单观察](public-api-receipt.json)确认该 Release 当时共有23个附件，新增两个协议资产均为
`uploaded`，大小和 API digest
匹配。此前[v2公开回验](../PUBLIC-READBACK-v2.md)的21附件结论仍是原时点有效的历史快照，原字节不改；新增协议包不属于旧13包/11终态运行的 v2
清单。此处未重新下载其余21个资产，也不承诺未来链接状态。

[本次复制及发布身份](publication-identity.json)记录所有小型副本的来源与摘要。本页不收录正在运行或后来完成的实验，也不把单个完整原生对照当成成对候选验收；后续结果应另建版本和
capsule。
