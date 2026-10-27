# 固定数据版本与消费链路验证

本记录说明原始数据读取、准备、索引与回查能否被独立消费，不是完整任务、质量、性能或科研结论。 各批使用下表明确固定的软件版本；后续文档订正不自动升级已经验证的软件身份。
原始文件、请求、图像、参考答案及完整索引均不在本文公开。

| 批次                       | 固定数据提交 / CI wheel SHA256                                                                                  | 研究端固定提交与回执                                                                                                                                                                | 已验证范围                                                                                                                     |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 文档理解与结构化           | `49506ccb80968a428d4025b0a7249dbcce316f1f` / `49a607dc61df7a287c3c2ee3290748870d28d4a34ca77abce29957ba40cb3eec` | [43ef0932 原始回执](https://github.com/intellistream/agent-workload-suite/blob/43ef0932a664ff81ff3198f5d1c1f936531567e7/docs/evidence/document-schema-input-consumer-20261023.json) | JSONSchemaBench三份原始schema、QuALITY三道原始问题，安装工件与固定Git两模式；旧来源九条回归                                    |
| 服务轨迹元数据             | `84b901f673c30ba8dce2c73038dc52e1c5cc9932` / `dec0c49b61df5a43d0ffebb2b3ef50fdcff506ff12f555ce3f7b1e6d367032c8` | [72068e98 原始回执](https://github.com/intellistream/agent-workload-suite/blob/72068e987e73f9f0933cbe0c5cb57f3c780214ed/docs/evidence/serving-metadata-consumer-20261023.json)      | Azure两文件、Mooncake两文件、BurstGPT首文件各三条，共15条，两模式；旧来源九条回归                                              |
| 代码、科学、视觉及历史来源 | `653ad3518e5fcd78c958d86ea39bac6d8714fb3a` / `54af5022625da0f92e0633c1a8a8b306f8b6011bd4ae70cf9d6560ee31edabe9` | [4906910e 原始回执](https://github.com/intellistream/agent-workload-suite/blob/4906910e9c5f9909d84e41d59f9f49dcb055a912/docs/evidence/public-seven-source-consumer-20261023.json)   | BigCodeBench、ScienceAgentBench、DS-1000、SWE-bench Verified、ToolACE、ChartQA、TraceLab各三条，共21条，两模式；旧来源九条回归 |

最近七源批次的工件metadata、摘要sidecar及30个软件payload均独立核对固定Git内容，文本仅归一CRLF/LF。
服务器实际安装该工件后，七个来源各三条原始输入与参考字段的读取、prepare和resolve往返通过。
研究端独立取得同一CI工件，锁定该提交与摘要；安装wheel和固定Git两种模式结果一致，旧来源指HumanEval、LongBench Qasper和BFCL simple_python。
研究侧的索引与特征转换只在研究仓实现、验证，不随本批迁入benchmark。

- [x] 最近七源批次数据CI成功：[主测试与lint](https://github.com/vLLM-HUST/vllm-hust-benchmark/actions/runs/39867689476)、[CPU正确性与工件](https://github.com/vLLM-HUST/vllm-hust-benchmark/actions/runs/39867689457)。主测试2416通过、4跳过。
- [x] 研究端固定提交4906910e的[测试、打包和论文构建](https://github.com/intellistream/agent-workload-suite/actions/runs/39869936643)及[准备链路](https://github.com/intellistream/agent-workload-suite/actions/runs/39869936570)全部成功。Linux消费85、组件941、harness及前批回归127项通过，无跳过。
- [x] 原PR继续为Draft，数据入口与消费端修改分别在原PR中交付，未合并到main。
- [ ] 上述消费检查不包含新的模型推理、原生任务执行或官方评分；这些须查各来源独立运行记录。

ToolACE是发布方合成历史工具对话，不是在线工具执行环境；TraceLab、Azure、Mooncake和BurstGPT上述记录是元数据，不补造请求正文或会话。
ChartQA保留原始图像字段，但当前NPU服务使用language-model-only，未验证视觉推理。
BigCodeBench生成三条正常终止；DS-1000三条中一条触及原始1024预算；两者功能评分仍未执行。 这些事实不因消费测试或CI通过而改变，来源级完成状态也不因此打勾。

CI工件保留期为30天；长期使用应从表中固定Git提交按安装文档自行构建，不能假定临时工件永久可下载。 既有候选工件、失败记录和原始回执保留，不用新验证重写历史身份。

## LoCoMo 固定版本增量

固定数据提交`04113e35405d959aaa6191c3d7a6fdfdd9ece942`，CI wheel
SHA256为`a1fb1efd188cee7ace85a3f911894509bcfd401c41cf4ecd7ca6ef36eb83c26c`，31个软件payload逐项对固定Git核验。
服务器实际安装后，原10会话/1986问题的输入和参考逐项对原JSON检查，展平前三题prepare/resolve通过。
研究端固定提交`b3e1f9f397afca1a4181b0969b34705f61046bca`的[独立消费回执](https://github.com/intellistream/agent-workload-suite/blob/b3e1f9f397afca1a4181b0969b34705f61046bca/docs/evidence/locomo-original-consumer-20261023.json)已核对：安装包与固定Git两模式各3题、旧来源各9条回归一致。
前三题均来自首会话，每题保留19个历史session和419个原始turn；不把这些历史轮次计为新模型调用。

数据端[主CI](https://github.com/vLLM-HUST/vllm-hust-benchmark/actions/runs/39873172357)为2419通过、4跳过，lint与[CPU smoke](https://github.com/vLLM-HUST/vllm-hust-benchmark/actions/runs/39873172330)成功。
研究端[测试、打包及论文构建](https://github.com/intellistream/agent-workload-suite/actions/runs/39873904563)和[准备链路](https://github.com/intellistream/agent-workload-suite/actions/runs/39873904546)全部成功，Linux消费92项、组件948项、harness及前批回归127项通过，无跳过。
研究侧多会话文本投影只用于其离线消费，明确不是官方模型提示词或tokenizer输入，不迁入benchmark。

上述固定工件与跨仓消费核验本身为零模型、零任务、零评分，不能覆盖[首次原题执行失败](validation/locomo-original-qa-failure-20261023.json)：HTTP500后引擎退出，没有完成答案或分数。随后按原模型和推理参数恢复服务，并仅将编译临时目录改到实例本地`/tmp`；[首批重试](validation/locomo-original-qa-recovery-20261023.json)及[追加三题](validation/locomo-original-qa-recovery-batch2-20261023.json)记录了原索引0–5共六题完成，发布方F1均值0.5337。该结果支持这六条原题的读取、生成、终止及评分链路，不能证明NFS故障根因已修复，也不代表全量LoCoMo或发布方GPT-4-Turbo复现。[六题汇总回执](validation/locomo-original-qa-sampled-20261023.json)不含原始题目和模型回答；原始任务和模型响应仍留在私有服务器，没有提交到公开仓库。
第5类兼容和视觉执行仍未核验，CC-BY-NC-4.0原始使用条款仍适用。
[固定工件汇总](validation/locomo-fixed-input-consumption-20261023.json)与先前候选、失败回执分别保存；文档更新不自动重锁数据版本或重写旧证据。
