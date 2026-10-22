# 按 workload 分类交付与 PR 重整计划（历史拆分方案）

> 2026-10-23 更新：用户已明确采用“几类 workload 合为一个完整交付批次”，不再逐来源或逐类别开 PR。本文第五节的逐类拆分次序停止采用；当前交付组织与既有 PR 处理见[合批交付调整](PR_BATCH_CONSOLIDATION.zh-CN.md)。本文分类、来源缺口和历史证据仍保留，不因合批改写验证结果。

日期：2026-10-22。本文是分类与实施计划，不表示新增真实验证。本轮不关闭、拆改或合并远端 PR，不覆盖原始回执。

## 一、范围与现状

现有登记范围为 36 个任务细类、89 个公开候选来源和 8 份企业数据；另有 17 个补接候选。候选不是已接入，不保证每个候选都适合作为该类的原生任务。

当前完成来源级 NPU 抽样验证的是 GSM8K、ARC-Challenge 及八份企业数据，每源三条。公开两源有任务评分；企业八源仅有回放/响应完整性检查，没有独立语义正确率。24
来源完成过本地三条输入读取，包括 16 个公开来源；本地读取不等于 NPU 执行。

上述为拟定计划时的初始截面，后续进展以[分类执行清单](WORKLOAD_EXECUTION_CHECKLIST.zh-CN.md)和[来源台账](DATASET_CHECKLIST.zh-CN.md)为准。已新增按类Draft
PR #405（规则业务telecom）及#406（独立推理），均未合并；旧PR尚未关闭、改基或拆分，不声称重整全部完成。MMLU-Pro
biology及tau2原三任务已补验证，公开来源现有四个抽样闭环；GSM8K增量覆盖数另按原ID去重。

分类采用三个层次，不强行互斥：

1. **任务类别**：下表保留现有 36 个稳定 ID，作为检索和分工索引。
1. **跨任务特征**：长输入、长输出、共享前缀、会话上下文、语义相似等；单独维护交付项，不伪装成新的任务语义。
1. **来源与执行协议**：原始闭环任务、固定请求回放、历史轨迹、仅元数据分别标记。历史轨迹不等于可重新执行的原生环境。

## 二、完整任务分类及来源计划

符号：`✓S` 为原始任务抽样执行及评分完成；`✓R` 为请求回放及完整性完成、语义未评估；`L`
为本地三条输入读取完成；其余为候选登记/待当前版本复核。下面的“首批”是建议优先验证来源，不表示其余来源被删除。各类尚缺的共同项是：逐子类确认源契约、适用执行/评分、安装说明及可复现回执；具体差异另列。

| ID / 任务类别                                     | 已登记公开候选（全部保留）                                                                                          | 已有证据及建议首批                                                                           | 本类主要缺项                                                        |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| general-single-turn-assistant / 通用单次助手请求  | WILDCHAT、LMSYS_CHAT、DOLLY、NATURALQUESTIONS、DUREADER、NATURAL_INSTRUCTIONS、OASST1、TULU3、ULTRACHAT、LONGWRITER | L：DOLLY、OASST1、LONGWRITER；首批 DOLLY 原生单轮子集                                        | 固定原生子任务；开放会话没有官方正确率时只能回放/完整性检查         |
| conversation-continuation / 对话续接              | WILDCHAT、LMSYS_CHAT、OASST1、BURSTGPT、ULTRACHAT、SPOKENWOZ、DAILYTALK                                             | L：OASST1、BURSTGPT；✓R：PAIO-CHAT、PAIO-REUSE-CONV（来源级，分类待逐记录确认）；首批 OASST1 | 历史上下文与真实连续交互分开；不能伪造跨请求会话                    |
| rag-followup / 检索条件下问答与跟进               | QRECC、QASPER、NARRATIVEQA、DUREADER、NATURALQUESTIONS、LONGBENCH、LONGMEMEVAL                                      | 首批 QRECC                                                                                   | 原生对话、检索库与检索协议；单轮 QA 不自动证明跟进                  |
| document-grounded-analysis / 文档条件分析         | LONGBENCH、LONGBENCH_V2、INFINITEBENCH、QUALITY、QASPER、NARRATIVEQA、NATURALQUESTIONS、DUREADER、LONGALPACA、DOLLY | L：LONGBENCH_V2、DOLLY；首批 LONGBENCH_V2                                                    | 原始上下文与官方评分；企业长输入不自动归文档分析                    |
| standalone-reasoning / 独立推理                   | GSM8K、MATH500、BBH、GPQA、MMLU_PRO、AI2_ARC、OPENMATH2、OPENTHOUGHTS3、LABBENCH                                    | ✓S：GSM8K 3/3正确，ARC-Challenge 2/3正确；L：MMLU_PRO；先交现有两源                          | ARC-Easy 未跑；其他源未完成服务器评分；知识选择题与多步推理分别标签 |
| structured-generation / 结构化生成转换            | JSONSCHEMABENCH、STRUCTEVAL、SPIDER、SCHEMASTORE、BFCL、TOOLACE、XLAM_TOOL、NATURAL_INSTRUCTIONS                    | L：STRUCTEVAL、TOOLACE；✓R：PAIO-JSON；首批 STRUCTEVAL                                       | schema 资源不等于任务集；格式有效不等于抽取/转换正确                |
| code-generation-with-tests / 单题代码生成测试     | HUMANEVAL、MBPP、BIGCODEBENCH、DS1000、LIVECODEBENCH、OPENCODE、OPENTHOUGHTS3                                       | L：BIGCODEBENCH、DS1000；首批 BIGCODEBENCH                                                   | 隔离执行依赖、官方测试；不能仅检查代码文本生成                      |
| code-evaluation / 代码评判请求                    | LIVECODEBENCH、OPENCODE、BIGCODEBENCH                                                                               | ✓R：PAIO-CODE-EVAL；首批企业原始请求，公开补 CodeJudge-Eval                                  | 现有公开候选多偏代码生成，不能自造评判题；缺独立评判标签            |
| repository-completion / 仓库上下文补全            | REPOBENCH、CROSSCODEEVAL                                                                                            | 首批 CROSSCODEEVAL                                                                           | 仓库上下文、版本、原生补全协议与评分                                |
| interactive-code-editing / 交互式代码编辑         | TRACELAB                                                                                                            | L：TRACELAB；首批原始历史请求回放                                                            | 历史编辑轨迹不等于可重新运行的闭环编辑环境                          |
| stateful-environment-interaction / 状态化环境操作 | AGENTBENCH                                                                                                          | 首批官方 ALFWorld 环境                                                                       | 完整环境与终态评分，不复用 QBI overlay                              |
| database-query-update / 数据库查询更新            | AGENTBENCH                                                                                                          | 首批官方 DB 环境                                                                             | 数据库初始状态、更新隔离及官方判定                                  |
| knowledge-graph-traversal / 知识图谱遍历          | AGENTBENCH                                                                                                          | 首批官方 KG 环境                                                                             | 原始图服务、查询协议、评分                                          |
| web-navigation-interaction / 网页导航交互         | AGENTBENCH、MIND2WEB、WEBARENA、VISUALWEBARENA、WORKARENA                                                           | 首批 WEBARENA 可部署子环境                                                                   | 浏览器/站点恢复；演示轨迹不能冒充在线操作                           |
| os-command-execution / 操作系统命令执行           | AGENTBENCH                                                                                                          | 首批官方 OS 环境                                                                             | 隔离容器、工具权限及官方终态                                        |
| desktop-gui-interaction / 桌面GUI操作             | OSWORLD                                                                                                             | 首批 OSWORLD 原生任务                                                                        | 桌面环境、视觉能力、动作协议与安全隔离                              |
| application-api-workflow / 应用API多步工作流      | APPWORLD、BFCL、TOOLBENCH                                                                                           | 首批 APPWORLD                                                                                | 固定应用状态、实际 API 执行与最终状态评分                           |
| policy-constrained-service / 规则约束业务办理排障 | TAUBENCH                                                                                                            | 首批核对该目录 ID 绑定的官方 tau2 具体版本/domain                                            | 原 tau2 环境源码已缺失；重新安装固定官方环境，用户模型也走 NPU      |
| tool-call-generation / 工具选择参数生成           | BFCL、XLAM_TOOL、TOOLACE                                                                                            | L：TOOLACE；首批 BFCL 原生子集                                                               | 工具调用结构与执行正确性分开；不能扩称完整 Agent                    |
| evidence-search-synthesis / 证据检索综合          | GAIA、BFCL                                                                                                          | 首批 GAIA 可用能力范围内原始任务                                                             | 原题附件、公开 Agent 框架与真实搜索/文件工具；不使用 FrozenEvidence |
| scientific-experiment-planning / 科学实验规划     | DISCOVERYBENCH、PAPERBENCH、MLAGENTBENCH、SCIENCEWORLD、SCIENCEAGENTBENCH、LABBENCH                                 | L：SCIENCEAGENTBENCH；先核对原生任务是否确实要求实验规划                                     | 科学问答不自动属于规划；需要环境和规划/实验结果判定                 |
| simulation-analysis-verification / 仿真分析验证   | SCIENCEWORLD、DISCOVERYBENCH、BIXBENCH、NATUREBENCH、SCIENCEAGENTBENCH                                              | L：SCIENCEAGENTBENCH；首批 SCIENCEWORLD                                                      | 固定仿真/分析环境与原始判定；目录名不能证明任务适配                 |
| scientific-coding-workflow / 科学代码工作流       | SCIENCEAGENTBENCH、NATUREBENCH、MLEBENCH、MLAGENTBENCH、BIXBENCH、PAPERBENCH、DS1000                                | L：SCIENCEAGENTBENCH、DS1000；首批 SCIENCEAGENTBENCH                                         | 科学数据及执行环境；DS1000 单题不扩称长程科研 Agent                 |
| evolving-corpus-rag / 演化语料检索                | QRECC、LONGMEMEVAL_V2、TRACELAB                                                                                     | 现有仅候选；补 DRAGOn/TemporalWiki 真实版本资产                                              | 现有候选未证明完整更新/删除/查询事件；不得造时间线                  |
| conversation-memory-recall / 跨会话记忆           | LONGMEMEVAL、LOCOMO                                                                                                 | 首批 LONGMEMEVAL                                                                             | 原会话时间顺序、记忆输入/查询、官方评分                             |
| experience-memory-recall / 任务经历轨迹记忆       | LONGMEMEVAL_V2                                                                                                      | 先核对固定发行版是否有对应原生任务                                                           | 若只是对话记忆，不强行归经验记忆；可明确保持缺口                    |
| repository-agent-loop / 长程仓库Agent             | SWE_BENCH、SWE_VERIFIED、TRACELAB、SWESMITH                                                                         | L：SWE_VERIFIED、TRACELAB；首批 SWE_VERIFIED                                                 | 原始仓库环境、真实编辑/测试闭环、官方测试结果                       |
| image-understanding / 一般图像理解                | VISIONARENA、MMMU                                                                                                   | 首批 MMMU                                                                                    | 实际图像输入、可部署视觉模型与原生评分                              |
| visual-document-chart / 视觉文档图表              | CHARTQA、MMMU                                                                                                       | L：CHARTQA；首批 CHARTQA                                                                     | 图像字节及视觉推理；不能用 OCR 文本替代后冒充原生视觉               |
| video-understanding / 视频理解                    | VIDEOMME、VIDEOMME_V2、MVBENCH、LONGVIDEOBENCH                                                                      | 首批 VIDEOMME                                                                                | 原始视频、采帧/字幕协议、模型资源与评分                             |
| offline-audio-understanding / 离线音频理解        | LIBRISPEECH、FLEURS、WAVCAPS                                                                                        | 首批 FLEURS                                                                                  | 音频输入及匹配模型；ASR/理解任务分别记分                            |
| text-summarization / 文本摘要                     | XSUM、MULTINEWS、BILLSUM、XLSUM、DOLLY                                                                              | L：DOLLY；首批 XSUM                                                                          | 原始摘要任务和适用评分，不用普通聊天响应代替                        |
| text-translation / 文本翻译                       | FLORES200、NATURAL_INSTRUCTIONS                                                                                     | 首批 FLORES200 固定语向                                                                      | 原始语种/切分与评分协议                                             |
| text-extraction-classification / 文本抽取分类     | DOLLY、NATURAL_INSTRUCTIONS、TULU3                                                                                  | L：DOLLY；首批原生抽取子集                                                                   | 抽取与分类分别核验，不从格式通过推导语义通过                        |
| text-rewriting / 文本改写编辑                     | NATURAL_INSTRUCTIONS、TULU3、DOLLY                                                                                  | L：DOLLY；先筛原生改写任务                                                                   | 仅目录候选，原任务和可用质量 oracle 尚待确定                        |
| audio-streaming-interaction / 音频流交互          | SPOKENWOZ、DAILYTALK、LIBRISPEECH、FLEURS、WAVCAPS                                                                  | 补 Full-Duplex-Bench 作为更直接候选                                                          | 文本对话/离线音频均不能证明双工与流式协议                           |

以上公共来源在 36 类中有重复引用，只维护一个取得/适配实现；不同子任务或协议分别验证。分类边界修正不删除来源，也不改写既有评分。

## 三、跨任务特征和未归入任务类的来源

| 独立交付项              | 来源                                                                                                               | 已有证据                                                                    | 还缺什么                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 长输入/长上下文         | PAIO-LONG-PREFILL-5000、PAIO-LONGTEXT-4000、LONGBENCH、LONGBENCH_V2、INFINITEBENCH；RULER 单列为上游构造型受控测试 | 企业两源 ✓R；LONGBENCH_V2 为 L；超长企业三条约7.7万～8.2万实际 prompt token | 公开原生任务评分；按真实输入记录长度，不把长输入等同文档理解   |
| 长输出                  | LONGWRITER                                                                                                         | L                                                                           | 原任务输出与评分；不强制凑输出长度                             |
| 共享前缀请求            | PAIO-PREFIX-SHARED-5000                                                                                            | ✓R                                                                          | 保留分组与原顺序；抽样实际 token 前缀核对；未验证 KV 命中/加速 |
| 会话上下文重用          | PAIO-REUSE-CONV-5000、PAIO-CHAT-1000                                                                               | ✓R                                                                          | 逐记录任务分类及真实跨请求关系；不从标签推断连续会话           |
| 语义相似输入            | PAIO-SEMANTIC-SIMILAR-5000、PAWS                                                                                   | 企业源 ✓R；PAWS 待复核                                                      | 相似标签和任务契约；语义相似不等于相同 token/KV 可复用         |
| 请求尺寸/到达与服务轨迹 | AZURE_LLM、MOONCAKE、BURSTGPT                                                                                      | BURSTGPT 为 L，另两源待复核                                                 | 真实字段与原始时间语义；无 prompt 的元数据不要求伪造模型回放   |

企业来源完整 ID
为：PAIO-CHAT-1000、PAIO-CODE-EVAL-1000、PAIO-JSON-500、PAIO-LONG-PREFILL-5000、PAIO-LONGTEXT-4000、PAIO-PREFIX-SHARED-5000、PAIO-REUSE-CONV-5000、PAIO-SEMANTIC-SIMILAR-5000。表中企业短名仅为显示缩写。

企业代码评判、JSON 和会话数据分别进入相应任务类 PR；剩余企业来源进入上述特征类 PR。八源验证回执完整保留，不能因为重新归类就变成新跑八遍。

## 四、17 个新增候选的完整去向

以下全部尚未完成本次获取、适配和真实执行；不加入已完成分母。

| 新候选                                       | 目标交付项                  | 关键边界                                      |
| -------------------------------------------- | --------------------------- | --------------------------------------------- |
| MULTIAGENTBENCH                              | 多 Agent 协作关系扩展       | 作为跨任务执行关系，不冒充互斥任务类          |
| DEEPPLANNING                                 | 约束规划扩展、应用API工作流 | 保留工具/数据库和官方约束评估                 |
| STREAMINGBENCH、OVOBENCH                     | 在线视频协议扩展            | 不能看未来帧，不等于离线整视频                |
| FULL_DUPLEX_BENCH                            | 音频流交互                  | 固定具体版本及双工协议                        |
| LIBRITTS、LJSPEECH                           | 语音生成扩展                | 文本到音频，不是音频识别                      |
| BEIR_SCIFACT、BEIR_NFCORPUS、MSMARCO_PASSAGE | 向量检索/重排扩展           | 固定 corpus/query/qrels；用途许可分别审查     |
| STSB                                         | 句对相似度扩展              | 不是生产检索请求，也不是 KV 重用证据          |
| DOCVQA                                       | 视觉文档                    | 原图及访问条件                                |
| CODEJUDGE_EVAL                               | 代码评判                    | 使用发布方原始评判任务和标签                  |
| DRAGON_PUBLIC、TEMPORALWIKI                  | 演化语料资产                | 只有真实版本/差异，不补造完整在线事件         |
| COPILOT_AGENT_2026                           | 仓库Agent请求元数据         | 无提示/工具内容时不能原请求回放；不迁观测分析 |
| AZURE_LMM_2025                               | 多模态请求元数据            | 尺寸/时间元数据不能当图像或视频任务           |

引用及来源说明沿用 [新增候选目录](gap-sources-20261021.json)，不在本轮重新宣称已核验下载或原生能力。

## 五、PR 重整与分批次序

### 现有 PR 如何处理

| 现有 PR | 建议调整                                                                                                                                               | 不丢失什么                                            |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| #379    | 收敛为 P0 共用底座：目录、读取/准备/消费、安全边界、CPU CI；GSM8K 专属执行脚本和回执移入独立推理 PR                                                    | 原始验证时间、模型身份、样本和评分不变                |
| #381    | 重整为“独立推理”类 PR，同时承接 GSM8K 与 ARC-Challenge，不再把单数据集当类别                                                                           | 两源全部回执、错误结果、原始配置限制                  |
| #383    | 作为拆分前的历史汇总入口，暂保留 Draft；其公共响应/schema 校验与可选依赖归共用验证底座，各来源脚本/回执按类别分配；替代 PR 创建并互相链接后才关闭本 PR | 八源回执、16K失败与128K配置、工件身份；不重写执行历史 |

确认本计划后再改分支/PR；不 force push 覆盖未检查的他人提交，不修改其他人的 PR。先比对最新 main 和相邻 PR。公共代码只能有一个所属底座
PR，各类别只调用它。重整可能改变文件所在提交，不改变已经执行的软件指纹；新增代码需重新验证受影响路径。

### 提交顺序

1. **P0 共用底座**：重整 #379，并明确基础/验证两层依赖。研究端 #37 的源码+wheel锁按新固定提交更新，重新双包安装消费检查，不把旧锁自动当作新包验证。
1. **第一波已有真实结果**：独立推理 → 代码评判 → 结构化输出 → 对话续接 → 长输入特征 → 共享前缀特征 → 会话上下文重用特征 → 语义相似输入。每个箭头项目独立
   PR；会话源共享回执但子类状态分别记录。
1. **第二波已有本地输入**：通用单轮、抽取分类、单题代码生成、文档分析、工具选择、科学代码、仓库 Agent、视觉图表、交互式编辑轨迹、长输出。每类独立提交，只有真实
   NPU/原生协议完成后才勾验证；轨迹回放只交回放范围。
1. **第三波需原生环境**：状态化环境、数据库、KG、网页、OS、GUI、应用API、规则业务、证据检索。AgentBench 的共享安装可复用，五类环境独立清单/PR，不用 ALFWorld
   一次执行勾五类。
1. **第四波其余任务/特殊模型**：RAG跟进、科学规划、仿真验证、演化语料、跨会话记忆、经历记忆、一般图像、视频、离线音频、摘要、翻译、改写、音频流。可并行准备，但不能因为困难从总清单消失。
1. **独立轨迹包及新增扩展**：AZURE_LLM/MOONCAKE/BURSTGPT 元数据；随后分批处理第四节所列新增任务/协议。元数据只验证读取和字段契约，明确不具备模型任务验证条件。

每波是排队优先级，不是一波一个大 PR，不绑定必须先跑完前一波才能交后一波。若同学已有实现，直接对接既有执行器，不重复另建。一个类别也允许“已可用部分”先交，但必须标题/清单注明子集，不能勾整类完成。

## 六、清单与打勾口径

重整后的清单以“类别 → 数据源 → 原生子集/协议”为主视图，另保留来源去重汇总。每行记录：公开/企业、版本、数据获取、读取适配、本地读取、NPU执行、评分或完整性、对应PR、证据、缺项。

- `[x] 数据源抽样闭环`：预选三条原始样本完成适用协议与核验；模型答错仍保留真实得分。
- `[ ] 类别首批交付`：该类在 PR 中事先声明的首批来源/子集、安装运行、证据及消费接口未全部闭环时不勾。候选余项仍列在“待补来源”，首批交付不等于候选全部完成。
- `[ ] 该类已纳入来源全部验证`：所有正式纳入该类的来源/子集均完成才勾；未审查候选单独登记，不能用调整分母隐藏缺项。
- `不适用`：纯元数据没有可执行 prompt；明确交付数据读取能力，不伪造任务成功勾。
- 每个类别至少包含一个源契约匹配的可用来源才能称已有数据支撑；没有合适来源的类别保持缺口，不用自造数据填充。

既有10来源勾选保留在来源层，但不会无条件复制到各任务类。企业请求格式通过不代表语义质量；共享前缀请求完成不代表引擎机制性能已验证。科研观测、特征挖掘和干预继续排除在本次迁移之外。

## 七、当前阻塞与恢复后的执行

当前授权 SSH 入口连接拒绝，无法继续 NPU 任务，也尚未复核服务器私有响应档案可恢复性。该原因与数据集本身不兼容分别记录，不修改历史验证事实。

恢复后先核对实例、数据卷和现有服务，再继续真实任务。遵照用户授权：每15分钟检查是否已有真实NPU任务，无任务时运行容易验证的真实数据小批次；有任务则不抢占。重复样本标复跑，不计新增来源，不能空转保活。原生任务环境准备可与真实推理串接，不能因整理文档再次遗漏资源空闲期限。

企业原文、模型响应、答案、gated 数据与权重不进公开提交。CI 只有 CPU 正确性 smoke，不承担性能或真实模型验收。全部 PR 默认 Draft，未经用户同意不自动合并。
