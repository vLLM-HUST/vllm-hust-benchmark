# 40类 workload、公开来源与企业数据迁移清单

更新日期：2026-10-23。沿用原文件更新，替代本文原“36类逐类拆PR”计划；不新增平行总清单，不删除历史回执。

## 一、范围、数量和阅读方法

目标是把已有可用的数据获取、原始读取适配、准备/索引/回查和必要原生执行入口提供给公共使用方；研究观测、特征提取、干预及分析留在研究仓。先交付已有能力，不等待全部扩展完成。

本次先整理完整方案，再按用户要求核对和补齐已有公共能力，仅做软件/消费验证；不扩数据集、不下载新载荷、不恢复远端实验、不创建或合并PR。公开来源指有公开发布定位，不代表无条件取得、可公开再分发或适用于商业验收。企业数据永不随公开仓库、wheel或CI发布。

| 统计对象 | 数量 | 口径 |
| --- | ---: | --- |
| 本清单建议任务family | 40 | 原36类＋约束规划、语音生成、向量表示/相似度/检索、重排4类；W01–W40仅为本文序号 |
| 已有公开来源目录 | 89 | 全部保留在第五节；任务、语料、schema、轨迹分别标注 |
| 前轮新增公开候选 | 17 | 第六节；不是已取得或已验证 |
| 本次必须补入的公开来源 | 4 | HLE-Verified、SWE-bench Pro、FrontierScience、Terminal-Bench 2.1；MMLU-Pro已在原89项，不重复计数 |
| 本清单公开来源合计 | 110 | 89＋17＋4；是来源条目数，不是独立血缘数、可运行benchmark数或完成数 |
| 企业私有资产 | 8 | 第四节全部列明；与110项公开来源分开统计 |
| 规划来源总数 | 118 | 110公开来源＋8私有资产；不得拿此数宣传全部可用 |

**代码和验证截面**：既有交付基线为 `c6b6a00c49dd901fd08c6b215448ea69d761f933`，包含53个读取适配/字段映射/原生引用定义；定义存在不等于原始资产验证或完整任务执行。本次已有能力补漏及两端收口版本另记第八节。MATH500后续本地未提交读取工作单列，不并入本轮收口提交或完成数。本文机器目录依据仍是89公开＋8企业及17候选，40类中的新增4类及4个必接新源仅在本方案登记，尚未同步为程序接口。

已提交公开来源中，29项有明确原数据取得/读取验证证据；其中20项有NPU运行尝试（不等于成功），另5项为元数据/历史对话数据核验、4项仅读取准备。旧来源台账勾选11项，但各自只覆盖声明子集且勾选条件存在历史差异，**不据此计算整体完成率**。企业8项各有3条NPU回放完整性记录，没有统一独立语义gold。

### 状态如何理解

- **已有可迁移/已迁代码**：当前分支已有实现，优先复用；尚需逐能力核对源仓与目标仓是否等价，不声称完整迁移审计已完成。
- **字段映射可复用**：已有字段契约，但没有足够原始载荷或端到端证据；不能称已经适配验证。
- **仅登记/需补接入**：有来源定位，不得当成已经下载、安装或运行。
- **[x]**：仅勾选文字明确限定的已完成步骤/样本；**[ ]**：相应步骤未完成。模型答错与执行失败分开，数据读取/CI/研究消费不能替代模型任务。
- 每个来源的“下一步”还包括通用交付要求：许可与版本/摘要核验、依赖与原生入口、CPU正确性smoke、固定安装工件和研究消费核验。纯元数据不要求虚构模型调用。

证据入口：[机器验证台账](dataset-validation-status.json)、[原来源清单](DATASET_CHECKLIST.zh-CN.md)、[执行记录](WORKLOAD_EXECUTION_CHECKLIST.zh-CN.md)、[固定工件与研究消费](DATA_CONSUMPTION_VALIDATION.zh-CN.md)。它们的旧文字若与新证据不同，按明确固定版本的原始回执核对，不能静默升级状态。本文不是重新运行报告，也不宣称旧台账已全部同步。

## 二、40类完整任务—来源对照

任务family表示“做什么”；执行关系/协议和负载特征另列在第七节，不混进40类。每条原任务按目标选择主类，再加辅助标签；同一数据源可跨类，但一次子集验证不得复制成整源或多类完成。

“候选”表示来源可定位、适配边界仍需核验；并非表内每个来源天然能支撑该类完整执行。权威性优先依据发布方原始任务、论文与官方评测；公共训练语料、上游合成数据和元数据不是同等级原生benchmark。历史合成family生成器不作为官方原始数据默认入口。

| 编号／任务family | 对应公开来源（点击查看逐源状态及链接） | 已有可复用能力与验证边界 | 还缺什么 |
| --- | --- | --- | --- |
| W01 通用问答与开放生成 | [WILDCHAT](#src-wildchat)、[LMSYS_CHAT](#src-lmsys-chat)、[DOLLY](#src-dolly)、[NATURALQUESTIONS](#src-naturalquestions)、[DUREADER](#src-dureader)、[NATURAL_INSTRUCTIONS](#src-natural-instructions)、[OASST1](#src-oasst1)、[TULU3](#src-tulu3)、[ULTRACHAT](#src-ultrachat)、[LONGWRITER](#src-longwriter) | DOLLY原行0–2及LongWriter-6k原行0–2已NPU回放、完整性核验；其他语料部分仅字段映射。 | 开放问答与写作保留子标签；训练答案不进入模型输入；未做语义评分或全量验证。 |
| W02 对话续接 | [WILDCHAT](#src-wildchat)、[LMSYS_CHAT](#src-lmsys-chat)、[OASST1](#src-oasst1)、[ULTRACHAT](#src-ultrachat)、[SPOKENWOZ](#src-spokenwoz)、[DAILYTALK](#src-dailytalk) | OASST1三条原父链续接正常终止，固定工件/研究消费已核；企业聊天/复用源见企业表。 | 历史对话和在线多轮交互分开；保持树分支、轮次、角色；BURSTGPT没有对话正文，不作本类任务源。 |
| W03 给定检索材料的问答与跟进 | [QRECC](#src-qrecc)、[QASPER](#src-qasper)、[NARRATIVEQA](#src-narrativeqa)、[DUREADER](#src-dureader)、[NATURALQUESTIONS](#src-naturalquestions)、[LONGBENCH](#src-longbench) | QReCC原test16451条读取准备、3条索引/消费通过；LongBench-Qasper三题已有评分。 | QReCC缺检索库和原生检索/阅读评分；Context是会话历史，不是检索证据；Qasper单轮不代表跟进已验。 |
| W04 文档分析与理解 | [LONGBENCH](#src-longbench)、[LONGBENCH_V2](#src-longbench-v2)、[INFINITEBENCH](#src-infinitebench)、[QUALITY](#src-quality)、[QASPER](#src-qasper)、[NARRATIVEQA](#src-narrativeqa)、[LONGALPACA](#src-longalpaca) | LongBench-Qasper三题官方F1；LongBench-v2三题已尝试/评分；QuALITY三题似然评分。 | v2存在原预算耗尽；QuALITY非发布方基线复现；其他子集及完整上下文/题型覆盖待补。 |
| W05 独立知识与推理求解 | [GSM8K](#src-gsm8k)、[MATH500](#src-math500)、[BBH](#src-bbh)、[GPQA](#src-gpqa)、[MMLU_PRO](#src-mmlu-pro)、[AI2_ARC](#src-ai2-arc)、[OPENMATH2](#src-openmath2)、[OPENTHOUGHTS3](#src-openthoughts3)、[LABBENCH](#src-labbench)、[HLE_VERIFIED](#src-hle-verified)、[FRONTIERSCIENCE](#src-frontierscience) | GSM8K历史去重204/1319；ARC两子集各3题；MMLU-Pro两学科各3题；BBH有提前终止失败。 | HLE-Verified与FrontierScience必须新增接入；科学推理按学科/赛道细分；训练语料不冒充独立评测集。 |
| W06 结构化生成与格式转换 | [JSONSCHEMABENCH](#src-jsonschemabench)、[STRUCTEVAL](#src-structeval)、[SPIDER](#src-spider)、[SCHEMASTORE](#src-schemastore)、[NATURAL_INSTRUCTIONS](#src-natural-instructions) | StructEval Text→JSON原50条执行评分；JSONSchemaBench三份原schema执行验证，1/3符合。 | SchemaStore仅schema资源，不是现成请求集；SQL静态生成与数据库执行分开；格式有效不等于语义正确。 |
| W07 摘要 | [XSUM](#src-xsum)、[MULTINEWS](#src-multinews)、[BILLSUM](#src-billsum)、[XLSUM](#src-xlsum)、[DOLLY](#src-dolly)、[NATURAL_INSTRUCTIONS](#src-natural-instructions) | NI原摘要子任务3/6493已执行评分；Dolly其他任务回放不自动覆盖摘要。 | 其他摘要来源需取得/适配；固定单/多文档、语言、长度及原评分。 |
| W08 翻译 | [FLORES200](#src-flores200)、[NATURAL_INSTRUCTIONS](#src-natural-instructions) | NI某原翻译子任务250/250已执行评分，不代表NI全量。 | FLORES语向、版本、原文/参考对齐与评分待补；不把模型训练参考当生成结果。 |
| W09 信息抽取与分类 | [DOLLY](#src-dolly)、[NATURAL_INSTRUCTIONS](#src-natural-instructions)、[TULU3](#src-tulu3)、[PAWS](#src-paws) | NI分类某任务102/102已执行评分；实体抽取3条已执行评分、最终工件/消费待闭环。 | 抽取与分类分子任务；PAWS为释义判别辅助源，当前原始下载受阻；Dolly其他类别不继承验证。 |
| W10 改写与编辑 | [NATURAL_INSTRUCTIONS](#src-natural-instructions)、[TULU3](#src-tulu3)、[DOLLY](#src-dolly) | NI原改写子任务3条已执行评分、候选安装回查通过。 | 最终CI工件与研究消费未闭环；保留编辑目标/参考，不将摘要或翻译重复当改写完成。 |
| W11 单题代码生成与测试 | [HUMANEVAL](#src-humaneval)、[MBPP](#src-mbpp)、[BIGCODEBENCH](#src-bigcodebench)、[DS1000](#src-ds1000)、[LIVECODEBENCH](#src-livecodebench)、[OPENCODE](#src-opencode)、[OPENTHOUGHTS3](#src-openthoughts3) | HumanEval、BigCodeBench、DS-1000各3条已NPU生成；有数据/读取和部分消费证据。 | 尚缺隔离功能测试及评分；DS-1000一条触及原1024预算；只生成代码不能勾完整任务。 |
| W12 代码评判 | [CODEJUDGE_EVAL](#src-codejudge-eval) | 企业PAIO-CODE-EVAL三条回放/完整性通过。公开CodeJudge-Eval官方来源已核，尚未接入。 | 需官方评判输入与标签、评分；不把代码生成题改写为自造评判题；企业无独立语义gold。 |
| W13 仓库上下文代码补全 | [REPOBENCH](#src-repobench)、[CROSSCODEEVAL](#src-crosscodeeval) | 目录已有；未有本次固定版本真实验证。 | 取得仓库/跨文件上下文，固定补全切分、原评测和安装/消费。 |
| W14 交互式代码编辑助手 | [TRACELAB](#src-tracelab) | 当前TraceLab选定文件只验证3条调用元数据读取/消费。 | 没有已核验的编辑请求正文、补丁和人类反馈序列；本类可执行数据仍缺，不能以元数据冒充。 |
| W15 长程仓库软件工程任务 | [SWE_BENCH](#src-swe-bench)、[SWE_VERIFIED](#src-swe-verified)、[SWESMITH](#src-swesmith)、[SWE_BENCH_PRO](#src-swe-bench-pro) | SWE-bench Verified原任务3条读取/索引/消费通过；其他来源未有完整闭环。 | SWE-bench Pro必须补接入；原仓库、依赖、修复Agent、测试容器及官方评分均须保留。 |
| W16 状态化环境操作 | [AGENTBENCH](#src-agentbench)、[SCIENCEWORLD](#src-scienceworld) | AgentBench ALFWorld FC new_std原0–2三题完成交互和官方评分。 | 仅3/109；其他环境未验；ARM构建/模型替换已披露；ScienceWorld依任务目标辅助归类。 |
| W17 数据库查询与更新执行 | [AGENTBENCH](#src-agentbench)、[SPIDER](#src-spider) | AgentBench目录存在，但ALFWorld结果不覆盖DB；Spider仅作为SQL及数据库辅助资产。 | 缺DB环境、初始状态、真实执行反馈、更新隔离与评分；静态SQL生成不能替代。 |
| W18 知识图谱查询与遍历 | [AGENTBENCH](#src-agentbench) | 只有来源与分类登记。 | 缺原KG数据/服务、遍历工具、原任务和评分；不能用普通文档RAG填充。 |
| W19 网页导航与交互 | [AGENTBENCH](#src-agentbench)、[MIND2WEB](#src-mind2web)、[WEBARENA](#src-webarena)、[VISUALWEBARENA](#src-visualwebarena)、[WORKARENA](#src-workarena) | 只有登记或字段契约，不存在本次完整网页闭环证据。 | 网站镜像、浏览器/视觉协议、站点状态重置及原评分；Mind2Web演示不冒充在线执行。 |
| W20 操作系统与终端任务 | [AGENTBENCH](#src-agentbench)、[TERMINAL_BENCH_2_1](#src-terminal-bench-2-1) | AgentBench OS为候选；Terminal-Bench 2.1已核官方定位，本次新增必须接入。 | 缺隔离容器、命令工具、实际终态/测试；终端套件跨代码/科学任务，按每题目标增辅助分类。 |
| W21 桌面GUI操作 | [OSWORLD](#src-osworld) | 已登记，当前未有原生桌面闭环验证。 | 需桌面/VM环境、截图和动作协议、匹配视觉模型、环境重置与评分。 |
| W22 应用API多步工作流 | [APPWORLD](#src-appworld)、[BFCL](#src-bfcl)、[TOOLBENCH](#src-toolbench)、[DEEPPLANNING](#src-deepplanning) | BFCL仅静态simple_python三题已验，不能覆盖本类；其他来源待补。 | AppWorld/ToolBench实际调用、状态和终态判定；BFCL选择真正stateful子集；保留工具服务依赖。 |
| W23 规则约束业务办理与排障 | [TAUBENCH](#src-taubench) | 目录TAUBENCH实际绑定官方tau2；telecom_full原前三任务闭环、奖励1/1/0。 | 其余domain/任务待跑；模型错误保留；用户/Agent模型与原环境协议、最终版本消费分别核验。 |
| W24 工具选择与参数生成 | [BFCL](#src-bfcl)、[XLAM_TOOL](#src-xlam-tool)、[TOOLACE](#src-toolace) | BFCL v4 simple_python三题官方AST匹配通过；ToolACE原历史对话3条读取/消费通过。 | 静态调用生成、历史对话、真实工具执行分开；参考调用列表不推出调用依赖或并行。 |
| W25 主动证据检索与综合 | [GAIA](#src-gaia)、[BFCL](#src-bfcl) | 官方原题与工具框架路线已确定；无本次官方GAIA完整任务闭环证据。 | 需原题附件、公开Agent框架、检索/网页/文件工具及官方评分；不使用QBI FrozenEvidence改题。 |
| W26 约束规划与重规划【新增】 | [DEEPPLANNING](#src-deepplanning) | Qwen官方原生工具/数据库及约束任务已核，尚未取得适配。 | 固定旅行/购物与语言版本、数据库、工具、全局约束评分；首轮运行不代表重规划行为已验证。 |
| W27 科学实验规划 | [DISCOVERYBENCH](#src-discoverybench)、[SCIENCEWORLD](#src-scienceworld)、[LABBENCH](#src-labbench)、[MLAGENTBENCH](#src-mlagentbench)、[SCIENCEAGENTBENCH](#src-scienceagentbench) | 候选来源存在；ScienceAgentBench三条读取不证明实验规划已执行。 | 须逐题确认实际规划契约；LAB-Bench原设计/协议类题只支持静态规划评估，不是执行实验。 |
| W28 仿真、分析与验证流水线 | [SCIENCEWORLD](#src-scienceworld)、[DISCOVERYBENCH](#src-discoverybench)、[BIXBENCH](#src-bixbench)、[NATUREBENCH](#src-naturebench)、[SCIENCEAGENTBENCH](#src-scienceagentbench) | ScienceAgentBench三条原任务读取/消费已验；其他原生流水线未完成。 | 取得真实科学数据/仿真环境/依赖和结果判定；仅纳入确有该任务契约的子集。 |
| W29 科学代码工作流 | [SCIENCEAGENTBENCH](#src-scienceagentbench)、[NATUREBENCH](#src-naturebench)、[MLEBENCH](#src-mlebench)、[MLAGENTBENCH](#src-mlagentbench)、[BIXBENCH](#src-bixbench)、[PAPERBENCH](#src-paperbench)、[DS1000](#src-ds1000) | ScienceAgentBench三条读取/消费；DS-1000仅单题生成，不能代表完整科研工作流。 | 科学数据、工具运行、结果/复现实验判定未完成；MLE/PaperBench等可能需额外计算资源，不能承诺双NPU即可全跑。 |
| W30 演化语料检索 | [DRAGON_PUBLIC](#src-dragon-public)、[TEMPORALWIKI](#src-temporalwiki) | 新增公开候选尚未取得/适配；原QReCC等静态数据不作为本类直接证据。 | DRAGOn月度快照与TemporalWiki时间资产不等于完整在线查询/增删事件；缺任务-版本绑定则保持缺口。 |
| W31 跨会话对话记忆 | [LONGMEMEVAL](#src-longmemeval)、[LOCOMO](#src-locomo) | LoCoMo原六题生成和发布方F1已完成；LongMemEval已有字段映射，资产/运行未核。 | LoCoMo其他1980题、第5类与视觉兼容待验；保留会话顺序，不从记忆题推出缓存收益。 |
| W32 任务经历与环境轨迹记忆 | [LONGMEMEVAL_V2](#src-longmemeval-v2) | 已核官方LongMemEval-V2是Web/企业Agent长期记忆任务；当前仅登记。 | 取得原任务轨迹/截图/问题、历史链接与原评分；不能把HF自动imagefolder视图当完整任务包。 |
| W33 一般图像理解 | [VISIONARENA](#src-visionarena)、[MMMU](#src-mmmu) | 目录与部分字段映射存在，未有当前NPU视觉执行证据。 | 原图、题目/会话、匹配模型、原评分或完整性协议；纯文本替代不算视觉验证。 |
| W34 视觉文档与图表理解 | [CHARTQA](#src-chartqa)、[MMMU](#src-mmmu)、[DOCVQA](#src-docvqa) | ChartQA原测试文件三条图像/问题/参考读取和消费通过，未推理。 | 视觉模型、原图/版式、OCR或切片协议及评分；DocVQA需按官方下载/使用条件取得。 |
| W35 视频理解 | [VIDEOMME](#src-videomme)、[VIDEOMME_V2](#src-videomme-v2)、[MVBENCH](#src-mvbench)、[LONGVIDEOBENCH](#src-longvideobench)、[STREAMINGBENCH](#src-streamingbench)、[OVOBENCH](#src-ovobench) | 现有离线视频候选未验证；在线协议补StreamingBench/OVO-Bench。 | 原视频/时间戳、采帧/字幕及评分；流式任务只可看当时可见前缀，不能喂完整未来视频。 |
| W36 离线音频识别与理解 | [LIBRISPEECH](#src-librispeech)、[FLEURS](#src-fleurs)、[WAVCAPS](#src-wavcaps) | LibriSpeech/FLEURS有字段契约，原音频与任务未在当前版本验证。 | ASR、音频描述/问答分别记录；取得波形、采样和文本参考，匹配音频模型与原评分。 |
| W37 音频流交互 | [FULL_DUPLEX_BENCH](#src-full-duplex-bench)、[SPOKENWOZ](#src-spokenwoz)、[DAILYTALK](#src-dailytalk) | Full-Duplex-Bench为更直接原生协议候选；后两者仅提供录制对话辅助语料。 | 固定双工版本、实时输入输出、轮换/重叠/打断与原评分；录音回放或离线ASR不能替代。 |
| W38 文本到语音生成【新增】 | [LIBRITTS](#src-libritts)、[LJSPEECH](#src-ljspeech) | 发布方TTS语料定位已核，未取得/适配。 | 文本/说话人/波形配对、TTS模型与实际音频输出；质量及性能各自核验，不冒充对话语音。 |
| W39 向量表示与相似度／检索任务【新增】 | [BEIR_SCIFACT](#src-beir-scifact)、[BEIR_NFCORPUS](#src-beir-nfcorpus)、[MSMARCO_PASSAGE](#src-msmarco-passage)、[STSB](#src-stsb)、[PAWS](#src-paws) | 公开候选已有定位；PAWS仍下载受阻，不是检索语料。 | 分别固定编码、句对相似度、检索子任务；corpus/query/qrels、向量模型、指标待补；PAWS只辅助释义分类。 |
| W40 相关性重排【新增】 | [BEIR_SCIFACT](#src-beir-scifact)、[BEIR_NFCORPUS](#src-beir-nfcorpus)、[MSMARCO_PASSAGE](#src-msmarco-passage) | 尚无当前版本适配或评分；不能从文本问答通过推导。 | 固定候选集合、相关性标签、ranker输入/分数及原评价；MS MARCO用途条款需单独核验。 |

W26、W38、W39、W40为新增建议类；其余36类保留原稳定family ID，不因本文重排编号改写历史回执。四个新增ID建议为 `constraint-planning`、`text-to-speech-generation`、`embedding-retrieval`、`reranking`，待实现时审查接口兼容。W39内部编码、相似度和检索分别验证，不能相互替代。W06仅在结构/格式本身是主要任务目标时作主类；其他任务的JSON约束作为输出标签，避免重复计数。

## 三、五项必须纳入的来源

| 来源及权威定位 | 归类 | 当前状态 | 必须保留的原始任务和执行要求 |
| --- | --- | --- | --- |
| [MMLU-Pro官方代码](https://github.com/TIGER-AI-Lab/MMLU-Pro)／[数据](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | W05：学科知识/推理 | 已登记、已迁读取及执行入口；biology/chemistry各3题验证，不再重复新增来源 | 固定revision、学科、题型/选项、参考答案、原始评分；余12学科和全量未验 |
| [HLE-Verified论文](https://arxiv.org/abs/2602.13964)／[论文指向的数据](https://huggingface.co/datasets/skylenage-ai/HLE-Verified) | W05：跨学科专家级难题；按实际输入加视觉标签 | 官方发布定位已核；需补获取、适配、执行/评分和消费 | 区分发布方验证/修订/不确定状态，固定发行版本；不能拿原HLE或其他Verified版本冒充；保留图像和参考 |
| [SWE-bench Pro官方代码与公开版入口](https://github.com/scaleapi/SWE-bench_Pro-os) | W15：长程仓库软件工程 | 新增必接；未取得或验证 | 公开奖励/测试和任务实例按发行版固定，真实仓库编辑及功能测试；不混SWE-bench Verified/Pro Verified，不取私有商业集 |
| [FrontierScience官方介绍](https://openai.com/index/frontierscience/)／[官方数据](https://huggingface.co/datasets/openai/frontierscience) | W05：科学推理；Olympiad/Research分标签 | 新增必接；未取得或验证 | 两赛道分开，保留原题、参考与评分rubric；这是专家级科学推理，不据名称冒充W27–W29实验/仿真/代码闭环 |
| [Terminal-Bench 2.1官方仓库](https://github.com/harbor-framework/terminal-bench-2-1) | 套件入口W20；逐题按目标辅助标W11/W15/W28/W29等 | 新增必接；未取得或验证 | 固定2.1任务、容器、Harbor/Agent和验证测试；终端只是接口，不能把每题简化成静态命令生成；ARM/依赖兼容尚待验证 |

以上“官方已核”只指2026-10-23复核发布方页面/任务定位；不是原始数据已下载或评测已复现。原题有缺陷或环境不兼容时保留实例与原因，不为跑通改题或默默剔除。

## 四、8份企业私有资产完整清单

来源均为企业交付，部分由真实请求衍生或构造，不笼统称未经加工的生产流量；企业资产本身不是完整benchmark。可公开的是获准的读取/适配代码、配置说明及脱敏摘要；请求正文、响应、答案、附件与完整索引不公开。下表记录名中的数量不是本次全量验证数。

| 企业资产ID／版本 | 对应任务或跨任务特征 | 已有可迁移能力 | 已验证范围 | 仍缺事项／限制 |
| --- | --- | --- | --- | --- |
| `PAIO-CHAT-1000`／v1 | W02为主；逐请求确认客服/问答子类 | 已迁：原请求包装读取/回放；保留原请求体 | [x] 原3条NPU回放、响应完整性；无语义正确率；[回执](validation/paio-chat-1000-npu-20261022.json) | 逐记录任务/会话关系；没有独立语义gold，不能宣称多轮业务闭环 |
| `PAIO-CODE-EVAL-1000`／v1 | W12代码评判 | 已迁：原请求包装读取/回放；不重造评判prompt | [x] 原3条NPU回放、响应完整性；无语义正确率；[回执](validation/paio-code-eval-1000-npu-20261022.json) | 缺独立评判参考和评分；代码评判不等于执行被评代码 |
| `PAIO-JSON-500`／v1 | W06；内容若是抽取则辅标W09 | 已迁：原请求包装读取/回放；保留原schema/指令 | [x] 原3条NPU回放、响应完整性；无语义正确率；[回执](validation/paio-json-500-npu-20261022.json) | 缺独立抽取/转换语义gold；JSON解析通过不等于任务正确 |
| `PAIO-LONG-PREFILL-5000`／v2 | 长输入/prefill特征；任务语义逐记录确认 | 已迁：原messages/API字段投影、读取/回放 | [x] 原3条NPU回放、响应完整性；无语义正确率；[回执](validation/paio-long-prefill-5000-npu-20261022.json) | 长输入不是独立语义类；未验证prefill加速或全量 |
| `PAIO-LONGTEXT-4000`／超长文本交付 | 超长输入特征；不得默认归W04 | 已迁：原messages/API字段投影、读取/回放 | [x] 原3条NPU回放、响应完整性；无语义正确率；[回执](validation/paio-longtext-4000-npu-20261022.json) | 已测三条长输入不代表全4000条；原失败/新配置分别保留 |
| `PAIO-PREFIX-SHARED-5000`／v2 | 共享前缀特征；任务主类逐记录确认 | 已迁：原messages读取/回放；保留来源身份 | [x] 原3条NPU回放、响应完整性；无语义正确率；[回执](validation/paio-prefix-shared-5000-npu-20261022.json) | 需核真实token前缀/分组/顺序；未证明KV命中、复用关系或加速 |
| `PAIO-REUSE-CONV-5000`／v2 | W02候选＋会话上下文重用特征 | 已迁：原messages读取/回放；不伪造连续交互 | [x] 原3条NPU回放、响应完整性；无语义正确率；[回执](validation/paio-reuse-conv-5000-npu-20261022.json) | 需核会话ID/追加关系；标签不能替代真实跨请求依赖证据 |
| `PAIO-SEMANTIC-SIMILAR-5000`／v2 | 语义相似输入特征；不直接当W39检索任务 | 已迁：原messages读取/回放；保留原字段 | [x] 原3条NPU回放、响应完整性；无语义正确率；[回执](validation/paio-semantic-similar-5000-npu-20261022.json) | 缺独立相似度gold；语义相似不等于相同token或KV可复用 |

八份均只继承各自真实回执范围；重新归类不重跑八遍、不改变历史模型/预算。企业原始数据后续只通过授权外部目录绑定；公开使用者没有这些载荷时应明确提示未配置，不能用伪造样本代替原数据。

## 五、原89个公开来源逐项盘点

全部来源按ID列出，官方定位沿用既有来源目录；本轮重点复核五项必接源及扩展边界，没有宣称对89项都重新完成下载、许可或官方性审计。混合/衍生/构造数据保留血缘，不当成独立生产流量。部分来源在多类中重复出现，本节只计一次。

“代码可复用”检查依据当前已提交适配注册与原生入口，**不是完整的源→目标能力等价审计**；旧获取脚本/准备脚本是否全部有等价目标实现仍需逐项核查。默认“未复核”也不等于历史从未有人运行。

| 来源ID／原始定位 | 数据性质 | 类别/维度 | 已有可迁移／需补接入 | 真实验证证据（仅所述范围） | 仍缺什么 |
| --- | --- | --- | --- | --- | --- |
| <a id="src-agentbench"></a>[AGENTBENCH](https://github.com/THUDM/AgentBench) | 原生任务／数据资产 | W16、W17、W18、W19、W20 | 已迁：ALFWorld原生引用/准备 | [x] ALFWorld原3题交互＋官方评分；其他环境未验；[记录](validation/agentbench-alfworld-public-handoff-npu-20261023.json) | DB/KG/OS/Web原生环境、其他ALFWorld任务；ARM/模型替换不等于官方容器复现。 |
| <a id="src-ai2-arc"></a>[AI2_ARC](https://huggingface.co/datasets/allenai/ai2_arc) | 原生任务／数据资产 | W05 | 已迁：数据读取适配；非完整执行承诺 | [x] Challenge/Easy各3题执行评分（2/3、1/3）；[记录](validation/arc-challenge-npu-20261022.json) | 两个子集剩余题目；不合并成代表性质量结论。 |
| <a id="src-appworld"></a>[APPWORLD](https://github.com/StonyBrookNLP/appworld) | 原生任务／数据资产 | W22 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W22。 |
| <a id="src-azure-llm"></a>[AZURE_LLM](https://github.com/Azure/AzurePublicDataset) | 历史轨迹／元数据 | 跨任务关系／运行维度 | 已有字段映射可复用；原数据适配待复核 | [x] 2023两文件28185条元数据字段检查；各3条索引回查/消费；[记录](validation/original-serving-traces-reader-20261023.json) | 其他年份/文件；无原prompt，模型执行与任务评分不适用。 |
| <a id="src-bbh"></a>[BBH](https://github.com/suzgunmirac/BIG-Bench-Hard) | 原生任务／数据资产 | W05 | 已迁：数据读取适配；非完整执行承诺 | [ ] Boolean三题NPU尝试并评分0/3；提前停止、无最终答案；[记录](validation/bbh-boolean-original-npu-20261023.json) | 排查原harness停止协议；最终工件消费、其他任务；失败记录保留。 |
| <a id="src-bfcl"></a>[BFCL](https://github.com/ShishirPatil/gorilla) | 原生任务／数据资产 | W22、W24、W25 | 已迁：数据读取适配；非完整执行承诺 | [x] v4 simple_python原3题官方AST匹配；[记录](validation/bfcl-v4-simple-python-public-handoff-npu-20261023.json) | 其他类别/全量；AST匹配不代表工具执行。 |
| <a id="src-bigcodebench"></a>[BIGCODEBENCH](https://huggingface.co/datasets/bigcode/bigcodebench) | 原生任务／数据资产 | W11 | 已迁：数据读取适配；非完整执行承诺 | [ ] 原3题生成正常终止；已读取/消费；[记录](validation/bigcodebench-original-generation-npu-20261023.json) | 隔离功能测试、官方评分和更广覆盖；不能计完整代码任务。 |
| <a id="src-billsum"></a>[BILLSUM](https://github.com/FiscalNote/BillSum) | 原生任务／数据资产 | W07 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W07。 |
| <a id="src-bixbench"></a>[BIXBENCH](https://github.com/Future-House/BixBench) | 历史轨迹／元数据 | W28、W29 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W28、W29。 |
| <a id="src-burstgpt"></a>[BURSTGPT](https://github.com/HPMLL/BurstGPT) | 历史轨迹／元数据 | 跨任务关系／运行维度 | 已迁：数据读取适配；非完整执行承诺 | [x] 首个原文件3条元数据prepare/resolve及消费；[记录](validation/burstgpt-original-metadata-reader-20261023.json) | 其他文件/全量；无原对话，不能模型回放。 |
| <a id="src-chartqa"></a>[CHARTQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) | 原生任务／数据资产 | W34 | 已迁：数据读取适配；非完整执行承诺 | [x] 原测试文件3条图像/问题/参考读取与消费；无视觉运行；[记录](validation/public-original-assets-server-reader-20261023.json) | 视觉模型、真实图像推理与原评分；其他分片。 |
| <a id="src-crosscodeeval"></a>[CROSSCODEEVAL](https://github.com/amazon-science/cceval) | 原生任务／数据资产 | W13 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W13。 |
| <a id="src-dailytalk"></a>[DAILYTALK](https://github.com/keonlee9420/DailyTalk) | 对话／训练语料 | W02、W37 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 原音频/文本/说话人配对及许可；录制双人对话不能当实时打断验证。 |
| <a id="src-discoverybench"></a>[DISCOVERYBENCH](https://github.com/allenai/discoverybench) | 原生任务／数据资产 | W27、W28 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W27、W28。 |
| <a id="src-dolly"></a>[DOLLY](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 对话／训练语料 | W01、W07、W09、W10 | 已迁：数据读取适配；非完整执行承诺 | [x] train原0–2回放及响应完整性；无语义评分；[记录](validation/dolly-original-npu-20261022.json) | 原其他子任务、全量和适用语义核验；不是官方评测复现。 |
| <a id="src-ds1000"></a>[DS1000](https://huggingface.co/datasets/xlangai/DS-1000) | 原生任务／数据资产 | W11、W29 | 已迁：数据读取适配；非完整执行承诺 | [ ] Pandas原3题生成：stop/length/stop；读取/消费通过；[记录](validation/ds1000-original-generation-handoff-20261023.json) | 原依赖环境与隔离功能评分；保留原1024预算终态。 |
| <a id="src-dureader"></a>[DUREADER](https://github.com/baidu/DuReader) | 原生任务／数据资产 | W01、W03 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W01、W03。 |
| <a id="src-fleurs"></a>[FLEURS](https://huggingface.co/datasets/google/fleurs) | 原生任务／数据资产 | W36 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；复核现有字段映射并补原生准备、适用执行/评分和固定包/消费验证；本类细项见W36。 |
| <a id="src-flores200"></a>[FLORES200](https://huggingface.co/datasets/facebook/flores) | 原生任务／数据资产 | W08 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W08。 |
| <a id="src-gaia"></a>[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) | 原生任务／数据资产 | W25 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 官方原题/附件、框架、工具、原生终止和评分；旧QBI衍生执行不能当原版验证。 |
| <a id="src-gpqa"></a>[GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | 原生任务／数据资产 | W05 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 先核官方访问条件和所选split；原题/答案与原评分、模型执行均待验证。 |
| <a id="src-gsm8k"></a>[GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 原生任务／数据资产 | W05 | 已迁：数据读取适配；非完整执行承诺 | [x] 历史去重204/1319题执行评分；有索引/消费证据；[记录](validation/gsm8k-npu-20261022.json) | 非全量、配置不一，不合并准确率；不为整理清单重复跑已验证ID。 |
| <a id="src-humaneval"></a>[HUMANEVAL](https://github.com/openai/human-eval) | 原生任务／数据资产 | W11 | 已迁：数据读取适配；非完整执行承诺 | [x] 原164题取得；[ ] 原3题仅生成，未运行测试；[记录](validation/humaneval-original-generation-npu-20261023.json) | 安全沙箱/功能测试/官方评分；原容器权限阻塞保留。 |
| <a id="src-infinitebench"></a>[INFINITEBENCH](https://github.com/OpenBMB/InfiniteBench) | 原生任务／数据资产 | W04 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W04。 |
| <a id="src-jsonschemabench"></a>[JSONSCHEMABENCH](https://github.com/guidance-ai/jsonschemabench) | schema资源，不是现成请求 | W06 | 已迁：数据读取适配；非完整执行承诺 | [x] 原3份schema执行验证，1/3符合；固定包/消费已核；[记录](validation/jsonschemabench-original-schema-npu-20261023.json) | 另2份失败保留；全量/其他schema；不是参考OpenAIEngine复现。 |
| <a id="src-labbench"></a>[LABBENCH](https://github.com/Future-House/LAB-Bench) | 原生任务／数据资产 | W05、W27 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W05、W27。 |
| <a id="src-librispeech"></a>[LIBRISPEECH](https://huggingface.co/datasets/openslr/librispeech_asr) | 原生任务／数据资产 | W36 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；复核现有字段映射并补原生准备、适用执行/评分和固定包/消费验证；本类细项见W36。 |
| <a id="src-livecodebench"></a>[LIVECODEBENCH](https://github.com/LiveCodeBench/LiveCodeBench) | 原生任务／数据资产 | W11 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W11。 |
| <a id="src-lmsys-chat"></a>[LMSYS_CHAT](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 对话／训练语料 | W01、W02 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；复核现有字段映射并补原生准备、适用执行/评分和固定包/消费验证；本类细项见W01、W02。 |
| <a id="src-locomo"></a>[LOCOMO](https://github.com/snap-research/locomo) | 发布方构造／合成语料 | W31 | 已迁：数据读取适配；非完整执行承诺 | [x] 原0–5六题执行＋发布方F1；固定数据/消费已核；[记录](validation/locomo-original-qa-sampled-20261023.json) | 其余1980题、第5类与视觉兼容；历史500/引擎退出保留，未证根因修复。 |
| <a id="src-longalpaca"></a>[LONGALPACA](https://huggingface.co/datasets/Yukang/LongAlpaca-12k) | 发布方构造／合成语料 | W04 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 发布方构造长问答训练语料；固定原上下文与参考，不能当自然在线流量。 |
| <a id="src-longbench"></a>[LONGBENCH](https://huggingface.co/datasets/THUDM/LongBench) | 原生任务／数据资产 | W03、W04 | 已迁：数据读取适配；非完整执行承诺 | [x] v1 Qasper原3/200题＋官方F1（22.52%）；[记录](validation/longbench-v1-qasper-public-handoff-npu-20261023.json) | 其他子集和全量；F1不等于正确率或质量达标。 |
| <a id="src-longbench-v2"></a>[LONGBENCH_V2](https://huggingface.co/datasets/THUDM/LongBench-v2) | 原生任务／数据资产 | W04 | 已迁：数据读取适配；非完整执行承诺 | [ ] 原3题执行评分0/3，其中1题触及原预算；[记录](validation/longbench-v2-original-attempt-npu-20261022.json) | Medium/Long资源与终止状态、其他子集；不把模型错误等同执行故障。 |
| <a id="src-longmemeval"></a>[LONGMEMEVAL](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | 发布方构造／合成语料 | W31 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；复核现有字段映射并补原生准备、适用执行/评分和固定包/消费验证；本类细项见W31。 |
| <a id="src-longmemeval-v2"></a>[LONGMEMEVAL_V2](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2) | 原生任务／数据资产 | W32 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 原始历史轨迹/截图/问题及关联、数据revision、记忆评分；已核官方存在但未取得。 |
| <a id="src-longvideobench"></a>[LONGVIDEOBENCH](https://huggingface.co/datasets/longvideobench/LongVideoBench) | 原生任务／数据资产 | W35 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W35。 |
| <a id="src-longwriter"></a>[LONGWRITER](https://huggingface.co/datasets/zai-org/LongWriter-6k) | 发布方构造／合成语料 | W01 | 已迁：数据读取适配；非完整执行承诺 | [x] LongWriter-6k原3条回放/完整性；无质量评分；[记录](validation/longwriter-original-npu-20261022.json) | 原题篇幅和质量、全量；不是LongBench-Write。 |
| <a id="src-math500"></a>[MATH500](https://github.com/openai/prm800k/tree/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/math_splits) | 原生任务／数据资产 | W05 | 本地未提交读取适配；不是已提交成果 | [x] 本地未提交：固定文件500题读取/唯一ID检查；[ ] 执行未归档闭环 | 现有远端尝试未归档入此提交，不能写成已验证或确定从未运行；勿重复已有ID；最终包、评分、消费待补。 |
| <a id="src-mbpp"></a>[MBPP](https://github.com/google-research/google-research) | 原生任务／数据资产 | W11 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W11。 |
| <a id="src-mind2web"></a>[MIND2WEB](https://huggingface.co/datasets/osunlp/Mind2Web) | 原生任务／数据资产 | W19 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W19。 |
| <a id="src-mlagentbench"></a>[MLAGENTBENCH](https://github.com/snap-stanford/MLAgentBench) | 原生任务／数据资产 | W27、W29 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 原实验仓库/数据/执行依赖和原指标；不能以科学问答代替。 |
| <a id="src-mlebench"></a>[MLEBENCH](https://github.com/openai/mle-bench) | 原生任务／数据资产 | W29 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 原比赛数据访问条件、训练/执行环境与原评分；不可承诺双NPU能完成所有计算。 |
| <a id="src-mmlu-pro"></a>[MMLU_PRO](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | 原生任务／数据资产 | W05 | 已迁：数据读取适配；非完整执行承诺 | [x] biology/chemistry各原3题，6/12032；六题答对；[记录](validation/mmlu-pro-biology-npu-20261022.json) | 其他12学科及全量；不因已验证而重复接入另一份MMLU-Pro。 |
| <a id="src-mmmu"></a>[MMMU](https://huggingface.co/datasets/MMMU/MMMU) | 原生任务／数据资产 | W33、W34 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；复核现有字段映射并补原生准备、适用执行/评分和固定包/消费验证；本类细项见W33、W34。 |
| <a id="src-mooncake"></a>[MOONCAKE](https://github.com/kvcache-ai/Mooncake) | 历史轨迹／元数据 | 跨任务关系／运行维度 | 已有字段映射可复用；原数据适配待复核 | [x] FAST25两文件35639条元数据字段/索引；各3条消费；[记录](validation/original-serving-traces-reader-20261023.json) | 其他文件/合成轨迹；无原prompt，不做任务执行或缓存效果声明。 |
| <a id="src-multinews"></a>[MULTINEWS](https://github.com/Alex-Fabbri/Multi-News) | 原生任务／数据资产 | W07 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W07。 |
| <a id="src-mvbench"></a>[MVBENCH](https://huggingface.co/datasets/OpenGVLab/MVBench) | 原生任务／数据资产 | W35 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W35。 |
| <a id="src-narrativeqa"></a>[NARRATIVEQA](https://huggingface.co/datasets/deepmind/narrativeqa) | 原生任务／数据资产 | W03、W04 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；复核现有字段映射并补原生准备、适用执行/评分和固定包/消费验证；本类细项见W03、W04。 |
| <a id="src-natural-instructions"></a>[NATURAL_INSTRUCTIONS](https://github.com/allenai/natural-instructions) | 对话／训练语料 | W01、W06、W07、W08、W09、W10 | 已迁：数据读取适配；非完整执行承诺 | [x] 分类某任务102条、翻译某任务250条、摘要某任务3条；改写/抽取各3条已执行评分；[记录](validation/natural-instructions-original-npu-20261023.json) | 新增改写/抽取批最终工件及消费；原任务ID/定义按回执保留，其他NI任务未验。 |
| <a id="src-naturalquestions"></a>[NATURALQUESTIONS](https://github.com/google-research-datasets/natural-questions) | 原生任务／数据资产 | W01、W03 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W01、W03。 |
| <a id="src-naturebench"></a>[NATUREBENCH](https://github.com/FrontisAI/NatureBench) | 历史轨迹／元数据 | W28、W29 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 原论文对应科学数据、环境与原任务/结果判定；仅核官方定位，未取得或执行。 |
| <a id="src-oasst1"></a>[OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 对话／训练语料 | W01、W02 | 已迁：数据读取适配；非完整执行承诺 | [x] 3条原父链续接正常stop＋固定包/消费；无语义评分；[记录](validation/oasst1-original-history-npu-20261023.json) | 仅两棵树；无实时交互、独立质量评分或全量结论。 |
| <a id="src-opencode"></a>[OPENCODE](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | 发布方构造／合成语料 | W11 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 发布方合成代码训练语料；不将训练输出当执行结果，需实际代码测试/环境。 |
| <a id="src-openmath2"></a>[OPENMATH2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | 发布方构造／合成语料 | W05 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 发布方合成训练语料；原始字段/去重/许可与回放适配待核，不当独立评测集。 |
| <a id="src-openthoughts3"></a>[OPENTHOUGHTS3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | 发布方构造／合成语料 | W05、W11 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 发布方合成推理训练语料；选实际代码或推理子集，核原始任务和参考隔离。 |
| <a id="src-osworld"></a>[OSWORLD](https://github.com/xlang-ai/OSWorld) | 原生任务／数据资产 | W21 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W21。 |
| <a id="src-paperbench"></a>[PAPERBENCH](https://github.com/openai/frontier-evals/tree/main/project/paperbench) | 原生任务／数据资产 | W29 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 论文/仓库/复现实验及原评分依赖；部分任务可能需额外计算资源。 |
| <a id="src-paws"></a>[PAWS](https://huggingface.co/datasets/google-research-datasets/paws) | 原生任务／数据资产 | W09、W39 | 已有字段映射可复用；原数据适配待复核 | [ ] 下载403/超时，未取得原载荷；[记录](validation/paws-original-acquisition-20261023.json) | 按原源恢复取得；不以重构QQP或未核镜像代替；未验证适配。 |
| <a id="src-qasper"></a>[QASPER](https://huggingface.co/datasets/allenai/qasper) | 原生任务／数据资产 | W03、W04 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W03、W04。 |
| <a id="src-qrecc"></a>[QRECC](https://github.com/apple/ml-qrecc) | 对话／训练语料 | W03 | 已有字段映射可复用；原数据适配待复核 | [x] 原test16451条准备；3条索引/研究消费；未运行任务 | 原检索库、检索/阅读原生执行与评分；不能把Context当检索证据。 |
| <a id="src-quality"></a>[QUALITY](https://github.com/nyu-mll/quality) | 原生任务／数据资产 | W04 | 已迁：数据读取适配；非完整执行承诺 | [x] dev2086问题读取；3题NPU似然与固定harness评分1/3（归一化2/3）；消费见固定工件记录；[记录](validation/quality-original-mc-npu-20261023.json) | 不是发布方基线；其他题/全量，最终锁身份逐项确认；台账消费待核旧注释需后续同步。 |
| <a id="src-repobench"></a>[REPOBENCH](https://github.com/Leolty/repobench) | 原生任务／数据资产 | W13 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W13。 |
| <a id="src-ruler"></a>[RULER](https://github.com/NVIDIA/RULER) | 发布方构造／合成语料 | 跨任务关系／运行维度 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 上游生成器/种子/长度/任务规则与原评分；标为官方构造受控探针，不是自然真实请求。 |
| <a id="src-schemastore"></a>[SCHEMASTORE](https://github.com/SchemaStore/schemastore) | schema资源，不是现成请求 | W06 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 只能迁schema资产；若无上游任务协议，不自编prompt冒充原始benchmark。 |
| <a id="src-scienceagentbench"></a>[SCIENCEAGENTBENCH](https://huggingface.co/datasets/osunlp/ScienceAgentBench) | 原生任务／数据资产 | W27、W28、W29 | 已迁：数据读取适配；非完整执行承诺 | [x] 原选定文件3任务读取/prepare/resolve/研究消费；未执行；[记录](validation/public-original-assets-server-reader-20261023.json) | 科学任务数据、环境和原生执行/评分；读取不能勾科学工作流。 |
| <a id="src-scienceworld"></a>[SCIENCEWORLD](https://github.com/allenai/ScienceWorld) | 原生任务／数据资产 | W16、W27、W28 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W16、W27、W28。 |
| <a id="src-spider"></a>[SPIDER](https://github.com/taoyds/spider) | 原生任务／数据资产 | W06、W17 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 固定原SQL任务/数据库与原执行评分；SQL文本生成和交互DB任务必须分开。 |
| <a id="src-spokenwoz"></a>[SPOKENWOZ](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/spokenwoz) | 对话／训练语料 | W02、W37 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 原录制语音/对话/状态的配对和协议；不是已运行的实时双工系统。 |
| <a id="src-structeval"></a>[STRUCTEVAL](https://huggingface.co/datasets/TIGER-Lab/StructEval) | 原生任务／数据资产 | W06 | 已迁：数据读取适配；非完整执行承诺 | [x] Text→JSON原50/50条执行评分；不是整套2035条；[记录](validation/structeval-json-original-npu-20261022.json) | 其他格式/转换/视觉子集；正式质量结论需按原指标。 |
| <a id="src-swe-bench"></a>[SWE_BENCH](https://huggingface.co/datasets/princeton-nlp/SWE-bench) | 原生任务／数据资产 | W15 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；复核现有字段映射并补原生准备、适用执行/评分和固定包/消费验证；本类细项见W15。 |
| <a id="src-swe-verified"></a>[SWE_VERIFIED](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | 原生任务／数据资产 | W15 | 已迁：数据读取适配；非完整执行承诺 | [x] 原选定文件3任务读取/索引/研究消费；未执行；[记录](validation/public-original-assets-server-reader-20261023.json) | 仓库初始化、Agent修复、测试环境与官方评分。 |
| <a id="src-swesmith"></a>[SWESMITH](https://github.com/SWE-bench/SWE-smith) | 发布方构造／合成语料 | W15 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 发布方生成仓库任务/轨迹；固定实例、仓库、测试与许可，非自然生产请求。 |
| <a id="src-taubench"></a>[TAUBENCH](https://github.com/sierra-research/tau2-bench) | 原生任务／数据资产 | W23 | 已迁：原生任务引用/执行；注册注释待同步 | [x] 官方tau2 telecom_full前3任务交互/评分，奖励1/1/0；[记录](validation/tau2-reference-handoff-npu-20261022.json) | 其他任务/domain、最终固定版本消费锁；不是原tau-bench全套。 |
| <a id="src-toolace"></a>[TOOLACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 发布方构造／合成语料 | W24 | 已迁：数据读取适配；非完整执行承诺 | [x] 发布方合成历史工具对话3条读取/消费；[记录](validation/public-original-assets-server-reader-20261023.json) | 原对话结构与其他数据；不虚构在线episode、执行环境或任务评分。 |
| <a id="src-toolbench"></a>[TOOLBENCH](https://github.com/OpenBMB/ToolBench) | 发布方构造／合成语料 | W22 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 固定原工具任务、API/环境可用性与原评分；外部服务失效不能改题隐瞒。 |
| <a id="src-tracelab"></a>[TRACELAB](https://github.com/uw-syfi/TraceLab) | 历史轨迹／元数据 | W14 | 已迁：数据读取适配；非完整执行承诺 | [x] 选定文件3条调用元数据读取/消费；[记录](validation/public-original-assets-server-reader-20261023.json) | 未含已核正文/补丁/工具结果；不等于真实代码编辑闭环。 |
| <a id="src-tulu3"></a>[TULU3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | 对话／训练语料 | W01、W09、W10 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；复核现有字段映射并补原生准备、适用执行/评分和固定包/消费验证；本类细项见W01、W09、W10。 |
| <a id="src-ultrachat"></a>[ULTRACHAT](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 发布方构造／合成语料 | W01、W02 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 发布方合成对话；取得原数据、检查完整消息和参考隔离，原任务回放/消费待验。 |
| <a id="src-videomme"></a>[VIDEOMME](https://huggingface.co/datasets/lmms-lab/Video-MME) | 原生任务／数据资产 | W35 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W35。 |
| <a id="src-videomme-v2"></a>[VIDEOMME_V2](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2) | 原生任务／数据资产 | W35 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W35。 |
| <a id="src-visionarena"></a>[VISIONARENA](https://huggingface.co/datasets/lmarena-ai/VisionArena-Chat) | 对话／训练语料 | W33 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；复核现有字段映射并补原生准备、适用执行/评分和固定包/消费验证；本类细项见W33。 |
| <a id="src-visualwebarena"></a>[VISUALWEBARENA](https://github.com/web-arena-x/visualwebarena) | 原生任务／数据资产 | W19 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W19。 |
| <a id="src-wavcaps"></a>[WAVCAPS](https://huggingface.co/datasets/cvssp/WavCaps) | 原生任务／数据资产 | W36 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W36。 |
| <a id="src-webarena"></a>[WEBARENA](https://github.com/web-arena-x/webarena) | 原生任务／数据资产 | W19 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W19。 |
| <a id="src-wildchat"></a>[WILDCHAT](https://huggingface.co/datasets/allenai/WildChat-1M) | 对话／训练语料 | W01、W02 | 已有字段映射可复用；原数据适配待复核 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；复核现有字段映射并补原生准备、适用执行/评分和固定包/消费验证；本类细项见W01、W02。 |
| <a id="src-workarena"></a>[WORKARENA](https://github.com/ServiceNow/WorkArena) | 原生任务／数据资产 | W19 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W19。 |
| <a id="src-xlam-tool"></a>[XLAM_TOOL](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 发布方构造／合成语料 | W24 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W24。 |
| <a id="src-xlsum"></a>[XLSUM](https://huggingface.co/datasets/csebuetnlp/xlsum) | 原生任务／数据资产 | W07 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W07。 |
| <a id="src-xsum"></a>[XSUM](https://github.com/EdinburghNLP/XSum) | 原生任务／数据资产 | W07 | 仅登记；需补接入 | [ ] 当前固定迁移版本未有取得/原生运行验证；不推断历史全无 | 取得并校验原发布文件/所选子集；补原生准备、适用执行/评分和固定包/消费验证；本类细项见W07。 |

### 需要特别纠正的旧映射

- QReCC是对话检索任务，不能仅因有历史Context就当“演化语料”。
- 当前TraceLab已验证的是调用元数据，不足以支持“原始代码编辑请求已经能回放”。
- AgentBench仅ALFWorld子集完成，不能把它的DB、KG、OS、Web类别一并勾选。
- BFCL静态AST匹配不证明工具执行；ToolACE合成历史对话不等于原生Agent环境。
- SchemaStore是schema资源；RULER是发布方构造的受控探针；训练语料、原生评测、真实请求各自标注。
- 旧QBI改造的GAIA/AgentBench/tau2是历史派生配置，不作为官方原始任务的默认入口或原版验证证据。
- LongMemEval-V2官方发布包含Web/企业Agent经历记忆任务；要取得完整任务/轨迹关系，不把HF自动转换的图片视图误认成完整数据。

## 六、扩展公开来源清单：17个既有候选＋4个新增必接

这21项保留在扩展列表，不混入原89项已验证分母；分类接入顺序可以调整，但必须来源不能被候选替换或删掉。所有尚无已取得载荷、实现或真实执行证据的项目继续保持未完成。

### 6.1 前轮17个候选

| 来源ID／发布方定位 | 对应类别/维度 | 可复用程度与验证 | 所缺资产/协议 |
| --- | --- | --- | --- |
| <a id="src-multiagentbench"></a>[MULTIAGENTBENCH](https://github.com/ulab-uiuc/MARBLE) | 跨任务关系／运行维度 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 原场景/环境、协作与竞争协议、任务评分；多Agent是跨任务关系，不额外凑第41类。 |
| <a id="src-deepplanning"></a>[DEEPPLANNING](https://qwenlm.github.io/Qwen-Agent/en/benchmarks/deepplanning/) | W22、W26 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 原旅行/购物任务、语言版本、数据库/工具、原全局约束评估；不能改为静态聊天。 |
| <a id="src-streamingbench"></a>[STREAMINGBENCH](https://streamingbench.github.io/) | W35 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 原视频、查询时间戳、允许可见前缀与评分；不看未来帧。 |
| <a id="src-ovobench"></a>[OVOBENCH](https://github.com/JoeLeelyf/OVO-Bench) | W35 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 原视频、时间边界、任务模式与评分；离线整视频不能替代在线协议。 |
| <a id="src-full-duplex-bench"></a>[FULL_DUPLEX_BENCH](https://full-duplex-bench.github.io/) | W37 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 固定具体版本、原音频/元数据、双工client/server、打断/重叠及原评分。 |
| <a id="src-libritts"></a>[LIBRITTS](https://openslr.org/60) | W38 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 官方split、文本/说话人/波形配对、语音生成入口及评估；保留归属和许可。 |
| <a id="src-ljspeech"></a>[LJSPEECH](https://keithito.com/LJ-Speech-Dataset/) | W38 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 1.1版元数据/文本/原波形配对；单说话人朗读不代表全部TTS或双工任务。 |
| <a id="src-beir-scifact"></a>[BEIR_SCIFACT](https://github.com/beir-cellar/beir) | W39、W40 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | BEIR固定版本corpus/query/qrels及检索/重排协议；与原始SciFact事实核验任务区分。 |
| <a id="src-beir-nfcorpus"></a>[BEIR_NFCORPUS](https://github.com/beir-cellar/beir) | W39、W40 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | corpus/query/qrels、候选集及检索/重排协议；用途许可单独核验。 |
| <a id="src-msmarco-passage"></a>[MSMARCO_PASSAGE](https://microsoft.github.io/msmarco/) | W39、W40 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 固定passage ranking发行版及原候选/相关性标签；发布页限非商业研究，不默认适用商业验收。 |
| <a id="src-stsb"></a>[STSB](https://www.sbert.net/docs/sentence_transformer/dataset_overview.html) | W39 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 原句对/连续相似度标签、明确数据版本与相似度评价；不是生产检索流量。 |
| <a id="src-docvqa"></a>[DOCVQA](https://site.docvqa.org/datasets/docvqa) | W34 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 原文档图像、问题/答案、官方下载条件、视觉模型及原评分。 |
| <a id="src-codejudge-eval"></a>[CODEJUDGE_EVAL](https://aclanthology.org/2025.coling-main.7/) | W12 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 发布方原始代码评判任务、标签、原评估；与其他同名CodeJudge框架区分。 |
| <a id="src-dragon-public"></a>[DRAGON_PUBLIC](https://github.com/RussianNLP/DRAGON) | W30 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 公开月度文本快照/问题；固定版本和合法数据来源；不能补造生产增删/在线事件。 |
| <a id="src-temporalwiki"></a>[TEMPORALWIKI](https://github.com/joeljang/temporalwiki) | W30 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 真实时间快照/差异与原协议；其原用途并非完整在线RAG请求轨迹。 |
| <a id="src-copilot-agent-2026"></a>[COPILOT_AGENT_2026](https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md) | 跨任务关系／运行维度 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 真实请求生命周期元数据/字段语义；无prompt/代码/工具结果时仅交元数据读取。 |
| <a id="src-azure-lmm-2025"></a>[AZURE_LMM_2025](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md) | 跨任务关系／运行维度 | [ ] 仅来源定位；需补获取、适配、验证及许可检查 | 原多模态请求元数据与时间/尺寸字段；不是图像/视频内容，不能原任务回放。 |

### 6.2 本次新增的4个必须来源

| 来源ID／原始定位 | 对应类别 | 可迁移／验证状态 | 还缺什么 |
| --- | --- | --- | --- |
| <a id="src-hle-verified"></a>[HLE_VERIFIED](https://huggingface.co/datasets/skylenage-ai/HLE-Verified)（HLE-Verified） | W05 | [ ] 新增必须接入；仅核发布定位，未取得/适配/运行 | 固定数据revision及验证/修订状态；保留图像与题型；原评测/评分器、可部署模型及原题抽样均待完成。 |
| <a id="src-swe-bench-pro"></a>[SWE_BENCH_PRO](https://github.com/scaleapi/SWE-bench_Pro-os)（SWE-bench Pro） | W15 | [ ] 新增必须接入；仅核发布定位，未取得/适配/运行 | 使用公开发行版，固定revision、仓库和测试容器；Agent闭环、功能测试、官方评分、资源预算和消费验证均待完成；不以SWE-bench Verified替代。 |
| <a id="src-frontierscience"></a>[FRONTIERSCIENCE](https://huggingface.co/datasets/openai/frontierscience)（FrontierScience） | W05 | [ ] 新增必须接入；仅核发布定位，未取得/适配/运行 | 分别固定Olympiad/Research任务与原评分协议；检查评分器依赖；不冒充实验/仿真执行；获取、适配及原题抽样待完成。 |
| <a id="src-terminal-bench-2-1"></a>[TERMINAL_BENCH_2_1](https://github.com/harbor-framework/terminal-bench-2-1)（Terminal-Bench 2.1） | W20 | [ ] 新增必须接入；仅核发布定位，未取得/适配/运行 | 固定2.1而非2.0/其他版本；Harbor/Agent、每题容器与原测试；核验ARM/NPU服务器兼容性；获取、端到端执行与消费待完成。 |

MMLU_PRO已在第五节，五项必接的总要求为“原MMLU_PRO＋本节四项”，不重复计为五个新来源。新增来源ID目前是方案标识，程序注册、适配和CI尚未实施。

## 七、跨任务执行协议与负载资产

这些项目同样属于完整覆盖清单，但不是额外互斥任务类别。

| 关系/协议/特征 | 对应来源 | 当前证据 | 待补内容 |
| --- | --- | --- | --- |
| 多Agent协作/竞争、分支汇合 | MULTIAGENTBENCH；依各原任务增加关系标签 | 仅定位 | 原协作协议与通信/环境；无现成真实轨迹不能虚构 |
| 在线视频可见性 | STREAMINGBENCH、OVOBENCH | 仅定位 | 原时间戳、历史前缀、输入推进与评分，不看未来 |
| 双工语音与打断 | FULL_DUPLEX_BENCH；SPOKENWOZ/DAILYTALK仅辅助录制语料 | 仅定位 | 版本化音频流协议、轮换/重叠/打断 |
| 长输入/长上下文 | LONGBENCH、LONGBENCH_V2、INFINITEBENCH、PAIO-LONG-PREFILL-5000、PAIO-LONGTEXT-4000 | LongBench子集/企业抽样已执行，v2预算终态保留 | 长度、任务、模型上限与完整输入分别记录；不把长输入叫文档理解 |
| 长输出 | LONGWRITER | 三条原训练请求回放完整性 | 未评原篇幅/质量；不强制凑输出，不当LongBench-Write |
| 共享前缀/会话追加 | PAIO-PREFIX-SHARED-5000、PAIO-REUSE-CONV-5000、原始多轮语料 | 企业各三条回放；不是复用验证 | 原token/分组/序列与会话关系；缓存效果属于另行系统实验 |
| 语义相似 | PAIO-SEMANTIC-SIMILAR-5000、PAWS、STSB | 企业三条回放；PAWS下载受阻；STS-B仅定位 | 原标签/任务协议；相似不等于KV可复用 |
| 到达、尺寸、突发和生命周期 | AZURE_LLM、MOONCAKE、BURSTGPT、TRACELAB；扩展COPILOT_AGENT_2026、AZURE_LMM_2025 | 前四项已有选定文件元数据检查/消费；后两项仅定位 | 保留时间/取消/重试含义；元数据不能补造prompt或模型输出 |
| 上游构造型受控长上下文探针 | RULER | 仅登记 | 固定上游生成器/种子/原指标；与自然数据分开，不继承真实流量属性 |

负载属性还包括输入输出比例、模态尺寸、工具等待、并行依赖、schema约束等。它们在可取得原字段范围内登记，不迁研究特征提取和观测系统。多机、多卡、PD分离、调度和缓存策略属于运行配置/实验因素，不是新数据集或新任务family。

## 八、逐能力迁移与研究侧接通打勾表

本节回答两个独立问题：**已有可复用能力是否迁全迁对；研究侧是否已经通过公共提供方接通。** 不用110项扩展来源的覆盖率替代这两个验收项。本轮停止扩数据集，PR不合并；40类仍是总体方案，新增4类没有实现或运行的部分继续保持未完成。

比对范围包括：研究消费分支原45条定义（27条主注册＋18条扩展）、研究主目录后续已有的数据工具、原89公开＋8企业来源，以及原manifest中38份固定数据文件身份。不是只比历史迁移起点，也不把尚未实现的候选算成遗失能力。源码路径在下表均相对于各自仓库根目录；源前缀`P`表示研究仓`workload-preparation/`，目标前缀`D`表示benchmark的同名组件目录。

三列分别记录代码、软件行为和研究消费：`[x]`只表示本行声明的范围；`—`为不适用；`[ ]`仍需完成。软件夹具不是公开原题或企业真实数据验证。原始模型执行记录保持第五节的范围，不因本次打勾升级。

### 8.1 已复用、等价替换和补漏

| 能力 | 研究侧已有入口/资产 → benchmark对应 | 处置 | 代码 | 行为核验 | 研究侧消费/剩余事项 |
| --- | --- | --- | --- | --- | --- |
| 原分类及稳定ID | `P/data/catalogs/workload_taxonomy_v1.json`、`v2.json` → `D/data/catalogs/`及`taxonomy.py` | 继承，不重造研究分类；v1 JSON一致，v2仅增NI摘要来源 | [x] | [x] 原定义对照；现有程序仍36类 | 已有消费；40类扩展方案尚未程序化，不勾新增4类 |
| 公开/企业来源目录 | `authoritative_datasets_v1.json`、`dataset_assets_v1.json` → 对应catalog、`provider.list_sources` | 保留89公开及8企业ID；不等于原数据已取得 | [x] | [x] 89源无漏ID；既有provider夹具核97项/8企业 | 既有固定包查询/消费有记录 |
| 原固定文件身份 | `source_manifests_v1.json` → 对应manifest、`assets.verify_file` | 研究主目录与消费分支各38份`role=data`文件，其来源ID/路径/SHA全部可在目标找到 | [x] | [x] 身份集合对照；非再次读取38份载荷 | 原真实文件消费按来源回执，不宣称38份全量任务均跑过 |
| 原45条适配定义 | `adapter_registry_v1.json`、`adapter_extensions_v1.json` → 对应注册与`adapters.py` | 45个ID均保留；42条普通数据适配复用，3条QBI派生绑定另列 | [x] | [x] 42条各1个软件夹具对比输入、outcome、runtime及不改原行 | 真实消费只覆盖回执声明的来源，不能据42个夹具称42源真实验证 |
| StructEval参考隔离 | 原`STRUCTEVAL`定义 → 目标同ID及`test_structeval_input_boundary.py` | 有意修正：VQA参考由注释移到outcome，防答案进入输入封装；没有丢弃参考 | [x] | [x] 专项答案隔离回归；普通输入/结果对照 | 原Text→JSON评分记录独立保留；其他子集未验 |
| WorkloadRecord原始封装 | `P/src/llm_serving_workloads/records.py` → `D/src/vllm_hust_workload_data/records.py` | 复用输入/outcome分离及原请求体接口，不复制研究特征 | [x] | [x] 源/目标普通适配对照、输入隔离测试 | 固定工件两模式消费已验证部分来源 |
| 原始文件读取与安全路径 | `assets.py` → 同名数据模块 | JSON/JSONL/gzip/CSV/Parquet、媒体字节、路径与SHA校验复用 | [x] | [x] 读取、越界、坏输入和完整性夹具 | 实际文件范围见数据及消费回执；不是全模态推理完成 |
| 通用显式字段映射 | `mapping_contract.py`的数据部分 → 同名数据模块、`input_views.py` | 只复用原字段到输入/参考的映射；研究features/归一化另留研究仓 | [x] | [x] 数据pipeline/mapping夹具 | 研究侧独立接归一化/特征；不迁计算逻辑 |
| 本地源绑定/冻结 | `bindings.py`、`freeze_source_manifests.py`中数据身份能力 → `bindings.bind_source` / `pipeline bind` | 等价替换；不再依赖研究`results/`回执注册表，也不自动重写研究catalog | [x] | [x] 显式revision/文件、SHA及绑定回查夹具 | 固定包消费路径支持显式manifest |
| 准备计划与索引 | `pipeline plan/run`的数据部分 → `D/.../pipeline.py` | 保留显式根目录、manifest、limit、失败/partial语义；不存研究特征 | [x] | [x] SQLite只存引用、失败不伪装完成、无隐式下载 | 研究pipeline消费provider；最终版本核验见8.3 |
| 原文回查与引用 | `consumer.resolve_input` → `D/.../consumer.py`，研究侧保留转接 | 数据提供方重建原输入/参考及原始关系 | [x] | [x] prepare/resolve及参考隔离夹具 | LoCoMo、代码、图表、文本、元数据等固定包回执已覆盖各自小样本 |
| 来源查询/下载计划 | `catalog.py` → 同名数据模块及`workload-data`入口 | 复用发布方/版本/文件查询；计划不自动下载 | [x] | [x] packaged catalog/provider测试 | 研究侧download-plan调用提供方 |
| 发布方元数据核验 | `P/analysis/verify_public_sources.py` → `D/source_tools/verify_public_sources.py` | 已迁，不是遗漏 | [x] | [x] 原三工具之一，AST一致 | — 操作工具；非研究模型任务 |
| 指定HF文件批量获取 | `materialize_public_datasets.py` → 同名`source_tools`脚本 | 已迁，保留显式选择、revision、gate与字节/SHA限制 | [x] | [x] AST与源一致；本轮不下载新数据 | — Git/sdist操作入口，不冒充wheel命令 |
| 单源检查/显式下载 | 研究主目录`download_dataset.py` → `D/source_tools/download_dataset.py` | **原遗漏，本轮补迁**；换namespace，下载必须显式回执；不改研究回执注册表 | [x] | [x] 默认只检查、不写缓存；缺回执在网络前拒绝的离线夹具 | — 取得后通过公共bind/pipeline消费；无本轮真实下载 |
| 固定release获取 | `acquire_release_datasets.py`＋`release_dataset_selection_v1.json` → 目标同名工具/catalog | **原遗漏，本轮补迁**；仅既有BURSTGPT/TraceLab选择，不新增源 | [x] | [x] 已有字节校验、CSV/gzip解析、拒绝错SHA覆盖夹具 | — 原有元数据消费证据保留；未重新下载release |
| 逐文件全量适配核验 | `validate_source_adapters.py` → 目标同名工具 | **原遗漏，本轮补迁**；保留省略limit时逐文件全读与读后SHA核对 | [x] | [x] 限定样本、参考不泄漏与汇总回执夹具 | — 验证工具，不代替原生执行/评分 |
| 嵌入媒体字节校验 | `validate_embedded_media.py` → 目标同名工具 | **原遗漏，本轮补迁**；去研究catalog构建器依赖，改用公共SHA工具 | [x] | [x] Parquet含字节/空字节的软件夹具 | — 仅存在性；不是图像解码、视觉质量或模型验证 |
| 企业资产校验 | `validate_paio_assets.py` → `D/source_tools/validate_paio_assets.py` | 已迁，聚合/脱敏的结构和数据完整性校验；不是观测/机制分析 | [x] | [x] AST一致、企业校验软件回归；原8源回执保留 | 原数据授权外部绑定；不随Git/wheel公开 |
| 原请求回放 | 企业原请求体/准备接口 → `execution.py`、根`run_source_request_samples.py` | 只回放已提供原请求；模型绑定差异明确记录，不改题 | [x] | [x] HTTP/SSE/终止软件测试及历史8源NPU回放 | 研究只消费原请求，不把HTTP返回当任务正确 |
| 原生任务引用 | 原`runtime_refs.py`中的可公开任务身份 → `provider`的tau2/AgentBench原生引用及研究`native_task_refs.py` | 按官方原始资产重新绑定，不照搬QBI私有根目录/派生题 | [x] | [x] 官方引用/资产身份软件测试；历史原生小样本另列 | AgentBench/tau2固定引用消费有记录；GAIA原版闭环仍缺，不假装完成 |
| 原生执行器/评分衔接 | 已有官方任务环境及本仓根脚本 → `configs/workload-native/`、对应`run_*_original*.py` | 保留已做通的原生入口/外部依赖；没有的保持缺口，不扩成新执行器 | [x] 已有部分 | [x] 已有入口软件测试；真实执行仅第五节子集 | 研究观测/结果分析不迁；GAIA及代码功能评分等原缺口仍未完成 |
| 软件检查/可安装性 | `validate_pipeline.py`数据部分、`verify_portable_wheel.py`数据检查 → 目标pipeline/输入核验脚本、CPU smoke和研究双包检查 | 等价替换数据检查；研究schema/特征/论文构建仍在研究仓 | [x] | [x] 干净快照229通过、1归档Git守卫跳过；该守卫在真实checkout另1通过；7工具help及sdist边界通过 | [x] 最终包独立安装＋研究侧两模式消费；仅软件验证，见8.3 |
| 纯数据来源汇总 | `public_source_inventory.py`的数据汇总 → 公共manifest/`source_spec`字段、`catalog`/provider查询及研究侧等价配方 | 等价替换，不误划为研究分析排除；保留逐文件nullable行数/单位，不跨单位合并成任务数 | [x] 原字段公开API可得 | [x] 研究侧对97条登记项的无原数据读取配方通过 | [x] 无独有缺失parser或正常调用仍依赖旧汇总器；旧19组历史报告不当40类 |
| 批量只读适配核验 | `validate_refactor_assets.py`的数据循环 → 公共pipeline＋`source_tools/validate_source_adapters.py` | 等价替换数据段，保留显式limit/错误；QBI历史段留研究仓 | [x] | [x] pipeline/逐源校验离线回归 | [x] 研究侧确认可复用；不是重新执行全部真实来源 |

上述新补工具属于Git源码/sdist入口；wheel只提供公共数据包，不能声称工具自动装成命令。完整用法见[原数据获取文档](../../workload-preparation/docs/dataset-downloads.md)。

### 8.2 有意排除，不属于“遗漏”

| 已有内容 | 处置与去向 | 边界核验 |
| --- | --- | --- |
| `normalized.py`、`family_features.py`、`input_measurements.py`、`family_features_v1.json` | 留研究仓；公共侧只保留原字段视图，不计算科研特征 | [x] 数据模块allowlist及依赖边界测试；研究消费测试另验其自身转换 |
| `full_cycle_trace.py`、`*observation*`、`*_runtime_hooks.py`、资源probe/采样、SDK/引擎patch工具 | 留研究仓，不迁观测、干预或采集 | [x] 目标核心无这些模块和依赖；本次没有启动采集/模型 |
| `trace_analysis.py`、`trace_validation.py`、研究结果/数据库/指标/ownership及分析构建器 | 留研究仓；不以公开数据迁移夹带科研分析 | [x] 不进入数据包；基础数据校验与科研分析明确区分 |
| QBI旧`runtime_refs/history`、私有根目录、FrozenEvidence/派生task | 保留具名历史，不作为官方原版默认入口；源ID仍保留但执行契约不同 | [x] 三个原绑定有意替换；缺原版能力如GAIA继续未完成，不拿历史通过冒充 |
| `freeze_source_manifests.py`对研究`results/`的扫描/回填，`build_refactor_handoff.py`的特征/研究聚合 | 不迁研究结果依赖；数据身份冻结由显式bind替代 | [x] 原路径对应已明确；没有把研究回执注册表装进公共包 |
| `retire_generators.py`及旧synthetic family生成器 | 保留停用记录，不迁为默认数据或fallback | [x] legacy处置catalog保留，生成器模块缺席回归通过 |
| 企业原始数据、gated载荷、私有答案、模型权重、第三方完整环境 | 外部授权取得；不进入Git/wheel/CI | [x] 包/仓库数据边界检查；需要原环境的任务明确外部依赖 |
| 本轮之前未提交的MATH500增量及其他无关改动 | 保留原工作区，本轮不提交、不拿来补完成率 | [x] 收口代码提交显式选路径；干净固定快照须排除这些改动 |

### 8.3 两端核对和收口状态

- [x] 已向研究线程核对双重目标，停止新增数据集、禁止合并PR；分类以本40类方案为目标，原36类是当前机器接口基线。
- [x] 已核原45条定义ID、89公开源ID、38份固定文件身份；42条普通适配的软件输入/outcome/runtime对照通过。三份原获取/校验脚本AST一致。此为软件对照，不是42份真实载荷验证。
- [x] 已补上述4个遗漏工具及既有release选择，数据层/边界等工作区CPU检查241通过；无新下载、无模型、无性能测试。
- [x] 研究侧已有固定版本数据包→研究消费链路；既有LoCoMo等真实文件回查/安装与源码两模式消费记录保留。
- [x] 新增工具后的最终固定干净快照构建、安装、数据包边界与研究侧独立消费通过。数据提交`1a9776865546d85edd667425661b4a5603696e35`；本地wheel SHA256为`c2b73006c3d954b78e4dde9bac1c0acca75758aeab264a3b779386d22a0122d0`。32个包payload与Git内容一致，但存在CRLF/LF差异，不声明逐字节一致或Linux CI工件已验证。
- [x] 研究侧本地提交`89959d2a1faa9be3da1b9979fcf1076f6a68ab62`独立验证：安装包/固定源码各71项消费夹具，无跳过；核心输入/参考及双索引回查通过；完整安装检查92项、补迁工具7项通过。回执在研究仓`docs/evidence/migration-closure-consumer-20261023.json`；原真实数据回执不改、不升级。
- [x] 研究侧禁用旧reader模块后，顶层入口、pipeline、回查及七批研究脚本正常链路仍通过。旧`adapters.py`、`bindings.py`未物理删除，保留显式兼容/测试调用及具名QBI历史，不是自动回退或第二个正常数据owner。
- [x] 两端已核定：45条定义无漏ID、42条普通数据适配的软件对照通过，4个原遗漏工具补齐；来源汇总/批量校验用公共接口等价替换。研究观测/特征/分析与QBI历史有意不迁。原生执行/评分未做通的部分仍是原有缺口，不由本轮软件通过升级。
- [ ] 原Draft PR更新发布、远端固定提交可取得及两端相应CI确认；这是发布收口，不与上述本地能力/消费收口混为一个勾。不新开PR、不合并。

核验脚本：[逐能力软件对照](../../scripts/audit_workload_existing_capabilities.py)；[固定快照审计回执](validation/migration-existing-capability-audit-20261023.json)。运行需要用户明确指定可信研究组件路径；不遍历私有数据、不获取载荷、不运行研究观测。打勾分为“原已有能力迁移与本地接通”及“远端发布/CI”，后者未完成前不宣称公共PR交付已完成。

## 九、接下来怎么迁与验证

### 9.1 先交已有能力，不等待110项补齐

- [x] 原已有能力→目标实现对应与处置见第八节；目录ID和软件行为分开，不对候选升级。
- [x] 旧获取/准备脚本已复用、补迁或明确等价替换；目标下载计划仍不冒充实际取得。
- [x] 已有结果按原任务ID/版本复用，本轮没有重跑GSM8K、八份企业数据或同一子集。
- [ ] 将未评分、预算终止、原生环境缺失与模型错误分开；核对旧台账矛盾后同步，不篡改原始回执。
- [x] 最终本地可安装包固定版本、研究侧独立安装及源码消费核验通过；远端PR/CI另记，消费测试不是新的模型/官方评分。
- [ ] 40类方案通过审阅后再同步机器分类/来源目录、适配与测试；不把本文编辑称为已完成接入。

### 9.2 分批补接入，不按单数据集开PR

| 建议批次 | 相关workload/来源 | 交付重点 |
| --- | --- | --- |
| 已有公共能力迁移收口 | 已实现读取/准备/原生引用、当前全部历史回执、8份企业外部绑定 | 完整性审计、依赖、使用链路、最终固定包与消费联调；不扩研究观测 |
| 推理与文本补全 | MMLU-Pro余子集、HLE-Verified、FrontierScience及未闭环文本来源 | 原始题/模态/参考、评分依赖；数类合批，已有来源不重复接入 |
| 仓库与终端原生任务 | SWE-bench Pro、Terminal-Bench 2.1及已有代码/仓库执行缺口 | 原仓库/容器、Agent闭环、隔离测试、评分；代码生成不当功能测试 |
| 工具、检索和规划 | GAIA官方原题、DB/KG/Web/API、DeepPlanning | 复用官方环境/公开框架，工具与附件完整，不为模型能力不足改题 |
| 专用模型/协议与其他扩展 | 图像/视频/音频、TTS、向量检索/重排、长期记忆、演化语料、多Agent | 按共用模型/环境拆成少量相关批次；资源不足写明，不假装都能双卡执行 |

批次是后续建议，不表示本次创建/重排了远端PR；沿用现有PR并先查他人进展，避免重复。旧“每来源/每类一个PR”的规则停止采用；历史合批记录见[原合批说明](PR_BATCH_CONSOLIDATION.zh-CN.md)，其中旧状态不作为本次实时PR结论。所有PR不自动合并。

### 9.3 统一验证完成口径

1. **数据层**：官方原文件/版本/摘要可追溯；输入、参考、媒体、会话树、任务/环境关系完整；prepare/index/resolve往返一致。只拿少数字段不是原任务已接入。
2. **任务层**：每个实际执行来源至少预选3条完整原任务；小且便宜的子集尽量全跑，大集按原ID抽样，说明抽样不具统计代表性。静态题执行原输入和适用评分；有状态任务执行原生闭环、终止与评分；代码须功能测试。元数据仅验证数据契约，模型任务不适用。
3. **质量层**：模型正确性、格式符合率、任务奖励与链路是否正常分开。答案错不能伪称环境坏，也不能随意换题直到答对；原预算终止如实保留。
4. **交付层**：可安装依赖、必要原生入口、使用说明、CPU正确性smoke、固定软件身份及研究消费。CI不做性能测试；真实NPU回执不能拿CI替代。
5. **迁移整体**：既有可复用能力迁全迁对＋研究侧已独立接通；来源级少量读取通过不能宣布整个迁移完成。后续新候选另行追踪，不阻塞首批已有成果交付。

## 十、此次文档审阅检查

- [x] 列齐40个任务family并区分执行协议/负载特征。
- [x] 原89公开来源逐项保留，17候选另列，新增4个必接来源另列；MMLU-Pro不重复计数。
- [x] 8份企业资产逐一列出，公开数据与私有载荷边界分开。
- [x] 明确五项必须来源的官方定位、分类、原生协议及未完成事项。
- [x] 区分已迁代码、字段映射、数据取得/读取、NPU尝试、原生执行/评分与研究消费；不把任何局部验证说成整类完成。
- [ ] 40类和新增来源尚未程序化接入；后续按审阅结果实施。
- [x] 原已有数据能力逐项审计、补漏和最终固定版本本地研究消费已完成；边界及证据见第八节。
- [ ] 原PR远端发布和相应CI仍待确认，不自动合并，不将本地通过写成远端验收完成。
