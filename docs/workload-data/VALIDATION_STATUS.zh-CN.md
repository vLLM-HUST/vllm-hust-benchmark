# 首批接入验证状态

更新：2026-10-24。逐来源进度见[可打勾清单](DATASET_CHECKLIST.zh-CN.md)。

当前统一计划及整体打勾状态只维护在[40类完整迁移清单](WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md)。本文是固定证据的导读，不另设一套迁移完成表。“原能力迁移/消费接通”与“各来源原生任务验证”是两个层次；下列历史软件数量不累计为当前测试数。

## 已验证

- 固定提交`e2de9812`的CPU
  smoke通过，其CI软件归档ID`12550944059`在服务器校验发布摘要与wheel摘要后独立安装，BBH全27变体各前三条共81条及BurstGPT三文件各前三条共9条原记录回查通过。见[固定工件回执](validation/ci-installed-reader-e2de9812-20261024.json)。这不是研究侧的新联调或新增模型执行；该提交主CI仍暴露格式/公开摘要误报及导航索引过期问题，修复中的后续提交不能冒用本次工件身份。
- 2026-10-24在服务器当前候选源码快照复现CPU
  smoke的九组pytest检查：418通过、1跳过（无Git元数据的归档快照不执行Git跟踪检查）；实际本地checkout另行`git ls-files`确认组件无跟踪的JSONL/CSV/Parquet。不是GitHub
  CI结果，不新增模型任务。初次快照缺配置文件及两份既有runner字节不一致的问题已补齐重跑，无需改历史回执或测试。见[软件回归回执](validation/server-software-regression-20261024.json)。
- 当前软件专项130项通过：原文回查、输入/答案隔离、失败事务、模型显式绑定、流式终止、响应/原schema校验、清单与脚本指纹和数据-only边界。夹具不是模型执行证据。
- 安装包对24来源各3条完成本地读取、准备及原文回查，共72条；该步骤没有模型调用或评分。
- GSM8K历史执行回执按实际`doc_id`去重覆盖原test:0–200，共201条；另完成尾部test:1316–1318三条，固定revision累计204/1319个唯一题号。尾部三题strict/flexible均正确；跨配置不汇总准确率，也不解读为全量或性能结论。lm-eval尾部补测的effective
  count异常另列。
- ARC-Challenge与ARC-Easy均按原版多选似然协议抽测test:0–2，各3题、每题4个候选请求；Challenge 2/3、Easy
  1/3，分别保留，不合并成全量正确率。两子集test共6/3548，未跑全量。见[复现说明](ARC_CHALLENGE.zh-CN.md)。
- 全部8份企业文件SHA核验，每源前三条共24条在双卡NPU完成回放及响应完整性/原声明格式检查。服务端prompt
  token数与预检24/24一致；未截断输入或改变显式预算。没有独立语义gold，不称任务准确率，见[企业说明](ENTERPRISE_REQUESTS.zh-CN.md)。
- 本地现有Qwen3.5-27B的11份完整权重、9份配置/tokenizer文件与发布方固定提交匹配，见[工件记录](validation/qwen35-27b-deployable-artifact-20261022.json)。
- 底座912eaee6（#379）和ARC增量ef0afae6（#381）的CPU smoke、lint、整库tests全部通过；企业增量CI单独核验。底座Linux整库2161 passed、4
  skipped。

## 尚未完成及证据边界

- AgentBench已有ALFWorld FC
  `new_std`原索引0–2的原生交互和官方评分[交接回执](validation/agentbench-alfworld-public-handoff-npu-20261023.json)，3/109、得分1/1/1；不覆盖DB/KG/OS/Web等环境，也不称ARM构建与官方Docker等价。tau2已有`telecom_full`前三条双角色原生交互和评分[交接回执](validation/tau2-reference-handoff-npu-20261022.json)，reward为1/1/0；其他domain/mode未完成。旧环境探测失败保留为历史，不能据此否定后来具名环境的成功。GAIA官方原版闭环仍未完成；QBI753/165/solo114不提供默认回退。
- 研究端配套[PR #37](https://github.com/intellistream/llm-workload-suite/pull/37)的已验证提交为`6ca466e9a278ae5b3dfa33ce380cd9a23ee7c772`，锁定数据源码`1a9776865546d85edd667425661b4a5603696e35`；[消费检查](https://github.com/intellistream/llm-workload-suite/actions/runs/39958054179)及[test/package/paper](https://github.com/intellistream/llm-workload-suite/actions/runs/39958054031)均成功，Linux安装消费92、组件955、harness及七批127项通过、无跳过。它是软件消费证据，零新模型/原生任务。原`b3e1f9f`/`04113e35`身份继续按[历史消费记录](DATA_CONSUMPTION_VALIDATION.zh-CN.md)保留，不改写为新版本验证。
- 未完成状态分开读：HumanEval/BigCodeBench只有生成、缺隔离功能评分；DS-1000另有一条原预算终止及原环境缺口；LongBench
  v2三题有评分但含一条`length`，来源级仍未勾；BBH复用Boolean三题评分0/3，2026-10-24新增26变体78条NPU原生生成、固定harness评分及原响应离线复核，覆盖全部27变体81个唯一原题，非全量评测；最终固定工件/文档回归待收口，整个来源仍未勾。低分不是执行故障，见[新增回执](validation/bbh-all-task-variants-original-npu-20261024.json)。JSONSchemaBench三条原schema生成与原validator运行完成，仅一条符合schema，不能写成三条格式均正确。OASST1只有历史续接，没有在线交互或独立语义评分；服务轨迹按资产交付，不虚构prompt或模型任务。这些状态与逐源台账及相应回执一致，未改原始回执。
- 初始16K/思考配置的企业v1预算终止及上下文拒绝继续保留在[既往尝试](validation/enterprise-v1-attempts-20261022.json)。新128K/非思考配置单列，不作为因果性能对比。
- 整库Windows旧回归1836 passed、258 failed、60 errors、11
  skipped，包含fcntl/脚本平台/编码等，未逐项归因，不称Windows整库全绿。新增源码指纹检查采用规范LF，避免把Windows CRLF检出当成代码漂移。
- CPU smoke不取企业/gated数据、不加载模型、不做性能测试。目录候选不等于已取得/已适配/已完成服务器任务；真实任务、原始请求、静态语料和历史轨迹单位不同，不汇总成同一种任务数量。
- 历史连接事件：2026-10-21 21:13
  UTC起，授权SSH入口返回连接拒绝，TCP复核相同；不能仅凭此确认回收原因。已完成NPU结果的脱敏回执、输入/模型校验及代码指纹已取回本地；原始响应只在受控服务器data卷，当前可恢复性未重新核实。恢复入口后先检查已有原始归档，不因观察失败擅自重启已完成任务，不用模拟或CPU推理补造证据。

## 使用

数据与依赖另行取得，运行输出使用新建私有目录，不提交原题、答案、模型响应。基础安装见[固定提交wheel入口](INSTALLATION.zh-CN.md)，任务/数据用法见各来源说明；来源不具备原生执行器或评分时明确列为未验证，不能用普通聊天响应冒充。
