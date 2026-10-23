# 首批接入验证状态

更新：2026-10-23。逐来源进度见[可打勾清单](DATASET_CHECKLIST.zh-CN.md)。

## 已验证

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

- AgentBench、GAIA、tau2官方执行器仍待逐环境接入。既有tau2环境仅留可编辑安装元数据，其源码路径已缺失，不能称可运行的官方环境。QBI753/165/solo114仅历史派生登记，不提供默认回退，本次未重跑。
- 研究端配套[PR #37](https://github.com/intellistream/agent-workload-suite/pull/37)当前head为`b3e1f9f`，四项CI均通过，仍为Draft未合并；已锁定上游数据包提交`04113e35`/wheel
  SHA256
  `a1fb1efd188cee7ace85a3f911894509bcfd401c41cf4ecd7ca6ef36eb83c26c`，完成固定安装包消费及旧来源回归。Linux安装消费92项、组件948项、harness及七批127项通过，无跳过。此为软件消费链验证，model_calls/native_task_runs均为0，不冒充NPU新任务。
- 初始16K/思考配置的企业v1预算终止及上下文拒绝继续保留在[既往尝试](validation/enterprise-v1-attempts-20261022.json)。新128K/非思考配置单列，不作为因果性能对比。
- 整库Windows旧回归1836 passed、258 failed、60 errors、11
  skipped，包含fcntl/脚本平台/编码等，未逐项归因，不称Windows整库全绿。新增源码指纹检查采用规范LF，避免把Windows CRLF检出当成代码漂移。
- CPU smoke不取企业/gated数据、不加载模型、不做性能测试。目录候选不等于已取得/已适配/已完成服务器任务；真实任务、原始请求、静态语料和历史轨迹单位不同，不汇总成同一种任务数量。
- 2026-10-21 21:13
  UTC起，授权SSH入口返回连接拒绝，TCP复核相同；不能仅凭此确认回收原因。已完成NPU结果的脱敏回执、输入/模型校验及代码指纹已取回本地；原始响应只在受控服务器data卷，当前可恢复性未重新核实。恢复入口后先检查已有原始归档，不因观察失败擅自重启已完成任务，不用模拟或CPU推理补造证据。

## 使用

数据与依赖另行取得，运行输出使用新建私有目录，不提交原题、答案、模型响应。基础安装见[固定提交wheel入口](INSTALLATION.zh-CN.md)，任务/数据用法见各来源说明；来源不具备原生执行器或评分时明确列为未验证，不能用普通聊天响应冒充。
