# Agent实验工作流

目标是及时发现失败、阻止带着失败扩大实验、保留可诊断证据，并交付可读报告；不能保证未知故障不发生。沿用离线批量实验契约、现有RIDE部署入口和四层报告脚本，不另写调度器。

## 顺序和结果

1. 配置：冻结run-id、完整参数、输入及源码SHA。检查整批time0到达、无客户端并发限制、输出预留上下文，以及等待上限早于Agent寿命。等待截止并不是公平性修复，仍需实际进展验证。
2. 身份：核实EngineCore实际加载源码、完整解析后的SchedulerConfig、各worker身份及真实图重放。只有文件在磁盘上不算通过。
3. 机制：同输入最小对照，分别保存逐步预算、对齐和缓存分配证据。计时阶段不挂临时trace；完成后核对观察器移除、引擎排空。
4. 首次回复覆盖：用目标批次的全部program同时提交短验证，两组逐program记录发送、首次返回、异常。截止时间在实验前登记。任一program没有返回或超过截止时间，即阻止正式长窗口。只测几个请求不能替代整批覆盖；此阶段通过也不证明后续无饥饿。
5. 正式对照：同输入、同预算，保留所有完成、失败、取消和未完成记录。编排SUCCEEDED不等于Agent提交成功。明确窗口吞吐与完整批次耗时的区别。
6. 归档：原包及每个成员独立SHA核验。已有作业或提交意图不可重复提交。
7. 报告：Controller、Engine（含设备）、Program、Request四层图文；包含未完成请求、零回复program及缺失项。图需目视检查，数字从原件计算，按现有plot_run风格生成。
8. 交付：原文档图文读回，图片数量逐项匹配；公开备份和其他登记分别记录回执。未查询CI明确写NOT_CHECKED。

## 可执行入口

`experiments/workload/agent_experiment_workflow.py --plan PLAN --through liveness`

默认只执行到首次回复覆盖。完整执行需显式指定`--through delivery`。计划JSON包含`run_id`、新的`run_dir`、`protocol`、`source_root`、`source_files`、`stages`及`required_destinations`。每个stage声明现有命令的参数数组、有限的`timeout_seconds`及回执相对路径；不用shell拼接。完整字段由入口的`validate_protocol`和`validate_receipt`定义，失败回归见`tests/test_agent_experiment_workflow.py`。

回执必须携带本run-id、协议SHA及非空`evidence_files`文件SHA。环境变量`EXPERIMENT_RUN_ID`和`EXPERIMENT_PROTOCOL_SHA256`传给现有命令适配器。状态位于run目录下`workflow/state.json`，命令原始输出位于各阶段log。不可手填PASS代替真实适配器采集。

## 故障、恢复和边界

执行前独占锁并持久化提交意图；恢复时重验已通过的回执，跳过已执行命令。命令超时、中断或失败后状态为BLOCKED，并保存本地工作流元数据故障包。UNKNOWN表示远端结果未知，绝不自动再次提交；先只读核对已有作业和原件。

当前入口未提供远端作业自动协调/取消，也未将全部历史部署脚本迁入统一计划。故障包仅含本地工作流元数据，不冒充完整远端归档。正式作业失败后的原始日志归档和负结果报告，仍使用已有唯一提交意图的归档入口独立完成，不能通过重新运行pair来恢复。这些边界未补齐前，不能声称已经实现无人值守的完整实验平台。

## 本次问题的回归

2026-10-21的128program对照中，Controller组17个program没有一次正常模型回复，且Controller等待默认1800秒晚于Agent1200秒寿命。现有回归分别验证：等待上限冲突会在配置检查失败；即使其他program很快，任何一个零回复也会阻止正式阶段。此前结果保留为失败证据，不修改历史协议来让其通过。

单元测试验证检查逻辑，不是实机性能或公平性证明；下一轮仍需真实首次回复覆盖验收。

## 修改后的软件验收

统一入口为`python tools/check.py --output-dir results/checks/<唯一目录>`，依赖通过`.[test,upstream]`显式安装；不得依赖某个开发环境碰巧已有的库。CI与本地运行同一入口，覆盖Python3.11/3.12，失败时保存各阶段log、JUnit XML和result.json。

- `test_agent_experiment_workflow.py`：实际CLI和子进程贯穿8阶段，验证失败测量仍能归档/报告/交付、恢复不重复执行、损坏产物被拒绝、超时保留UNKNOWN和提交意图。外部服务回执使用明确标记的fixture，不是实机证据。
- `test_wait_aging_http.py`：实际FastAPI端点和固定上游router通过HTTP传输适配器完成入队、等待优先提升、恢复、转发、释放；异常截止返回504且不转发；取消时清理自身队列并保留无关program。后端为fake engine，不替代NPU验证。
- `test_first_reply_qualification.py`：真实loopback HTTP测试服务器验证两组首次回复收集、失败停止、释放和输入/worker漂移；不调用实际模型。
- `test_agent_report_pipeline.py`：从小型原件fixture生成指标、设备分析、零回复审计、9张图、内嵌HTML及文件SHA；覆盖全零回复、负向输出变化和缺失设备数据，不能用成功生成一张图代替整个链路。
- `test_standard_checks.py`：验证标准入口首错即停、保留日志、不覆盖已有结果，以及安装后命令绑定当前Python环境。

所有代码改动增加对应回归并通过标准入口；单元、集成、安装/构建和真实NPU验收分开报告。可选且未修改的Ray、transformers、ECPA测试若未安装依赖而跳过，必须在日志中显示原因；不将它们算作通过。CI因账户、额度或runner未启动时同样不能报告通过。
