# 数据消费契约（0.1.0）

发行包 `vllm-hust-workload-data`，命名空间 `vllm_hust_workload_data`，不替换科研包。使用
`provider.provider_identity()` 核对版本、安装路径和来源身份。

`pipeline.plan()` 只判断绑定是否可尝试；`pipeline.run()` 生成 `workload-source-index/v1` 输入索引，不运行模型。数据库只含
sources、records 两张表；不含观测、事件、关系、特征或答案内容。

`pipeline.prepared_rows(database, source_id=...)` 产出 record_id、source_id、source_unit、runtime_kind。用
`consumer.resolve_input(database, record_id, data_root=...)` 回查原始文件；读取前后核验
SHA，文件改变即失败。不同来源的行、会话和任务单位不能直接相加。

`record.input_record()` 提供输入侧投影；参考答案通过独立的 `record.outcome_record()` 获取，只交评分器，不交模型。请求体仅在原始
request_replay 记录上使用 `request_body()`，不得把静态任务伪装成原始 API 请求。

来源到分类的候选关系不是逐记录分类结论。按原生子类型及证据类型做显式选择；不根据模型生成答案反推类别。外部新来源需审阅 mapping、bind 文件身份后再执行准备，不能静默构造缺失字段。

官方 AgentBench、GAIA、tau2 需要各自的原版执行器。当前目录显式登记缺口；不导入 QBI 753/165/114 派生运行库、历史或研究划分。GAIA
保持原题及原附件，不用本地整理材料改题。

执行器、模型版本、模板、工具、预算、终止和评分由调用方显式固定。请求回放成功、读取成功、模型答对和完整任务成功分开记录。软件测试、历史证据与当前服务器抽样不能相互替代。

科研消费者继续拥有观测、特征、研究分析及机制代码，不得为方便消费把这些模块搬回数据底座。完整命令见[组件说明](../README.md)。
