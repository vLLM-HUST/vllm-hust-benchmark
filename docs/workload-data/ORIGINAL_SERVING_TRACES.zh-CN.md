# 原始服务轨迹：Azure 2023 与 Mooncake FAST25

## 范围与用途

本批提供原始到达时间、请求尺寸和发布方匿名前缀块标识的读取及准备能力。 Azure 的代码/会话轨迹和 Mooncake 的会话/工具代理轨迹均不包含原始 prompt。
因此不得伪造正文后称为原始任务回放，也不具有语义正确率或官方任务评分。 研究侧可以消费这些元数据，但分析器、缓存模拟器、观测和机制干预不随本批迁移。

## 原始来源与已验证范围

| 来源                         | 原始发布路径                                   | 记录数 | 实际核验                               |
| ---------------------------- | ---------------------------------------------- | -----: | -------------------------------------- |
| Azure 2023 Code              | data/AzureLLMInferenceTrace_code.csv           |   8819 | 全量原字段相等、全量建索引、前三条回查 |
| Azure 2023 Conversation      | data/AzureLLMInferenceTrace_conv.csv           |  19366 | 全量原字段相等、全量建索引、前三条回查 |
| Mooncake FAST25 Conversation | FAST25-release/traces/conversation_trace.jsonl |  12031 | 全量原字段相等、全量建索引、前三条回查 |
| Mooncake FAST25 Tool/Agent   | FAST25-release/traces/toolagent_trace.jsonl    |  23608 | 全量原字段相等、全量建索引、前三条回查 |

Azure 固定版本：`Azure/AzurePublicDataset@215becdacba1ce682c7368642ce97d5a332de7a6`。 Mooncake
固定版本：`kvcache-ai/Mooncake@531790322667942f0368933d0d5960f6e4a4c87b`。 四文件合计 63824
条，全量读取及建索引验证已在服务器安装的固定软件工件执行。 每文件的回查样本预选原始行 0、1、2，合计十二条，不替代全部记录逐条 resolve。 该验证没有调用模型、评分器或模拟器。

## 字段语义与不可推导结论

Azure 保留原始 TIMESTAMP、ContextTokens、GeneratedTokens，CSV 字段保持字符串； TIMESTAMP
不因文件名或采集日期被擅自补时区。发布方说明该数据用于 Splitwise 论文中的请求尺寸和到达特征，原始正文因隐私不提供。

Mooncake 保留 timestamp、input_length、output_length、hash_ids。按固定发布方说明， timestamp 为相对到达毫秒，hash_ids
为重映射的前缀块标识、块大小 512 tokens。 这些标识不代表本机已产生 KV、缓存命中或实测加速；不得跨独立文件臆造相同会话或
全局块身份。本批不重排记录、不生成新的到达时间，不从输出尺寸伪造输出文本。

Mooncake synthetic_trace 是发布方从公开数据构造并生成 Poisson 到达的独立轨迹， 本批未取得/验证；历史 arxiv-trace 也未混入 FAST25 两份原始轨迹。
Azure 2024/2025 不属于此次 Azure 2023 范围。

## 当前交付状态

- [x] 官方固定版本四文件已私有取得。
- [x] 使用现有 CI wheel 在服务器全量读取、逐字段比对及建索引。
- [x] 每文件前三条 prepare/resolve 往返，保持原始记录相等。
- [x] 确认记录类型为 request_metadata，拒绝取得可回放模型请求体。
- [ ] 本批固定来源目录、安装文档和回执正式纳入版本。
- [ ] 本批最终 CI 工件与指定研究消费端联调。
- [ ] BurstGPT 当前服务器取得/安装验证。

尚未为本批创建新 PR，准备将数种相关场景合批交付。原始文件不进入公开提交。 Azure 发布说明为 CC-BY，使用者应保留其引用并核实具体条款；Mooncake 的仓库代码
许可不被自动解释为匿名轨迹的无限制再分发许可。
