# 原始服务轨迹：Azure 2023、Mooncake FAST25 与 BurstGPT

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

## 安装与准备

按[安装说明](INSTALLATION.zh-CN.md)取得指定成功 CI 的数据组件 wheel，核对提交和 SHA256 后安装。CSV/JSONL 读取不需要
pyarrow、模型库或科研仓库；本例仅使用标准库。 原始文件由使用者按来源目录的固定 URL 取得，并直接放在同一个私有目录中：

```text
/private/original-traces/
  AzureLLMInferenceTrace_code.csv
  AzureLLMInferenceTrace_conv.csv
  conversation_trace.jsonl
  toolagent_trace.jsonl
```

将以下配置保存为 `trace-inputs.json`；输出目录必须与原始数据目录分离且尚不存在：

```json
{
  "sources": [
    {"source_id": "AZURE_LLM", "data_root": "/private/original-traces"},
    {"source_id": "MOONCAKE", "data_root": "/private/original-traces"}
  ]
}
```

```bash
python -m vllm_hust_workload_data.pipeline plan --config trace-inputs.json
python -m vllm_hust_workload_data.pipeline run --config trace-inputs.json --output /private/trace-index-new
```

原始元数据保存在原文件中，索引记录来源版本、文件摘要和行引用，不把输入正文复制到研究结果。 消费端可使用现有接口回查，并检查明确的 `request_metadata` 类型：

```python
from pathlib import Path
from vllm_hust_workload_data.pipeline import prepared_rows
from vllm_hust_workload_data.consumer import resolve_input

database = Path("/private/trace-index-new/workload.sqlite")
reference = next(prepared_rows(database))
record = resolve_input(database, reference["record_id"], data_root=Path("/private/original-traces"))
assert record.runtime["kind"] == "request_metadata"
assert record.task["input"] == {}
# 原字段位于 record.replay["annotations"]；不生成或请求模型正文。
```

取得不同版本或其他原始文件时，须使用既有 `pipeline bind` 显式固定新版本和摘要， 不能替换同名文件后沿用本批验证结论。本批没有额外的调度器、性能回放器或分析功能。

## 当前交付状态

候选 wheel `08304cddf20fe00788a0835928d9709ce91e70138b36eab3520cd8555f19aba6` 已在服务器独立安装，并实际执行上述无额外
manifest 的 `pipeline plan`、`pipeline run` 命令；63824 条索引和每文件前三条回查通过，默认固定目录读取计数相符。 这是本地候选工件，不替代本批最终 CI
工件的回归与消费端验收。

BurstGPT 补充范围：官方 v2.0 Release 的 `BurstGPT_1.csv` 已取得，52283111 bytes，
摘要与已有固定目录相符，绑定时计数1429737条。原始行0、1、2的读取、prepare、resolve 在服务器通过。未验证所有记录的字段语义，也未取得/验证本批中的其他 BurstGPT 文件。
服务器直连两次90秒超时的失败保留，随后从本地官方 Release 下载并校验传输至服务器， 未换用重造数据。它同样没有原始 prompt，不执行模型或任务评分。

- [x] 官方固定版本四文件已私有取得。
- [x] 使用现有 CI wheel 在服务器全量读取、逐字段比对及建索引。
- [x] 每文件前三条 prepare/resolve 往返，保持原始记录相等。
- [x] 确认记录类型为 request_metadata，拒绝取得可回放模型请求体。
- [x] 本批固定来源目录、安装说明、CPU 回查夹具和无正文回执已在本地提交。
- [ ] 本批最终 CI 工件与指定研究消费端联调。
- [x] BurstGPT_1 原始字节核验及当前服务器三条安装读取/prepare/resolve；其他文件及全量未完成。

尚未为本批创建新 PR，准备将数种相关场景合批交付。原始文件不进入公开提交。 Azure 发布说明为 CC-BY，使用者应保留其引用并核实具体条款；Mooncake 的仓库代码
许可不被自动解释为匿名轨迹的无限制再分发许可。
