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

### Mooncake 固定版官方回放工具复核

固定提交`531790322667942f0368933d0d5960f6e4a4c87b`还包含官方[KVCache Storage Benchmark v1](https://github.com/kvcache-ai/Mooncake/tree/531790322667942f0368933d0d5960f6e4a4c87b/benchmarks/storage_benchmark_v1)。它读取Conversation或Tool/Agent
trace的`hash_ids`，按发布方布局在磁盘页上模拟读写并报告存储I/O指标；不会构造prompt、调用模型或向vLLM-HUST服务发请求。`--model glm5/kimi-k2.6`是KV布局预设，非模型推理。

因此它可作为该匿名服务轨迹的官方存储回放入口；结果只说明所选存储配置下的KV页I/O行为，不能作为推理引擎的原生模型任务、在线请求回放或引擎吞吐结果。工具默认`--max-pages 2000`，发布代码在trace所需页数超过该值时会报告容量不足/页映射边界；运行前必须记录实际trace最大页ID与页面容量。默认`--file-mode single`会创建并写入`data.bin`，只能指定新的、独立的私有`storage-dir`。本轮发现该工具但尚未运行；待研究线程释放资源后再安排专用磁盘I/O窗口。

2026-10-24只读容量预检：原conversation与toolagent文件分别有12031/23608条，最大`hash_id`为182789/183299。固定`glm5`
MLA布局按512-token页计算，每个逻辑页56,229,888 bytes；若保持全局page ID一一映射，需要183300页、约9.37 TiB，超过该机当时约1.42
TB的`/data`可用空间。官方默认2000页配置最多约104.74
GiB，但会把超出容量的ID按槽位映射，不能把所得命中率解释为无冲突的真实前缀复用率。此次未启动写入；后续若采用容量受限官方profile，只报告明确配置下的存储I/O与映射边界。

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

### 2026-10-24：Azure 数据资产复核

本轮 Azure 2023 两文件在独立 Python 3.11.15 环境全量读取、建索引共 28185 条，每文件预选原始行 0、1、2 共六条回查原字段一致。所有样本均为
`request_metadata`，拒绝生成模型请求体。没有调用模型、评分器或制造 prompt。这是**服务轨迹数据资产可用**，不是原生推理 benchmark
跑通；其他来源不沿用本次完成结论。

固定发布仓库的完整目录包含发布说明、分析 notebook 及两份 CSV，本次未识别出该版本专用的原始请求回放入口；分析 notebook
不作为推理执行器。固定文件摘要及实际验证范围见[脱敏回执](validation/azure-serving-trace-delivery-20261024.json)。

复现时在仓库外建立私有目录并下载固定原件：

```bash
mkdir -p /private/original-traces
curl --fail --location https://raw.githubusercontent.com/Azure/AzurePublicDataset/215becdacba1ce682c7368642ce97d5a332de7a6/data/AzureLLMInferenceTrace_code.csv --output /private/original-traces/AzureLLMInferenceTrace_code.csv
curl --fail --location https://raw.githubusercontent.com/Azure/AzurePublicDataset/215becdacba1ce682c7368642ce97d5a332de7a6/data/AzureLLMInferenceTrace_conv.csv --output /private/original-traces/AzureLLMInferenceTrace_conv.csv
sha256sum /private/original-traces/AzureLLMInferenceTrace_code.csv /private/original-traces/AzureLLMInferenceTrace_conv.csv
python3.11 -m venv /private/trace-reader-env
/private/trace-reader-env/bin/python -I -m pip install --no-index --no-deps /private/wheels/vllm_hust_workload_data-0.1.0-py3-none-any.whl
```

下载前确认目标文件尚不存在，避免覆盖已有原件。安装使用固定数据组件提交 `1a9776865546d85edd667425661b4a5603696e35` 对应 wheel，SHA256 为
`c2b73006c3d954b78e4dde9bac1c0acca75758aeab264a3b779386d22a0122d0`；仅版本号相同不足以确认软件身份。CSV
读取不需要模型或媒体依赖。通过安装说明取得该工件并核对摘要；自行重建工件应另记摘要，不冒充此工件。

将上述配置中的 `sources` 仅保留 `AZURE_LLM`，使用独立输出目录，执行：

```bash
/private/trace-reader-env/bin/python -I -m vllm_hust_workload_data.pipeline plan --config trace-inputs.json
/private/trace-reader-env/bin/python -I -m vllm_hust_workload_data.pipeline run --config trace-inputs.json --output /private/azure-index-new
```

预期状态为 `PREPARED_REFERENCES_ONLY`，索引条数 28185；这是正确的数据资产终态，不应解释为模型任务已评分。沿用前述 `resolve_input`
接口逐文件回查原始前三行，并与 CSV 原字段逐项比较。使用 `python -I` 避免已有 CANN 等环境的 `PYTHONPATH`
污染客户端；不修改其他服务的环境。原文件和完整索引均留在私有目录。

### 历史批次记录（不替代本轮交付核验）

候选 wheel `08304cddf20fe00788a0835928d9709ce91e70138b36eab3520cd8555f19aba6` 已在服务器独立安装，并实际执行上述无额外
manifest 的 `pipeline plan`、`pipeline run` 命令；63824 条索引和每文件前三条回查通过，默认固定目录读取计数相符。 这是本地候选工件，不替代本批最终 CI
工件的回归与消费端验收。

### BurstGPT本轮补充（2026-10-24）

官方v2.0的三个保留失败记录的完整发布文件均已在服务器核验字节数、发布SHA和全CSV解码：`BurstGPT_1.csv` 1429737行、`BurstGPT_2.csv` 3858436行、`BurstGPT_3.csv` 5344021行，合计10632194行；不代表三文件互不重叠的独立请求总数。首文件复用，不重下载；其余两文件通过官方GitHub Release API资产341027213、341026736的`application/octet-stream`下载，所得字节与发布摘要一致。之后复用`acquire_release_datasets.py --source BURSTGPT --data-root EXTERNAL_DATA_ROOT --output EXTERNAL_REPORT`重新核验全部三文件。见[资产回执](validation/burstgpt-original-release-assets-20261024.json)，其中`download_url`表示规范发布地址，不表示本次实际传输绕过API。

此证据仅证明原资产完整取得、全文件可解码，不替代全部文件适配/prepare/resolve或模型回放。发布方存在`example/`示例，用外部ShareGPT提示池按轨迹到达/长度构造模拟压力，不能称为原始请求重放；该官方示例的依赖与运行范围尚待复核，不宣称无官方工具或已跑过示例。原CSV无prompt，不能自行补prompt伪装原请求。

随后三文件完整绑定均匹配原发布SHA与行数，各取原行0、1、2，共9条完成安装包读取→prepare→resolve往返，输入与历史结果分离且逐项一致；9条均明确为`request_metadata`，请求体接口按预期拒绝，不产生模型请求。见[原元数据往返回执](validation/burstgpt-original-roundtrips-20261024.json)。首文件跨数据根目录的软链接被安全检查拒绝；使用同一NFS内原文件的硬链接形成合法外部数据根目录，不放宽越界校验，不复制正文进仓库。官方示例适用性和最终软件交付仍待收口，不勾整个来源。

### 旧首文件尝试（保留历史范围）

BurstGPT 旧范围：官方 v2.0 Release 的 `BurstGPT_1.csv` 已取得，52283111 bytes，
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
