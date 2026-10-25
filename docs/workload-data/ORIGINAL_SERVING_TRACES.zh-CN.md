# 原始服务轨迹：Azure 2023、Mooncake FAST25、BurstGPT 与 TraceLab

## 范围与用途

本批提供原始到达时间、请求尺寸和发布方匿名前缀块标识的读取及准备能力。 Azure 的代码/会话轨迹和 Mooncake 的会话/工具代理轨迹均不包含原始 prompt。
因此不得伪造正文后称为原始任务回放，也不具有语义正确率或官方任务评分。 研究侧可以消费这些元数据，但分析器、缓存模拟器、观测和机制干预不随本批迁移。

## TraceLab：原调用元数据与官方合成会话回放

固定发布`uw-syfi/TraceLab@v0.0.2`的`syfi_coding_trace.jsonl.gz`共100939722字节，SHA256为`11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65`。现已在服务器逐条对比665453条原记录的全部字段与顺序，包括session/round标识、可选turn标识、tokens、timing_events和tools。它们原样位于`record.replay["annotations"]`；task.input为空、outcome为None，request_body拒绝伪造请求。665453是原文件行数，不是独立父任务或成功任务数。随后以固定CI工件完成全部665453条SQLite索引，并核验原行0、1、2、665450、665451、665452的prepare→resolve，见[全量索引回执](validation/tracelab-full-index-20261024.json)。未运行科研分析或模型，历史三条准备回执不回写。

[固定发布源码](https://github.com/uw-syfi/TraceLab/tree/61fea8f97277d0aa247cc2e4c7e31f90ed2bea1c)中的20份replay、示例及许可文件已在服务器取得，逐文件Git
blob身份一致，不在本仓重新分发源码、原数据或研究分析模块。[数据审计](validation/tracelab-cpu-audit-20261024.json)与[源码准备回执](validation/tracelab-official-replay-preparation-20261024.json)分别保留最初数据核验与源码获取的历史状态，不能据此证明构建/回放通过。原数据按发布方[数据许可](https://github.com/uw-syfi/TraceLab/blob/61fea8f97277d0aa247cc2e4c7e31f90ed2bea1c/LICENSE-DATASET.md)标为CC-BY-4.0，代码为Apache-2.0，二者不能混用；使用时保留来源与归属，不公开本地原始载荷。

### 官方入口与尚缺的运行条件

元数据准备可直接使用已安装的数据组件。先将官方原压缩文件放在独立私有目录，建立 `tracelab-inputs.json`（不填 `limit` 即全量）：

```json
{"sources": [{"source_id": "TRACELAB", "data_root": "/private/original-tracelab", "files": ["syfi_coding_trace.jsonl.gz"]}]}
```

```sh
python -I -m vllm_hust_workload_data.pipeline plan --config tracelab-inputs.json
python -I -m vllm_hust_workload_data.pipeline run --config tracelab-inputs.json --output /private/tracelab-index-new
```

输出须使用不存在的新目录，并与原数据目录分离。索引不替代原文件；下游回查还需保留原文件和绑定版本。此入口只准备调用元数据，不调用模型或启动官方回放器。安装及工件核验见[安装说明](INSTALLATION.zh-CN.md)。

官方[session_runner](https://github.com/uw-syfi/TraceLab/blob/61fea8f97277d0aa247cc2e4c7e31f90ed2bea1c/replay/README.md)按会话顺序提交token-id
completions，用独立文本池模拟输入；每轮等响应及工具等待后推进。它不是原私有prompt或真实工具执行。官方[CSV准备器](https://github.com/uw-syfi/TraceLab/blob/61fea8f97277d0aa247cc2e4c7e31f90ed2bea1c/artifacts/trace_facts/csv_export/README.md)还会做发布方定义的会话分组、排序、长度下限处理和带种子的合成到达；这些转换参数必须记录，不能称原始请求时间。本仓不复制图表、特征或缓存研究分析。

服务器初始未找到cargo/rustc；本轮已在独立`/tmp`前缀安装官方摘要核验的Rust1.90.0，不修改全局工具链或模型环境。首次按原锁文件构建因找不到OpenSSL开发文件退出101，保留[首次构建回执](validation/tracelab-original-build-attempt-20261024.json)。随后仅为构建进程绑定现有Miniforge
OpenSSL3.6.4的头文件/库，原Cargo.lock双CPU并发构建退出0，原源码与锁文件全部SHA不变，见[锁定构建回执](validation/tracelab-original-locked-build-20261024.json)。实际系统为openEuler24.03
LTS-SP3 ARM64；构建使用独立CARGO_HOME/CARGO_TARGET_DIR，运行客户端也需可找到同一OpenSSL动态库，不向模型服务环境注入路径。

官方转换器依赖的`syfi_coding_trace.duckdb`现已由官方GitHub Release
API资产488289840直接流式传至服务器私有目录，未在本地保存数据文件。160182272字节及SHA256
`a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744`均与发布方摘要一致，见[原DuckDB资产回执](validation/tracelab-duckdb-original-20261024.json)。随后补取同一固定提交的官方转换器、数据库工具、schema及锁文件共六份原文件，逐文件核对Git
blob；以只读方式打开原数据库并完成全量转换，两次各665453行、CSV逐字节一致，原数据库SHA不变，见[全量官方转换回执](validation/tracelab-official-csv-preparation-20261024.json)。历史资产回执仍保留取得当时未打开/转换的状态，不回写。真实服务接入、缓存明细探测与NPU回放仍需验证，不写成完整回放环境已就绪。

转换器的实际隔离环境为Linux
ARM64、Python3.12.13、DuckDB1.5.3、NumPy2.4.6；两项依赖版本与固定版`uv.lock`一致，`pip check`通过。这是该转换器所需的最小依赖环境，不是上游所有科研/网页功能的完整`uv sync`环境。可在独立环境复现，不要安装进模型服务环境：

```sh
python3.12 -m venv /tmp/tracelab-csv-env
/tmp/tracelab-csv-env/bin/python -m pip install duckdb==1.5.3 numpy==2.4.6
/tmp/tracelab-csv-env/bin/python -m pip check
# 在上述固定提交的官方源码根目录运行；输出路径须未使用。
/tmp/tracelab-csv-env/bin/python artifacts/trace_facts/csv_export/convert.py \
  --db /private/syfi_coding_trace.duckdb -o /private/coding-trace-new.csv \
  --provider all --session-order shuffle --arrival-pattern poisson \
  --arrival-rate 1 --seed 0 --tool-latency-source wall
```

不设置`--max-sessions`，保留全部发布记录；这里的shuffle、到达过程、长度下限及工具等待转换均为原官方逻辑，不是原请求正文或原始到达时间。两次完整转换不产生模型请求，也不构成回放或性能结论。

独立文本资产按固定版官方README推荐选择enwik9，已从[原发布地址](https://mattmahoney.net/dc/enwik9.zip)直接取得至服务器，未改写文本。原解压文件10亿字节、MD5和SHA1与[发布方校验页](https://mattmahoney.net/dc/textdata.html)一致，并核验完整UTF8解码与ZIP
CRC；本机另记SHA256，见[辅助文本资产回执](validation/tracelab-independent-text-pool-20261024.json)。它仅为官方合成会话回放的独立文本池，不是TraceLab原私有prompt，不增加本项目的数据集清单条目；原文件仍不公开分发，使用者应遵守Wikipedia派生文本的适用许可。取得文本不等于已创建token池，更不证明默认1亿token池容量满足实际分词结果。

随后以独立CPU诊断程序直接引用未修改的上游`tokens.rs`及原锁构建的Rust依赖，加载当前Qwen3.5的原tokenizer.json并调用上游`load_tokenizer/build_token_pool`，完整得到默认100000000个token，未降低池容量；原源码与tokenizer的SHA不变。该默认值也与全量/固定三会话的上游自动容量公式一致。见[CPU诊断回执](validation/tracelab-original-token-pool-cpu-20261024.json)。此项不是原`session_runner`的执行，更不是服务缓存探测、NPU回放或性能结果；诊断没有网络模型客户端，也没有模型请求。

诊断返回完整结构化成功输出后，取回文件时SSH连接拒绝，因此公开诊断回执由此前已经捕获的完整stdout恢复，并明确注明未完成SCP传输核验；不推断服务器被回收或其当前进程状态。原构建、CSV及dry-run历史回执不回写。

满足上述条件后，在固定版上游目录按原入口构建和预检：

```sh
env PATH="/private/rust-1.90.0/bin:$PATH" \
  CARGO_HOME=/private/tracelab-cargo-cache CARGO_TARGET_DIR=/private/tracelab-target \
  OPENSSL_DIR=/root/miniforge3 LD_LIBRARY_PATH=/root/miniforge3/lib \
  cargo build --locked --release --jobs 2 --manifest-path replay/Cargo.toml --bin session_runner
LD_LIBRARY_PATH=/root/miniforge3/lib /private/tracelab-target/release/session_runner \
  --trace /private/publisher-export.csv --text-file /private/text-corpus \
  --tokenizer /private/model/tokenizer.json --model LOCAL_MODEL_ALIAS \
  --dry-run --max-model-len 131072 --summary-path /private/replay-dry-run.json
```

dry-run不联系模型，也不加载文本池/tokenizer，成功退出不证明这些资产已存在或上下文适配。须检查summary的overflow记录，并逐轮确认完整输入加原请求输出预算可容纳；超限记阻塞，不截断、不换题。特别注意原版`--fail-on-context-overflow`实际会跳过轮次，本项目固定任务验证不能启用它来凑通过。真实运行时移除dry-run并提供批准的本地`--base-url .../v1`，仍需先验证完整所选会话和原等待/输出设置。

上述全量官方dry-run已退出0，识别8058会话/665453轮，但完整输入加原输出预算有336898轮超过当前131072窗口，因此不代表当前部署能跑全量。另按官方`--max-sessions 3`固定抽样完成dry-run：上游使用BTreeMap，选的是会话ID字典序前三个，而非CSV前三条或挑容量合适的会话；完整保留这三会话20轮，最大输入+原输出预算46646，无超限。未启用跳轮选项，也未调用模型。见[dry-run回执](validation/tracelab-original-dry-run-20261024.json)。注意官方summary只检查prompt长度，本项目另核验了原输出预算，不能拿summary中的零实际请求/零失败当成真实通过。

官方回放器启动时会用两次模型请求检查prefix-cache和cached-token明细；服务不支持则终止，不能绕过该检查冒充通过。作者示例CSV只能验证示例链路，不是全部发布轨迹。NPU回放尚未完成，不把已完成的构建、CSV转换或dry-run等同于原生回放；研究线程占卡期间不启动模型或探测请求。

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

2026-10-26补充：9.37TiB不是官方工具接通验证的唯一门槛。官方2000页默认profile保持原始hash ID和全部选中请求，按作者映射规则运行，
可单独验证工具能否使用，但不声称无冲突KV复用。两个场景各自保留独立数据文件时，容量上限合计224,919,552,000字节；这是静态配置计算，不是当前服务器可用空间。

不写KV页的公开预检入口如下；原始文件留在外部服务器，工具目录必须是固定提交的`benchmarks/storage_benchmark_v1`子目录：

```bash
python scripts/prepare_mooncake_storage_plan.py \
  --author-tool /private/Mooncake/benchmarks/storage_benchmark_v1 \
  --trace-root /private/original-traces \
  --storage-root /private/fresh-mooncake-storage \
  --output /private/fresh-mooncake-plan --count 3
```

校验九个作者模块及两份原始轨迹的字节身份与完整字段/行数，固定原顺序前三请求，生成两个原生CLI命令；`--full`不设置作者max-requests限制。 计划保留原始时间戳、长度和hash
IDs，默认`replay-scales=0`仅为作者未配速profile，不冒充真实到达速率。
预检只解析原模型配置字典推导容量，不导入执行作者代码，不创建未来storage目录，不执行命令、不写data.bin。
生成的`plan-private.json`包含路径、配置和容量边界；空间观察只用于后续资源核验，执行前仍须确认独立目录、足够余量及研究任务释放磁盘窗口。
软件夹具验证不证明原文件在当前服务器存在或官方回放通过，来源继续不勾选。`glm5`在此仅为KV布局预设，不是部署了GLM-5模型。

确认研究任务已释放专用磁盘窗口，并复核服务器容量后，实际回放入口如下（目前只有软件与模拟进程验证，没有真实回放通过证据）：

```bash
python scripts/run_mooncake_storage_original.py \
  --author-tool /private/Mooncake/benchmarks/storage_benchmark_v1 \
  --trace-root /private/original-traces \
  --storage-root /private/fresh-mooncake-storage \
  --output /private/fresh-mooncake-run \
  --storage-budget-bytes 250000000000 --io-window-confirmed --count 3
```

仅允许Linux服务器执行，预算是对两个场景最大保留文件容量的显式上限，不是实际I/O流量；另需至少2GiB空闲余量。
目录均须新建且与原始资产分离。九个原模块逐字节复制到私有干净目录后运行，不修改作者实现，也不把原始轨迹复制或改写成另一份请求。
`--io-window-confirmed`是操作者确认，不是程序自动证明研究任务已结束；不能因为旧线程交接或状态文件存在就勾选。
每个命令的退出码及最终请求数、token总数、读写页操作总数均须与固定原任务一致，部分结果或跳过的I/O不会因退出码为0而通过。
这些是执行完整性计数，不是研究观测、特征分析或引擎性能结论。失败和超时保留原日志、目录及数据文件，不回退、重试或自动删除。
如conversation已有效通过，可在新的私有目录下加`--scenarios toolagent`只补该场景，不重跑已有成果；单场景默认容量112,459,776,000字节，预算可相应具名设置。
完整文件选用`--full`，不改变题内页操作；仍需记录明确外围deadline和资源窗口。两种场景与研究侧固定工件消费齐全前，来源不打勾。

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

官方固定提交`7eb2c4f8350f8a6985272386f5c14af1f678b299`的example说明及客户端代码已核对，见[官方示例适用性复核](validation/burstgpt-official-demo-review-20261024.json)。它显式另读ShareGPT等prompt池，按trace长度映射外部例子，内部维度上限1024并有长度裁剪；这属于官方模拟压力工具，不是原始请求回放。CSV按20行chunk读取，prompt_num不是严格读取条数；surplus默认16384可能超出已读轨迹，不能直接套默认参数。客户端调用旧`/generate`，当前服务OpenAPI确认不提供该接口，因此尚未运行官方示例。后续需明确兼容传输/支持服务后单独验证，保持外部prompt来源披露；不修改原CSV、不把外部prompt冒充BurstGPT原请求。此项仍未勾选，不把模拟工具未执行隐去。

### BurstGPT原官方示例兼容入口（2026-10-25）

新增`scripts/run_burstgpt_original_example.py`，在回环地址提供旧式非流式`/generate`，转发到已部署模型的`/v1/completions`。只适配传输，不改原prompt、temperature0、ignore_eos=False及原工具实际使用的1024-token输出预算。旧接口返回`prompt + 生成后缀`；这与chat模板请求不同，不自动改成chat请求或附加思维提示。合法stop/length及零token的空stop均保留，不换题重试。桥接错误或缺少任一选定请求的原生响应都会使整批失败，不能被原工具的HTTP错误日志吞掉后误记成功。尚不提供流式桥接；非流式路径没有TTFT测量结论。

入口核验固定提交18份原文件的Git blob、三份原trace的发布SHA，以及官方自带外部prompt池的Git
blob。该池已在服务器取得，共6473720字节、300条，见[准备回执](validation/burstgpt-official-example-preparation-20261025.json)。它不是BurstGPT原私有请求。原工具长度映射使用池内既有长度字段，未重新按Qwen分词；桥接另存实际服务usage，不能把这些字段混为实测token长度。原工具的out_len是返回文本字符数，也不能当生成token数。固定根LICENSE为CC-BY-4.0，而example说明称MIT；使用者须分别核查数据、示例代码及外部prompt池条款，本仓不重新分发这些原件。

在服务器取得固定版源码及原trace，安装独立Python3.12客户端；不向模型服务环境安装这些依赖：

```sh
python3.12 -m venv /private/burstgpt-client
/private/burstgpt-client/bin/python -m pip install -r configs/workload-native/burstgpt-original-demo-py312-dependencies.txt
/private/burstgpt-client/bin/python -m pip check
/private/burstgpt-client/bin/python -I scripts/run_burstgpt_original_example.py \
  --source-root /private/BurstGPT-fixed \
  --prompt-pool /private/BurstGPT-fixed/example/preprocess_data/shareGPT.json \
  --trace /private/original-traces/BurstGPT_1.csv \
  --output /private/fresh-burstgpt-native --count 20 --scale 1 \
  --endpoint http://127.0.0.1:19466/v1 --model workload-qwen35-27b-128k-nonthinking
```

`--source-root`须是固定提交`7eb2c4f8350f8a6985272386f5c14af1f678b299`的根目录，至少包含example子树和LICENSE，不能指向修改过的客户端。默认选前20条，`--count N`可显式选原前N条或全部发布行；不允许越过原行数进入上游Gamma模拟分支。原读取器仍按20行块加载；surplus=N恰好只发N条，不把多读行当已执行。输出必须是不存在的新私有目录，含原工具日志、每次请求/响应及终态回执，不得公开原载荷。

本轮在模型结果产生前固定三个文件各原前20条，scale=1。原第1份的选定时间戳为5–1072秒；原官方CLI及现有双NPU服务现已完成20条模拟请求，12条length、8条stop。直接重建原查询生成器，20条原查询、原预算、保存响应摘要、旧接口返回映射、官方detail日志及最终summary均核对一致；审计模型调用0，见[原生保存结果审计](validation/burstgpt-release1-native-audit-20261025.json)。保留length，不为stop重试；没有模型质量或吞吐结论。

第2、3份开头时间戳延续全局时轴，分别约5270414秒和19440110秒；上游按启动后等待时间直接sleep，scale=1对应约61天和225天的初始等待。未静默减去首时间戳、压缩到达或跳过原行；这两份的模拟执行仍待适用的显式调度配置，元数据资产核验不受影响。官方`--scale`是RPS缩放参数，若后续选择它必须单列配置和改变后的到达时间，不能称原始到达回放。旧适用性回执保留当时未实现兼容层的历史状态；当前仍不勾整个BurstGPT来源。

官方v2.0的三个保留失败记录的完整发布文件均已在服务器核验字节数、发布SHA和全CSV解码：`BurstGPT_1.csv` 1429737行、`BurstGPT_2.csv`
3858436行、`BurstGPT_3.csv` 5344021行，合计10632194行；不代表三文件互不重叠的独立请求总数。首文件复用，不重下载；其余两文件通过官方GitHub Release
API资产341027213、341026736的`application/octet-stream`下载，所得字节与发布摘要一致。之后复用`acquire_release_datasets.py --source BURSTGPT --data-root EXTERNAL_DATA_ROOT --output EXTERNAL_REPORT`重新核验全部三文件。见[资产回执](validation/burstgpt-original-release-assets-20261024.json)，其中`download_url`表示规范发布地址，不表示本次实际传输绕过API。

此证据仅证明原资产完整取得、全文件可解码，不替代全部文件适配/prepare/resolve或模型回放。发布方存在`example/`示例，用外部ShareGPT提示池按轨迹到达/长度构造模拟压力，不能称为原始请求重放；该官方示例的依赖与运行范围尚待复核，不宣称无官方工具或已跑过示例。原CSV无prompt，不能自行补prompt伪装原请求。

随后三文件完整绑定均匹配原发布SHA与行数，各取原行0、1、2，共9条完成安装包读取→prepare→resolve往返，输入与历史结果分离且逐项一致；9条均明确为`request_metadata`，请求体接口按预期拒绝，不产生模型请求。见[原元数据往返回执](validation/burstgpt-original-roundtrips-20261024.json)。首文件跨数据根目录的软链接被安全检查拒绝；使用同一NFS内原文件的硬链接形成合法外部数据根目录，不放宽越界校验，不复制正文进仓库。官方示例适用性和最终软件交付仍待收口，不勾整个来源。

### 旧首文件尝试（保留历史范围）

以下保留当时的文件范围、勾选与PR状态，不是当前总进度；当前以本节前面的新回执、机器清单及原PR #432为准。

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
