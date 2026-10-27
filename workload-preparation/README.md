# vLLM-HUST Workload Data

## 按来源找到准备与执行入口

分类导航（只查元数据，不下载数据或宣称可运行）：

```bash
python -m vllm_hust_benchmark.workload_data classes --scheme delivery-plan
python -m vllm_hust_benchmark.workload_data class-plan W11 --scheme delivery-plan
python -m vllm_hust_benchmark.workload_data delivery-assets
python -m vllm_hust_benchmark.workload_data source-plan BURSTGPT
```

`delivery-plan`覆盖本次40类任务及118个登记来源；其中15项作为服务轨迹、受控探针、 跨任务关系或企业私有请求资产单列，不硬造语义任务分类。企业仅提供索引和边界，
不包含原始请求。类下来源是候选关联，不表示该来源全部记录属于该类，也不表示已适配、 取得数据或原生跑通。研究默认38类入口保持不变；当前交付状态以机器清单为准。

`source-plan SOURCE_ID`同时返回已索引的准备、生成、原生profile、隔离评分或数据读取入口，
按role区分。路径相对于benchmark仓库根目录，执行脚本不会随纯数据wheel打包；先按相应 文档准备外部原始资产和独立环境，再查看脚本帮助。数据读取入口不是原生benchmark执行器。
索引包含尚未验证的实现，空列表也不表示已经审计为“没有实现”；完成状态仍只看机器清单， 不能从脚本存在推断原生验证通过。

检索/重排新增入口：`BEIR_SCIFACT`、`BEIR_NFCORPUS`的原zip准备、原文档/查询/qrels读取，以及`scripts/run_beir_original.py`的全量test检索和dense
top100重排，见[原生协议与环境](../docs/workload-data/PUBLIC_ASSET_SERVER_READINESS.zh-CN.md#beir-%E5%8E%9F%E5%A7%8B%E6%A3%80%E7%B4%A2%E9%9B%86%E5%90%88%E4%B8%8E%E9%87%8D%E6%8E%92%E5%8D%8F%E8%AE%AE)。原集合不是chat请求，qrels只作评分参考。原全量NPU运行、最终公开入口交接和固定CI工件/研究消费均已验证，范围限定为上述两集合的原test检索及dense
top100重排。

| 场景                   | 入口与验证范围                                                                                                                                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 长文档理解             | [QuALITY 原始问题](../docs/workload-data/QUALITY_ORIGINAL_QUESTION_INPUTS.zh-CN.md)：问题级读取、完整文章、MC 执行与评分；不是发布方训练基线复现                                                                                                                   |
| 结构化输出             | [JSONSchemaBench 原始约束](../docs/workload-data/JSONSCHEMABENCH_ORIGINAL_SCHEMA_INPUTS.zh-CN.md)：不改 schema；三条生成结束，原约束符合仅1/3，失败保留                                                                                                            |
| 独立推理               | [MATH500 原始任务](../docs/workload-data/MATH500_ORIGINAL_TASK.zh-CN.md)：500条固定 held-out 问题按来源 ID 读取；problem 独立作为输入，answer/solution 保持在 outcome；三区分测试样本、H4 Inspect 配方与 PRM800K 确定性评分器                                      |
| 检索跟进、文本任务     | [QReCC 准备](../docs/workload-data/QRECC_ORIGINAL_PREPARATION.zh-CN.md)、[原始 NI](../docs/workload-data/NATURAL_INSTRUCTIONS_ORIGINAL_INPUTS.zh-CN.md)、[编辑与实体抽取](../docs/workload-data/NI_EDITING_EXTRACTION_ORIGINAL.zh-CN.md)：分来源区分准备与实际评分 |
| 到达、尺寸、匿名块标识 | [Azure、Mooncake、BurstGPT 原始轨迹](../docs/workload-data/ORIGINAL_SERVING_TRACES.zh-CN.md)：仅元数据，无原始 prompt，不生成伪回放或推导实测 KV 命中                                                                                                              |

代码类新增入口见[BigCodeBench 原始生成阶段](../docs/workload-data/BIGCODEBENCH_ORIGINAL_GENERATION.zh-CN.md)： 三条原题
NPU 生成与公开入口复跑完成，功能执行/评分未完成。
代码、科学、图表与历史来源的[服务器准备记录](../docs/workload-data/PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)
仅证明七源21条读取/准备/回查，不能代替其原生任务执行。

[LoCoMo原始长会话问答](../docs/workload-data/LOCOMO_ORIGINAL_INPUTS.zh-CN.md)保留完整会话和参考隔离，原生完整上下文文本QA已验证108个唯一原题，含显式缺键兼容；不是全1986题或RAG/图像任务完成，须遵守CC-BY-NC-4.0条款。

[独立QASPER原发布版准备](../docs/workload-data/PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)保留三split共5049个原问题、完整论文与全部答案/证据标注；提供归档提取、读取和索引，原生模型及答案/证据评分尚未完成，不能与LongBench派生200题混计。

上表是使用导航，不代表全来源或全类别完成。精确已验证范围见各来源说明、
[类别执行清单](../docs/workload-data/WORKLOAD_EXECUTION_CHECKLIST.zh-CN.md)和对应 PR 的固定工件记录。

## 其他已接入来源与边界

OASST1 原始消息树的显式历史续接视图见[使用与验证边界](../docs/workload-data/OASST1_ORIGINAL_CONTINUATION.zh-CN.md)。
它保留原始父子关系、排除目标回复；不是实时交互环境。通用索引可显式选择 `history_targets`；对应固定工件的研究消费回执见类别清单，不能把历史续接等同实时会话。

BFCL v4
simple_python的原始JSONL读取和独立静态评分入口见[接入与验证](../docs/workload-data/BFCL_V4_SIMPLE_PYTHON_ORIGINAL.zh-CN.md)。
显式格式贯通原行回查；评分参考不进入任务输入，原三题NPU/官方AST评分不代表真实工具执行或其他类别。

LongBench v1 Qasper原文准备和执行见[使用与验证](../docs/workload-data/LONGBENCH_V1_QASPER_ORIGINAL.zh-CN.md)。
`workload-data prepare-longbench-qasper`只复制并核验原始归档成员，任务输入与评分参考分离；仅Qasper原三题完成NPU/官方评分，不代表其他子集或全量。

AgentBench FC
ALFWorld原生引用入口见[接入与验证](../docs/workload-data/AGENTBENCH_ALFWORLD_ORIGINAL_PREPARATION.zh-CN.md)。`workload-data agentbench-native-tasks`只核验官方索引/游戏字节并输出元数据引用；Python>=3.10数据组件和独立Python
3.9原生任务环境分离，不把闭环任务退化为静态请求，也不迁入科研观测。

数据来源目录、原始文件读取适配、可校验输入引用和显式请求回放入口。独立发行包 `vllm-hust-workload-data==0.1.0`，导入名
`vllm_hust_workload_data`；不覆盖科研仓库的 `llm_serving_workloads`。

不包含观测钩子、采集、研究特征计算、机制干预或 QBI
历史执行入口。目录登记、读取适配与模型执行/官方评分是不同状态，见[逐项验证清单](../docs/workload-data/DATASET_CHECKLIST.zh-CN.md)。

## 安装与来源查询

从本仓库根目录运行：

```sh
python -m pip install -e .
python -m pip install -e "./workload-preparation[data]"
workload-data identity
workload-data sources
workload-data classes
python -m vllm_hust_workload_data.catalog source GSM8K
python -m vllm_hust_workload_data.catalog download-plan GSM8K
```

数据由使用者按发布方条款单独取得，放在仓库外。此组件不分发企业载荷、gated 数据、模型或第三方环境。下载计划是来源信息，不保证已经下载或具备执行器。

## 固定本地文件与准备输入引用

```sh
python -m vllm_hust_workload_data.pipeline bind GSM8K --data-root /data/gsm8k --files test.parquet --revision ORIGINAL_REVISION --output /data/bindings/gsm8k.json
```

编辑配置（路径相对于配置文件；输出目录必须不存在）：

```json
{"sources":[{"source_id":"GSM8K","data_root":"/data/gsm8k","manifest":"/data/bindings/gsm8k.json","limit":3}]}
```

```sh
python -m vllm_hust_workload_data.pipeline plan --config /data/input.json
python -m vllm_hust_workload_data.pipeline run --config /data/input.json --output /data/prepared/run-001
```

输出 `workload.sqlite` 与
`manifest.json`，仅保存可回查输入引用、来源/适配器身份和准备状态，不保存原始请求、答案、事件或研究特征。默认失败即停止；`--allow-partial`
才允许明确标注的部分结果。文件 SHA 校验只固定本地字节，不证明发布方认证。

## 下游直接读取

```python
from pathlib import Path
from vllm_hust_workload_data.pipeline import prepared_rows
from vllm_hust_workload_data.consumer import resolve_input

database = Path("/data/prepared/run-001/workload.sqlite")
for row in prepared_rows(database, source_id="GSM8K"):
    record = resolve_input(database, row["record_id"], data_root=Path("/data/gsm8k"))
    original_input = record.input_record()
    # 交给相应任务的原版执行器；不要把参考答案拼进输入。
```

静态任务不是完整 Agent episode；任务执行器负责原生环境、工具、终止和评分。AgentBench、GAIA、tau2 当前明确要求官方执行器适配，不回退到 QBI
派生任务。真实请求回放仅接受原始请求体；显式 `--model` 只绑定服务端模型并记录差异，不修改消息、预算和流式标志。流式响应必须收到
`[DONE]`。`python -m vllm_hust_workload_data.execution --help` 查看入口，单个 HTTP 响应不代表任务评分通过。

科研消费者保留自己的观测、特征和分析代码，只从本组件读取原始输入。参考[消费契约](docs/consumer_contract.md)、[迁移计划](../docs/workload-data/MIGRATION_PLAN.zh-CN.md)。

## 软件检查

```sh
python -m pip install -e "./workload-preparation[dev,data]"
python -m pytest -q workload-preparation/tests tests/test_workload_data_boundary.py tests/test_workload_gap_catalog.py tests/test_workload_validation_checklist.py
```

上述测试是 CPU 软件正确性检查，不加载模型，不证明真实数据集完整执行或性能提升。

独立知识问答任务的原版候选似然执行示例见 [AI2-ARC Challenge/Easy 抽样复现](../docs/workload-data/ARC_CHALLENGE.zh-CN.md)，可用
`--suite` 选择子集，与聊天生成协议分开。

通用单轮指令语料的原始字段生成入口见
[Dolly 接入说明](../docs/workload-data/DOLLY_ORIGINAL_INSTRUCTIONS.zh-CN.md)：原instruction/context不改，参考response隔离，提示格式与模型替换显式声明；只验证生成完整性，不冒充官方质量评分。

长输出运行维度见
[LongWriter-6k 原消息接入](../docs/workload-data/LONGWRITER_ORIGINAL_MESSAGES.zh-CN.md)：保留原始user消息，隔离最后assistant参考；不要将训练语料生成接入称为LongBench-Write官方评分或长输出能力验收。
