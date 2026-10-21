# vLLM-HUST Workload Data

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
