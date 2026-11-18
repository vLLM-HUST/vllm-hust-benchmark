# 生信原生任务代表模型 Benchmark

本入口为基于 vLLM-HUST 的多种生信模型推理引擎提供项目评测。首批执行 ESM-2 150M 与 ProGen2-small 的表征、任务头、变异、生成和评分，采用各发布物现有的单
Ascend NPU、FP16 eager 原生运行配置。

任务驱动和模型专属验证器位于 `vLLM-HUST/scientist-lightweight-vllm` 的 `experiments/benchmark/representative/`。本仓库的
`vllm_hust_benchmark.bio` 负责跨仓配置绑定、进程编排、验收与归档，`bio_report` 从原始记录重算统计与独立数值结果。

## 使用方式

使用固定模型运行环境中的 Python，指定 scientist 工作区。CPU 分析依赖在 `bio` 可选依赖组中。

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.bio check \
  --scientist-root /root/scientist-lightweight-vllm-workspace
PYTHONPATH=src python -m vllm_hust_benchmark.bio run \
  --scientist-root /root/scientist-lightweight-vllm-workspace \
  --output .local/bio/representative-001
PYTHONPATH=src python -m vllm_hust_benchmark.bio verify \
  --scientist-root /root/scientist-lightweight-vllm-workspace \
  --output .local/bio/representative-001
PYTHONPATH=src python -m vllm_hust_benchmark.bio report \
  --scientist-root /root/scientist-lightweight-vllm-workspace \
  --output .local/bio/representative-001 \
  --destination .local/bio/representative-001-recomputed
```

`run --pilot` 执行预运行并在结果中保留其身份。完整运行采用版本化 suite 声明的实例数、预热与重复预算。`--python`
指定执行原生驱动的解释器，`--suite`、`--data`、`--assets`、`--upstream` 指定配置和本地资产。

## 当前测量协议

配置维护在 scientist 工作区的 `docs/benchmark/phase2_suite.json`，详细协议为同目录
`phase2_protocol.zh-CN.md`。39个用例使用三个带摘要的 UniProt 序列快照和明确窗口，覆盖七类任务的短中长、边界、不等长 batch、采样、目标排除与状态恢复。

每个模型执行三个独立实例，每用例保留首次调用、两次预热和十次稳态调用。两个模型还分别执行额外实例的主机异常注入与关闭。模型专属八类输入拒绝和随后恢复在每个成功实例中执行。

公开调用延迟使用 prepared input 到 CPU 输出的单调时钟。吞吐使用实际稳态循环窗口，窗口含请求间观测与记录开销。报告保留逐实例与合并样本、p50/p90、实例级 bootstrap
区间、设备 allocated/reserved/峰值和 worker RSS。

原始权重与同设备独立 HF/作者参考提供数值依据。JSONL、原生输出及参考张量共同支持独立重算，源码归档覆盖两个工作区的本地修改。`verify`
检查任务完整性、数值、时钟、统计、文件摘要和来源身份。

## 本地验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_bio_benchmark.py' -v
```

本入口的实验与原件进入所选项目评测目录，交付状态由对应 `result.json`、`summary.json` 与 `report.md` 共同记录。对现有测量结果的复核通过独立输出目录生成。
