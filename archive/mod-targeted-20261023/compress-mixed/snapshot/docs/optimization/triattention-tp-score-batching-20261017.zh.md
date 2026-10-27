# TriAttention TP 评分合并与 SWE 配置筛选

[English](triattention-tp-score-batching-20261017.md) | 简体中文 |
[优化记录索引](README.zh.md)

日期：2026-10-17。范围：固定的 Qwen3.5-35B-A3B BF16、TP=2、Ascend 910B2
宿主上的 0.10.0 源码实现。这是工程记录，不能外推到所有模型、宿主或请求组合。

## 问题与实现

[0.9.0 SWE 配对](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261008-v090-swe-pair.json)
的逐请求解码 P90 提升 1.05%，但每卡输出吞吐下降 2.71%。原 TP=2 评分路径
对每个采样的注意力层分别执行 HCCL MAX 归约。Qwen3.5 采用 stride 2 时，
一次压缩选择要为五个采样层执行五次 collective。

新实现预分配连续的评分工作区，将各采样层归一化后的评分写入独立行，
对整个工作区执行一次 HCCL MAX，再沿层轴聚合。MAX 对每个层/token 项独立
生效，因此合并通信与逐层跨 rank 归约得到相同的结果。Ascend 归一化算子
使用与 Torch 回退路径一致的最小方差。TriAttention 的校准评分、V3 保护窗口、
分段配额及全部十个全注意力层的 KV 搬运均保留。实现见
[method.py](../../src/vllm_ascend_kvcompress/methods/triattention/method.py)
和[kernels.py](../../src/vllm_ascend_kvcompress/methods/triattention/kernels.py)。

此项代码改动减少每次压缩事件的同步工作。服务性能还取决于**何时**触发压缩，
因此候选筛选保持同一份优化源码，只调整物理 KV 预算和请求输出门槛。
门槛检查的是请求配置的 `max_tokens`；2048-token 门槛会跳过输出上限较小的请求。

## 固定负载与正确性检查

四组均使用相同的优化后 method/kernel 源码、模型 revision
`712cf74392b05026a6db2bf213d343747d1f6d45`、准备好的 SWE 负载哈希
`8044561ffa1bb430bea8f778ef814d96649321e1a92654b95f64263b996d5e85`、
宿主源码、两张 910B2、C4 和分别运行的 900 秒窗口。配置上下文为 262,144
token，实际最长 prompt 为 52,505。B0 加载插件但将压缩设为不可触发
（`kv_budget=262656`、`recompute_window=1024`）。B1 候选均使用 stride 2
和 1024-token 重算窗口。APC、真实 MTP2、异步调度、`mamba_cache_mode=align`
及 FULL_AND_PIECEWISE 保持一致。

匹配宿主环境中 210 项测试通过；TP=2 数值测试检查了合并评分后的跨 rank
一致性。各服务组分别通过 60 秒检查，并以 0 个失败请求完成 900 秒窗口。
所选候选通过四项带真实压缩的确定性 32K 检索检查。其服务日志记录 20 次
生命周期内的 scheduler 提交和 40 次 TP rank 确认。这些检查不能证明 SWE
任务的语义答案质量。

## 服务测量

| 组别 | KV 预算 | 请求输出门槛 | 每卡输出 token/s | 逐请求解码 P90 token/s | TTFT P95 ms | 窗口内完成数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0，关闭压缩 | 262656 | 512 | 74.426 | 47.843 | 1561 | 218 |
| B1，初始配置 | 8064 | 512 | 73.154（−1.71%） | 48.560（+1.50%） | 1828（+17.15%） | 214 |
| B1，缩小预算 | 4608 | 1024 | 74.098（−0.44%） | 48.426（+1.22%） | 1827（+17.06%） | 217 |
| B1，所选配置 | 4608 | 2048 | **75.418（+1.33%）** | **48.616（+1.62%）** | 1696（+8.65%） | 218 |

括号内百分比均相对于 B0。所选预算相当于四个 1152-token 注意力页。前两种
候选虽然提升了解码指标，但没有同时提高每卡吞吐。所选配置在一组匹配配对中
实现双指标提升，但 TTFT P95 仍退化。这只是一种负载上的单组观察，没有
可重复性或统计显著性估计。较高的请求输出门槛会改变触发压缩的请求集合，
因此收益不能全部归因于 HCCL 改动。

[聚合证据](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261017-v010-swe-pair.json)
包含源码和宿主身份、精确指标、run ID 及原始报告 SHA-256。
[测试规程](../benchmark/frontier-benchmarking.zh.md)说明指标计算与资格检查。
