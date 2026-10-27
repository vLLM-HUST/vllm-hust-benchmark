# PyramidKV：按层预算的有损KV筛选

2026-10-26，只读源码及作者报告，未运行设备。源码 `394844c96de5a93c1ec5830a743965d9360515f2`。

## 机制

是此前审阅的Ascend KVCompress共享adapter上的外部method，不是另一套完整生命周期系统。仅压缩full-attention KV，Qwen3.5
GDN递归状态不动。不是量化位宽，也不是缓存迁移，而是丢弃历史token的KV。

final-prefill末尾query window（默认8行）对完整历史K重新计算QK/sqrt(d)、causal mask、FP32
softmax；对query窗口求和、GQA组内求mean，再max_pool1d（kernel7）和top-k。每个KV head可选不同历史位置；最近8个token全部保留。

“金字塔”是浅层保留多、深层保留少。当前method按参与的full-attention层序号分配预算，非按GDN/attention混合模型所有层盲目编号。默认max_capacity_prompt512、beta20；超过4096
prompt tokens才压缩。max_capacity不是所有层硬上限：历史预算B=capacity-window，最少约B/beta，最多2B-B/beta，再随层递减并加回window。

## 执行和成本

共享adapter跨chunk捕获最终query窗口，完整final-prefill forward和sampling成功后发布观察结果，再在同步压缩事务中调用method。不是每个decode
token持续重打分；当前method拒绝重复压缩。

逐层gather完整历史K以打分，执行matmul/softmax/pool/topk；按head索引读取选中K/V并index_copy到独立私有目标页。V只取选中行，不全量gather。之后提交每层不同physical
length，用于decode和Graph metadata；MTP辅助cache也要物化。没有独立异步压缩流水的证据，不能把device kernel异步发射等同于不占关键路径。

为获取最后8行Q，APC准入最多命中prompt_len-8，即便缓存已有这些token仍需留尾部重算。经济账是一次额外打分/搬运/元数据成本，换之后decode少读KV，且承担有损精度与可能的MTP接受率变化。

## 当前已读实测

2026-10-09固定paired evaluation：Qwen3.5-35B-A3B
BF16、TP2、910B2、CANN9.1、APC/MTP2/async/chunked-prefill/FULL_AND_PIECEWISE/mamba
align。源版本明确记录，不当作HEAD独立重测。

原512/beta20配置质量未通过；先修复共享adapter FULL Graph cache-write
slot错误，再用开发集选择1368/beta2并在holdout前冻结。最终409请求/arm全部完成，325压缩commit。物理allocation上界2048不意味着每层恰保留2048。

- 开发/holdout各150题，平均QA-F1分别下降0.59/0.26分，满足预声明mean\<=3、各task\<=5分的容忍门槛；不是无损。
- 8K/24K长输入、并发1/4，输出吞吐分别下降10.3%/4.0%/8.8%/2.6%。
- sampled peak KV pool利用率29.12%→21.70%，但整卡峰值HBM约增加410MiB，因为还含预留pool、graph/workspace。
- 两边都完成32K、并发1/4，没证明多承载了请求。
- 全run MTP接受率89.79%→83.74%，不能把全部性能退化仅归因某个单因素。
- 1024-token组没有触发压缩，其正向波动不是压缩收益。

当前有实际实现和限定栈功能/质量证据，但性能未获正向资格；不同于无benchmark的小补丁。本轮不预写Fletcher最终comment。

## 固定来源

- [README](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/blob/394844c96de5a93c1ec5830a743965d9360515f2/README.md)
- [预算与打分](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/blob/394844c96de5a93c1ec5830a743965d9360515f2/src/vllm_ascend_pyramidkv/provider.py)
- [当前method、物理复制](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/blob/394844c96de5a93c1ec5830a743965d9360515f2/src/vllm_ascend_pyramidkv/method.py)
- [共享宿主合同](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/blob/394844c96de5a93c1ec5830a743965d9360515f2/docs/host-contract.md)
- [完整paired报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9b81690f767aca56524ec6651535a9b8d484a4c3/archive/mod-git-20261022/pyramidkv-website-source/snapshot/evidence/current/2026-10-09-paired-evaluation/README.md)
