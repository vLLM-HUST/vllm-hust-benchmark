# Quantized KV Cache：INT8与KIVI INT4

2026-10-26，只读。HEAD
`da171fcbf9c5e79710083df70cb445fb548871b7`；未安装或实测。与PyramidKV不同：保留token，减少KV表示位宽，仍有量化误差，不是无损压缩。

## INT8路径

插件按CLI kv-cache-dtype分派attention
implementation。INT8的`_prepare_int8_scales`在layer首次收到K/V时沿token维amax，得到各channel对称scale；`_int8_scales_ready`后不再更新。所谓dynamic是运行时初始化，不是每请求/每token持续动态校准，后续超范围会clamp。具体首次输入可能受宿主warmup/capture流程影响，尚未验证。

写入执行multiply/round/clamp/int8转换；decode将INT8 paged KV和antiquant scale交给fused
attention，在线反量化，不先把整个历史还原成稠密cache。Prefill/chunked-prefill有需要时gather和反量化。原始数据部分约2x缩小，不当作全服务显存减半。

## INT4路径

KIVI：最近默认128token保持全精度，旧历史INT4；K沿token组量化，V沿head维组量化，非对称min/max，默认group128。Triton打包；两张宿主byte
buffer切成K/V数据及scale/min共6视图，不为视图本身复制。

关键缺口：当前attention读路径是**分页gather→纯torch解包/反量化→生成稠密TND K/V→覆盖残差尾→fused attention**，不是在attention
kernel内直接消费INT4。因此省常驻容量，但仍有稠密临时张量及读写，不能从存储压缩比推导attention带宽加速。

head128/group128单侧每token/head：64bytes INT4 + 8bytes FP32
scale/min=72bytes，对比FP16的256bytes为3.56x；此为历史区字节预算，额外残差窗口与读路径临时内存不在比值内。

## 当前证据质量

作者2026-10-04
Qwen3.5-35B-A3B、TP2+EP、910B2、CANN9.1，eager、无APC/无chunked-prefill，max_num_seqs16；三组8请求并发、每请求输出128。原始JSON提示长度只有15–30tokens，**不是长上下文KV带宽基准**，也几乎没有长历史区可摊销。

表中吞吐auto2.2、INT8 2.3、INT4
2.0tok/s，单路87.1/85.9/99.8s。宿主GDN因算子缺失走torch/Python兜底，主导耗时；即使三组共享瓶颈，也不能据此证明正常生产栈KV优化收益。

“8192上下文可并发90.14x→126.20x”不是实际完成90/126并发的压测：只发8条、max_num_seqs16，是容量估计口径。INT8/INT4估计完全相同，作者自己列为待查。稠密模型2x/3.56x早期结果原始数据未保留，仅有文字记录，不能当独立已核实容量收益。

README明确模型级质量基线未建立；“输出连贯”或HTTP200不足以证明质量，文档还保留INT8答错数值比较的样例。未做正式任务打分，不据单例断言普遍掉分。

## 集成和判断

需要宿主CLI literal、storage dtype、page size、cache shape四处配套修改；非匹配宿主会在CacheConfig构造阶段拒绝。非NPU/context
parallel等fail closed，不是装wheel即可通用。FP8只有写内核无读路径，不算可用方案。

INT8在线antiquant方向比INT4显式全历史还原更接近真正省decode带宽；当前INT4主要是容量型原型，融合读路径尚未完成。性能、容量、精度三者应分别验收，现有短prompt+GDN兜底数据不足以做采用判断。暂不预写Fletcher最终comment。

## 固定来源

- [README](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/blob/da171fcbf9c5e79710083df70cb445fb548871b7/README.md)
- [INT8 backend](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/blob/da171fcbf9c5e79710083df70cb445fb548871b7/src/vllm_ascend_quantized_kv_cache/methods/int8_dynamic/attention_backend.py)
- [KIVI backend](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/blob/da171fcbf9c5e79710083df70cb445fb548871b7/src/vllm_ascend_quantized_kv_cache/methods/kivi_int4/attention_backend.py)
- [benchmark及限制](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/blob/da171fcbf9c5e79710083df70cb445fb548871b7/docs/benchmark.md)
- [原始JSON](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/tree/da171fcbf9c5e79710083df70cb445fb548871b7/benchmarks/qwen3.5-35b-a3b)
- [serving修复和失败记录](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/blob/da171fcbf9c5e79710083df70cb445fb548871b7/docs/serving-verification-20261003.md)

## 最终评审 comment

Fletcher/Lumi：**受 Ascend AIC/AIV 协作限制，额外主动量化/反量化往返通常不值得做；更有意义的是产出与消费链路直接使用量化表示，否则会增加 GM 读写。**
这是工程投入判断而非本轮普遍性能证明。KIVI 的显式稠密还原直接落入该问题；INT8 online antiquant 的物理路径仍需区分，不能凭接口名推断零额外 GM
搬运。[集中comment](../comments.md)
