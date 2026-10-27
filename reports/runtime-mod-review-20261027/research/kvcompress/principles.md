# Ascend KVCompress：TriAttention token 淘汰与物理 KV 压紧

日期：2026-10-26；源码
`vLLM-HUST/vllm-ascend-kvcompress-hust@3ddff1c2afee1eccc3d1b65def61167752298f5f`。本轮只读关键代码、设计与冻结汇总证据，不是独立硬件/质量复现。网页抓取仍显示
0.9，直接 Git checkout 为 0.10；实现判断以此固定 Git 对象为准。

## 机制

当前主方法是 TriAttention。这里的“压缩”是选择并删除历史 token 的 K/V，不是无损编码，也不是降低数值 bit width。保留的 K/V 仍为 dense BF16/FP16。

原论文用校准得到的 pre-RoPE query 分布中心与幅值统计，结合 RoPE 相位/距离信息估计未来 query 对不同 key 的重要性。插件的 `scoring.py` 读取实际
post-RoPE keys，结合模型校准的 query 均值、幅值、频率与未来位置 offset 打分；partial-RoPE 模型还计入未旋转维度的 Q·K 项。当前 TriAttention
不依赖每个请求实时观测完整 query window，不能与可扩展 method API 的 query hooks 混为一谈。

V3 选择策略保留开头和最近窗口，中间分成多个位置段、按比例分配淘汰数量，再按分数挑选。这样避免单纯 global top-k 把某些位置段全部挤掉；它仍不是语义完整性保证。

压缩过程为：

1. 物理 KV 达到 `kv_budget + recompute_window` 阈值，并满足请求输出长度门槛。
1. 对采样层打分，聚合，并在 TP ranks 之间同步，产生共享的保留位置。
1. 将每个 full-attention 层所选 K/V gather 到私有目标块，返回缩短后的物理长度。
1. worker/scheduler 通过 transaction 确认切换，释放旧引用；async 已排队批次仍需的源块延后释放。
1. 后续 attention 使用更短物理长度，但 RoPE/逻辑位置继续保持语义长度，不把原来的第 30000 token 改成第 4608 token 的位置。

当前 Qwen3.5 示例：budget 4608，recompute window 1024，保护前 128/最近 512，8 个位置段，打分层 stride 2，`max_tokens`
请求上限低于 2048 时跳过压缩。请求上限不等于实际生成长度。Qwen3.5 只压 full-attention 层，Gated-DeltaNet 状态留给宿主。

## 它能省什么、不能保证什么

可以减少后续 decode attention 读取的 KV token 数，并使部分原占用块重新可用。是否降低进程已预留 HBM、增加可服务并发或避免 OOM，不能仅由 token budget
推出；还要看预分配 pool、共享源块引用和私有目标块所需瞬时容量。

主要代价是打分、选择、TP 同步、搬运以及 transaction 协调。短输出很难摊薄这笔成本，因此有较高的请求输出门槛。删除的 KV 不会由一次精确 verifier
补回；长程事实、代码依赖或检索所需 token 可能被误删。

APC 不是可以无条件照常使用：符合压缩条件的请求会按 method 声明限制前缀复用，保证尾部窗口执行。当前设计已经有 APC/async 的专项适配，不应只凭 README
早期通用启动例子断言它们一概不支持。KV transfer/PD、量化 KV、多种并行组合仍有拒绝门槛；MTP2 默认拒绝，实验需显式开启。

## 已核对的冻结证据

2026-10-17，Qwen3.5-35B-A3B BF16，2×910B2/TP2，APC、真实 MTP2、async、align、FULL_AND_PIECEWISE；SWE C4，900
秒。冻结 JSON 明确保留了三个候选，不只成功的一项：

| 候选                               | 输出吞吐/芯片变化 | decode tokens/s P90 变化 | TTFT P95 变化 |
| ---------------------------------- | ----------------: | -----------------------: | ------------: |
| 8064 budget，512 输出门槛          |            -1.71% |                   +1.50% |       +17.15% |
| 4608 budget，1024 输出门槛         |            -0.44% |                   +1.22% |       +17.06% |
| 4608 budget，2048 输出门槛（选中） |            +1.33% |                   +1.62% |        +8.65% |

decode 指标是 tokens/s 分位数，不能写成 decode P90 延迟下降。选中候选仅一轮，不能证明重复性或显著性。其服务生命周期记录 20 次 scheduler commits、40
次 TP rank acknowledgements；不把这个计数强行等同于测量窗口内压缩请求占比。配置上下文 262144，但实测最长 prompt 52505。

质量证据为 32K 检索 fixture 4/4 正确、四次压缩，作者明确不等于真实 SWE 语义质量。性能 baseline 的超大 budget 使压缩 commits
为零，不应随意解释成完全卸载插件的原生 baseline。

## Lumi 的初步判断（非 Fletcher 已定 comment）

机制有实际作用，值得与“只改配置但未生效”的 MOD 区分；对长上下文、长生成、attention KV
带宽/容量受限场景有合理研究动机。但当前插件尚未证明足以采用的服务收益：收益微小且只一轮、TTFT 退化、质量缺口大，并且配置主动绕过短请求。不能把原论文在另一模型/平台的结果直接继承为 Ascend
插件收益。

在要求精度保证的部署中，目前不推荐采用这条有损路径；是否保留为质量可评估的研究方向，留待 Fletcher 判断。本轮不启动实验。

## 固定来源

- [原论文](https://arxiv.org/abs/2604.04921)：算法动机，不是本插件验收。
- [打分](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/3ddff1c2afee1eccc3d1b65def61167752298f5f/src/vllm_ascend_kvcompress/methods/triattention/scoring.py)
- [选择/物理搬运调用](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/3ddff1c2afee1eccc3d1b65def61167752298f5f/src/vllm_ascend_kvcompress/methods/triattention/method.py)
- [scheduler transaction](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/3ddff1c2afee1eccc3d1b65def61167752298f5f/src/vllm_ascend_kvcompress/stateful.py)
- [方法合同](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/3ddff1c2afee1eccc3d1b65def61167752298f5f/docs/design/methods.md)
- [冻结 v0.10 汇总证据](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261017-v010-swe-pair.json)

## 打分、搬运的执行时机与摊销条件

源码链路：`plugin.sample_tokens` 先调用原始 `sample_tokens`，随后在返回 output 前调用
`provider.compress_scheduled_requests`；后者逐个处理 scheduler 授权的压缩计划，调用 `method.compress`。因此不是插在每个
transformer 层的 forward 内，而是 attention/model 执行后的采样包装路径，属于 step 的串行附加工作。首次压缩要求完整 prompt
结束；后续达到物理长度阈值可再次压缩，并非每 token 都压。

`compress` 内顺序为：打分 → 归一化/聚合 → TP all-reduce(MAX) → 分段 top-k 与位置排序 → 构造 source/destination slots →
逐层物化。0.10 将多个打分层的分数放入连续 workspace，合并为一次 TP collective；这减少 collective 次数，不消除同步依赖。

`cache.materialize_token_slots` 对每层依次执行：gather K 到持久 K workspace、gather V 到持久 V workspace、再把
workspace 写入私有目标槽位。连续 cache 使用 `index_select`/`index_copy_`，不是只改 block table，也不是 CPU
offload。仅就理想字节流量，保留 K+V 数据会经历源读、workspace 写、workspace 读、目标写；不计缓存命中时约四倍保留数据大小，还未计打分读取和临时张量开销。

本轮读取的压缩路径没有插件自建后台线程或独立 NPU stream/event pipeline，也未看到 KV Tiering 那样的显式 device-wide
synchronize。应准确描述为“调用路径串行、设备算子按当前执行上下文排队”，不应误说每个 PyTorch 调用都让 CPU 等到设备执行完成。async scheduling
支持是事务与在途批次的兼容，不等于压缩开销被自动隐藏。

收益确有机制基础：后续 full-attention metadata 采用更短物理长度，因此可以少读 KV；GDN 和其余计算并不同比下降。经济性须满足：

`压缩后到下次压缩/请求结束前的累计 attention 节省 > 打分 + 选择 + TP 通信 + 物化 + 状态协调开销`。

不能用压缩率直接代替服务收益。如果剩余输出短、attention 占比低或反复压缩的成本太大，就可能亏本。现有约 1% 的单轮服务收益不能独立量化各阶段的摊销，尚缺我们核实的分阶段设备时间与 HBM
流量证据。

[采样包装入口](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/3ddff1c2afee1eccc3d1b65def61167752298f5f/src/vllm_ascend_kvcompress/plugin.py)；[provider 时机](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/3ddff1c2afee1eccc3d1b65def61167752298f5f/src/vllm_ascend_kvcompress/provider.py)；[gather/scatter 实现](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/3ddff1c2afee1eccc3d1b65def61167752298f5f/src/vllm_ascend_kvcompress/methods/triattention/cache.py)。

## 共同评审 comment（2026-10-26）

**打分器和数据搬运有额外成本，然后还有精度损失，历史实测收益非常有限，做这个事情没什么性价比。**

此为当前实现及已读证据下的采用判断。算法有损、不保证精确 full-attention 语义；本轮未独立测量具体任务的精度下降幅度。
