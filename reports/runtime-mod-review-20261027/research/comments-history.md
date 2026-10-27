# 共同评审 comments

记录 Fletcher 的研究判断和取舍；与源码事实、作者实验及我们的实测分开。Comments 仅保存在本仓库，不向上游 issue、PR 或其他公共渠道发布。

## SimLLM — 不采用其近似 KV 匹配路径（2026-10-26）

Fletcher：**“它的 KV 匹配完全不靠谱，只在新闻之类的场景有点用，支撑不起任何要求精度的场景。”**

结论：不推荐给要求精度保证的任务。新闻类、容忍近似的场景仅是可能探索方向，未作实验验证；不把该判断扩展到严格相同前缀缓存。[机制与证据](simllm/principles.md)

## Pipeline Microbatch — PASS（2026-10-26）

Fletcher：**“这玩意也没有用，现在 PP 不是主流技术。pass。”**

结论：不再投入深入研究或复现。该方向不符合当前关注的部署与投入优先级，已读配置也未显示性能收益；这是研究取舍，不是对所有 PP
部署价值的普遍否定。[机制与证据](pipeline-microbatch/principles.md)

## 跳过范围

- BetterScale：Fletcher 自己开发，本轮跳过，不代表负面评价。
- BidKV：按 Fletcher 要求跳过，本轮不作技术评价。

## KV Materialization Arrival Control — 重算亏本（2026-10-26）

**我们的 comment：重算亏本。**

语境：已有可复用缓存，且有在线流量和其他活跃任务时，不考虑主动放弃缓存去重算。

论据：[SGLang HiCache: Fast Hierarchical KV Caching with Your Favorite Storage Backends](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)。读者自行参阅。

## KV Tiering — 生产不可用（2026-10-26）

**我们的 comment：设备搬运侧都是同步搬运，对于计算的干扰非常大，生产不可用。**

适用范围：本轮审阅的 Ascend `torch_sync` 设备搬运路径；不扩展到未审阅的 CUDA/XPU
宿主实现。同步复制与设备同步由源码确认；干扰程度与生产采用结论是我们的评审判断，并非本轮性能实测。[传输链路分析](kv-tiering/principles.md)

## Ascend KVCompress — 性价比低（2026-10-26）

**我们的 comment：打分器和数据搬运有额外成本，然后还有精度损失，历史实测收益非常有限，做这个事情没什么性价比。**

这是对当前有损 KV 淘汰实现及已读历史证据的采用判断。精度损失指其不保留精确 full-attention
语义；本轮未独立测量具体任务掉分幅度。[机制、执行开销与历史结果](kvcompress/principles.md)

## DiffSpec — 不采用（2026-10-26）

**我们的 comment：没有用，草稿接受率太低了。**

针对已读 Qwen TP4 配置：历史验收接受率 19.29%，相对 target-only 性能退化，因此不继续投入。不是所有模型与 draft
配对的普遍结论，本轮未重测。[机制与历史结果](diffspec/principles.md)

## MuxPD — PASS，吞吐不合格（2026-10-26）

**我们的 comment：没有用，直接枪毙。用 mixed/chunked prefill，并在容量与延迟预算允许时扩大 mixed
batch，比按层暂停这条路线更值得投入；当前吞吐完全不合格。**

Fletcher 的判断：没有看到放弃 mixed 的充分理由。按层暂停需要保留中间状态，又增加调度开销，得不偿失，不继续研究或复现。

技术口径：保留的是完整 prompt 在当前层的 hidden states/residual 等中间激活，不是额外生成 tokens；不意味着每次暂停必然发生 host 搬运。Mixed
batching 与 chunked prefill 可组合，但 mixed 本身不保证消除任意长 P 的阻塞，扩大 batch
也要受显存和出词延迟约束。这些边界不改变本轮采用结论。已读作者实用原生参考下 mixed 吞吐 −38.254%，不符合我们的效率要求。[机制与证据](muxpd/principles.md)

## OPset — 去清零有用，融合需区分证据（2026-10-26）

**我们的 comment：GEMM 多余清零删除还是有用的。GEMM+SwiGLU 不能仅凭融合就认为有访存收益，Ascend Cube/Vector 协作仍要考虑 GM 路径与有效流水。**

保留作者指定配置下的正收益数字，但不把它解释成已证明消除 GM 中间读写。尚未检查编译后 IR/设备流水来确定收益来源。[机制、数据与修正](opset/principles.md)

## OPset — 最终取舍：融合不采用，去清零应是小补丁（2026-10-26）

**我们的 comment：普通 ACLNN 分开跑就更快了，这个 GEMM+SwiGLU 融合对我们的部署没有采用价值。缓冲区用 empty 而不是 zeros 确实有用，但不值得单独做成一个
MOD，就是一个小补丁的事。**

依据：ascend180 物理2/7的独立算子测试中，融合耗时为普通ACLNN分开执行的1.48–1.82倍；收益只相对较慢的batch-invariant
Triton基线成立。我们认为该基线不足以支持普通部署的优化价值，不推断作者有意挑弱基线，也未证明ACLNN满足相同batch-invariant合同。这里讨论的融合准确名称为GEMM+SwiGLU，而非ReLU。[实测与限制](opset/reproduction/README.md)

## Core Attention Boundary — 收益不清（2026-10-26）

**我们的 comment：目前缺 benchmark，收益不清，没有看到什么实质性的实现。**

这里的“没有看到实质性实现”指未见有证据支撑的实质性优化；仓库确有CPU元数据边界辅助函数替换，不是完全没有代码。[源码分析与证据状态](core-attention-boundary/principles.md)

## GDN State Codec — 参考 BetterScale 既有 LeapQuant 实验（2026-10-26）

**我们的 comment：可以参考 BetterScale 里做过的 LeapQuant 实验。此前已经做过相当详细的工作，在 Ascend 上拿不到稳定收益，不重复读论文或重做这轮研究。**

Fletcher 补充背景：这就是此前在 BetterScale 探索过的方向，同学将其集成并做成 MOD；主要架构障碍是 Ascend AIV/AIC 无法共享
SRAM，协作的数据搬运成本使该方向不容易获得收益。这是 Fletcher 对既有工作的回顾，本轮未重新查阅 BetterScale
实验材料，也未独立复核其全部结论。[当前仓库机制与证据](gdn-state-codec/principles.md)

## Prefix Router — 有需求，值得做（2026-10-26）

**我们的 comment：缓存感知路由确实有需求，也值得做；最好连接分布式 KV 存储，优先复用已有缓存，而不是重算。**

方向判断：不应只在“路由到已有本地缓存的节点”和“换个节点重新prefill”之间选择，应与分布式KV存储结合，把可获取的缓存位置、传输成本与节点负载纳入决策。当前仓库主要实现本地缓存事件索引与请求路由，不代表已具备分布式KV传输能力；此comment认可方向，不等于当前版本通过生产验收。[机制与现状](prefix-router/principles.md)

## PyramidKV — 当前整体性价比不足（2026-10-26）

**共同判断（Fletcher认可）：有实质实现和测量，但KV占用减少没有转化为整体收益；额外打分、搬运及精度代价使当前方案不划算，与此前KVCompress的性价比问题一致。**
已读固定配置长输入吞吐下降2.6%–10.3%，不推广为所有未来实现的定理。[机制与证据](pyramidkv/principles.md)

## Quantized KV Cache — 不看好额外量化/反量化往返（2026-10-26）

**我们的 comment：由于 Ascend AIC/AIV 的限制，需要主动量化/反量化的实现通常没什么用；除非能做到产出与消费都保持量化状态，否则只是白白增加 GM 读写。**

这是我们的架构与投入判断，重点反对独立量化/反量化、物化稠密中间结果带来的额外 GM 往返，不将其写成所有量化路径必然负收益的实测定理。当前 KIVI INT4 的 gather→反量化→稠密
attention 路径直接体现此问题；INT8 的 fused attention online antiquant 不等同于显式全历史还原，但是否真正减少总 GM
流量、抵消写入量化成本仍需物理路径证据，不能仅凭“融合”认定收益。[机制与证据](quantized-kv/principles.md)

## vSpec — 35B独立实测更新：有收益，不能按原先预期一概否定（2026-10-27）

**这是依据新实测记录的整体配置观察，不代表通过验收；当前意见见下方更新。**
用户指定的Qwen3.5-35B-A3B/MTP路径，在ascend87固定56题三轮交错复测中，vSpec相对原生target-only的耗时中位数比为1.53×，相对原生固定MTP2仍为1.17×；按输出tokens/s也有相近收益。此负载不支持“没有加速”或“接受率太低所以没有价值”的判断。

固定280题得分为原生203、原生MTP2
202、vSpec193；配对区间跨0，约三分之一题截断，且原生56题重复本身出现40/40/37的分数波动。尚不能把分差可靠归因于vSpec，也不能据此证明精度无损。此结论只覆盖35B/MTP与本机栈，不推广到vSpec所有proposer；不自动扩全量。[完整结果与证据边界](speculative-mmlu-pro/results-qwen35.md)

## DiffSpec — 链路失败与有效短对照分开报告（2026-10-27）

先前“不采用/接受率低”的comment是历史评审取舍。此次独立B32测试中，原生Eagle与DiffSpec短检查共享连续token0退化；原生Eagle的280题结果已严重受其污染，不能把低分归罪于DiffSpec插件。已停掉无效的DiffSpec长跑与性能比较，保留失败证据并做小型定位。另确认当前默认auto/linear检索门禁不会执行attention-score
retrieval；不把只加载了proposer/cache就说成完整检索机制生效。[过程](speculative-mmlu-pro/progress.md)

后续将缓存预算降至0.4并统一B16后，三臂56题首轮均无退化、无抢占。原生/Eagle/DiffSpec输出吞吐分别约232.1/95.6/91.3
tokens/s，接受率后两者约16.9%/14.7%；当前默认artifact在该负载未显示加速，但仅一轮，且检索计数仍为0，不泛化为完整检索思路的否证。旧C04/B32质量轮Eagle发生1488次抢占，已停止继续同配置长跑。DiffSpec有效三臂280题尚未完成；Fletcher于2026-10-27决定停止追加实验，保留现有证据，不补跑或扩大结论。[当前有效结果与未完成范围](speculative-mmlu-pro/results-diffspec.md)

## vSpec — 当前状态不合格（2026-10-27，最终评审更新）

**Fletcher 的 comment：当前状态仍不合格。** 作者对投机解码 graph 还做了 runtime 优化，现有 vSpec 与原生 MTP
对照没有控制这一变量，不能把整体提速归功于自适应投机策略。280 题准确率从原生 72.50%、MTP2 72.14% 降到 68.93%，分别下降 3.57、3.21
个百分点；这一质量表现不予接受。

保留整体配置的速度实测，不将其改写成“完全没有加速”；但性能归因缺少控制变量，精度亦未通过验收。截断、配对区间和原生重复波动仍限制因果结论，不宣称已经证明算法必然损失精度。不追加实验。

## OPset — ACLNN为验收基线：融合不合格（2026-10-27）

**Fletcher 的 comment：直接比较融合与普通 ACLNN 分开执行，不以较弱 Triton 为验收基线。两卡三个shape的融合时延增加48.36%–82.30%，不合格。**
百分比为(融合/ACLNN−1)×100%，按实测汇总表计算。去多余清零仍是有用的小补丁，不与融合结果混为一谈。[两卡实测](opset/reproduction/README.md)

## DiffSpec — 当前评语（2026-10-27）

**性能收益不合格，需要重新评估项目可行性。**

基于当前有效 B16/C04
三臂56题首轮；不是完整280题结论，也不是对完整检索算法的否证。停止追加实验的决定不变。[有效证据](speculative-mmlu-pro/results-diffspec.md)

## StateAxis — 三项评语（2026-10-27）

- **Async KV Transfer：原理上正确。** 需要适配 Qwen3.5-35B-A3B，并找到合适的缓存压力 workload 做进一步验证。
- **Automatic Prefix Cache：有潜力。** 主要潜力来自 recurrent state checkpointing，可在特定数据集上支持前缀匹配；需要高效的异步
  recurrent state checkpointing，不能干扰计算。这是待验证方向，不声称当前索引已完成这一能力。
- **Pipelined Weight Loading：很实用的小技术。** 建议推广到系统默认主线，做成 default-in（默认启用）。

以上是方向和采用建议，不自动启动新的适配或设备实验。[机制与证据边界](stateaxis-data-paths/principles.md)
