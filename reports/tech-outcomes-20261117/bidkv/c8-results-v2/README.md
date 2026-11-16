# 默认True配置C8：完整配对数字原件

本目录使用最终v4源、默认CPU绑定True和共同拓扑修复，独立于旧False的c8-results-v1。两臂全部683条原SSE已受控独立重放：Native334（16/225/16/77）、Candidate349（16/236/16/81），0失败，client/controller/preflight真实wait均0、模型释放后进入下一阶段。公共只含数字、配置和受控原件SHA；不声称公开读取了私有SSE。

| 阶段         | Native请求 | BidKV请求 | Native tok/s | BidKV tok/s | 比值变化 |
| ------------ | ---------: | --------: | -----------: | ----------: | -------: |
| 4096-warmup  |         16 |        16 |    65.293885 |   65.600610 | +0.4698% |
| 4096-formal  |        225 |       236 |   189.513189 |  198.781805 | +4.8907% |
| 16384-warmup |         16 |        16 |    60.590861 |   61.973642 | +2.2822% |
| 16384-formal |         77 |        81 |    64.708895 |   68.301798 | +5.5524% |

**上表分母是整个phase的end−start，包含排空在途响应，分子也是包含drain的accepted输出。**
paired-summary原字段名window不表示只取固定\[0,300)的token窗口；不能把含drain
token直接除以300。原60秒预热/300秒正式提交窗口与全部在途响应保留，warmup也全部公开。这是独立固定256/ignoreEOS合成资格，不是五项有限EOS批次的统计替身。

两臂全部683条首generated与首content事件实际相同，因此本次原阶段两种计时数值一致；每臂phase均保留TTFT/TPOT样本量及(n−1)q分位数。usage缓存详情均null，未知值不写0；唯一盐及内部/外部prefix计数0另行交叉证明。

**Candidate策略实际启用，但两臂journal的calls/decisions/selections均0。**
上表正比值不能归因于BidKV选victim机制，不能作为2x或普遍性能提升证据。只有一次预注册顺序批次，没有独立重复或性能效应置信区间。

[配对摘要](paired-summary.json)、Native与BidKV各自的requests.jsonl、phases.json、independent-verification.json、execution-identity.json、policy-custody-index.json及controlled-originals-index.json均受公共manifest守卫。初始化/清理与资格推理分开，完整五项/解题质量/32K长期稳定性另论。
