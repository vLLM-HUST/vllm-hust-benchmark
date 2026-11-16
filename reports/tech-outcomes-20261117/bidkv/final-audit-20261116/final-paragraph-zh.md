BidKV：已在 yanbo-81 双 Ascend910B2 上，按冻结 MOD 协议完成每臂 672 条 solver 和同一 Native judge 的 720
条判分请求。MMLU、HLE、Frontier、SWE trace、Terminal trace 的吞吐分别为
614.569→629.913、793.276→794.148、761.031→762.812、36.152→35.545、85.393→87.391 tok/s，变化为
+2.497%、+0.110%、+0.234%、−1.679%、+2.340%；同序 content TTFT P95 改善
−2.413%、+9.832%、−0.512%、−12.331%、−5.658%，请求等权 TPOT 均值改善
+1.809%、−2.105%、−0.295%、−2.072%、+1.017%。原任务质量为 MMLU 144/200→149/200、HLE
已判正确32/200→33/200（未决94→92）、Frontier 已判成功64/160→57/160（未决8→4）；两种 trace
不测解决率。正式负载的原生抢占和策略调用均为0，以上单批波动不能归因为 BidKV
机制收益；另列压力诊断可触发，但未建立稳定提速且质量无损的结论。[结果报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9930a0b772b79c92a487509767991a9a476f110d/reports/tech-outcomes-20261117/bidkv/five-results-v1/report/REPORT.md)｜[逐请求结果与原件索引](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/9930a0b772b79c92a487509767991a9a476f110d/reports/tech-outcomes-20261117/bidkv/five-results-v1)。
