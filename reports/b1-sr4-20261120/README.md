# B1-sr4 共享专家 MatMul–SwiGLU 场景实验

B1-sr4 基于冻结 B1-r8，接入周升晖
[固定的共享专家融合内核](https://github.com/vLLM-HUST/vllm-hust-weight-layout-research/tree/28e7e3e09053b3b1d5c5b8daf5d33a65bfb5f811/evidence/fusion)。双卡
Ascend 910B2（物理 1、6）、BF16 Qwen3.5-35B-A3B、TP2、E36/R54、MTP2、FULL，Host State 4 GiB/rank，KV 20
GiB/rank。独立作用范围为 BF16 小批量共享专家，双 rank 在 C1 和两轮 C8 正式流量均重放融合图。

场景型实验冻结。两轮同卡 C8 的 4K 端到端尾延迟下降，16K 吞吐提升，16K 首字 P95 两轮均回退。MMLU-Pro 与 FrontierScience 答对数下降。保留为 SR
实验记录，常规 r 准入未通过。 4K C8 的端到端 P95 两轮下降 3.17%、2.53%，P99 下降 6.47%、4.75%。16K C8 吞吐两轮增加 1.16%、0.79%，首字 P95
增加 1.57%、7.31%。

| 负载           | 轮次  | r8 → SR4 输出 tok/s |  E2E P95/P99 变化 |            答对数 |
| -------------- | ----- | ------------------: | ----------------: | ----------------: |
| MMLU200        | —     |   1445.91 → 1499.12 |   +5.39% / +1.81% | 147/200 → 143/200 |
| HLE200         | —     |   1029.14 → 1039.67 |   +0.37% / +2.50% |   52/200 → 60/200 |
| Frontier160    | —     |    988.82 → 1004.34 |   +0.39% / +0.87% |   66/160 → 62/160 |
| SWE-bench Pro  | —     |     136.70 → 110.74 | +13.98% / -55.23% |                 — |
| Terminal-Bench | —     |     217.13 → 243.29 |   +9.51% / +1.99% |                 — |
| 4K / C8        | Run 1 |     386.95 → 387.08 |   -3.17% / -6.47% |                 — |
| 16K / C8       | Run 1 |     159.63 → 161.48 |   +0.68% / +1.70% |                 — |
| 4K / C8        | Run 2 |     386.95 → 387.79 |   -2.53% / -4.75% |                 — |
| 16K / C8       | Run 2 |     159.63 → 160.89 |   -0.77% / -2.34% |                 — |

所有配对复用同卡封存 r8。C1 仅有候选值，4K/16K 分别为 99.59/66.30 tok/s，117/78 条请求全部完成。五套共 672 条推理和 360
条独立判分请求，传输零失败。HLE 两臂各有 1 条未决，FrontierScience 从 8 条增至 9 条。SWE 和 Terminal
是固定轨迹回放，统计服务性能。公开数据仅含统计值、运行身份和校验摘要。

[完整标量和校验索引](summary.json) · [网页发布数据](../../website-publication/data/sr-development-results-v1.json)
· [私有 Engine Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/b1-sr4-20261120)
