# wl38：MuxPD 默认模式 vs native，第二轮真实文本派生 workload（2026-11-07 冻结，任何运行之前）

只跑测试，不发 PR / issue。只测默认模式；不测 ROW、不设质量门槛、不扩展 TP8 / 其他模型；不重跑 13 个合成负载和 wl37 的三个 workload。

## workload（派生负载：正文为真实文本，到达 / 比例 / 输出长度为测试设置）

| 名称 | 速率 req/s | 时长 s | 组成 | 想检验的问题 |
|---|---|---|---|---|
| wl-docchat-r1.3 | 1.3 | 120 | qasper 0.2 + OASST 0.8 | wl37 A（1.0，收益消失）与 B（1.6，收益成立）之间的转折点 |
| wl-2wiki-r1.6 | 1.6 | 120 | 2wikimqa 0.2 + OASST 0.8 | 多文档 QA（输入 p50 约 6.8K，比 qasper 长），高负载 |
| wl-hotpot-r1.0 | 1.0 | 120 | hotpotqa 0.2 + OASST 0.8 | 很长的 prefill（输入 p50 约 15K），中等负载 |
| wl-govsum-r1.0 | 1.0 | 120 | gov_report 0.2 + OASST 0.8 | 长输入 + 长输出（摘要，预算 512） |
| wl-code-r1.0 | 1.0 | 120 | lcc 0.5 + repobench-p 0.5 | 代码补全：中长输入、极短输出，prefill 为主（可能是代价场景） |
| wl-chat-r1.5 | 1.5 | 90 | OASST 1.0 | 无长 prefill 的纯对话、未过载（可能是代价场景） |

长度用服务 tokenizer + 聊天模板实测（tools/profile_lb.py）。LongBench 请求 kind=long（TTFT SLO 3 s），OASST kind=short（1 s）。
输入 + 官方预算 > 16384 的行排除并记录（hotpotqa 67 行、repobench-p 120 行、gov_report 28 行、lcc 6 行、2wikimqa 5 行、qasper 3 行），不截断。
数据与 wl37 相同的固定版本文件（SHA256 在 build_plans.py 中校验）。

## 运行（与 #36 / wl37 完全相同的服务与负载设置）

TP2 卡 0,1：MuxPD -> native；TP4 卡 0-3：MuxPD -> native；每臂依次回放 6 个 workload 各一次，keepalive 停止，图模式 G16，
max-num-seqs 16，native max-num-batched-tokens 16384。loadgen / SLO（aggregate.slo()）/ 超时与 wl37 相同。失败保留，不重跑、不换样本、不改参数。

## “效果很好”的判定规则（运行前固定，同样应用于 wl37 三个 workload）

- 效果很好：TP2 和 TP4 都满足 Goodput 提升 >= 15 个百分点、输出吞吐 >= -2%、0 错误。
- 有收益：TP2 和 TP4 Goodput 都提升 >= 5 个百分点，或两边吞吐都 >= +5%（过载场景）。
- 代价：任一 TP Goodput 下降 >= 3 个百分点，或任一 TP 吞吐 <= -3%。
- 其余为持平。无论归类如何，TTFT / ITL p50 / 最大停顿的代价都一并列出；单轮结果，不作显著性结论。
