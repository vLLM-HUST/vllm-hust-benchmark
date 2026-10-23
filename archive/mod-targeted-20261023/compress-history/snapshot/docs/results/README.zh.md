# 结果与证据

[English](README.md) | 简体中文

[HTML 榜单](benchmark-leaderboard.html)读取[evidence](evidence)中的聚合、
带版本记录。页面区分 Frontier 实测点与历史实验，不是 vLLM-HUST 官网榜单。
运行 `python scripts/kvcompress_leaderboard.py` 生成 JavaScript 数据包，
再用 `python scripts/kvcompress_leaderboard.py --check` 核对。

## 已发布 0.9.0 的本地验证

2026-10-08 的 Qwen3.5-35B-A3B BF16/TP=2 SWE C4／900 秒配对有效，
0 个失败请求；60 次压缩提交均由两个 TP rank 确认。B0 到 B1 的每卡输出吞吐
**76.648 → 74.572 token/s（−2.71%）**，逐请求 P90 解码速度
**47.920 → 48.424 token/s（+1.05%）**。当前 SWE 目标的两项指标仅一项
提升。实际最长 prompt 为 52,505 token。详见[配对证据](evidence/kvcompress-qwen35-20261008-v090-swe-pair.json)
及[完整历史过程](../archive/results/frontier-benchmarking-0.9.zh.md)。

同次 AgentX 配对的解码 P90 提升，但输出吞吐基本持平，属于另一 cohort。
其[配对证据](evidence/kvcompress-qwen35-20261008-v090-agentx-pair.json)
保留原报告哈希及限制。两组都不是官网正式提交，也不能证明可重复性。

## 0.10.0 本地验证

2026-10-17 在固定宿主上完成 Qwen3.5-35B-A3B BF16/TP=2 SWE
C4／900 秒匹配配对：两侧均有效、0 个失败请求、窗口内各完成 218 个请求。
B0 到所选 B1 的每卡输出吞吐 **74.426 → 75.418 token/s（+1.33%）**，
逐请求 P90 解码速度 **47.843 → 48.616 token/s（+1.62%）**；
TTFT P95 则从 **1561 增至 1696 ms（+8.65%）**。候选采用
4608-token KV 预算、1024-token 重算窗口、stride 2 和 2048-token 请求
输出门槛。服务整个生命周期有 20 次压缩提交，均由两个 TP rank 确认。
实际最长 prompt 为 52,505 token。详见[聚合证据](evidence/kvcompress-qwen35-20261017-v010-swe-pair.json)
和[优化记录](../optimization/triattention-tp-score-batching-20261017.zh.md)。

另两种候选设置的解码 P90 提升，但每卡吞吐分别下降 1.71% 和 0.44%；
同一证据记录保留了它们的指标与报告哈希。所选候选通过四项带真实压缩的
确定性 32K 检索检查，但这不能代表 SWE 任务成功率。这只是一组本地配对，
不是可重复性证明或官网榜单提交。HTML 榜单保留不同模型
revision 的历史实测 cohort，不与本组结果混合。
