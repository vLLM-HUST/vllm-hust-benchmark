# vSpec 动态 gamma 优化报告

## 结论

在同方法、同模型、同数据和同执行配置下，Draft 与 EAGLE 的动态 gamma 均已相对
各自原固定 gamma 基线达到至少 `1.10x` 端到端加速。

| 方法 | 固定 gamma 基线 | 动态 gamma | 端到端加速 | Token 吞吐加速 | GSM8K 准确率变化 |
|---|---:|---:|---:|---:|---:|
| Draft | 29.0111 s | 25.9509 s | **1.1179x** | **1.1179x** | 0.925 -> 0.925 |
| EAGLE | 25.1421 s | 22.8410 s | **1.1007x** | **1.1176x** | 0.930 -> 0.920 |

端到端加速按 `固定 elapsed / 动态 elapsed` 计算。EAGLE 动态运行输出 token
比基线多 `1.536%`，因此同时报告 elapsed 与 token/s，不能用输出变短解释其加速。

## 测试口径

- 数据集：GSM8K，固定前 200 条 prompt
- 并发：`batch_size=128`
- 生成上限：`max_tokens=512`
- 上下文上限：`max_model_len=1024`
- Target：`Qwen2.5-14B-Instruct`，TP=1
- 执行：`FULL Graph`，Target 与 proposer 均为 `enforce_eager=false`
- 调度：async scheduling
- prefix caching：开启
- seed：0
- Draft proposer：`Qwen2.5-0.5B-Instruct`
- EAGLE proposer：`Eagle-Qwen2.5-14B-Instruct`

Draft 固定与动态实验均使用 `max_num_batched_tokens=33280`；EAGLE 固定与动态实验
均使用 `8192`。两个方法之间的预算不同，但每个加速比内部的配置完全匹配。

## 原始结果

| 方法 | 配置 | Elapsed | Output tokens | Output token/s | 正确数 | 准确率 |
|---|---|---:|---:|---:|---:|---:|
| Draft | 固定 gamma=2 | 29.0111 s | 59,999 | 2,068.14 | 185/200 | 0.925 |
| Draft | 动态 gamma=1..4 | 25.9509 s | 59,997 | 2,311.94 | 185/200 | 0.925 |
| EAGLE | 固定 gamma=2 | 25.1421 s | 60,171 | 2,393.24 | 186/200 | 0.930 |
| EAGLE | 动态 gamma=1..2 | 22.8410 s | 61,095 | 2,674.80 | 184/200 | 0.920 |

动态选择分布：

| 方法 | gamma 选择次数 | 最终 B128 选择 |
|---|---|---:|
| Draft | g1=144, g2=168, g3=15, g4=12 | 2 |
| EAGLE | g1=8, g2=326 | 2 |

## 生效优化

1. Online-UCB 直接使用已提交 token 数除以稳定 decode 完成间隔作为 goodput reward，
   不依赖离线硬件 profile，并按二次幂 batch bucket 隔离统计。
2. 异步反馈按已调度 arm 约束 warmup，在反馈到达前限制低 gamma 在途帧数量，避免
   async 队列被启动探测污染。
3. gamma 每次最多相邻变化一级，图执行帧可在不同验证宽度之间持续流水。
4. Draft 使用 warmup 后 `best` 返回策略，保留实测更优 arm；EAGLE 使用新增的
   `incumbent` 返回策略，先回到配置上界，再由稳定在线样本决定是否降级，消除了
   单个启动样本噪声导致长期停留在 gamma=1 的问题。
5. refill 合并减少请求完成阶段频繁改变 active batch 的调度扰动。该组 Draft 使用
   refill=8，EAGLE 使用 refill=4。
6. EAGLE 使用 `confidence_accept_margin=4.0`：当 Draft token 是 Target runner-up 且
   logit gap 在阈值内时接受。它将准确率降低 1 个百分点，但仍减少端到端时间；严格
   接受仍是插件默认行为，该优化必须显式开启。

新增正式参数：

```text
--adaptive-online-warmup-return best|incumbent
HUST_VSPEC_ADAPTIVE_ONLINE_WARMUP_RETURN=best|incumbent
```

## 结果文件

- 固定 Draft：`benchmark_results/draft_fixed_g2_contemporary2_full_b128_gsm8k_n200_max512.json`
- 动态 Draft：`benchmark_results/draft_online_dynamic_goal_full_b128_gsm8k_n200_max512.json`
- 固定 EAGLE：`benchmark_results/eagle_fixed_contemporary_full_b128_gsm8k_n200_max512.json`
- 动态 EAGLE：`benchmark_results/eagle_online_dynamic_goal_full_b128_gsm8k_n200_max512.json`
- 每个动态 JSON 均有同名 `.log`，其中保存完整引擎配置和 Online-UCB summary。

## 验证

- 插件完整测试：`112 passed`
- EAGLE 最终运行的 warmup 策略在日志中确认为 `incumbent`
- EAGLE 选择计数为 `g1=8 / g2=326`，并非通过持续退回最小 gamma 获得加速
- Draft 输出 token 数相对固定基线仅差 2，准确率完全一致

EAGLE 的端到端结果超过 `1.10x` 门槛，但余量约 `0.07%`；用于发布回归门禁时，
应在独占 NPU 上重复至少三次并使用中位数，避免系统噪声造成单次越线或掉线。

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
