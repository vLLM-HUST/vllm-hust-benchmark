# vSpec Adaptive NPU 验证报告

日期：2026-09-02

## 测试配置

- 数据集：GSM8K，固定 manifest 前 200 条
- Target：Qwen2.5-14B-Instruct，BF16，单 NPU
- Batch：128；`max_tokens=512`；`max_model_len=1024`
- 模式：FULL Graph、async scheduling、prefix caching
- Draft：Qwen2.5-0.5B-Instruct
- EAGLE：Eagle-Qwen2.5-14B-Instruct
- EAGLE 固定/动态两侧都启用 metadata cache 和 Graph event ordering
- 吞吐只统计 generation 区间，不包含模型加载和 Graph capture

## 结果

### Draft

| 方案 | Gamma | 耗时 (s) | Output tokens | Output tok/s | GSM8K |
|---|---:|---:|---:|---:|---:|
| 固定基线 | 3 | 30.864 | 60,361 | 1,955.68 | 186/200 (93.0%) |
| Adaptive（离线 policy） | B<=64: 4；否则 3 | 28.506 | 59,667 | 2,093.13 | 187/200 (93.5%) |
| Adaptive（在线时延校准） | B<=64: 4；否则 3 + 在线修正 | 28.602 | 60,029 | **2,098.78** | **187/200 (93.5%)** |

最新 Adaptive 相对同轮固定基线的 output token 吞吐提升 **7.32%**，请求完成
耗时降低 **7.33%**；相对上一版 Adaptive 吞吐再提升 **0.27%**。在线校准从已完成
的稳定 decode step 获取实测延迟，对 profile 预测误差做逐 gamma EWMA 修正。

### EAGLE

| 方案 | Gamma | 耗时 (s) | Output tokens | Output tok/s | GSM8K |
|---|---:|---:|---:|---:|---:|
| 固定基线 | 2 | 24.796 | 60,171 | 2,426.61 | 186/200 (93.0%) |
| vSpec Adaptive run 1 | B=1: 1；否则 2 | 24.727 | 60,033 | **2,427.85** | **187/200 (93.5%)** |
| vSpec Adaptive run 2 | B=1: 1；否则 2 | 24.639 | 60,033 | **2,436.49** | **187/200 (93.5%)** |
| Adaptive 均值 | - | 24.683 | 60,033 | **2,432.17** | **93.5%** |

两次 Adaptive 运行都超过同环境固定基线，均值吞吐提升 **0.23%**；按 200 个
请求的完成时间计算提升 **0.46%**。相对旧报告的 2,365.56 tok/s 提升
**2.82%**。EAGLE 的收益较小，应按该工作负载的稳定小幅优化理解。

### Gamma=0 Target-only 功能验证

该组只使用固定 manifest 中实际存在的 2 条 prompt、`max_tokens=32`、B8，目的是
验证动态关闭 Draft 的状态和图路径，不作为 B128/N200 吞吐结论。

| 方案 | 模式 | 耗时 (s) | Output tok/s | Decode Graph | 输出哈希 |
|---|---|---:|---:|---:|---|
| 真实 Target baseline | FULL_DECODE_ONLY + async | 1.773 | 36.10 | 31/31 | reference |
| Adaptive gamma=0 | FULL + async | 3.355 | 19.08 | 0/31（安全 eager） | 2/2 一致 |
| Adaptive gamma=0 | FULL_DECODE_ONLY + async | 1.968 | 32.52 | 31/31 | 2/2 一致 |

两种 Adaptive 模式的 Draft input、Draft metadata 和 Draft replay 累计值均为 0。
async 路径仍执行通用 next-token state preparation，否则下一帧会丢失本轮 Target
采样结果；真正的 Draft model forward 在 proposer 边界前返回。组合 FULL 的 width-1
图与最大投机验证宽度共享 mixed descriptor，插件因此只对该模式的 gamma=0 Target
step 强制 eager。FULL_DECODE_ONLY 按 query width 区分图键，可正确回放 width 1。

### Gamma=0 同请求恢复验证

`sync` 策略在 gamma=0 期间运行一个不提交候选 token 的 gamma=1 shadow Draft，
保持 Draft KV 与 Target 同步，从而允许同一请求执行 `0 -> 1 -> 2`。该组使用两条
固定 prompt，重点是输出与状态转换，不作为正式吞吐结论。

| 方案 | 模式 | max_tokens | 耗时 (s) | Output tok/s | 输出 |
|---|---|---:|---:|---:|---|
| Target baseline | Eager | 32 | 3.643 | 9.88 | reference |
| Adaptive gamma0 sync | Eager | 32 | 3.574 | 10.07 | 2/2 哈希一致 |
| Adaptive gamma0 sync | Eager | 12 | 2.249 | 7.11 | 2/2 前 12 token 一致 |
| Adaptive gamma0 sync | FULL_DECODE_ONLY + async | 12 | 1.655 | 9.67 | 2/2 哈希一致 |

FULL_DECODE_ONLY trace 确认实际发生 `0 -> 1 -> 2`。`sync` 的 gamma=0 不是零 Draft
计算；需要完全关闭 Draft 时仍应使用 `sticky`。

### 正 gamma 动态切换回归验证

该组继续使用 2 条 prompt 的 smoke manifest、B8、`max_tokens=32`，验证 async
流水中 `gamma=3 -> 4` 的 Target/Draft 多宽度切换，不作为吞吐结论。

| 方案 | 耗时 (s) | Output tok/s | 动态切换 | 稳定 decode Graph | 输出哈希 |
|---|---:|---:|---|---:|---|
| Target baseline | 1.773 | 36.10 | - | 31/31 | reference |
| Adaptive Draft | 3.091 | 20.70 | 3 -> 4 | 11/11 | 2/2 一致 |

切换帧仍消费上一异步帧的 q4 Target 输出，同时为下一帧生成 gamma=4 Draft。
插件在 FULL_DECODE_ONLY 下为 Target query width 分别隔离 Graph 参数，并在
capture bucket 不能被当前 uniform width 整除时将该帧转为 non-uniform/eager
分发，不再触发 Ascend dispatcher 断言。切换后的 11 个稳定 decode 调用全部
Graph replay，Draft eager fallback 为 0。
该 B2 实际并发下 Draft 成本高于 Target，性能应以上面的 B128/N200 正式结果为准。

### 组合 FULL 重复回放回归

早期实现把 FULL_DECODE_ONLY 的 Target width 隔离也用于组合 FULL。trace 定位到
第二个稳定 q4 step 卡在 Target `execute_model` 内：组合 FULL 的 mixed descriptor
共享 Ascend event/update family，替换全局 graph-parameter table 会破坏重复回放顺序。

修复后组合 FULL 恢复共享 Target table，只有 Draft table 按 gamma 隔离；
FULL_DECODE_ONLY 继续使用 width 隔离。以下回归均完成：

| 配置 | 在线校准 | 耗时 (s) | Output tokens | Output tok/s | 结果 |
|---|---|---:|---:|---:|---|
| B8/N2/max8 FULL+async | 关闭 | 0.647 | 16 | 24.74 | 完成，2/2 哈希稳定 |
| B8/N2/max8 FULL+async | 开启 | 1.243 | 16 | 12.87 | 完成，哈希相同 |
| B128/N8/max32 FULL+async | 开启 | 1.935 | 256 | 132.31 | 8/8 长度 32 |

前两项开启 trace 且样本极小，不能用其耗时比较校准开销；它们只验证执行边界和
输出一致性。B128/N200 正式结论以上方 Draft 表为准。

## 已实现优化

1. Scheduler 在每个 decode step 写入 batch-uniform runtime gamma，并从验收统计
   更新 AC EWMA；支持延迟模型 Goodput 决策和校准 batch policy。
2. async scheduling 在 gamma 变化后修复下一帧 speculative placeholders，D2H
   Draft token copy 使用动态宽度并清理尾部。
3. 为每个 gamma 独立创建并捕获 FULL Draft Graph；隔离 Ascend 进程级
   `_draft_graph_params`，避免多宽度 graph handle 串用。
4. 保留精确动态 capture sizes，并按实际 token 数检查 graph bucket；过渡 step
   或不安全 shape 明确回退到 eager body。若 padded bucket 不能整除当前动态
   uniform width，dispatcher 会转为非 uniform 分发而不是触发断言。
5. EAGLE 启用 speculative metadata cache 和 ACL Graph device-event ordering。
6. Benchmark 增加 GSM8K 答案提取和 exact-match，性能调优同时检查正确率。
7. `gamma=0` 使用 sticky request cohort，跳过 Draft prefill/forward；async 保留
   next-token 状态准备，FULL_DECODE_ONLY 注册 Target query width 1 到 max+1。
8. `gamma=0 sync` 用 shadow Draft 保持 KV 对齐，支持同请求恢复；在线完成时延
   校准用逐 gamma EWMA 修正离线 Goodput 模型。
9. DFlash 原生 parallel proposer 的动态 `num_speculative_tokens`、slot width 和
   dummy-run/capture hook 已接入；profiler 按一次并行 block forward 拟合，不复用
   串行 Draft 的 gamma 次累加公式。当前本机缺匹配 checkpoint，只完成 ABI/单测。

## 调优记录

- Draft `gamma=5` 在小 batch 尾部触发 NPU 地址越界，已从发布 profiles 移除。
- EAGLE `B<=8 -> gamma=1` 为 2,410.29 tok/s，切换过早；缩到 B=2 后为
  2,425.36 tok/s，最终 B=1 达到上表结果。
- trace 模式会逐 step 写 WARNING。B=2 trace 验证了单次 `2 -> 1` 切换和后续
  gamma=1 Graph dispatch，但吞吐测试必须关闭 trace。
- EAGLE uniform-state kernel 组合为 2,168.31 tok/s，本轮未采用。

## 结果文件

- `benchmark_results/draft_fixed_g3_b128_gsm8k_n200_max512_accuracy.json`
- `benchmark_results/draft_adaptive_g3_g4_b128_gsm8k_n200_max512_accuracy.json`
- `benchmark_results/draft_adaptive_online_latency_full_b128_gsm8k_n200_max512_fixed.json`
- `benchmark_results/eagle_fixed_g2_metadata_event_b128_gsm8k_n200_max512_accuracy.json`
- `benchmark_results/eagle_adaptive_g2_g1_b1_metadata_event_b128_gsm8k_n200_max512_accuracy.json`
- `benchmark_results/eagle_adaptive_g2_g1_b1_metadata_event_b128_gsm8k_n200_max512_accuracy_repeat2.json`
- `benchmark_results/target_baseline_full_async_b8_n8_max32.json`
- `benchmark_results/target_baseline_full_decode_async_b8_n8_max32.json`
- `benchmark_results/draft_adaptive_targetonly_full_async_b8_n8_max32_v4.json`
- `benchmark_results/draft_adaptive_targetonly_full_decode_async_b8_n8_max32.json`
- `benchmark_results/draft_adaptive_position_ac_full_decode_async_b8_n8_max32_v9.json`
- `benchmark_results/target_baseline_eager_b2_transition_max32.json`
- `benchmark_results/draft_adaptive_gamma0_sync_transition_eager_b2_max32.json`
- `benchmark_results/draft_adaptive_gamma0_sync_transition_full_decode_async_b2_max12_v2.json`
- `benchmark_results/draft_adaptive_full_async_b8_n2_max8_shared_target_params.json`
- `benchmark_results/draft_adaptive_full_async_b8_n2_max8_online_calibration.json`
- `benchmark_results/draft_adaptive_online_latency_full_async_b128_n8_max32_fixed.json`

正式 profiles：

- `profiles/qwen25_draft_gsm8k_graph_b128_g4.json`
- `profiles/qwen25_eagle_gsm8k_graph_b128.json`

`benchmarks/offline_ab.py` 是固定 gamma 和 Adaptive 的同口径入口。正式测试不要
添加 `--adaptive-trace`；EAGLE 运行还需启用 `--eagle-spec-metadata-cache` 和
`--graph-event-ordering`，或设置对应环境变量。

图模式实验不要导出 `TORCHDYNAMO_DISABLE=1`；实测该变量会导致
`aot_compile is not supported`，与 FULL/FULL_DECODE_ONLY 编译配置冲突。

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
