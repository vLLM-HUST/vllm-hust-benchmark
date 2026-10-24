# 最新软件栈 GSM8K FULL Graph 性能报告

测试时间：2026-09-04 UTC<br>
插件版本：0.12.0

> Draft 后续已完成性能恢复。安装版 `0.12.1` 使用正式 CLI 参数复测达到
> `2357.96 output tok/s`，详见
> [`draft_latest_stack_performance_recovery_20260904.md`](draft_latest_stack_performance_recovery_20260904.md)。
> 下文 `1715.70 tok/s` 保留为回退现场，不再代表当前推荐配置。

## 结论

- vSpec EAGLE Online Adaptive 达到 `2296.65 output tok/s`，相对 Target-only
  Graph 为 `1.4599x`，相对固定 EAGLE gamma=2 提升 `5.51%`。
- 固定 EAGLE gamma=2 达到 `2176.62 output tok/s`，相对 baseline 为
  `1.3836x`。
- 固定 Draft gamma=2 达到 `1762.50 output tok/s`，相对 baseline 为
  `1.1204x`。
- Draft Online Adaptive 达到 `1715.70 output tok/s`，相对 baseline 为
  `1.0906x`，但比固定 Draft gamma=2 低 `2.66%`。在这组 200 prompt 短测中，
  在线探索和切换开销没有被后续稳态完全摊薄。
- 五组 GSM8K 准确率为 `92.0%` 至 `94.0%`。本报告以吞吐为主指标，准确率
  作为输出质量监控项。

## 软件栈

| 组件 | 版本或 revision |
|---|---|
| Python | 3.11.16 |
| CANN | 9.1.0 |
| vLLM-HUST | `762f85b311fbab0bcf8921dd216f5093cd58b9b8` |
| vLLM-HUST distribution | `0.17.2rc1.dev5871+g762f85b31.empty` |
| vLLM-Ascend-HUST | `4e57439e58ed3d78e675f9fd7b4614fb183c5394` |
| vLLM-Ascend-HUST distribution | `0.25.1rc1+hust.20260903.4` |
| vLLM-HUST-vSpec | `0.12.0` |

## 测试口径

- 数据集：GSM8K 固定 manifest，前 200 条 prompt。
- Target：`Qwen2.5-14B-Instruct`。
- Draft：`Qwen2.5-0.5B-Instruct`；EAGLE：`Eagle-Qwen2.5-14B-Instruct`。
- 单卡串行：Target TP=1、Draft TP=1；没有使用 TP2。
- Batch=128，`max_tokens=512`，`max_model_len=1024`，
  `max_num_batched_tokens=33280`。
- BF16、greedy、seed=0、prefix caching、async scheduling。
- Target-only、Draft 和 EAGLE 均为 `enforce_eager=False`、FULL Graph。
- 动态 Draft 搜索 gamma 1 到 4；动态 EAGLE 搜索 gamma 1 到 2；每次最多改变一级。
- 生成计时不包含模型加载、编译和 Graph capture。

## 正式结果

`吞吐加速` 使用 output tokens/s 相除。由于不同运行的停止位置存在少量数值差异，
输出 token 总数并不完全相同，因此另列 `纯耗时比` 作为辅助指标。

| 方法 | Gamma | 耗时 (s) | 输出 token | 输出吞吐 (tok/s) | 吞吐加速 | 纯耗时比 | GSM8K | 与 baseline 逐条哈希一致 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Target-only Graph | 0 | 38.1468 | 60010 | 1573.13 | 1.0000x | 1.0000x | 188/200 (94.0%) | - |
| Draft fixed | 2 | 34.2815 | 60421 | 1762.50 | 1.1204x | 1.1128x | 186/200 (93.0%) | 110/200 |
| Draft Online Adaptive | 1-4 | 35.0335 | 60107 | 1715.70 | 1.0906x | 1.0889x | 187/200 (93.5%) | 107/200 |
| EAGLE fixed | 2 | 27.6681 | 60223 | 2176.62 | 1.3836x | 1.3787x | 184/200 (92.0%) | 97/200 |
| EAGLE Online Adaptive | 1-2 | 26.1856 | 60139 | 2296.65 | 1.4599x | 1.4568x | 186/200 (93.0%) | 106/200 |

## 动态控制行为

Draft Online Adaptive：

- 全程选择计数：gamma1=`2`、gamma2=`187`、gamma3=`131`、gamma4=`3`。
- gamma 切换 `5` 次，token-level acceptance 为 `97.47%`。
- B128 稳态选择 gamma=2；该桶测得 gamma2 平均 reward 为 `2.0185`。
- 动态吞吐是固定 gamma=2 的 `97.34%`。这组结果不支持用动态 Draft 替代已知
  固定最优档，但动态控制仍能在没有离线 profile 的情况下取得 baseline 加速。

EAGLE Online Adaptive：

- 全程选择计数：gamma1=`5`、gamma2=`363`。
- gamma 切换 `4` 次，token-level acceptance 为 `91.99%`。
- B128 桶 gamma1/gamma2 平均 reward 分别为 `1.9622`/`2.6239`，控制器稳定选择
  gamma=2。
- 动态吞吐是固定 gamma=2 的 `105.51%`，达到本轮最佳结果。

## 正确性说明

B2/N2/`max_tokens=16` 冒烟中，Draft 和 EAGLE 的输出均与 Target-only 逐条哈希
一致。B128 正式测试中，BF16 FULL Graph 的批量执行顺序会造成少量 argmax 分歧，
所以不能用逐条哈希完全一致作为吞吐测试门槛。正式组仍使用同一 prompt manifest、
greedy 参数和 seed，并单独记录 GSM8K 准确率。

## 最新宿主兼容修复

本轮真实 NPU 测试补齐三项最新版宿主 API 兼容：

1. EAGLE metadata cache 同时支持最新版两参数接口和旧版可选参数接口，并兼容
   单卡 runner 不再暴露 `pcp_size`/`dcp_size`。
2. Adaptive scheduler 在最新版删除 `_get_dynamic_sd_batch_size` 后，Draft/EAGLE
   使用实际调度请求数；旧宿主存在该 hook 时继续调用旧逻辑。
3. 最新版 `ACLGraphWrapper.__init__` 删除 `is_draft_model` 后，事件排序包装层按
   原始宿主签名过滤参数；旧版仍保留原参数。

每次正式 Graph 测试使用独立 `VLLM_CACHE_ROOT`。当前 CANN 9.1 栈首次编译会出现
AOT 函数无法落盘的警告，但不影响首次运行；不要复用该次运行产生的不完整 AOT
目录。

## 结果文件

- `benchmark_results/latest_stack_target_baseline_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/latest_stack_draft_fixed_g2_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/latest_stack_draft_online_g1to4_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/latest_stack_eagle_fixed_g2_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/latest_stack_eagle_online_g1to2_full_b128_gsm8k_n200_max512.json`

同名 `.log` 文件保留完整启动配置、编译、Graph capture、动态控制汇总和退出日志。

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
