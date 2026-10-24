# vSpec 默认 Adaptive 回归结果

日期：2026-09-07

## 结论

插件默认开启 Adaptive，gamma 不需要在运行时显式输入；默认上限为 4，在线控制器的候选范围为 `1..4`。本轮回归中，Draft 和 EAGLE 均在 FULL Graph 下正常运行，没有出现图回退。

在统一的 GSM8K B128 测试口径下：

- Draft Adaptive：相对 Target-only 加速 `1.509x`，耗时降低 `32.83%`。
- EAGLE Adaptive：相对 Target-only 加速 `1.487x`，耗时降低 `32.28%`。

## 测试口径

| 项目 | 配置 |
| --- | --- |
| 数据集 | GSM8K，manifest 中前 200 条 prompt |
| Target | `/data/shared-models/Qwen2.5-14B-Instruct` |
| Draft | `/data/shared-models/Qwen2.5-0.5B-Instruct` |
| EAGLE | `/data/shared-models/Eagle-Qwen2.5-14B-Instruct` |
| Batch | 128 |
| 最大生成长度 | 512 tokens |
| dtype | BF16 |
| Graph | FULL，精确 capture |
| 调度 | async scheduling 开启 |
| Prefix Cache | 开启 |
| 最大 batch token | 33280，三组一致 |
| 并行 | 单卡 TP1，NPU 7 |
| 软件栈 | vLLM-HUST + vLLM-Ascend-HUST，CANN 9.1 |

性能计时只覆盖 `llm.generate()`，不包含模型加载、torch.compile 和 Graph capture；baseline、Draft 和 EAGLE 使用同一份 prompt manifest 和同一组生成参数。

## 性能结果

| 方案 | Adaptive 配置 | 耗时 (s) | 输出 tokens | 吞吐 (tok/s) | 相对 baseline | GSM8K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 关闭投机，gamma=0 | 38.1951 | 59958 | 1569.78 | 1.000x | 186/200 = 93.0% |
| Draft | Online，gamma 上限 4，候选 `1..4` | 25.6553 | 60782 | 2369.18 | 1.509x | 182/200 = 91.0% |
| EAGLE | Online，gamma 上限 4，候选 `1..4` | 25.8669 | 60376 | 2334.10 | 1.487x | 187/200 = 93.5% |

本轮结果文件：

- `benchmark_results/default_gamma1to4_regression_20260907/target_baseline_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/default_gamma1to4_regression_20260907/draft_adaptive_g1to4_full_b128_gsm8k_n200_max512.json`
- `benchmark_results/default_gamma1to4_regression_20260907/eagle_adaptive_g1to4_full_b128_gsm8k_n200_max512.json`

对应完整日志与 JSON 同名，扩展名为 `.log`。

## 与上一轮结果对比

Draft 使用了上一轮已验证的 `refill=8` 和置信度 margin `5.25` 配置；本轮吞吐从 `2357.96` 提升到 `2369.18 tok/s`，提升约 `0.48%`，耗时从 `25.7774` 降到 `25.6553 s`。GSM8K 正确率保持 `91.0%`。

EAGLE 本轮将旧的 gamma 上限 2 改为默认上限 4，因此与上一轮 `1..2` 结果不是完全同配置对比；吞吐从 `2296.65` 提升到 `2334.10 tok/s`，提升约 `1.63%`，耗时从 `26.1856` 降到 `25.8669 s`，GSM8K 正确率从 `93.0%` 变为 `93.5%`。

## 默认行为校验

当前 `vllm-hust-vspec` CLI 的默认值为：

- `adaptive_speculation=true`
- `adaptive_policy=online`
- `adaptive_min_gamma=1`
- `gamma=4`，作为 Adaptive 的最大候选宽度
- `adaptive_max_gamma_step=1`
- `adaptive_full_graph=true`
- `adaptive_async=true`

因此用户可以只选择 `--mode draft` 或 `--mode eagle`，不必强制输入 `--gamma`。如果显式传入 gamma，它只覆盖候选上限；运行时仍由 Adaptive 控制器在 `1..gamma` 范围内在线选择。

## 验证状态

- `172 passed, 15 warnings`
- Extension manifest validate 通过
- vLLM-HUST host/protocol compatibility check 通过
- 实际 NPU 回归中 Draft/EAGLE 均完成 200 条请求并产生结果文件

回归时显式加载了 `/opt/vllm-hust-cann91/Ascend/cann-9.1.0/set_env.sh`。没有设置 `TORCHDYNAMO_DISABLE=1`，因为该变量会关闭 torch.compile 并使 FULL Graph 的 `aot_compile` 初始化失败；这不是插件性能或正确性失败。

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
