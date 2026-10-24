# Qwen3.5-35B-A3B MTP2 GSM8K 验证报告

## 结论

vSpec 已接入 Qwen3.5 原生 MTP，并完成 Qwen3.5-35B-A3B MTP2
验证。两轮相同 GSM8K B16/N200 配对测试中，MTP2 输出吞吐分别提升 `1.073x`
和 `1.174x`，两轮平均吞吐从 `413.32 tok/s` 提升到 `463.76 tok/s`，均值比为
`1.122x`。两轮共 400 个请求均成功。

严格图模式保护和日志证据同时确认 target 与 MTP proposer 均使用图执行，测试中
没有 eager 回退。

## 固定环境

| 项目 | 配置 |
|---|---|
| 模型 | `Qwen/Qwen3.5-35B-A3B` |
| 模型 revision | `59d61f3ce65a6d9863b86d2e96597125219dc754` |
| 本地目录 | `/workspace/models/Qwen3.5-35B-A3B` |
| 模型完整性 | 14/14 safetensors，索引 1811 个 tensor，包含 785 个 MTP tensor |
| vSpec | `0.13.3`，分支 `feature/qwen35-mtp` |
| vLLM-HUST | `762f85b311fbab0bcf8921dd216f5093cd58b9b8` |
| vLLM-Ascend-HUST | `4e57439e58ed3d78e675f9fd7b4614fb183c5394` |
| Python 环境 | `/opt/vllm-hust-cann91`，Python 3.11 |
| CANN | `9.1.0` |
| torch / torch-npu | `2.13.0+cpu` / `2.13.0rc1` |
| 并行 | TP2 + EP2，NPU `0,1` |
| dtype | BF16 |

模型 `text_config.mtp_num_hidden_layers=1`。MTP2 在同一个原生 MTP head 上执行两次
forward，这是当前宿主实现和模型卡推荐配置，不需要外部 draft checkpoint。
目前没有单独名为 “Qwen3.5 Frontier” 的公开 checkpoint；本报告将 Frontier
视为该模型系列的场景名称，实际验证对象严格固定为表中的官方 35B-A3B revision。

## 测试协议

- 数据集：GSM8K materialized JSONL，固定顺序前 200 条
- 并发上限：16
- 输出长度：256
- `request-rate=inf`，`temperature=0`，`seed=0`
- target-only 与 MTP2 使用相同 target、输入、TP/EP、内存和图配置
- `max-model-len=4096`，`max-num-batched-tokens=8192`，`block-size=128`
- target：`FULL_DECODE_ONLY`
- MTP proposer：`FULL`
- MTP target capture sizes：`3, 6, 9, 12, 18, 24, 48`
- prefix caching 与 async scheduling 均关闭
- 正式测试前执行 8 条、32 输出 token 的预热

复现实验：

```bash
scripts/benchmark_qwen35_mtp2_gsm8k.sh baseline
scripts/benchmark_qwen35_mtp2_gsm8k.sh mtp2
```

## 性能结果

| 轮次 | 方案 | 完成/失败 | 耗时 (s) | 输出吞吐 (tok/s) | 总吞吐 (tok/s) | 相对 baseline |
|---|---|---:|---:|---:|---:|---:|
| 1 | Target-only Graph | 200/0 | 120.363 | 425.38 | 558.76 | 1.000x |
| 1 | vSpec MTP2 Graph | 200/0 | 112.116 | 456.58 | 599.77 | **1.073x** |
| 2 | Target-only Graph | 200/0 | 127.596 | 401.27 | 527.08 | 1.000x |
| 2 | vSpec MTP2 Graph | 200/0 | 108.698 | 470.94 | 618.63 | **1.174x** |
| 均值 | Target-only Graph | 200/0 | 123.980 | 413.32 | 542.92 | 1.000x |
| 均值 | vSpec MTP2 Graph | 200/0 | 110.407 | 463.76 | 609.20 | **1.122x** |

两轮累计墙钟时间从 `247.96 s` 降至 `220.81 s`，加速 `1.123x`，耗时减少
`10.95%`。每轮 baseline 生成 51200 个输出 token；MTP2 生成 51190 个，其中一个
请求提前遇到 EOS。因此表中主比较采用已经按实际输出 token 归一化的吞吐，而不是
只比较墙钟时间。第二轮绝对性能相对第一轮发生波动，但同轮配对加速仍为正。

## MTP 验收

| 指标 | 第 1 轮 | 第 2 轮 |
|---|---:|---:|
| Draft rounds | 18214 | 18192 |
| Draft tokens | 36428 | 36384 |
| Accepted tokens | 32962 | 32989 |
| Token acceptance | 90.49% | 90.67% |
| Mean acceptance length | 2.810 | 2.813 |
| Position 1 acceptance | 94.76% | 94.83% |
| Position 2 acceptance | 86.21% | 86.51% |

## 无图回退证据

MTP server 日志包含：

```text
Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL, use_eagle=True
Graph capturing finished in 7 secs
```

引擎配置同时记录 `enforce_eager=False`、target
`cudagraph_mode=FULL_DECODE_ONLY` 和上述 7 个 capture size。vSpec 还在运行期检查
每个 decode 批次的 target dispatcher 及 proposer runnable；任何一侧未命中完整图都会
直接报错退出。基准脚本在运行结束后再次校验日志，因此不会把静默 eager fallback
计入有效结果。

## 环境问题与修复

最初运行错误地加载了 CANN 9.0，而当前 `vllm-ascend` 扩展是在 CANN 9.1 下构建。
这会依次暴露 `npu_gemma_rms_norm`、`moe_gating_top_k` 和
`npu_causal_conv1d_custom` 缺失。切换到
`/opt/vllm-hust-cann91/Ascend/cann-9.1.0/set_env.sh` 后，原生算子全部正确注册；
本报告的性能数据没有使用 Python fallback。插件现在会在启动时同时检查这三个
原生算子，缺失任意一个都会立即终止并提示 CANN/扩展版本不匹配。

当前宿主会提示 Mamba KV cache group 无法用于跨请求 prefix-cache reuse。测试本身已
显式关闭 prefix caching，因此该提示不影响本次结果。

## 原始数据

- Baseline：`benchmark_results/qwen35_mtp2_gsm8k/b16-n200-baseline-cann91-20260927/`
- MTP2：`benchmark_results/qwen35_mtp2_gsm8k/b16-n200-mtp2-cann91-20260927/`
- Baseline repeat：`benchmark_results/qwen35_mtp2_gsm8k/b16-n200-r2-cann91-20260927-baseline/`
- MTP2 repeat：`benchmark_results/qwen35_mtp2_gsm8k/b16-n200-r2-cann91-20260927-mtp2/`

## 上游参考

- [Qwen3.5-35B-A3B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [vLLM-Ascend Qwen3.5 MoE + MTP + FlashComm1 问题](https://github.com/vllm-project/vllm-ascend/issues/7996)

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
