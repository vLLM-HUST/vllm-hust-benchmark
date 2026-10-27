# Qwen3.5 Frontier 固定 MTP2 历史验证

## 结论

本报告记录 vSpec 0.14.2 的固定 MTP2 实验，不代表当前默认启动参数。当前版本默认开启
自动 gamma（候选 `2/4/6`）和自动 refill；B16 命令见
[`qwen35_mtp2_b16_graph_benchmark_commands.md`](qwen35_mtp2_b16_graph_benchmark_commands.md)。

vSpec 已完成 Qwen3.5-35B-A3B Frontier 固定 MTP2 路径的本地适配。历史服务保持
`TP2 + APC + MTP2 + async scheduling + FULL_AND_PIECEWISE + 256K`，没有切换到
EAGLE，也没有通过修改 gamma 获得结果。

GSM8K 200 条、并发 4 的同配置配对测试中，target-only 输出吞吐为
`171.4725 tok/s`，MTP2 为 `235.4372 tok/s`，加速 `1.3730x`。两侧输入 token
均为 16,054，输出 token 均为 51,190，200 个请求全部成功。

## 固定协议

| 项目 | 值 |
|---|---|
| Target | `Qwen3.5-35B-A3B` |
| 投机方法 | checkpoint 内置原生 MTP head |
| gamma | 固定 `2` |
| 并行 | TP2 + EP2，NPU `0,1` |
| dtype / KV dtype | BF16 / auto |
| 最大上下文 | 262,144 |
| 最大序列数 | 16 |
| 最大批 token | 8,192 |
| Prefix cache | 开启 |
| Chunked prefill | 开启 |
| Async scheduling | 开启 |
| Target graph | `FULL_AND_PIECEWISE` |
| MTP proposer graph | `FULL` |
| Target capture sizes | `3, 6, 9, 12, 18, 24, 48` |
| Adaptive | 关闭，固定 MTP2 |
| 本地 argmax reduction | 关闭 |

该协议已内置为 `qwen35-frontier-mtp2`，完整 TOML 为
[`configs/qwen35-35b-a3b-frontier-mtp2.toml`](../configs/qwen35-35b-a3b-frontier-mtp2.toml)。
协议会拒绝覆盖 method、gamma、TP、256K、APC、async 或 graph mode；模型路径、端口和
设备等部署参数仍可覆盖。

## 启动

直接启动：

```bash
source /opt/vllm-hust-cann91/Ascend/cann-9.1.0/set_env.sh
export VLLM_PLUGINS=ascend,vspec
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_AOT_COMPILE=0

vllm-hust-vspec \
  --protocol qwen35-frontier-mtp2 \
  --target-model /workspace/models/Qwen3.5-35B-A3B \
  --device 0,1
```

通过 Extension Manager 启动：

```bash
./manage.sh enable
./manage.sh run -- vllm-hust-vspec \
  --protocol qwen35-frontier-mtp2 \
  --target-model /workspace/models/Qwen3.5-35B-A3B \
  --device 0,1
```

当前本地开发分支为 `feature/qwen35-mtp`。协议不写死源码目录，已安装环境可直接使用；
源码开发时可额外传入 `--vllm-source` 和 `--ascend-source`。

## 回归命令

当前脚本默认依次运行 target-only 和自适应 MTP2，检查 token 数一致、校验图日志，并在
加速低于 `1.10x` 时返回非零状态：

```bash
scripts/benchmark_qwen35_frontier_mtp2_gsm8k.sh pair
```

也可以分别运行：

```bash
scripts/benchmark_qwen35_frontier_mtp2_gsm8k.sh baseline
scripts/benchmark_qwen35_frontier_mtp2_gsm8k.sh adaptive

# 复现本报告的历史固定 gamma=2、C4 口径
ADAPTIVE_SPECULATION=0 SPEC_GAMMA=2 MAX_CONCURRENCY=4 \
  scripts/benchmark_qwen35_frontier_mtp2_gsm8k.sh pair
```

默认测试为 GSM8K 固定顺序前 200 条、输出长度 256、`request-rate=inf`、
`temperature=0`、`seed=0`，正式测试前运行 8 条、输出长度 32 的预热。

## 正式结果

| 方案 | 完成/失败 | 耗时 (s) | 输出吞吐 (tok/s) | 总吞吐 (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target-only Graph | 200/0 | 298.532 | 171.4725 | 225.2490 | 556.38 | 21.23 | 1.000x |
| vSpec MTP2 Graph | 200/0 | 217.425 | 235.4372 | 309.2741 | 518.91 | 15.01 | **1.3730x** |

MTP2 的 token acceptance 为 `90.589%`，平均 acceptance length 为 `2.8118`；
36,394 个 draft token 中验收 32,969 个。逐位置验收率分别为 `94.884%` 和
`86.294%`。

## 256K 正确性

固定原生 MTP2（未使用 synthetic sampler）已完成 26 个长上下文检索 case：

| 项目 | 结果 |
|---|---:|
| 通过/总数 | **26/26** |
| 实际 prompt token 范围 | 8,180 - 249,995 |
| 目标长度 | 8K、16K、32K、48K、64K、80K、96K、112K、128K、160K、192K、224K、250K |
| Needle 位置 | 10%、90% |
| 总请求耗时 | 298.786 s |

测试使用与正式协议相同的 TP2、EP2、APC、async scheduling、target
`FULL_AND_PIECEWISE` 和 proposer `FULL`。26 个响应均精确返回预置 needle；服务日志
未出现 graph fallback、OOM、HCCL、context overflow、`RuntimeError` 或 traceback。
原始结果位于
`benchmark_results/qwen35_long_context/20260930-mtp2-26case/results.json`。

另有同一执行树下的 AgentX 256K C4 成对性能结果：固定 MTP2 为 `84.8839 tok/s`，
target-only 为 `52.6266 tok/s`，提升 `1.6129x`。该 AgentX 性能运行使用
SPEED-Bench coding AL 驱动的 synthetic sampler，不能替代本节的原生正确性结果；
完整口径见 [`agentx_256k_benchmark.md`](agentx_256k_benchmark.md#本地配对结果2026-09-30)。

## 并发边界

在相同服务配置上用 64 条、每条 256 token 做并发扫描：

| 最大请求并发 | Baseline (tok/s) | MTP2 (tok/s) | 加速 |
|---:|---:|---:|---:|
| 4 | 182.1091 | 234.1928 | **1.2860x** |
| 8 | 320.9424 | 354.7419 | **1.1053x** |
| 16 | 575.3663 | 569.7051 | 0.9902x |

因此当前收益区间是低到中等并发；TP2 在并发 16 已接近 target 饱和，额外 MTP forward
会抵消验收收益。服务仍保留 `max-num-seqs=16`，正式的 `1.3730x` 结果将客户端并发固定
为 4。该边界必须在部署容量规划中保留，不能把 C4 结果外推到所有并发。

## 正确性与图证据

- Target 启动日志同时包含 `mixed prefill-decode, PIECEWISE` 和 `decode, FULL` 捕获，
  7 个 MTP capture size 全部完成。
- Proposer 日志包含
  `Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL`。
- vSpec 的严格图保护会检查 target dispatcher 和 proposer runnable；任一 decode 批次
  回到 eager 都会立即报错。正式测试未触发该保护。
- TP2 hybrid KV 的 Mamba group ABI 已兼容当前宿主；200 个请求运行期间无 HCCL、KV
  ownership 或 graph replay 错误。
- 256K 配置启动后分配 1,690,828 个 KV token，可容纳约 `6.45x` 的 262,144-token
  请求，服务未因声明 256K 而降级。
- APC 长 prompt 重复测试中，第二次请求复用了一个 2,048-token hybrid cache block，
  日志中的 prefix-cache hit rate 从 0 提升到 9.5%。

最初的 `FULL_AND_PIECEWISE` 首请求失败来自可选的
`multistream_overlap_shared_expert` 路径，CANN 在 `SwiGlu_3_high_performance_27` 报告
无效 GM 地址。Frontier 协议已移除该非必要开关，但保留 EP；之后 target PIECEWISE、
target FULL 和 proposer FULL 均稳定重放。

## 原始数据

- 正式 baseline：
  `benchmark_results/qwen35_frontier_mtp2_gsm8k/20260927-c4-baseline-final/gsm8k.json`
- 正式 MTP2：
  `benchmark_results/qwen35_frontier_mtp2_gsm8k/20260927-c4-mtp2-final/gsm8k.json`
- 并发扫描：
  `benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-{baseline,mtp2}/`
- 256K 原生正确性：
  `benchmark_results/qwen35_long_context/20260930-mtp2-26case/results.json`
- AgentX 256K 成对结果：
  `benchmark_results/qwen35_agentx_256k/20260930-current-pair/`
- 稳定性与 APC 日志：`/tmp/vspec_qwen35_frontier_mtp2_stable.log`

这些结果由本地开发树生成，尚未提交、推送或发布。

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
