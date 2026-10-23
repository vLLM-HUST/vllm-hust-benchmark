# Qwen3.5 历史适配实验

[English](qwen3.5-20260920-adaptation-results.md) | 简体中文

> 2026-09-20/27 历史工程记录；当前实现请见设计说明和测试规程。

## 校准与启动

在两张空闲设备上生成模型绑定的校准产物：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m vllm_ascend_kvcompress.calibration \
  --model /workspace/models/Qwen--Qwen3.5-35B-A3B \
  --output artifacts/qwen3.5-35b-a3b-stats-v3.pt \
  --max-length 4096 --device npu:0 --device-map auto \
  --dtype bfloat16 --attn-implementation eager --local-files-only
```

以下 B0/B1 配方属于 2026-09-20 历史测试：使用 `mamba_cache_mode=none`，关闭
APC 与异步调度，不是当前 0.8 验收启动命令。使用
[`examples/qwen3.5-35b-a3b-triattention.json`](../../../examples/qwen3.5-35b-a3b-triattention.json)
配置插件，再启动 B1：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_PLUGINS=ascend,ascend_kvcompress
export VLLM_ASCEND_KVCOMPRESS_ENABLED=1
export VLLM_ASCEND_KVCOMPRESS_CONFIG=/absolute/path/qwen3.5-35b-a3b-triattention.json
# 仅当前已验证宿主使用旧式不可选 int[] GDN ABI 时需要。
export VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT=1
# 仅当容器 /dev/shm=64 MiB 时需要；并发 4 使用 40 MiB 环形区。
export VLLM_MQ_MAX_CHUNK_BYTES_MB=4
# 只把插件源码路径前置；不能覆盖包含 acl 的 CANN Python 路径。
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
vllm serve /workspace/models/Qwen--Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b --tensor-parallel-size 2 \
  --language-model-only --dtype bfloat16 --block-size 128 --mamba-cache-mode none \
  --max-model-len 32768 --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --generation-config vllm --enforce-eager
```

尽管命令请求 `--block-size 128`，宿主会分别报告 32,768-token scheduler 对齐和
2,048-token 全注意力页。scheduler bind 日志记录这两者，worker bind 日志还会记录
128-token 内核块。

B0 必须独立重启，但仍加载同一插件及模型绑定统计；把 `kv_budget` 设为 32,768、
`recompute_window` 设为 128，使 32,896-token 阈值严格大于服务上限，从而得到
零压缩工程对照。B0/B1 除配置文件中的压缩预算外保持同一命令。这样兼容挂钩和
运行栈完全一致，也无需修改任一宿主仓库。禁止用同一服务生命周期混跑两组结果。

插件会把显式设置的 `VLLM_MQ_MAX_CHUNK_BYTES_MB` 同步应用到 vLLM worker 回传
队列；上游只把它应用于 scheduler 广播队列。超出 chunk 的消息仍使用 socket，
因此该设置只缩小共享内存快速路径，不限制消息大小。不受 64 MiB 限制的部署应省略
此变量并保留上游默认值。

## 当前 0.8 工程状态

2026-09-27，同步后的宿主在 Qwen3.5 BF16/TP=2/eager、16,384-token 上限、
`mamba_cache_mode=align`、APC 和异步调度下完成四个非思考检索请求。答案均正确，
每个请求有一次压缩提交和两个 TP rank 回执；第二个相同请求新增 8,192 个服务端
prefix-hit token。两个不同请求由并发客户端提交，但尚未证明设备 batch 重叠。
详见[机器可读冒烟记录](../../results/evidence/kvcompress-qwen35-20260927-async-apc-align-smoke.json)
和[当前前沿测试协议](frontier-benchmarking-0.9.zh.md)。早期显式 MTP2 诊断的首条请求在
`aclnnCausalConv1d` 失败；修正目标状态列映射和扩展滚动卷积状态后，7K 未压缩对照
以及三条 10K 压缩检索均答对，三次压缩均有双 TP 回执。MTP2 仍默认拒绝启动，
只能显式作为实验功能开启；图模式、262,144-token 上下文和官方 900 秒负载未验证。
CPU 测试通过 135 项，另有 1 项环境跳过。详见
[MTP2 诊断](../../results/evidence/kvcompress-qwen35-20260927-mtp2-diagnostic.json)与
[修复后工程冒烟](../../results/evidence/kvcompress-qwen35-20260927-mtp2-apc-align-smoke.json)。

## 历史 0.7 测试状态与范围

2026-09-20 的工作树通过 114 项单元/契约测试，另有 1 项环境跳过。TP=2 的 16K 服务冒烟
已在两组完成：B0 保留全部 16,384 个输入 token；B1 把 16,384 个语义 token 提交为
8,192 个物理 token；两组均生成完整 1,024 个强制输出 token、命中 oracle，且无
静默截断。当时的模型级结论以公开长上下文和标准 A2/A3 套件为准。由于本目标使用 TP=2、
eager、显式旧 ABI 兼容桥和混合注意力，其结果属于工程证据，不能标为 V4.6 正式
验收。结果统一发布到 [HTML 榜单](../../results/benchmark-leaderboard.html)，并链接机器可读证据
和明确限制。

公开长上下文测试使用 `enable_thinking=false` 的冻结 chat template、
`--generation-config vllm` 和显式中性采样参数。B0/B1 各完成 403 个请求，合计
806 个请求全部成功且无静默截断：

| 数据集 | B0 → B1 质量 | 质量差 | 请求吞吐差 | 物理 KV 缩减 | 提交 / TP rank 回执 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2（110） | 47.27 → 47.27 accuracy | 0.00 pp | −5.83% | 63.76% | 110 / 220 |
| passage_retrieval_en（200） | 100.00 → 100.00 | 0.00 pp | −2.39% | 不适用（短输出绕过） | 0 / 0 |
| Qasper（93） | 51.07 → 50.64 F1 | −0.43 pp | −1.28% | 42.11% | 11 / 22 |

三项质量保持门槛均通过；详细 bootstrap 区间、时延与原始哈希见
[机器可读摘要](../../results/evidence/kvcompress-working-tree-20260920-qwen35-public-summary.json)。
服务运行时同一宿主的其他 NPU 还在执行标准矩阵，因此吞吐和时延差只作为探索性
数据；质量、请求完整性和压缩事务才是本次适配的主要结论。

## 移植的 A2/A3 工程矩阵

完整 commissioning 矩阵也已移植到该模型。profile 名保留 `FP16` 以维持工作负载
身份，但 Qwen3.5 实际使用 BF16、eager、TP=2 和显式旧 GDN ABI 兼容桥。因此这些
是补充工程证据，不是标准 graph-mode/TP=1 拓扑的正式执行。

A2 每个 cell、每轮测量 16 个请求，并发 4；B0/B1 各使用三轮独立冷生命周期：

| 形状 | RPS | B0 → B1 总吞吐中位数 tok/s | 变化 | 最小物理 KV 缩减 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 8K + 512 | 0.05 | 249.72 → 250.80 | +0.43% | 不适用：未跨阈值 | 通过 |
| 8K + 512 | 0.10 | 266.08 → 262.28 | −1.43% | 不适用：未跨阈值 | **失败** |
| 8K + 512 | 0.20 | 272.09 → 266.65 | −2.00% | 不适用：未跨阈值 | **失败** |
| 8K + 512 | 0.40 | 278.59 → 275.32 | −1.17% | 不适用：未跨阈值 | **失败** |
| 16K + 1,024 | 0.05 | 266.78 → 262.17 | −1.73% | 50.00% | **失败** |
| 16K + 1,024 | 0.10 | 274.73 → 269.07 | −2.06% | 50.00% | **失败** |
| 16K + 1,024 | 0.20 | 278.00 → 274.01 | −1.43% | 50.00% | **失败** |
| 16K + 1,024 | 0.40 | 275.39 → 273.54 | −0.67% | 50.00% | 通过 |

768 个实测组别请求和六个预热请求全部正确完成，失败、强制输出不足和静默截断均为
0。16K B1 共记录 192 次 scheduler 提交和精确的 384 次逐 rank 回执。两个 cell
通过；六个 cell 仅因超过冻结的 1% 总吞吐回退预算而失败。

A3 使用 30,720 输入 token 加 2,048 强制输出 token，并发 1，预热五分钟，测量
30 分钟并切分为六个窗口：

| 三轮中位数 | B0 | B1 | 变化 |
| --- | ---: | ---: | ---: |
| 总 token 吞吐 | 69.34 tok/s | 70.19 tok/s | +1.23% |
| 平均 TTFT | 1949.71 ms | 2009.13 ms | +3.05% |
| 平均 TPOT | 229.91 ms | 227.05 ms | −1.24% |
| 平均 E2E | 472.57 s | 466.77 s | −1.23% |

B0/B1 每轮都只完成 4 个正确请求，低于至少 24 个的要求；所有六窗口吞吐 CV 都是
100%，高于 5% 门槛，空窗口还使时延漂移无法计算。B1 仍实现最小 73.33% 物理 KV
缩减，24 次提交对应 48 次 TP rank 回执。因此 A3 结论为**失败**，尽管质量、事务、
缩减和吞吐回退检查均通过。

最初三个沙箱启动因无法枚举 Ascend 设备，在模型加载和请求执行前失败；其日志哈希
已保留，但不计入三轮有效生命周期。完整哈希和逐 cell 数值见
[标准矩阵机器可读摘要](../../results/evidence/kvcompress-working-tree-20260920-qwen35-standard-summary.json)。

这不能推翻 V3 论文对混合架构的风险提示：论文中的 Qwen3.5-27B/35B-A3B 在
V3-only 严格 NIAH 的中部和尾部位置失败，而后续 longctx rescue 尚未在这两个大
模型上验证。本次 `passage_retrieval_en` 因最大输出仅 32 token 按规则绕过压缩，
所以它只验证 bypass，不构成“压缩后 needle 检索”证据。部署前仍应增加经授权的
压缩态多位置 NIAH/生产检索门槛；当前结论严格限于已列出的公开任务。
