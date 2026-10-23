# 2026-09-20 历史工作树验证记录

[English](validation.md) | 简体中文

> **历史归档。** 本记录早于 0.8 Frontier 规程，不代表后续版本的发布验收。
> 新测量请从[测试规程索引](../../benchmark/README.zh.md)选择规程。

日期：2026-09-20。结论：**工程部分通过，不是 V4.6 正式验收**。完整 A2
commissioning 矩阵和三项公开质量任务通过；冻结的 A3 稳定性矩阵已完整执行，但未
通过最少完成数和六窗口吞吐 CV 门槛。该负结果原样保留，没有用新增轮次替换。补充
的 Qwen3.5 矩阵也已完成：A2 八个 cell 中两个通过，A3 稳定性失败，但模型输出和
压缩事务全部正确。

| 范围 | 结果 | 结论边界 |
| --- | --- | --- |
| 单元/契约套件 | 通过 | 仅代表仓库正确性 |
| Ascend 数值 smoke 与算子微基准 | 通过 | 合成 cache 形状，不是服务吞吐 |
| Qwen2.5 A2 八个 cell、每组各三轮冷启动 | 通过 | commissioning fixture 与同宿主对照 |
| Qwen2.5 A3、每组各三轮 30 分钟 | **失败** | 完成数和吞吐 CV 未通过 |
| Qwen2.5 公开 LongBench-v2/LongBench | 通过 | 每组单轮，主要证明质量和完整性 |
| Qwen3.5-35B-A3B 适配与公开任务 | 通过，补充证据 | 仅 TP=2/eager 工程证据 |
| Qwen3.5 移植 A2/A3 矩阵 | **失败**，补充证据 | 六个 A2 cell 吞吐失败，A3 稳定性失败 |
| V4.6 正式放行 | **不宣称** | 缺少官方 B0 与获授权 LONG-PUBLIC 证据 |

## 被测实现

当前工作树实现 TriAttention V3 位置策略：硬保护可配置的前缀与最近 token，再在等长
中间上下文分段间按比例分配驱逐额度。Qwen3.5 hybrid 路径只评分和压缩 10 个全注意力
层，Gated-DeltaNet 循环状态继续使用宿主生命周期。Qwen3.5 的 192 个非旋转 key
维度还使用校准后的直接 Q·K 内容项；TP rank 在选择前 all-reduce 每层分数，确保选择
完全相同的 token 集合。

两个模型系列都使用 schema-3 校准产物：

| 模型 | 校准 SHA-256 |
| --- | --- |
| Qwen2.5-14B-Instruct | `cf3be9376043bb9c47fb8beb461d3951c3d82caa90d33bffcfc4dff5d96a6f26` |
| Qwen3.5-35B-A3B | `c09e9ec41c34d876802865a5ef489c1d4514245a90b71d1574663ddcfa2c637b` |

本次没有修改 `vllm-hust` 或 `vllm-ascend-hust`。所有集成都在独立插件内完成。
Qwen3.5 的旧 `int[]` GDN ABI 兼容桥必须显式启用，并严格匹配已验证宿主 schema。

## 冻结环境

| 组件 | 已验证值 |
| --- | --- |
| 插件 | 基于 `de038e7` 的 0.7.0 发布候选 |
| vLLM-HUST | `6cdc0304a8ba`，`0.28.1.post1.dev260` |
| vLLM-Ascend-HUST | `5901bedbb718`，`0.25.1rc2.dev232+hust.20260903.4.g5901bedbb` |
| Python / torch / torch-npu | 3.11.16 / 2.10 / 2.10.post2 |
| 标准模型 | `/data/shared_models/Qwen--Qwen2.5-14B-Instruct`，FP16，TP=1 |
| 补充模型 | `/workspace/models/Qwen--Qwen3.5-35B-A3B`，BF16，TP=2，eager |
| 设备 | Ascend 910B2；本次只使用 0–5 卡 |

服务使用 `--generation-config vllm`。客户端固定 temperature 0、top-p 1、top-k -1、
min-p 0、presence/frequency penalty 0、repetition penalty 1、n=1、禁用 beam、
stop 为空、seed 0、流式回传 usage、`add_special_tokens=true`。

## 自动化与 NPU 检查

- `pytest`：114 passed、1 项环境跳过；
- Ruff 检查和格式检查：通过；
- Ascend 数值算子 smoke：通过；
- 2,176-token、8 KV head、BF16 新进程微基准：分页 K/V copy 相对参考 1.64×、
  直接评分 1.85×、融合聚合 1.35×。

算子结果只是隔离微基准，不是模型服务收益。跨版本规范化视图见
[HTML 测试榜单](../../results/benchmark-leaderboard.html)。

## Qwen2.5 标准 commissioning 矩阵

确定性 fixture SHA-256 为
`6911a586376fbb9472be9017d68660cd3fb38dda2a054dd76cd2b11c408e208e`。
B0/B1 各运行三轮独立冷服务生命周期；A2 每个 cell、每轮测量 16 个请求，并发 4。

| Profile | RPS | B0 → B1 总吞吐中位数 tok/s | 变化 | 质量 | 最小物理 KV 缩减 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8K + 512 | 0.05 | 439.13 → 438.90 | −0.05% | 100% → 100% | 不适用：未跨阈值 | 通过 |
| 8K + 512 | 0.10 | 829.81 → 828.68 | −0.14% | 100% → 100% | 不适用：未跨阈值 | 通过 |
| 8K + 512 | 0.20 | 1294.64 → 1290.71 | −0.30% | 100% → 100% | 不适用：未跨阈值 | 通过 |
| 8K + 512 | 0.40 | 1370.00 → 1368.79 | −0.09% | 100% → 100% | 不适用：未跨阈值 | 通过 |
| 16K + 1,024 | 0.05 | 820.79 → 826.60 | +0.71% | 100% → 100% | 50.00% | 通过 |
| 16K + 1,024 | 0.10 | 1085.70 → 1232.85 | +13.55% | 100% → 100% | 50.00% | 通过 |
| 16K + 1,024 | 0.20 | 1128.82 → 1300.52 | +15.21% | 100% → 100% | 50.00% | 通过 |
| 16K + 1,024 | 0.40 | 1147.92 → 1335.78 | +16.36% | 100% → 100% | 50.00% | 通过 |

A2 实测的 768 个组别请求全部完成，无失败、强制输出不足或静默截断；六个预热请求
也全部正常。16K B1 共记录 192 次 scheduler 提交和 192 次 worker 回执。设备 HBM
峰值为 87%；vLLM 启动时预留 KV 池，因此不宣称预留池缩小。8K cell 未跨过 9,216
语义 token 阈值，按规范把 M3 标为不适用。

## Qwen2.5 A3 稳定性负结果

A3 使用精确 30,720 输入 token 加 2,048 强制输出 token，并发 1，预热五分钟，
测量 30 分钟并切分为六个五分钟窗口。B0/B1 各运行三轮独立冷生命周期。

| 三轮中位数 | B0 | B1 | 变化 |
| --- | ---: | ---: | ---: |
| 总 token 吞吐 | 407.91 tok/s | 461.62 tok/s | +13.17% |
| 平均 TTFT | 6304.90 ms | 6318.90 ms | +0.22% |
| 平均 TPOT | 36.16 ms | 31.58 ms | −12.68% |
| 平均 E2E | 80.32 s | 70.96 s | −11.66% |

实测 147 个组别请求全部正确；失败、OOM、短输出和静默截断均为 0。B1 记录 156 次
提交和 156 次回执，最小物理 KV 缩减 73.33%。TTFT/TPOT 中位数与 p99 漂移门槛
全部通过，但 A3 仍因以下两个冻结条件失败：

- B0 每轮只完成 23 个请求，低于至少 24 个的要求；B1 每轮完成 26 个；
- B0 每轮六窗口吞吐 CV 都是 12.86%，B1 都是 8.94%，均高于 5%。

这是测试门槛失败，不是崩溃或数据损坏；没有用额外轮次替换失败结果。

## 公开长上下文质量与性能

Qwen2.5 使用所有可接纳且未截断样本：LongBench-v2 116 条、段落检索 200 条、
Qasper 压力子集 93 条。两组共 818 个请求全部完成。

| 数据集 | B0 → B1 质量 | 请求吞吐 | 平均 E2E | 物理 block | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| LongBench-v2 | 40.52 → 39.66 accuracy（−0.86 pp） | +7.78% | −6.99% | −61.96% | 通过 |
| 段落检索 | 98.75 → 98.75（0.00 pp） | +2.41% | −2.33% | 绕过 | 通过 |
| Qasper | 43.60 → 43.36 F1（−0.24 pp） | +1.35% | −1.41% | 触发样本 −38.73% | 通过 |

三项质量差都在冻结的 1 pp 门槛内。B1 有 127 次提交和 127 次回执；触发压缩的
样本从 20,666 个源 block 降至 8,128 个目标 block（−60.67%）。这些是每组单轮、
且宿主其他 NPU 同期有负载的工程结果，性能差只作探索性数据。详见
[公开长上下文 Benchmark](public-long-context-benchmarks.zh.md)。

## Qwen3.5 补充适配

已下载模型快照 revision 为 `59d61f3ce65a6d9863b86d2e96597125219dc754`；
14 个权重分片含 71,903,655,008 tensor bytes。TP=2 16K smoke 通过，物理 KV
缩减 50%。非思考公开测试的 806 个组别请求全部完成，无错误或静默截断，三项质量
门槛全部通过：LongBench-v2 准确率不变，段落检索保持 100% 且绕过压缩，Qasper
变化 −0.43 pp。TP=2 事务证据严格匹配：121 次 scheduler 提交对应 242 次逐 rank
回执。

这些公开任务的性能没有优于 B0；当前宿主还要求 eager 和显式旧 GDN ABI 兼容桥。
因此 Qwen3.5 只能作为 hybrid 兼容性补充证据，不能替代标准 Qwen2.5 矩阵。

移植的 A2 矩阵中，768 个实测组别请求全部正确，192 次提交对应 384 次 TP rank
回执；只有 8K@0.05 和 16K@0.40 通过，其他六个 cell 超过冻结的 1% 总吞吐回退
预算。移植的 A3 中，B0/B1 每轮均完成 4 个正确请求，总吞吐中位数提升 1.23%，但
未达到 24 个完成数，六窗口吞吐 CV 为 100%，且空窗口导致时延漂移无法计算。B1
仍实现至少 73.33% 物理 KV 缩减，24 次提交对应 48 次回执。详见
[Qwen3.5 适配说明](../../design/qwen3.5-35b-a3b-adaptation.zh.md)。

## 放行边界

当前工作树不能宣称 V4.6 正式通过或生产就绪：

1. B0 是当前宿主的无压缩工程对照，不是规定的官方 vLLM 0.18 与匹配官方 Ascend；
2. A2/A3 使用仓库 commissioning fixture，而非独立批准、签名的 LONG-PUBLIC；
3. Qwen2.5 A3 和移植的 Qwen3.5 A3 均未通过最少完成数和六窗口吞吐 CV 门槛；
   Qwen3.5 A2 另有六个 cell 未通过吞吐回退门槛；
4. 公开性能每组只有一次运行，且同机有并行负载；
5. 设备预留 HBM 未下降，也未独立测量系统 cost/token；
6. Qwen3.5 使用非标准 TP=2/eager 拓扑和宿主特定 ABI 桥，压缩态严格多位置 NIAH
   仍是部署门槛。

机器可读记录：

- [Qwen2.5 A2/A3 摘要](../../results/evidence/kvcompress-working-tree-20260920-qwen25-standard-summary.json)
- [Qwen2.5 公开基准摘要](../../results/evidence/kvcompress-working-tree-20260920-qwen25-public-summary.json)
- [Qwen3.5 公开基准摘要](../../results/evidence/kvcompress-working-tree-20260920-qwen35-public-summary.json)
- [Qwen3.5 移植 A2/A3 摘要](../../results/evidence/kvcompress-working-tree-20260920-qwen35-standard-summary.json)
- [榜单历史记录](../../results/evidence/leaderboard-history.json)
