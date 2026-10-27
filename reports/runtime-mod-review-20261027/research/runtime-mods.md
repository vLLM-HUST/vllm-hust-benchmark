# 运行时 MOD 清单

观察日期：2026-10-26。以下是官网目录事实，不是我们的源码审计或复现实验结论。

## 范围与来源

- [工坊页面](https://vllm-hust.sage.org.ai/plugins.html)
- [组件 registry](https://vllm-hust.sage.org.ai/data/ecosystem.json?v=state-scheduler-archive-20261018)
  与
  [taxonomy](https://vllm-hust.sage.org.ai/data/mod-taxonomy.json?v=state-scheduler-archive-20261018)
- 按 `public_surface != false` 且 `taxonomy.kind == runtime_mod` 筛选，共 **24 个组件**；本次这些组件的项目 ID 互不重复。
- 不把 connector、外部 KV 服务、工具、控制面和平台 profile 自动纳入。运行时 MOD 是网站分类，真实部署边界仍须核对。
- [结构化摘录](catalog/2026-10-26-runtime-mods.json) 保留接入面、版本、平台和原始证据入口。URL 查询参数不是内容版本保证，摘录代表此次读取。

## 完整索引

| MOD / 源码入口                                                                                                  | 目录中的职责                  | 效果标签（原值）                | 证据入口                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ADM](https://github.com/vLLM-HUST/ascend-distributed-metadata)                                                 | `dp_metadata_sync`            | `measured`                      | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-ascend-distributed-metadata/snapshot/qualifications/qwen3moe-dp4-int64-word-20260928.json)                   |
| [BidKV](https://github.com/vLLM-HUST/vllm-hust-bidkv)                                                           | `scheduler_policy`            | `measured`                      | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-bidkv/snapshot/results/qwen35-evoscientist-20260929/README.md#high-concurrency-kv-pressure-result) |
| [DiffSpec](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec)                                              | `decoding_mechanism`          | `not-beneficial-in-tested-cell` | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-ascend-hust-diffspec/snapshot/docs/evidence/sage-mate-20260904-tp4-graph.md)                            |
| [vSpec](https://github.com/vLLM-HUST/vllm-hust-vSpec)                                                           | `decoding_mechanism`          | `measured`                      | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-vSpec/snapshot/docs/gsm8k_b128_recovery_and_arc_easy_20260913.md)                                  |
| [LatchMoE](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE)                                              | `operator_runtime`            | `not-beneficial-in-tested-cell` | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/f57dbc2b1020c6f50b7a0a36db43fcd8a0bd0198/reports/vllm-hust-midterm-20261019/b1-mod-source-audit-20261020.json)                                                                                |
| [OpFuse](https://github.com/vLLM-HUST/vllm-hust-opfuse)                                                         | `operator_fusion`             | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-hust-opfuse)                                                                                                                                                                                                      |
| [Ascend KV Compression](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust)                               | `kv_state_manager`            | `not-beneficial-in-tested-cell` | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-website/snapshot/docs/FRONTIER-QWEN35-KVCOMPRESS.md)                                               |
| [Prefix Router](https://github.com/vLLM-HUST/vllm-hust-prefix-router)                                           | `request_router`              | `beneficial`                    | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/f57dbc2b1020c6f50b7a0a36db43fcd8a0bd0198/website-publication/data/leaderboard_prefix_routing_mean_evidence.json)                                                                              |
| [KV Tiering](https://github.com/vLLM-HUST/vllm-hust-kv-tiering)                                                 | `kv_residency_provider`       | `inconclusive`                  | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-website-extra/snapshot/docs/FRONTIER-KV-TIERING-20260926.md)                                       |
| [KNorm](https://github.com/vLLM-HUST/vllm-hust-knorm)                                                           | `kv_compression_method`       | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-hust-knorm)                                                                                                                                                                                                       |
| [PyramidKV Ascend](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust)                                     | `kv_compression_provider`     | `validated`                     | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/0dd2d1293e8d0f2e23c1e3c5bfe0a38e588a890c/archive/mod-git-20261022/pyramidkv-website-source/snapshot/evidence/current/2026-10-09-paired-evaluation)                                            |
| [Quantized KV Cache](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust)                          | `kv_quantization_research`    | `validated`                     | [原始材料](https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust/tree/ae7b44bfee695d0ba580040dffd10eb802104485#%E5%B7%B2%E9%AA%8C%E8%AF%81%E7%8E%AF%E5%A2%83)                                                                                       |
| [SimLLM](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust)                                                  | `kv_reuse_policy`             | `beneficial`                    | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-ascend-simllm-hust/snapshot/benchmarks/results/qwen35_latency_20261008/README.md)                       |
| [Split-Batch / Full-Graph Parallel](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust)                  | `graph_execution_policy`      | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust)                                                                                                                                                                                          |
| [Pipeline Microbatch](https://github.com/vLLM-HUST/vllm-hust-pipeline-microbatch)                               | `pipeline_scheduling_policy`  | `inconclusive`                  | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-website/snapshot/docs/FRONTIER-QWEN35-PIPELINE-K8S.md)                                             |
| [KV Materialization Arrival Control](https://github.com/vLLM-HUST/vllm-hust-kv-materialization-arrival-control) | `kv_materialization_policy`   | `measured`                      | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-website/snapshot/docs/FRONTIER-QWEN35-KV-MATERIALIZATION.md)                                       |
| [BetterScale](https://github.com/vLLM-HUST/BetterScale)                                                         | `graph_execution_policy`      | `measured`                      | [原始材料](https://vllm-hust.sage.org.ai/betterscale.html#service-throughput)                                                                                                                                                                                  |
| [StateAxis Automatic Prefix Cache](https://github.com/vLLM-HUST/stateaxis-automatic-prefix-cache)               | `prefix_cache_policy`         | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-automatic-prefix-cache/snapshot/evidence/ISSUE95_BOUNDED_APC_CLOSURE_20260815.md)                                 |
| [StateAxis Async KV Transfer](https://github.com/vLLM-HUST/stateaxis-async-kv-transfer)                         | `kv_transfer_overlap`         | `measured`                      | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-async-kv-transfer/snapshot/evidence/issue76-async-transfer-20260815-r5/result.json)                               |
| [StateAxis Pipelined Weight Loading](https://github.com/vLLM-HUST/stateaxis-pipelined-weight-loading)           | `model_weight_loading`        | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/snapshot/evidence/qwen35-loader-sharded-qualification-20261012/RESULT.json)              |
| [GDN State Codec](https://github.com/vLLM-HUST/gdn-state-codec)                                                 | `recurrent_state_compression` | `not-beneficial-in-tested-cell` | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/gdn-state-codec/snapshot/evidence/qwen35-agentx-v020-expanded-20261009/ROUND-RESULTS.md)                                    |
| [OPset](https://github.com/vLLM-HUST/vllm-hust-opset)                                                           | `operator_runtime`            | `beneficial`                    | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/vllm-hust-opset/snapshot/docs/optimizations/linear-swiglu-graph.md)                                                         |
| [Core Attention Boundary](https://github.com/xmdhb/vllm-hust-core-attention-boundary)                           | `attention_execution_policy`  | `preview`                       | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/e759add873a1f0a3d2f54b004c2eb650fc811245/archive/mod-git-20261022/website-source-vllm-hust-website-extra/snapshot/docs/FRONTIER-QWEN35-OP02-CORE-ATTENTION-BOUNDARY.md)                       |
| [MuxPD](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust)                                                    | `graph_execution_policy`      | `validated`                     | [原始材料](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/19081ff39a25c3c8ac821ae7885688ea24a03e90/reports/muxpd-ecpa-20261025/RELEASE-QUALIFICATION.md)                                                                                                |

## 读表规则

- `measured` 只表示有测量；不等于正收益。`validated` 也不能自行解释为性能验收。
- `beneficial` 和 `not-beneficial-in-tested-cell` 均须绑定具体模型、版本、硬件和 workload；不可外推到整个插件。
- `preview`、`inconclusive` 是重要边界，不应被兼容性标签覆盖。
- taxonomy evidence、compatibility status 与 public effect 是不同维度；有差异时保留原值，沿证据链接核实，不取最乐观者。

## 接下来的研究问题（我们的工作假设）

按真正修改的资源与决策分组，而不是按宣传名称判断相似性：

1. **调度与路由**：BidKV、Prefix Router、KV Materialization Arrival Control。分别追问请求放置、抢占与 KV 到达是否修改同一个决策点。
1. **KV 与其他状态**：Ascend KV Compression、KV Tiering、KNorm、PyramidKV Ascend、Quantized KV
   Cache、SimLLM、StateAxis Automatic Prefix Cache、StateAxis Async KV Transfer、GDN State
   Codec。先区分减少字节、改变驻留、复用和传输重叠；recurrent state 不直接当作普通 KV。
1. **执行组织**：OpFuse、Split-Batch / Full-Graph Parallel、Pipeline Microbatch、BetterScale、MuxPD。追踪
   scheduler → runner → graph 的所有权和并发时序。
1. **推测解码**：DiffSpec、vSpec。核对 draft 成本、接受率、验证成本与实际服务吞吐，不以接受率独立证明加速。
1. **算子与注意力**：LatchMoE、OPset、Core Attention Boundary。确认替换路径、形状门限、数值语义与回退。
1. **通信与启动**：ADM、StateAxis Pipelined Weight Loading。区分服务稳态收益与冷启动收益。

Fletcher 已选择先研究 **vSpec**。见 [问题、原理与实现定位](vspec/principles.md)。随后研究
[SimLLM](simllm/principles.md)；BetterScale 由 Fletcher 开发，本轮跳过。BidKV 也跳过，接着研究
[Pipeline Microbatch](pipeline-microbatch/principles.md)。其余分组是导航，不规定研究顺序。

## 单项研究的落笔标准

每项写清：原始问题与基线；算法与状态；精确源码提交和调用链；接入契约及进程归属；配置与生效标志；失败、禁用和回退；兼容/冲突；性能与正确性证据；仍待核对的问题。

结论明确标记为 **官网声明 / 源码观察 / 我们的推断 / 我们的实测**。目录本身记录 2026-10-26 首轮官网盘点；当时尚未运行插件。后续已开展多项源码研究与重点复现，但不代表 24
项均完成验收；当前进展与证据边界见[研究总报告](report.md)。
