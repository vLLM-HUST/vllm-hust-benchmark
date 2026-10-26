# MuxPD v5：完整质量结果、实际原生参考与混合 profiler

本次追加七个终态运行和一个完整支持胶囊。**没有证据表明当前插件总体优于原生 vLLM：固定质量不退化门槛失败，实际原生参考吞吐更高；插件的明确收益是本负载中更短的长 P 到达期间最坏 decode 停顿。** Grouped P 相对 raw 的吞吐增量属于插件内部优化，不等于超过原生。所有数字限定于保存的 TP4/EP4、910B2、BF16、Qwen3.5 配方和请求，不是 HTTP Goodput 或容量测量。

## 固定质量：收集完成，答案不退化失败

| 模式 | 正确 / 200 | 自然停止 | 达到 8192 上限 | 相对两原生共同正确集合新增错题 |
| --- | --- | --- | --- | --- |
| Native1 | 165 | 191 | 9 | — |
| Native2 | 165 | 191 | 9 | — |
| Raw | 166 | 193 | 7 | 3 |
| Group | 166 | 193 | 7 | 3 |

两个新进程原生控制的全部 200 份 token 数组、文本和终止记录相同。Raw/group 同样逐题完全相同；group/native 仅 11/200 完整 token 数组相同，189 份不同。相对原生，raw/group 有三项丢失和四项新增正确答案。提高总正确数不能抵消预冻结的“原生共同正确题不得新增错题”否决条件。

原严格无截断门槛保持 FAIL；在候选输出前冻结、另行命名的预算内 MCQ 补充答案及组合门槛也保持 FAIL。Length、无效和缺失均按错误计，分母始终 200。候选 collection 和实际混合覆盖各自通过，raw/group 各 25 批、每批十次绑定请求/阶段/时间顺序的 P 进展与 D 退休事务；这些不能改变答案门槛或证明物理 kernel overlap。补充协议依据原生校准另行预声明，并非把旧严格失败改名通过。

[完整 group 逐题审查](analysis/group/review.json)、[原始审查文字](originals/GROUP-QUALITY-README-v1.md.source)、[严格结果](analysis/group/strict-result-v1.json)和[补充结果](analysis/group/supplementary-result-v1.json)保留全部正反变化、输出身份和冻结条件。Root 执行 group 评分；现存 JSON 和执行身份被绑定，没有为未保存的 CLI stdout 伪造本地日志。原生双控制和 raw 原件沿用本目录既有副本与 Release 胶囊。

## 性能：短停顿收益与吞吐、TTFT 代价同时存在

主同步矩阵 N1/R1/G1/G2/R2/N2 保持 [v4 的原始 ABBA](../performance-lifecycle-v4/README.md)。Grouped P 相对 raw 的混合吞吐均值增量为 **+3.466%**，两对分别 **+5.913% / +1.047%**。这不是原生相对优势，也不是总体显著性证明。

另外两个 fresh 原生参考使用实际默认 async 调度与 npugraphEx 后端：async=None 解析为 True、队列 2、`step_with_batch_queue`、npugraphEx=True/static=False、真实内部 `NPUWorker`，不加载插件 worker。它们仍显式使用 HCCL=true 等本次固定配方，所以是实用默认后端参考，不是“所有设置默认”或厂商最优配置。两个进程单独描述，不加入主 ABBA；async 与后端同时变化，不能分离单因素成本。

| 指标 | 实用原生参考均值 | Raw 均值 | Group 均值 | Group 相对实用原生 |
| --- | --- | --- | --- | --- |
| Decode 输出 token/s | 368.844 | 217.889 | 218.525 | −40.754% |
| Mixed 输出 token/s | 194.555 | 116.105 | 120.129 | −38.254% |
| 长 P TTFT，ms | 455.777 | 621.827 | 616.651 | +35.297% |
| Mixed seed gap P99，ms | 32.575 | 63.483 | 60.640 | +86.157% |

冻结的、原生 N1 校准后且候选性能输出前声明的停顿补充指标显示：四个测量 mixed 窗口中，group 最大 seed token 间隔为 **75.486–86.950 ms**，实用原生为 **423.349–445.610 ms**。Group 每条流没有超过 100 ms 的间隔；原生每窗口每流恰一次。P99 仍更差；该孤立停顿观察补充原指标，不能替换原指标，也不构成 SLO Goodput。

[原完整指标报告](originals/NATIVE-REFERENCE-REPORT-v2.md.source)、[48 个保留窗口](analysis/native-reference/all-window-summaries.json)、[全部输出轨迹比较](analysis/native-reference/retained-token-comparisons.json)和[实际后端读回](analysis/native-reference/all-backend-counter-readbacks.json)不省略暖机、进程漂移或负面结果。ND2 相对 ND1 混合吞吐下降 4.068%。Decode 输出相同；mixed 中原生同步、原生 async、raw/group 三种输出轨迹不同，故不能把总耗时差全归于调度或后端成本。

原冻结 v2 offline analyzer 因 Ascend 初始化展开三个默认 True 配置键而拒绝。首次失败完整保留；另行版本的只读适配核验精确源代码、原启动配置 SHA 和 `fuse_muls_add`、`fuse_norm_quant`、`fuse_qknorm_rope` 三项 True 后，保留全部剩余检查。没有修改原运行、冻结 producer、原字典或指标，也没有把运行改成重跑后通过。

## 混合 profiler：可观测分组成立，但没有减少 kernel launch 数

Raw/group 各一次真实混合捕获，都从四条 seed D 各完成 32 token 后、长 P 入队前开始，到 P 首 token 停止。冻结 capture auditor 和匹配比较均通过；各 rank 有十次 D graph replay，group 有十次 P group return，与十个实际 P/D 调度步骤对应。全部十份 warmup/capture 完整请求输出数组相同。

| 每 rank 的原生 API 次数 | Raw | Group | 差值 |
| --- | --- | --- | --- |
| `aclrtLaunchKernelWithHostArgs` | 2854 | 2924 | +70 |
| inplace copy | 909 | 1039 | +130 |
| memcpy | 303 | 363 | +60 |
| record event | 462 | 462 | 0 |
| wait event | 346 | 346 | 0 |

因此不能把另行无 profiler 测得的 +3.466% 吞吐增量解释成更少原生 kernel launch。没有声称 NPU 物理 overlap、计算资源配额生效或“无 profiler 开销”；资源分区在当前配方中仍关闭。完整 [364 行 API 比较](analysis/mixed-profile/all-runtime-api-comparisons-v1.csv.source)、[capture 对比](analysis/mixed-profile/comparison-v1.json)及[原报告](originals/MIXED-PROFILE-README-v1.md.source)保留所有计数与持续时间。

原捕获日志的 daemon 导出异常仍在原 run 胶囊。终态后只对副本做原生 offline export，八份 trace 全部解析；全部约 729 MiB 未压缩副本树以及解析日志、脚本、回执进入支持胶囊，未写入 Git。原 run 目录前后库存不变。

## 来源、发布范围与处置边界

七个新 run 都绑定实际设备源 `f7ed21e183b27d2141176f3e7a5c8a5ff4ede07d`，Core `d0f22d2bda562156e4dbf433ce645e1769b4f804`、Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989`；既有 Ascend build 文件差异按原协议绑定。`98eac12dc2595a8742a81b58e96b899fed905194` 仍只有既有 source bridge 范围；后续文档提交不会被冒充成新实测 head。

新增七项为 N2/raw/group 质量、ND1/ND2、raw/group mixed profiler；N1 已在 v3，主矩阵及失败的 default-v1 观察器已在 v4。八个新胶囊使用完整原件清单和 SHA256；七个原 run 胶囊直接复用，支持胶囊完整保留失败的 offline analyzer、精确修正代码和派生数据。所有原终态 exit 0/清理通过都仅是 bounded cleanup：日志中的四 worker SIGTERM fallback 不删除，也不称全局无信号 graceful drain。

历史严格质量失败、20-block cancellation 失败、旧元数据误判、default-v1 首窗口前失败与旧 README pending 均保留原字节；新证据通过新版本追加。#11 的关闭仅表示既有有界调查与 opt-in 缓解完成，不能推出 #23 质量已通过。#23 保持开放的已知质量限制；本报告不把 PR10 变成已合并或生产默认。#3 的有限优化处置应同时记录上述收益与代价。源码支持审计中的 batch invariance/HCCL strict 并非已验证修复，本轮没有追加筛选治疗或重写质量标准。

本版本准备时旧 57 个远端资产和 400 个既有 canonical 文件为保护基线。新增资产仍需上传后的独立匿名下载、压缩包 SHA 和逐成员核验；本地校验不是远端可下载证明。下载入口、原件清单与 SHA 见 [release-assets-v5.json](release-assets-v5.json)、[archive-index-v5.json](archive-index-v5.json)和 [SHA256SUMS-v5](SHA256SUMS-v5)。准备时的 pending 草稿另存，最终状态以本报告及后来单独的公开读回回执为准。
