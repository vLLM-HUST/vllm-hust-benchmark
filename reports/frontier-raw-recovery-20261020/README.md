# Frontier 原始数据归档与复核（2026-10-20）

本目录收录 **56 个运行的原始文件**：45 个已复核曲线点，覆盖 9 条完整并发曲线， 另含 11 个历史或排除记录。原始文件不因效果不佳或无效状态而丢弃。
这是历史数据回收与复算，不是新增实验，也不改变现行验收合同。 汇报选图可以侧重代表性成果，数据归档保留完整系列，不删除收益较低或回退点。

## 复算入口

```bash
python3 reports/frontier-raw-recovery-20261020/verify.py
```

仅需 Python 3.9+ 标准库，不联网、不使用 NPU、不运行模型。 脚本校验压缩包及每个原文件的 SHA-256、run ID、并发、时长、卡数、负载哈希， 并从
`requests.jsonl` 的 `chunks` 逐项重算 `0 <= timestamp < 900` 窗口的输出 token。 不计入测量结束后的排空 token。

- [manifest.json](manifest.json)：每个运行的配置身份、源路径、文件哈希和复算值。
- [SHA256SUMS](SHA256SUMS)：归档包校验清单。
- [raw/](raw/)：按 run ID 保存的确定性 gzip/tar 包，每包含原始 `config.json`、`summary.json`、`requests.jsonl`
  三个文件，内容字节未改写。
- [verify.py](verify.py)：无需原服务器即可复算，不依赖网站展示结果。
- [workloads.json](workloads.json) 与 [workloads/](workloads/)： 两个原始预处理负载文件的压缩归档及校验值，覆盖所收录运行的负载哈希。

原文件保留在来源服务器，本次仅复制归档。未包含凭据、模型权重、环境文件或运行缓存。 清单中的主机别名和路径用于追溯；离线复算不依赖它们。

为遵守仓库 1000 KB 单文件检查，大于 900000 字节的压缩包按字节分块， 以 `.part000` 等文件存储。按 `manifest.json` 中 `archive_parts`
顺序连接即可恢复 原压缩包；每块及拼接后的完整包均有 SHA-256。`verify.py` 自动在内存中拼接， 不要求手工解包。小包保持一个 `.tar.gz` 文件，原始 JSON/JSONL
字节不变。

## A. BetterScale 完整曲线

对应已有登记：
[dataset_validation_qwen35_frontier_betterscale_900s.json](../../leaderboard-data/dataset-validation/dataset_validation_qwen35_frontier_betterscale_900s.json)。
共 10 个运行，包括五个 Native 基线和五个 BetterScale 点。

Qwen3.5-35B-A3B BF16，Ascend 910B2 ×2，TP2/PP1/DP1，APC、自然 MTP2， 每个并发一个 900 秒窗口。下表为 **双卡总输出
token/s**，不是每卡吞吐。

| 客户端并发 | Native vLLM 0.25.1 | BetterScale |   比值 |
| ---------: | -----------------: | ----------: | -----: |
|          1 |           100.8333 |    133.0522 | 1.3195 |
|          2 |           157.5178 |    252.0611 | 1.6002 |
|          4 |           219.3700 |    424.0089 | 1.9328 |
|          8 |           307.7978 |    615.7244 | 2.0004 |
|         16 |           443.9078 |    885.9611 | 1.9958 |

### 对照边界

- 两臂负载文件 SHA-256 相同，tokenizer fingerprint、客户端 seed、 模型 checkpoint、vLLM/Ascend 基础源码提交和双卡规格相符。
- Native 来自 2026-09-24，BetterScale 来自 2026-09-27（UTC）： **同负载、同规格历史部署对照，不是同批次随机配对消融**。 Native 与
  BetterScale 的共享主机、运行时间不同，不作硬件噪声已消除的声明。
- Native 为 `FULL_AND_PIECEWISE`，BetterScale 为 `FULL`、 E16/R20 驻留策略和均衡注意力组合。图捕获、调度及配置差异属于部署方案，
  不把全部收益归因于一个算子或独立 attention 改动。
- BetterScale C16 测量 `283e06d`，C1/2/4/8 测量 `eee35fd` 默认集成。 保留已有系列的真实版本记录。集成说明称数值程序和内核未变，
  这不等于对集成版本重新测过 C16，也不是五点完全相同的源码 SHA。
- 每点仅一次窗口，不构成重复性统计或答案质量评测。262144 是服务端容量， 不表示每个请求都占满 256K。
- Native 配置中 `runtime_parameters` 保留了早期历史参数； `old_parameters_scope` 明确以本次 `command` 覆盖旧的 synthetic
  acceptance 和 chat template 描述。保留原记录，不擅自重写历史元数据。

原始记录从 91 服务器旧容器的 `workspace/.../swe-frontier-sequential-20260924` 与
`rootfs/root/.../attention-response` 回收。Native C16 使用正式重试
`e52aadd2f98746d2b7da71ccd235ebb9`，不是只有配置、没有结果的初次尝试。 BetterScale 服务端配置另在 86 服务器旧容器 `ascend-probes`
中核对。

## B. 统一 MOD 曲线

对应已有登记：
[dataset_validation_qwen35_frontier_unified_900s.json](../../leaderboard-data/dataset-validation/dataset_validation_qwen35_frontier_unified_900s.json)。

| 系列              | 归档点数 | 并发               |
| ----------------- | -------: | ------------------ |
| Native            |        5 | 1 / 2 / 4 / 8 / 16 |
| BidKV             |        5 | 1 / 2 / 4 / 8 / 16 |
| DLA               |        5 | 1 / 2 / 4 / 8 / 16 |
| KV 分层迁移       |        5 | 1 / 2 / 4 / 8 / 16 |
| KV 压缩           |        5 | 1 / 2 / 4 / 8 / 16 |
| KV 物化与到达控制 |        5 | 1 / 2 / 4 / 8 / 16 |
| PegaFlow          |        5 | 1 / 2 / 4 / 8 / 16 |

这些 35 个点独立于 A 组。Native 基线不是 A 组 Native，不能相互替换， 也不能跨组挑选最优点、拼曲线或累加 MOD 收益。 所有登记点均从原始窗口重新计数，完整结果见清单。
部分系列的负载原文件哈希、客户端 seed 及 tokenizer 元信息不同， 清单逐运行保留这些差异，不把该统一登记自动当作严格单变量消融。

来源包括 180 裸机迁移目录 `frontier-mods-qwen35-20260925`， profiling 的 PegaFlow/KVCompress 正式目录，evaluation 的
KV 物化正式目录。

## C. 请求完成进度与其他已入库实验

补充记录在清单中标为 `supplementary`、`performance_eligible: false`： Mooncake 5 个历史点、phase16 旧版 KVCompress 5
个点，以及 `C4-invalid-source-worktree-mutation-1` 的 1 个无效运行。 保留原始 `valid` 和失败数，不把“文件完整可校验”误当成“性能结论有效”。

不重复上传已有原始包：

- [中期 MMLU 原始记录](../vllm-hust-midterm-20261019/)： `aligned-mmlu-b0.tar.gz` 中的 `mmlu-b0-r2` 为 Native；
  `microflow-school028/mmlu-b1-microflow-fa-on-r10-evidence.tar.gz` 为汇报所用组合配置的三次 ON 实测中位轮次。 200
  请求完成进度由逐请求 `start_times + latencies` 构造，整批时间使用 原始 benchmark duration。它与 A/B 的 SWE 900 秒吞吐实验独立。
- [BetterScale + solver mixed pool](../betterscale-solver-mixed-pool-20261020/)： 已有六种配置和重复测量登记；没有
  matched Native-OFF， 不用它生成 Native 对照收益。

## 发布与保全

本归档提供可复算的底层证据，不替代各实验的适用边界、质量评测和验收判定。 新结果应新增运行或版本记录，不覆盖本目录的原始包。 未收录尚在运行的实验、模型缓存及无关用户目录。其余未定位或未复核的数据
不在本次“56 个已归档运行”的统计中。
