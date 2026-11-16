# PrefixRouter：技术成果补证（2026-11-17）

本报告关联 [benchmark #664](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664) 和
[PrefixRouter 专项要求](https://github.com/vLLM-HUST/vllm-hust-prefix-router/issues/1#issuecomment-6846015789)。以下各行是不同版本、不同实验，不跨行计算收益。

| 填报项目                                                                     | 可引用结论及限制                                                                                                                                                                 | 固定证据                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 金山表原有成果：历史集成实现，单机 2×910B2、双 TP1、Qwen2.5-7B-Instruct BF16 | 与同部署固定种子随机路由比较，缓存命中率 24.44%→47.88%，P95 TTFT（三轮中位数）223.80→174.80 ms，请求吞吐 3.9542→3.9538 req/s。两侧均开原生前缀缓存，各轮 400/400 成功。          | [原报告、实测 SHA 与复现命令](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/d4e688d6d841963e7d9f8ccade2b60abc33ad012/submissions/pr-154-merge-gates-prefix-routing-benefit-2chip-68cf893a4-03a12f9bdd/README.md) · [逐轮原件](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/d4e688d6d841963e7d9f8ccade2b60abc33ad012/submissions/pr-154-merge-gates-prefix-routing-benefit-2chip-68cf893a4-03a12f9bdd/raw) · [缓存计数和汇总](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/d4e688d6d841963e7d9f8ccade2b60abc33ad012/submissions/pr-154-merge-gates-prefix-routing-benefit-2chip-68cf893a4-03a12f9bdd/comparison.json) |
| 历史 dev4，35B SWE，四卡双 TP2                                               | Random→Prefix 平均输出吞吐在 C4/C8/C16 分别变化 +6.34%/+14.72%/+17.52%。旧负载未在本次工作区恢复；历史 Prometheus 完整性缺项保留，不能据此独立验证原生命中率。                   | [冻结的逐轮数据与缺损清单](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/f57dbc2b1020c6f50b7a0a36db43fcd8a0bd0198/website-publication/data/leaderboard_prefix_routing_mean_evidence.json)                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 独立插件，11 月 15 日 eager 诊断实测，35B SWE C4                             | Prefix 相对 Random 的 TTFT P95 三轮均值降低 36.98%，输出吞吐变化 −0.56%。Lifecycle 相对 Prefix 的输出吞吐变化 +1.28%。只覆盖本次模型、负载、补丁和运行配置，不主张稳定吞吐优势。 | [九个窗口](swe-c4/windows.csv) · [三组对照](swe-c4/comparisons.csv) · [逐请求数值摘录](swe-c4/timings.csv)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 独立插件可靠性                                                               | 已有 CPU、包、协议复核；首轮原生 restart 失败记录保留，原生故障包由维护者持有，不视为最终发布 wheel 原生故障矩阵已通过。                                                         | [固定验收与缺项](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/70b28cc53b764e9555f341d7979c2a490208c219/archive/prefix-router-reliability-20261017/README.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

### 11 月 15 日 eager 诊断实测的环境与方法

- 单机 4×910B2，两个 TP2+EP 副本，Qwen3.5-35B-A3B BF16。vLLM 源码
  `e204c0f2818702314a96f8040eab134089888866`，Ascend 源码
  `82f586f8dcb5f92f8f3d55727bc05b3e2887606d`，保留宿主已有差异并加兼容补丁；三策略共用 eager、APC、Mamba align。见
  [软件版本](swe-c4/server-metadata.json)、[权重及 tokenizer 的哈希](swe-c4/model.sha256)、[后端完整实测命令](swe-c4/scripts/serve_backend.sh)。
- 插件基于
  [5ba9d710](https://github.com/vLLM-HUST/vllm-hust-prefix-router/commit/5ba9d710d8cb2e8b1a584b9453d25a61f6470c5d)，加固定种子
  Random、opt-in salt 哈希补丁；实验 wheel SHA-256 为
  `5a9d407b509d77d77a57ce9f197ac9c7712e9c5b67ad8aa06ad9e30d096feca4`。没有验证未修改主线、后续节点负载策略或正式发布包的性能。
- 客户端
  [swe-prefix-reuse 固定版本](https://github.com/vLLM-HUST/swe-prefix-reuse/tree/ef43ced21e870e2faa1122f89f831ddf39443bf6)
  加观测补丁；[NVIDIA Open-SWE-Traces 固定版本](https://huggingface.co/datasets/nvidia/Open-SWE-Traces/tree/fb0c0dccc7a5cce79b3f6de891848acdede36685)，准备
  8 条轨迹，workload SHA-256 为 `5350408c234545cd9e1202857708b7b79873dbdc9a8964f1aa26dcbb503d53a5`。C4
  为全入口持续维持四个在途请求。各窗口仅推进到前四条轨迹，未完成整条会话，未评分 SWE 任务质量或 SLO。
- 每次重启两个后端及 Router，空缓存开始，统一引擎内部预热。三轮顺序为 R/P/L、P/L/R、L/R/P；Random seed 20261115。三策略均走同一外部入口；每个窗口目录保留
  router.json、Router/后端/客户端 argv.json 和实测
  summary.json。对照配置：[Random](swe-c4/r1-random/router.json)、[Prefix](swe-c4/r1-prefix/router.json)、[Lifecycle](swe-c4/r1-lifecycle/router.json)。实际事件源为无版本
  best-effort 流，未验证完整故障恢复。
- 每窗口 900 秒内收到的输出 token 数除以 900 为输出吞吐；排空期的 token 不进入分子。每窗口 34 个请求，30 个窗口内完成、4 个排空完成，共 306/306
  成功。TTFT/TPOT/E2E 取窗口内完成请求的 P95，再对三轮等权平均；变化百分比按两组三轮均值之比减一。

| 策略        | 输出 token/s（均值 ± 样本标准差） | TTFT P95（ms） | TPOT P95（ms/token） | E2E P95（s） |
| ----------- | --------------------------------: | -------------: | -------------------: | -----------: |
| Random      |                      15.71 ± 0.20 |        2579.99 |               259.16 |       301.31 |
| Prefix-only |                      15.62 ± 0.11 |        1625.91 |               260.78 |       297.53 |
| Lifecycle   |                      15.82 ± 0.11 |        1610.08 |               254.47 |       294.90 |

首 token 等待缩短，但 Prefix 的吞吐略低，Lifecycle/Random 的吞吐差异为 +0.72%。三轮不支持普遍吞吐获益。历史 dev4
与本次的负载和宿主均有差异，绝对吞吐差距不能归因于插件。正式窗口 fallback 为零；预检阶段有导入、KV 接口和启动故障，后续修复后冻结配置，不能把“正式请求零失败”解释为从未有启动失败。

### 离线核验与原件范围

在本目录运行 `python verify.py`（仅标准库）。程序核对公开文件哈希、九份客户端原始 summary，使用不含题文和输出 ID
的逐请求时刻、计数摘录重新计算吞吐、延迟与三组对照。[原件清单](originals-manifest.json) 给出选定文件的归档相对路径、字节数、SHA-256
和是否原样收录。timings.csv 的 in_window 由原始 chunk 计数按单调时间 0≤t\<900 汇总，属于数值摘录，不是完整原始请求日志。

原件调阅联系人为插件维护者 [@Amber1qq](https://github.com/Amber1qq)。完整请求、chunk、原始遥测、启动日志和实验补丁在项目的
swe-c4-20261114/evidence 证据目录留存；数值摘录支持离线复算，哈希不替代原件访问。需进一步核对时，通过
[PrefixRouter #1](https://github.com/vLLM-HUST/vllm-hust-prefix-router/issues/1) 联系维护者，按相对路径和哈希调阅。11
月 16 日的整理与复算没有再次占用设备，也不构成生产资格验收。
