# A3 长上下文 INT8 KV 配对观察（2026-11-03～04）

这是 A3 长度与延迟门槛下的合成负载工程对比，不是正式数据集验收或公开 v0.18 基线成绩。
**并发 64 的全窗口吞吐收益通过四轮复现；所有运行的延迟门槛均未通过；未评估答案质量。**
两侧均为同一 HUST 开发栈，仅改变 KV dtype（BF16/INT8），不是 upstream vLLM 与 vllm-hust 的比较。

## 已测结果

主指标为客户端在完整 1800 秒窗口收到的服务端累计 token 用量增量，包括窗口结束时尚未完成的请求。
不将 SSE 分块数当作 token 数，不删除前期 Prefill。第一批没有 token 事件，只保留完成请求吞吐，不能与新口径混合。

| 并发 | 配对轮数 | 每轮全窗口 INT8 吞吐变化 | 判断 |
| --- | --- | --- | --- |
| 40 | 2 | -14.46%、-13.52% | 负收益 |
| 48 | 2 | -5.52%、+1.37% | 不稳定 |
| 56 | 2 | -1.80%、-2.65% | 负收益 |
| 60 | 2 | +3.72%、+0.37% | 小幅收益，未满足候选门槛 |
| 64 | 4 | +10.74%、+11.57%、+9.23%、+4.37% | 四轮复现 |

并发64四轮 BF16/INT8 吞吐分别为249.315/276.0994、245.6261/274.0372、246.4039/269.1406、255.0039/266.1511 token/s。
配对提升率算术平均8.9772%；两组平均吞吐249.0872→271.3571 token/s，提升8.9406%。不要混淆两种聚合。

并发64全部请求加权平均：TTFT 194.8046→50.4957秒（-74.08%）；TPOT 141.8039→199.2075毫秒（+40.48%）；完整请求耗时485.0781→458.2742秒（-5.53%）。
KV容量1,293,312→2,305,832 tokens（约+78.3%）；日志最大Running 41→64。容量扩大降低排队，不意味着逐token生成变快。

预先约定工程规则：四轮配对收益均>0，配对收益算术平均≥3%，请求无失败。64是最低的已测且四轮确认点；61～63未测，不声称精确最小阈值或严格单调性，也不声称统计显著性。

## 配置与验收边界

完整结构化配置见[configuration.json](configuration.json)，所有30组标量及初始负收益见[results.json](results.json)。

- Qwen3.5-35B-A3B，2×Ascend 910B2，TP2，BF16权重；基线KV `auto`，候选KV `int8`。
- 每请求30720输入+2048输出，重复合成文本；temperature=0、seed=17、ignore_eos=true。
- 固定并发闭环，每组单请求预热后1800秒发起窗口，再排空；同一并发按BF16→INT8、INT8→BF16反转顺序。
- max_model_len=32768，max_num_batched_tokens=2048，gpu_memory_utilization=0.80，chunked prefill开启，prefix caching关闭。
- compilation mode=3，FULL_DECODE_ONLY，capture sizes与max_num_seqs见归档runner。没有enforce-eager回退。
- TTFT mean/P95/P99门槛为30/45/60秒；TPOT为40/60/80毫秒。30组全部未通过，7463个测量请求全部成功。
- 质量/答案一致性未评估。APC、MTP和其他异步设置不能由此推导为已验收。
- 服务日志记录的引擎版本保留在每组results中；实验时插件/Ascend精确提交、dirty diff、模型权重hash未完整冻结。不以归档时源码版本冒充实验版本。
- 910B2为实验执行期间本会话npu-smi核验结果；原始目录没有同时保存设备清单文件。

## 原始证据及复核

[evidence-manifest.json](evidence-manifest.json)记录三个原目录、压缩包SHA256及每个原文件的字节数和SHA256。大包拆为900000字节分片，各分片另有SHA256，按清单顺序拼接后得到原gzip字节：

- `initial-originals.tar.gz`：并发16/32/64初始6组，完成请求吞吐口径。
- `boundary-originals.tar.gz`：并发40/48的8组，包含启动时脚本快照、真实窗口token事件。
- `confirmation-originals.tar.gz`：并发56/60/64的16组，自适应控制器、脚本、原判定和全部事件。
- `summaries/`保存JSON语义副本并规范末尾换行，方便浏览；压缩包内保留原summary/config字节。
- `legacy-scripts-captured-at-archive.tar.gz`为初始脚本的归档时副本，不冒充启动时冻结快照。

运行 `python scripts/verify_a3_int8_kv.py` 验证压缩包成员、窗口token计数与网站卡片数据。

## 网站接入

本报告属于专题工程观察，不进入official-baselines或正式leaderboard。
网站数据源为 `website-publication/data/a3-int8-kv-20261104.json`，卡片ID为 `quantized-kv-cache-migration`。
负责人保留原登记。公开描述必须同时包含：指定负载、并发64、四轮收益、TPOT代价、延迟未通过及质量未评估。

采用个人分支→预览→页面检查→PR合并流程；禁止直接推送main。数据文件存在不等于已部署，发布状态由后续发布回执证明。
