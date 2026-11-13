# PegaFlow 技术成果补证：历史索引与 2026-11-13 中断记录

本次提交是**增补归档和失败披露**。新增正式 900 秒对照点为 **0**，没有新增可引用的吞吐提升结果。 历史五档数值可复算，但 Native/Candidate 的 seed、prepared
workload hash、记录的 tokenizer 软件版本不同， 且测量不在同期；不能把历史发布比值升级为严格匹配的 A/B 因果收益。

任务：[benchmark #664](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664)、
[PegaFlow #18](https://github.com/vLLM-HUST/pegaflow-hust/issues/18)。
代码与贡献归属更新：[PegaFlow PR #49](https://github.com/vLLM-HUST/pegaflow-hust/pull/49)，固定提交
`e2cdc2f7b32d8294b54a74545c05d103e5a5ecaf`。现有历史原件保持不变，不重复上传 Release。

## 历史五档：公开原件与复算

| 并发 | Native tok/s | Candidate tok/s | 发布记录比值变化 | Candidate 校验请求数 |
| ---- | -----------: | --------------: | ---------------: | -------------------: |
| C1   |    91.545556 |       65.523333 |        -28.4254% |                  118 |
| C2   |   152.721111 |      167.143333 |         +9.4435% |                  252 |
| C4   |   217.475556 |      237.967778 |         +9.4228% |                  354 |
| C8   |   291.106667 |      361.441111 |        +24.1611% |                  527 |
| C16  |   359.750000 |      459.715556 |        +27.7875% |                  706 |

未四舍五入比值的几何平均变化为 `+6.3422978866%`；**C1 退化保留**。 每档各一次 900 秒观察，不是置信区间、统计显著结果或当前版本性能。 Candidate 的 1957
条请求已逐条检查 token IDs、usage、预算、chunk token 数一致， 按 `[0,900)` 的实际 chunk 时间重算输出 token 后除以 900。 这是流协议与
token 数检查，**不是 SWE 解题正确率，也不是缓存数值等价证明**。

| 并发 | Native 窗口（北京时间）                               | Candidate 窗口（北京时间）                            |
| ---- | ----------------------------------------------------- | ----------------------------------------------------- |
| C1   | 2026-09-27T11:16:49+08:00 — 2026-09-27T11:31:49+08:00 | 2026-09-29T09:17:46+08:00 — 2026-09-29T09:32:46+08:00 |
| C2   | 2026-09-27T11:31:51+08:00 — 2026-09-27T11:46:51+08:00 | 2026-09-29T09:33:08+08:00 — 2026-09-29T09:48:08+08:00 |
| C4   | 2026-09-27T11:46:57+08:00 — 2026-09-27T12:01:57+08:00 | 2026-09-29T09:49:22+08:00 — 2026-09-29T10:04:22+08:00 |
| C8   | 2026-09-27T12:02:11+08:00 — 2026-09-27T12:17:11+08:00 | 2026-09-29T10:04:34+08:00 — 2026-09-29T10:19:34+08:00 |
| C16  | 2026-09-27T12:18:50+08:00 — 2026-09-27T12:33:50+08:00 | 2026-09-29T10:19:57+08:00 — 2026-09-29T10:34:57+08:00 |

每档 run/point ID、Native 原始请求文件的已记录 SHA、公开元数据 JSON pointer、Candidate 原始目录、runtime SHA、seed、workload
hash、逐档结果见 [historical-index.json](historical-index.json)。 原始链接、Release 大小/校验和及 Native 公开元数据重建方式见
[sources.json](sources.json)。 Native 索引来自固定 benchmark 提交中的公开 summary/config 抽取；其完整请求流没有在本次取回，
该公开记录也没有给出精确的 Native runtime SHA，报告明确保留缺项，不能用 Candidate SHA 代填。

历史 Candidate：Qwen3.5-35B-A3B BF16，模型修订
`712cf74392b05026a6db2bf213d343747d1f6d45`；2×910B2，TP2/PP1/DP1，EP 关闭； 上下文上限 262144、max sequences
16、batch tokens 4096；每卡 KV `26038239232` bytes， host pinned pool 30 GiB；APC、Mamba align、async、自然
MTP2、thinking、temperature 0、 FULL_AND_PIECEWISE。PegaFlow `cd64ecc283ff856a44437a9a25659929ef3a0653`；
vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804`； vLLM-Ascend
`03766ac696fde5ab1980d80ca0b8543d3580c989`。 精确启动命令、环境与原始日志沿用 [sources.json](sources.json)
指向的原件，而非由本报告重新编造。

## 11 月 13 日新尝试：配置、失败与缺项

本次使用实际分配的 **2×Ascend 910B2**，物理设备 0/7、CANN 逻辑 0/1；容器 RAM 上限 **64 GiB**。 PegaFlow 基于 main
`2e4cd2e9a244f7aefd732a442f9662a2acc88049` 加 [已记录兼容补丁](config/compatibility-patch.json)，仅构建
`ascend`，**未验证 RDMA**。 框架使用干净的官方镜像源码：vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、 vLLM-Ascend
`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。 模型修订与历史一致，14 个权重分片在停机前完成全量 SHA-256 匹配；完整远端 hash/build
receipt 尚未取回， 因此本提交不声称完成 formal artifact custody。

1. 初始每卡 24.25 GiB KV 在 Native 图捕获阶段 NPU OOM；此后两臂统一改为 **20 GiB/卡**， 其余主要控制为 BF16、TP2/PP1/DP1、max
   context 262144、max sequences 16、batch 4096、 APC、Mamba align、async、MTP2、thinking、seed
   17、FULL_AND_PIECEWISE（capture 3/6/12/24/48）。
1. 原 PegaFlow 在稀疏设备节点下误查物理 1，worker NUMA 注册失败；修复后已启动， 为两个 NUMA 节点各分配 15 GiB pinned
   pool。兼容修复的启动成功不等于性能或输出正确性通过。
1. Native 的 8 组首次输入冷/重复 32-token 检查通过；C2/60s 资格检查失败 0、1779 个窗口 token， 仅为短资格检查。其原始请求流尚未取回，不能作为正式
   performance point。
1. Candidate 前四组与 Native 相同；第 5 组 1637-token 输入，冷/重复输出在从 0 计数的 token 20 开始分歧，重复输出与 Native
   相同。流程在正式计时前停止。原探针短于 2048-token 缓存块， reset-prefix-cache 接口返回 404，因此无法判断为真实外存恢复损坏；数值波动与缓存问题均待排查。
1. 随后安排两臂各 10 对重复输入诊断。北京时间 **18:27** 工作区被资源管理员主动停止， 最后一次观察时 Native 仍在启动，未取回诊断结果，也未取回远端完整原始日志/证据包。

已保存的公开诊断字段及 token IDs 在 [diagnostics/](diagnostics/)；它们是**明确标注的抽取记录**， 不是完整原始 HTTP
流、原始请求流或完整服务日志。配置见 [config/](config/)； [drivers/](drivers/) 保留控制器和探针的原始源码快照，`.py.source` 字节不可由
formatter 改写。 这些快照依赖未完整取回的 run-env/prepared/build receipts，属于失败尝试的执行来源， 不能作为已经成功的、可直接启动的 formal
复现入口。

兼容补丁以 `.patch.gz` 保存，解压后的 SHA-256 与原执行记录一致，以保留 diff 上下文空白。 原始控制器中的公开模型 revision SHA 在仓库秘密扫描 baseline
中逐条标为已审核误报；未关闭扫描或修改源码快照。

## 验证与后续

```bash
python reports/tech-outcomes-20261117/pegaflow/verify_public_evidence.py
python reports/tech-outcomes-20261117/pegaflow/verify_public_evidence.py --archive /path/to/pegaflow.tar.gz
```

第一条验证本提交所有公开文件的大小/SHA、字段边界和历史比值；第二条还校验原 Release 字节并重新读取 Candidate 全量请求流计算窗口吞吐。验证不填补 Native
私有原件或中断实验的缺项。

继续需要：确认 91 仍分配可用（2 卡、64 GiB 容器 RAM、30 GiB host pool）；恢复后先复核持久目录、 设备映射及全部源/模型身份，完成 Native/Candidate
重复性诊断和真实 load/save 检查。 改动确定性、图模式或 MTP 必须两臂一致并重新固定计划。随后五档同期计时本身需 150 分钟， 加启动、资格检查、条件触发的反向复测预留约 3–4
小时；资源与正确性检查通过前不能承诺完成时间。 完整原件恢复后另行增补，不修改本次已公开的失败记录。此提交不关闭旧 T1–T5、TailGuard/RDMA 或 SWE 质量验收。
