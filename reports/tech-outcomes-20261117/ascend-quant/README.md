# tech-outcomes-20261117 / ascend-quant

状态：**DRAFT（本地证据整理，尚未提交 GitHub）** 整理日期：2026-11-13 范围：Qwen3.5-35B-A3B 混合量化制品（W4A8 experts + GDN W8A8
linear）的吞吐、PPL、任务分数证据映射。

> 本文只把已有实测结果绑定到制品、设备总量、评分器、输出和原始记录。 未测到的项目在“缺项”中明确列出，不用推算值补齐。

## 可引用结论（一句）

在 Ascend 910B3 单卡、同一 SWE 前缀复用负载、同一 **18.85 GiB** KV pin 下， W8A8 与 W4A8 专家 + GDN W8A8 的 C16 窗口吞吐分别为
**205.58 / 203.39 output tok/s** （差 1.1%，不构成加速结论）；W4A8 的部署收益是把权重从 BF16 的 66.99 GiB 降到 22.38
GiB，从而单卡可把 KV 开到 33.49 GiB，容量组 C16 达 404.81 output tok/s （该组不能与 18.85 GiB 的 W8A8 做同口径比较）。精度方面，**单卡
TP1 同日同卡实测**： W8A8 PPL **7.1785**、W4A8+GDN **7.4009**，W4A8 相对 W8A8 **+3.10%**； 双卡 TP2 的 W4A8 7.9114
与 TP1 差 6.45%，不能用于单卡结论。长上下文任务级 8K/32K/128K + 260K 均已测，无可见退化；但 BF16 锚点 PPL 显示 W4A8 比 BF16 差
+2.1%～+2.8%、比 W8A8 差 +1.95%/+2.14%/+4.38%（4K/32K/128K）。完整 262,144 任务级未测（capacity smoke 已通过）。
MMLU-Pro 12,032 题（单卡 TP1）：主制品 **79.17%**，相对 BF16 79.86% 为 **−0.69 pp**（p=0.0082），相对官方 W8A8 79.50% 为
−0.32 pp（p=0.238，不显著）， 相对 gcy W4A8（GDN bf16）79.75% 为 −0.57 pp（p=0.0287）；GDN 三投影 int8
存在小幅精度代价，不能写成"无差异"。 19.22 GiB 同 KV 附录：W4A8/GDN 与 W8A8 均为 820,136 tokens；C1 90.02 vs 89.63、 C2 144.19
vs 151.14、C4 208.55 vs 201.33、C8 320.17 vs 307.03、C16 212.66 vs 206.13， 方向不一致，不构成加速结论。满 262,144：两臂
capacity smoke 均通过（prompt 262,143

- 1 token，HTTP 200，约 57 s）；261K NIAH C1/C2 未触发；260.8K 长生成 C3 触发 （W4A8 2/5 vs W8A8 5/5，n=5，3 条截断；260K
  方向相反），不声称稳定退化或无退化。

## 1. 制品身份

### 1.1 主制品（下文所有 W4A8/GDN 数据均指这一份）

| 项               | 值                                                                                                                          |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 路径             | `/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot`                                                                                      |
| 基座             | Qwen3.5-35B-A3B @ `712cf74392b05026a6db2bf213d343747d1f6d45`                                                                |
| 配方             | `qwen35-35b-a3b-w4a8-gdnw8a8`                                                                                               |
| 配方 YAML        | `src/ascend_quant_toolkit/recipes/yaml/qwen35-35b-a3b-w4a8-gdnw8a8.yaml`                                                    |
| 配方 sha256      | `0bcef0f56ddebea8f34841127e6ac5b369ffee20853e5a7d148c344387fd9384`                                                          |
| 权重分片         | 6 个 `quant_model_weights-00001..00006-of-00006.safetensors`                                                                |
| 分片合计         | 24,047,130,272 B（artifact record 口径）                                                                                    |
| 报告口径制品大小 | **22.38 GiB**（生成脚本/周报口径；对应字节 24,096,662,520 B，按 2^30 精确换算为 22.44 GiB，源文档存在约 0.06 GiB 的口径差） |
| 量化类型         | `W4A8_DYNAMIC` 122,880；`W8A8_DYNAMIC` 360–361；`FLOAT` 1,611；`W8A16` 0                                                    |
| 量化范围         | MLP experts gate/up/down：W4A8；30 层 GDN `linear_attn.in_proj_qkv/in_proj_z/out_proj`：W8A8；10 层 `self_attn` q/k/v：W8A8 |
| BF16 保留        | `self_attn.o_proj`、`in_proj_ba`、`shared_expert`、`lm_head`、`visual`、`mtp`                                               |
| 验证结果         | `ascend-quant-toolkit verify-artifact`：`valid: true`（2026-10-28）                                                         |
| Artifact record  | `vllm-ascend-quant-hust/docs/artifacts/qwen35-w4a8-gdnw8a8-tp1-20261023.md`                                                 |

分片 sha256（来自 artifact record）：

| 分片                                             |          字节 | sha256                                                             |
| ------------------------------------------------ | ------------: | ------------------------------------------------------------------ |
| `quant_model_weights-00001-of-00006.safetensors` | 4,292,174,512 | `3db59b946b0d406c3c7c9419ac5a88feb08214651799a919f72622ef991e88de` |
| `quant_model_weights-00002-of-00006.safetensors` | 4,298,505,816 | `dfb536f7946787b66a91919825b273affcbb8bf2f53eab8a56d75e5b3f032624` |
| `quant_model_weights-00003-of-00006.safetensors` | 4,298,560,512 | `7301f4beec2aff283873057dad8d4bf24f457124b9a8ec5c0b0c17e4d055fab3` |
| `quant_model_weights-00004-of-00006.safetensors` | 4,298,560,392 | `c18f7806d82d32ef8c3c14984adb54f34b9ac877904bdccd0fb1c1d9f7d7b5e5` |
| `quant_model_weights-00005-of-00006.safetensors` | 4,296,557,224 | `35f9775bf2f999e8c418278d3c3ef78c955d60d5635713f5203050f820d9d8f8` |
| `quant_model_weights-00006-of-00006.safetensors` | 2,562,771,816 | `84410167399a32558aa0acb2808a09b9cc6fd5bbf62fc19f5fc4827b3e44c450` |

### 1.2 对照制品

| 用途            | 路径                                              | 制品大小/量化                                                           |
| --------------- | ------------------------------------------------- | ----------------------------------------------------------------------- |
| BF16 基线       | `/data/jxd/Qwen3.5-35B-A3B`                       | 71,927,122,971 B = 66.99 GiB，14 分片                                   |
| 已发布 W8A8     | `/data/jxd/Qwen3.5-35B-A3B-W8A8`                  | 39,810,890,505 B（目录）= 37.08 GiB；官方 W8A8 配方                     |
| dense-plus 消融 | `/data/jxd/Qwen3.5-35B-A3B-W4A8-DenseW8A8`        | 23,888,188,915 B；在 W4A8/GDN 上再把 o_proj + shared_expert 量化成 W8A8 |
| lm_head 消融    | `/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot-LMHeadW8A8` | 在 W4A8/GDN 上把 lm_head 量化成 W8A8                                    |

## 2. 指标—证据映射（核心表）

每一行都必须带“设备总量”和“原始记录”。没有原始记录支撑的指标不进入引用。

本表保留采集时的源侧路径；公开归档后的逐文件对应关系以 **evidence-map.csv** 和 **originals-manifest.json** 为准。

| #   | 指标                              | 值                                                                     |                  制品 | 设备总量                           | 评分器/协议                                                                     | 输出                               | 原始记录                                                                                                 |
| --- | --------------------------------- | ---------------------------------------------------------------------- | --------------------: | ---------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1   | 单卡部署权重                      | 66.99 GiB → 22.38 GiB                                                  |       BF16 → W4A8/GDN | 1×910B3（TP1 服务）                | 字节数/U 口径                                                                   | 权重字节 + KV pin                  | `docs/artifacts/qwen35...md`；`gen_w4a8gdn_tp1_curve.py` KV_NOTE；`du -sb`                               |
| 2   | 窗口吞吐 C1                       | 90.1878 output tok/s                                                   |              W4A8/GDN | **1×910B3**（TP1）                 | `swe-prefix-reuse` 900 s 窗口，C1                                               | `summary.json`                     | `/data/jxd/swe-prefix-reuse/results/w4a8ours-tp1-pinmax-c1-900s/`（summary sha256 `0feaa4...`）          |
| 3   | 窗口吞吐 C2                       | 142.9478 output tok/s                                                  |                  同上 | **1×910B3**                        | 同上，C2                                                                        | `summary.json`                     | `...c2-900s/`（sha256 `774a08...`）                                                                      |
| 4   | 窗口吞吐 C4                       | 203.7911 output tok/s                                                  |                  同上 | **1×910B3**                        | 同上，C4                                                                        | `summary.json`                     | `...c4-900s/`（sha256 `150713...`）                                                                      |
| 5   | 窗口吞吐 C8                       | 312.7022 output tok/s                                                  |                  同上 | **1×910B3**                        | 同上，C8                                                                        | `summary.json`                     | `...c8-900s/`（sha256 `94bae0...`）                                                                      |
| 6   | 窗口吞吐 C16                      | 404.8133 output tok/s                                                  |                  同上 | **1×910B3**                        | 同上，C16                                                                       | `summary.json`                     | `...c16-900s/`（sha256 `9b2cb4...`）                                                                     |
| 7   | PPL（bf16）                       | 7.1694                                                                 |                  BF16 | **2×910B3**（TP2）                 | wikitext-2，298 chunk，296,755 token，并发 8，同 session 配对 ΔNLL              | `ppl_bf16_20261022.json`           | `/data/jxd/quant-Qwen3.5-35B-A3B/logs/ppl_bf16_20261022.json`（sha256 `7b03a5...`）                      |
| 8   | PPL（W8A8）                       | 7.1750                                                                 |             官方 W8A8 | **2×910B3**（TP2）                 | 同上                                                                            | `ppl_w8a8.json`                    | `.../logs/ppl_w8a8.json`（sha256 `2a773c...`）                                                           |
| 9   | PPL（experts int4，GDN bf16）     | 7.8551                                                                 |              gcy W4A8 | **2×910B3**（TP2）                 | 同上                                                                            | `ppl_w4a8gcy.json`                 | `.../logs/ppl_w4a8gcy.json`（sha256 `8952ab...`）                                                        |
| 10  | PPL（本制品）                     | **7.9070**（raw 7.907）                                                |              W4A8/GDN | **2×910B3**（TP2）                 | 同上                                                                            | `ppl_w8a8nr.json`                  | `.../logs/ppl_w8a8nr.json`（sha256 `e3900f...`）                                                         |
| 11  | HLE-Verified Gold                 | 145 / 668 = **21.71%**                                                 |              W4A8/GDN | **2×910B3**（TP2+EP2，卡 2,3）     | `strict_json`；judge = 同一制品 `qwen35-a3b-w8a8nr-tp2`；unresolved=17 计入分母 | `public/report.json`               | `/data/jxd/hle-verified-public-20261106-ep2/`（report sha256 `44027b...`，整包 `SHA256SUMS`）            |
| 12  | HLE-Verified Revision             | 302 / 1143 = **26.42%**                                                |              W4A8/GDN | **2×910B3**（TP2+EP2，卡 2,3）     | 同上；unresolved=38 计入分母                                                    | `public/report.json`               | 同上；sealed run `hle-revision-ep2-w4a8ours-tp2-20261106`                                                |
| 13  | FrontierScience Olympiad          | 60 / 100 = **60.00%**                                                  |              W4A8/GDN | **2×910B3**（TP2，卡 0,1）         | `mmlu_extract_first`，success_threshold=7                                       | `results/frontierscience.csv`      | `/data/jxd/hust-eval-client-frontierscience/runs/fs-w8a8nr-tp2-20261106/`；evidence bundle `0144d0fa...` |
| 14  | FrontierScience Research          | 1 / 60 = **1.67%**；mean_research_score **2.366**                      |              W4A8/GDN | **2×910B3**（TP2，卡 0,1）         | 同上                                                                            | `results/frontierscience.csv`      | 同上；run `fs-w8a8nr-tp2-20261106`                                                                       |
| 15  | 长上下文 NIAH 8K/32K/128K         | 单针 100%/100%/100%；多针 90%/90%（W8A8）vs 100%/90%（W4A8）           |      W8A8 vs W4A8/GDN | 1×910B3                            | 预注册 P2-3；NIAH，temperature 0，串行                                          | `niah_{w8a8,w4a8}.jsonl`           | `long-context/run_20261113T111012Z/out/`                                                                 |
| 16  | 长生成 32K/128K                   | 32K 80% vs 100%；128K 60% vs 60%                                       |      W8A8 vs W4A8/GDN | 1×910B3                            | 预注册 P2-3；exact substring / ROUGE-L / distinct-4                             | `longgen_{w8a8,w4a8}.jsonl`        | 同上；`stability.json`                                                                                   |
| 17  | 长上下文 260K 扩展                | NIAH 30/30=100% vs 29/30=96.7%；长生成 80% vs 100%                     |      W8A8 vs W4A8/GDN | 1×910B3                            | P2-3 客户端；260,000 token prompt + 1024 生成                                   | `niah/longgen_{w8a8,w4a8}.jsonl`   | `long-context-256k/run_20261116T042838Z/out/`                                                            |
| 18  | BF16 锚点长上下文 PPL 4K/32K/128K | W4A8 vs BF16 +2.09%/+2.50%/+2.83%；vs W8A8 +1.95%/+2.14%/+4.38%        | W4A8/GDN vs BF16/W8A8 | BF16 2×910B3 TP2；量化 1×910B3 TP1 | P2-2 teacher-forcing 逐位置 NLL；KV 8 GiB；bt 4096；串行                        | `longctx_ppl.csv` / `longctx.json` | `p2-2-bf16-anchor/run_20261116T071251Z/out/`                                                             |
| 19  | 长文本性能 C1 8K/32K/128K/260K    | TTFT ~0.66/2.7/18/57 s；W4A8 vs W8A8 TPOT +0.6%~+2.2%，total tok/s −1% |      W8A8 vs W4A8/GDN | 1×910B3（TP1）                     | `vllm bench serve` random，输出 256，1 warmup +3 reps                           | `longperf_summary.csv`             | `long-context-perf/run_20261116T075647Z/out/`                                                            |

> 表中的“设备总量”按实际运行计：吞吐曲线 1 卡；PPL 2 卡（BF16 67 GiB 装不下单卡，A/B 两边同为 TP2）；HLE/FS 2 卡。 不能把 HLE/FS 的 TP2
> 分数写成单卡结果，也不能把吞吐曲线的 1 卡数据写成 TP2 结果。

## 3. W4A8 / W8A8 官方对照状态

### 3.1 必须满足的条件

- 单卡（TP1）
- 同模型（Qwen3.5-35B-A3B；同一权重来源/修订）
- 同负载（同一份 `prepared/qwen35.json`，sha256 `ac1fbf98...`）
- 同 KV 预算
- 同客户端/协议（60 s 资格 + 900 s 正式窗口，seed 0，chips 1）

### 3.2 W8A8 单卡 KV 上限

| 口径                                               |            KV |  tokens | 262144 并发 | 来源                                                                      |
| -------------------------------------------------- | ------------: | ------: | ----------: | ------------------------------------------------------------------------- |
| 已发布单卡协议 util=0.95，显式 pin `20242376704` B | **18.85 GiB** | 805,156 |       3.07× | `results/tp1-c*-900s/config.json`；`swe_server_w8a8_20261014T032044Z.log` |
| util=0.99 实测上限                                 | **19.22 GiB** | 820,136 |       3.13× | `logs/swe_server_w8a8_20261023T145955Z.log`                               |

- **18.85 GiB** 是与已发布协议一致的显式 pin，也是做同 KV 对照时最稳妥的公共上限。
- **19.22 GiB** 是 util=0.99 下测到的绝对上限；同一日志还给出 `--kv-cache-memory=18.58 GiB`（fit requested）和
  `18.83 GiB`（fully utilize free memory） 两个建议值，19.22 GiB 的可复现性不如 18.85 GiB。
- W8A8 服务端记账：weights 38.18 GiB、peak activation 2.85 GiB、NPU graph 0.49 GiB、 non-torch 0.10
  GiB；权重体积决定了它不可能达到 W4A8 的 33.49 GiB。
- 因此，同 KV 对照的公共预算只能取 **≤18.85 GiB**（协议一致）或 **≤19.22 GiB** （两臂都改用 util=0.99 且保持同样开销）。W4A8 臂必须从 33.49
  GiB 降到这个公共值。

### 3.3 已有的同 KV（18.85 GiB）对照

两份结果目录的 `config.json` 都记录 `tensor_parallel_size=1`、
`gpu_memory_utilization=0.95`、`kv_cache_memory_bytes=20242376704`（18.85 GiB）， 同一份
`prepared/qwen35.json`（sha256 `ac1fbf98...`），同一 60s+900s 协议：

| C   | W8A8 @18.85 GiB | W4A8/GDN @18.85 GiB | 差值（W4A8 vs W8A8） |
| --- | --------------: | ------------------: | -------------------: |
| C1  |           88.03 |               90.17 |                +2.4% |
| C2  |          148.91 |              141.34 |                −5.1% |
| C4  |          198.66 |              202.80 |                +2.1% |
| C8  |          304.70 |              302.80 |                −0.6% |
| C16 |      **205.58** |          **203.39** |                −1.1% |

原始目录：

- W8A8：`/data/jxd/swe-prefix-reuse/results/tp1-c{1,2,4,8,16}-900s/`
- W4A8/GDN：`/data/jxd/swe-prefix-reuse/results/w4a8ours-tp1-c{1,2,4,8,16}-900s/`

这组满足“单卡、同模型、同负载、同 KV 预算”。C16 的 205.58 vs 203.39 差 1.1%， 落在单次 900 s 观测的噪声量级内；**不据此声称任何加速或减速**。

### 3.4 容量组与消融（不能与 18.85 GiB 组比较）

| 组     | 制品       | 设备    |                       KV 预算 |    C1 |     C2 |     C4 |     C8 |    C16 |
| ------ | ---------- | ------- | ----------------------------: | ----: | -----: | -----: | -----: | -----: |
| 容量组 | W4A8/GDN   | 1×910B3 | 35,956,596,497 B（33.49 GiB） | 90.19 | 142.95 | 203.79 | 312.70 | 404.81 |
| 消融   | dense-plus | 1×910B3 |              36,165,070,102 B | 87.14 | 134.18 | 197.61 | 300.44 | 375.47 |

容量组的 404.81 是“把权重省下的空间换成 KV”后的结果，不是同 KV 加速； 对外必须与 18.85 GiB 的 W8A8 点分开呈现，不能写成 404.81 vs 205.58 的加速比。

### 3.5 单卡部署证据（可直接引用）

- BF16 目录 71,927,122,971 B = **66.99 GiB**，超过单卡 64 GiB HBM，BF16 部署需要 TP2。
- W4A8/GDN 制品按报告口径为 **22.38 GiB**（源字节 24,096,662,520 B；精确换算 22.44 GiB， 引用时保留源口径）；单卡显式 pin KV
  35,956,596,497 B = **33.49 GiB**； 按源口径合计 55.87 GiB，落在 `0.95 × 60.96 GiB = 57.91 GiB` 的预算内。
- 该配置实际以 `TP1 / ASCEND_RT_VISIBLE_DEVICES=1` 跑完 C1–C16（见第 2 节原始结果）。
- 证据：`gen_w4a8gdn_tp1_curve.py` 的 `KV_NOTE`；`qwen35-w4a8-gdnw8a8-tp1-20261023/README.md` 第 127 行；
  `run_denseplus_ab_sweep.sh` 的启动参数。

### 3.6 19.22 GiB 同 KV 附录（2026-11-16 补测）

W8A8 的绝对上限组：W8A8 在 util=0.99 下自动估算到 19.22 GiB / 820,136 tokens （2026-10-23）；W4A8/GDN 在 2026-11-16
用显式 pin `--kv-cache-memory-bytes 20637317857` （19.22 GiB）复现同样的 **820,136 tokens**，两臂同
workload/同协议（60 s + 900 s，C1–C16）。

| C   | W8A8 @19.22 GiB（auto） | W4A8/GDN @19.22 GiB（pin） | Δ（W4A8 vs W8A8） |
| --- | ----------------------: | -------------------------: | ----------------: |
| C1  |                   89.63 |                      90.02 |             +0.4% |
| C2  |                  151.14 |                     144.19 |             −4.6% |
| C4  |                  201.33 |                     208.55 |             +3.6% |
| C8  |                  307.03 |                     320.17 |             +4.3% |
| C16 |                  206.13 |                     212.66 |             +3.2% |

两臂五档均 `valid=true`、`failed_requests=0`。方向不一致（C2 为负，其余为正）， 且单次 900 s 窗口的波动量级在
±3–5%；**不构成加速或减速结论**。正式对照仍用 18.85 GiB 组；19.22 GiB 仅作 W8A8 绝对上限附录。原始记录：
`data/results/perf/{w8a8,w4a8gdn}-19.22g/`，`evidence-map.csv` 的 `tput-19.22-*` 行。

### 3.7 #547 矩阵（gcy W4A8 线，已有实验，不重跑）

gcy W4A8 = experts int4 + GDN bf16（`/data/gcy/Qwen3.5-35B-A3B-W4A8`，25,099,795,178 B， 6
分片），**不是**主制品 GDN-W8A8；引用时不要混称"W4A8"。

| ID        | 实验                       | 臂                    | 设备 / KV               | 结果                                                                                                                      | 证据                                            |
| --------- | -------------------------- | --------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| P0-1      | SWE prefix-reuse C1–C16    | W8A8 vs gcy W4A8match | 1×910B3 / **18.85 GiB** | 88.95/149.61/195.85/302.75/194.80 vs 86.56/151.26/201.28/320.42/200.81；方向不一致，无加速结论                            | `data/results/gcy-547/p0/ab-*`                  |
| P0-2/P0-3 | MMLU-Pro 200-request burst | W8A8 vs gcy W4A8      | 1×910B3 / **18.85 GiB** | 441.67 vs 457.53 tok/s（+3.59%）；两次极差 12–15%；TPOT 29.89 vs 28.68 ms；不判为吞吐差异                                 | `data/results/gcy-547/p0/mmlu-*`                |
| P0 容量   | 容量组 C16                 | gcy W4A8max           | 1×910B3 / **32.55 GiB** | 394.34 vs 同权重 18.85 GiB 的 200.81（+96.4%）；收益来自 KV 容量，不是 int4                                               | `data/results/gcy-547/p0/ab-w4a8max-c16/`       |
| P1-1      | MMLU-Pro 12,032            | W8A8 vs gcy W4A8      | TP1                     | 79.496% vs 79.746%；McNemar +0.249 pp，p=0.351；未检出差异                                                                | `data/results/mmlu/`                            |
| P1-2      | 逐层误差分解               | W4A8/W8A8 vs BF16     | 离线                    | W4A8 rel_fro 11.55% vs W8A8 0.886%（13.0×）；无坏层/坏专家；近同粒度下限                                                  | `data/results/gcy-547/p1-2/`；大 CSV/JSONL 受控 |
| P1-3      | 反量化/算子四段探针        | W4A8/W8A8             | 1×910B3                 | 算子 0.42–0.51% vs 0.37%；C−B≈A−B；不是误差来源；scale_bias 是契约                                                        | `data/results/gcy-547/p1-3/`                    |
| P2-1      | 稀疏激活路由               | BF16/W8A8/gcy W4A8    | 2×910B3（TP2+EP）       | token×层完全一致 68.48%/50.75%；翻转与 ΔNLL 偏相关 −0.006；翻转无害                                                       | `data/results/gcy-547/p2-1/`；大 npz 受控       |
| P2-2      | 长上下文 PPL 4K/32K/128K   | BF16/W8A8/gcy W4A8    | 1×/2×910B3              | 4K +0.13%/+0.81%；32K +0.34%/−3.11%；128K −1.48%/−5.01%；核心 75% token 无 W4A8 优势                                      | `data/results/gcy-547/p2-2/`                    |
| P2-3      | NIAH + 长生成 8K/32K/128K  | W8A8 vs gcy W4A8      | 1×910B3 / **8 GiB**     | 单针 100/100/100 vs 100/99/100；多针 32K 90% vs 80%、128K 90% vs 90%；长生成 32K 80% vs 100%、128K 60% vs 60%；无一致优势 | `data/results/gcy-547/p2-3/`；cases 受控        |

结论边界：P1-2/P1-3/P2-1/P2-2 是归因实验；现有证据**不支持 int4 算子加速**， 也不支持"int4 降精度"的普遍结论；gcy 线的长上下文只覆盖 8K/32K/128K。
大原始：`layer_error_experts.csv`、`layer_error.jsonl`、`routed_*.npz`、 `niah_cases.jsonl` 在
`controlled-location.json`；15 份 `requests.jsonl` 在 Release `gcy-p0-swe-requests.tar.gz`。

## 4. 精度（PPL）与任务分数

### 4.1 PPL（wikitext-2，TP2，同 session bf16 基线，逐 chunk 配对 ΔNLL）

| checkpoint   | 量化范围                                 |        PPL |    vs bf16 |
| ------------ | ---------------------------------------- | ---------: | ---------: |
| bf16         | —                                        |     7.1694 |          — |
| W8A8         | experts + attn q/k/v W8A8                |     7.1750 |     +0.08% |
| experts int4 | experts W4A8 + attn q/k/v W8A8，GDN bf16 |     7.8551 |      +9.6% |
| **本制品**   | 上者 + GDN 三投影 W8A8                   | **7.9070** | **+10.3%** |

要点：以上是 **TP2（2×910B3）** 精度，不能当作单卡精度。GDN 三个投影的 int8 权重 相对 experts-only 多付 **+0.66%**
PPL。协议上限：`max-model-len 4096`、 999/1000-token chunk；**不覆盖长上下文**。

#### 单卡（TP1，1×910B3）

| checkpoint             |        PPL | 说明                                               |
| ---------------------- | ---------: | -------------------------------------------------- |
| **W8A8**               | **7.1785** | 2026-11-13，卡 7，TP1；298/298，failed=0           |
| **W4A8/GDN（本制品）** | **7.4009** | 2026-11-13，卡 7，TP1；复现 11-02 的 7.4005/7.4030 |
| BF16                   |     不可行 | 66.99 GiB 超过单卡 64 GiB HBM                      |

**单卡正式精度对照：W4A8 7.4009 vs W8A8 7.1785，+3.10%。**
原始记录：`logs/ppl_w8a8_tp1_20261113T101838Z.json`（sha256 `bf4a10d0...`）、
`logs/ppl_w8a8nr_tp1_20261113T101838Z.json`（sha256 `2c1f8e47...`）。

同一 W4A8 制品的 TP1 vs TP2：**7.4009（TP1）vs 7.9114（TP2）**，TP2 高约 6.45%； W8A8 的 TP1 vs TP2 为 7.1785 vs
7.1750（+0.05%），基本一致。说明 TP 依赖主要出现在 W4A8/GDN 路径；双卡 W4A8 数字不能用于单卡部署结论。

### 4.2 任务分数

| 基准                     | 制品     | 设备               |                                  评分器 | 结果                                    | 备注                                              |
| ------------------------ | -------- | ------------------ | --------------------------------------: | --------------------------------------- | ------------------------------------------------- |
| HLE-Verified Gold        | W4A8/GDN | 2×910B3（TP2+EP2） | `strict_json`，自评分（judge = 同制品） | 145/668 = 21.71%（95% 半宽 3.13pp）     | `formal_acceptance=false`；unresolved 17 计入分母 |
| HLE-Verified Revision    | 同上     | 2×910B3（TP2+EP2） |                                    同上 | 302/1143 = 26.42%（95% 半宽 2.56pp）    | unresolved 38 计入分母                            |
| FrontierScience Olympiad | 同上     | 2×910B3（TP2）     |       `mmlu_extract_first`，threshold 7 | 60/100 = 60.00%                         | 单次运行                                          |
| FrontierScience Research | 同上     | 2×910B3（TP2）     |                                    同上 | 1/60 = 1.67%；mean_research_score 2.366 | n=60，单次运行                                    |

注意：

- HLE 的两个分数来自 **EP2 最终包** `hle-verified-public-20261106-ep2`；早期 2026-11-05 包为 Gold 112/668 =
  16.77%、Revision 266/1143 = 23.27%，不要与 EP2 混用。
- HLE 包 `formal_acceptance=false`，是单 trial、Qwen substitute、self-judge，不是官方 leaderboard 复现。
- FS 的 Research 轨样本仅 60，且分数很低；引用时必须带 n 和阈值。
- 任务分数和吞吐曲线不是同一设备总量：任务分数是 2 卡，吞吐曲线是 1 卡。

### 4.2.1 MMLU-Pro 12,032 题（2026-11-14 补测，主制品）

| 臂                     |           设备 |        正确/总数 |      准确率 | 95% Wilson     |
| ---------------------- | -------------: | ---------------: | ----------: | -------------- |
| BF16（已发布）         | 2×910B3（TP2） |     9,609/12,032 |     79.862% | [79.14, 80.57] |
| 官方 W8A8              | 1×910B3（TP1） |     9,565/12,032 |     79.496% | [78.77, 80.21] |
| gcy W4A8（GDN bf16）   | 1×910B3（TP1） |     9,595/12,032 |     79.746% | [79.02, 80.45] |
| **W4A8+GDN（本制品）** | 1×910B3（TP1） | **9,526/12,032** | **79.172%** | [78.44, 79.89] |

逐题配对（McNemar exact，同一 12,032 题）：

| 对比                  |         Δ | Wald 95%         |          p |
| --------------------- | --------: | ---------------- | ---------: |
| W4A8+GDN vs BF16      | −0.690 pp | [−1.195, −0.184] | **0.0082** |
| W4A8+GDN vs 官方 W8A8 | −0.324 pp | [−0.849, +0.201] |      0.238 |
| W4A8+GDN vs gcy W4A8  | −0.573 pp | [−1.080, −0.067] | **0.0287** |

结论：GDN 三投影的 int8 量化在 MMLU-Pro 上引入约 0.6–0.7 pp 的小幅精度代价 （vs BF16 p=0.008，vs experts-only W4A8
p=0.029）；与官方 W8A8 的差异（−0.32 pp） 在 12,032 题上未达显著（p=0.24）。这与 PPL 的方向一致（单卡 W4A8+GDN 7.4009 vs W8A8
7.1785，+3.10%）。BF16 是已发布 TP2，量化三臂 TP1，跨拓扑比较带此限定。
原始记录：`data/results/mmlu/`、`pair-analysis.md`；`results.json.gz` 在 Release。

### 4.3 长上下文稳定性（8K/32K/128K + 260K 扩展）

预注册 P2-3 套件：单卡 7、TP1、同 KV 8 GiB、`max-model-len 133120`、关前缀缓存、 请求串行、temperature 0。

| 测试           |       W8A8 |         W4A8/GDN |
| -------------- | ---------: | ---------------: |
| NIAH 单针 8K   |    100/100 |          100/100 |
| NIAH 单针 32K  |    100/100 |          100/100 |
| NIAH 单针 128K |      25/25 |            25/25 |
| NIAH 多针 32K  |       9/10 |            10/10 |
| NIAH 多针 128K |       9/10 |             9/10 |
| 长生成 32K     | 8/10 = 80% | **10/10 = 100%** |
| 长生成 128K    |  3/5 = 60% |        3/5 = 60% |

预注册 C1/C2 未触发；C3 机械触发但方向是 W4A8 更好（32K 复现率 +20 pp）， 不是退化。结论：**8K/32K/128K 上没有观察到 W4A8 相对 W8A8
的长上下文退化**。

#### 260K 扩展（260,000-token prompt + 1024 生成，256K class）

单卡 7、TP1、KV 8 GiB、`max-model-len 262144`、关前缀缓存、串行、temperature 0。

| 测试           |         W8A8 |       W4A8/GDN |
| -------------- | -----------: | -------------: |
| NIAH 单针 260K | 25/25 = 100% |   25/25 = 100% |
| NIAH 多针 260K |   5/5 = 100% |      4/5 = 80% |
| NIAH 合计      | 30/30 = 100% |  29/30 = 96.7% |
| 长生成 260K    |    4/5 = 80% | **5/5 = 100%** |
| ROUGE-L        |        0.607 |          0.754 |
| distinct-4     |        0.547 |          0.714 |

C1/C2 未触发；C3 方向仍是 W4A8 更好（长生成 100% vs 80%），不是退化。 结论：**260K prompt 上没有观察到 W4A8 相对 W8A8
的退化**；n=5，不能声称更好。 **完整 262,144 任务级未测**（capacity smoke 已通过），不能外推到满上下文任务级。

#### BF16 锚点长上下文 PPL（4K/32K/128K）

P2-2 teacher-forcing 逐位置 NLL，复用已有 BF16/W8A8，再补跑我们的 W4A8+GDN。 BF16 是 TP2（2 卡，66.99 GiB 单卡放不下），量化臂是
TP1 卡 7；同 KV 8 GiB、 `max-model-len 133120`、关前缀缓存、bt 4096、串行。

| 长度 |   BF16 |   W8A8 | W4A8+GDN | W4A8 vs BF16 | W4A8 vs W8A8 |
| ---- | -----: | -----: | -------: | -----------: | -----------: |
| 4K   | 6.5036 | 6.5122 |   6.6395 |       +2.09% |       +1.95% |
| 32K  | 6.8861 | 6.9099 |   7.0580 |       +2.50% |       +2.14% |
| 128K | 6.5150 | 6.4183 |   6.6994 |       +2.83% |       +4.38% |

W4A8 的逐 token 中位数 ΔNLL vs BF16 为 +0.00099/+0.00107/+0.00094； W8A8 中位数≈0。W8A8 在 128K 的聚合负号由极少数极端
token 拉出，不能读成“量化更好”。 结论：**BF16 锚点下 W4A8 有可测的长上下文 PPL 代价，且随长度上升；任务级 NIAH/
长生成接近天花板，没有显示这个差异，两种口径必须同时报告。** BF16 TP2 / 量化 TP1 的跨拓扑差异按 P2-2 规则只作参考；W4A8 vs W8A8 同拓扑可比。

原始记录：`long-context/run_20261113T111012Z/out/{niah,longgen}_{w8a8,w4a8}.jsonl`、
`long-context-256k/run_20261116T042838Z/out/{niah,longgen}_{w8a8,w4a8}.jsonl`、 `stability.json`。

### 4.4 长文本性能（C1 单请求）

单卡 7、TP1、同 KV 18.85 GiB、`max-model-len 262144`、关前缀缓存、`max-num-seqs 1`； `vllm bench serve` random，输入
8K/32K/128K/260K，输出 256，1 warmup +3 reps。

| 输入 | W8A8 TTFT / TPOT / out tok/s | W4A8+GDN TTFT / TPOT / out tok/s |
| ---- | ---------------------------: | -------------------------------: |
| 8K   |      699 ms / 16.4 ms / 52.7 |          657 ms / 16.5 ms / 52.5 |
| 32K  |     2728 ms / 16.7 ms / 36.6 |         2744 ms / 16.9 ms / 36.3 |
| 128K |    18093 ms / 18.0 ms / 11.3 |        18237 ms / 18.4 ms / 11.2 |
| 260K |     57060 ms / 19.5 ms / 4.1 |         57624 ms / 19.8 ms / 4.1 |

W4A8 与 W8A8 的 TTFT 基本一致；W4A8 的 TPOT 略高 +0.6%～+2.2%， total tok/s 低约 1%。**没有看到 W4A8 在长文本上更快**；这是 C1
单请求， 不是并发吞吐，也未采集 HBM。完整表见 [long-context-performance.md](long-context-performance.md)。

### 4.5 满 262,144 与 261K/260.8K 任务级（2026-11-16 补测）

**精确满上下文 capacity smoke**：两臂各发一个 prompt=262,143 token、`max_tokens=1` 的 `/v1/completions` 请求（总计
262,144 token）。两臂均 HTTP 200，`usage.prompt_tokens =262,143`、`completion_tokens=1`，耗时约 57.3 s；W8A8 与
W4A8/GDN 都完整接受了 262,144 上下文。原始记录：`data/results/longctx-262k/capacity_smoke_*.json`。

**任务级 NIAH（body 261,000，单针/多针）**：

| 类型      |                    W8A8 |               W4A8/GDN |
| --------- | ----------------------: | ---------------------: |
| 单针 261K | 24/25 = 96%（1 条截断） |           25/25 = 100% |
| 多针 261K |   3/5 = 60%（3 条截断） | 5/5 = 100%（1 条截断） |

预注册 C1/C2 均未触发（n=25/5，Wilson CI 重叠）；W4A8 名义上更好，但多针 n=5， 不能据此声称优势。

**长生成（body 260,800，目标段落逐字复述，greedy，上限 1024 token）**：

| 臂       |     复现率 | ROUGE-L | distinct-4 | 重复 10-gram | 输出中位字符 | 截断 |
| -------- | ---------: | ------: | ---------: | -----------: | -----------: | ---: |
| W8A8     | 5/5 = 100% |   0.606 |      0.731 |        0.231 |        2,115 |    1 |
| W4A8/GDN |  2/5 = 40% |   0.539 |      0.941 |        0.016 |        4,074 |    3 |

预注册 C3 **触发**（exact −60 pp；distinct-4 相对 +28.8%）。W4A8 输出更长、更多样， 但逐字复述率更低，且 3/5 在 1024 token
上限截断；n=5，且在 260,000 时方向相反 （260K：W4A8 5/5 vs W8A8 4/5）。因此：

- **不声称 W4A8 在 260.8K 稳定退化**；
- **也不再把“无退化”外推到 262,144 任务级**。安全表述是：满 262,144 容量两臂均通过； 260K 任务级无可见退化；260.8K 长生成出现 C3
  触发（n=5，高方差），需在结论边界中保留。 原始记录：`data/results/longctx-262k/`，`evidence-map.csv` 的 `longctx-262k-*` 行。

## 5. 原始记录与固定引用

### 5.1 已有报告/归档

以下优先给本报告归档路径；固定提交/Release 链接与源侧路径见 `originals-manifest.json`（`source_path` 字段）与 5.3。

| 内容                                    | 位置                                                                                  | 固定引用                                                                                                                                               |
| --------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 吞吐曲线报告（C1–C16、协议、限制）      | `vLLM-HUST/vllm-hust-benchmark/reports/qwen35-w4a8-gdnw8a8-tp1-20261023/README.md`    | 仓库当前 `data/w4a8-gdnw8a8-tp1-sweprefix-20261023` 分支 `83956a6749dcea75e69e170600075092a6442d6b`；`main` `51563c0e6b65f5563475b539dad54f236f7e0470` |
| 制品记录（配方、分片哈希、verify 结果） | `vLLM-HUST/vllm-ascend-quant-hust/docs/artifacts/qwen35-w4a8-gdnw8a8-tp1-20261023.md` | 工具包分支 `feat/toolkit-0.5.0-qwen35-w4a8` @ `b93c073a60a1e6b28dc29c68b56358b8c8c27e88`                                                               |
| 0.4.2 发布记录                          | `docs/releases/ascend-quant-toolkit-0.4.2-pypi-release-20261013.md`                   | tag `toolkit-v0.4.2`，peeled commit `50ba3f8fdad986fa647b0d55918d13b47a25ec42`；GitHub Release `411366927`                                             |
| HLE EP2 公开包                          | `data/results/hle/`（本报告归档；源侧路径见 `originals-manifest.json`）               | `report.json` sha256 `44027b80...`；整包 tar.gz                                                                                                        |
| HLE EP2 sealed runs                     | HLE 包内 `manifest.json`（`data/results/hle/` 整包）                                  | 见包内清单；不复制新证据                                                                                                                               |
| FrontierScience 运行                    | `data/results/frontierscience/`                                                       | full run tar.gz；evidence bundle `0144d0fa271b0bf52e840abd20d43ab6a8d491627feaa55e52238ae8d14ee8f9`；table sha256 `6edc7165...`                        |
| 曲线原始 summary                        | `data/results/perf/`                                                                  | 见 `evidence-map.csv` 逐档 SHA                                                                                                                         |
| PPL 原始 JSON                           | `data/results/ppl/`                                                                   | 见 `evidence-map.csv` 逐文件 SHA                                                                                                                       |
| workload                                | `data/original/swe-prefix-reuse/prepared-qwen35.json`                                 | sha256 `ac1fbf9885076c369fd25c5525d1163ff8b7945972cdf4c0c6d2d523ab7c1172`                                                                              |
| gcy #547 矩阵                           | `data/results/gcy-547/`                                                               | P0/P1/P2 已有实验；大原始受控/Release                                                                                                                  |
| MOD 仓文档                              | `vllm-ascend-quant-hust/docs/tech-outcomes-20261117/`                                 | 本地 commit `7bf54a11...`；push 后固定链接                                                                                                             |

### 5.2 复现命令（摘要）

```bash
# 单卡 W4A8/GDN 服务（C1-C16 同一实例）
ASCEND_RT_VISIBLE_DEVICES=1 TP=1 PORT=8111 MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.95 \
LABEL=w8a8nr EXTRA_SERVER_ARGS='--kv-cache-memory-bytes 35956596497 --max-num-seqs 16 \
--max-num-batched-tokens 4096 --speculative-config {"method":"mtp","num_speculative_tokens":2} \
--compilation-config {"max_cudagraph_capture_size":256,"cudagraph_mode":"FULL"}' \
./run_swe_server.sh

# 客户端：每档 60 s 资格 + 900 s 正式
swe-prefix-reuse run --workload prepared/qwen35.json \
  --endpoint http://127.0.0.1:8111/v1/completions --model qwen35-a3b-denseplus \
  --concurrency <C> --duration 900 --chips 1 --server-max-context 262144 \
  --server-metadata <metadata.json> --seed 0 --timeout 3600 \
  --output results/<tag>-c<C>-900s
```

### 5.3 本轮公开归档（2026-11-16）

- 小文件：`data/`，见 `originals-manifest.json` 与 `SHA256SUMS`
- 大文件：Release `tech-outcomes-20261117-ascend-quant`，见 `release-manifest.json`
- 受控原件：`controlled-location.json`（模型权重等）
- 指标到文件的映射：`evidence-map.csv`

### 5.4 MOD 仓配套文档（2026-11-17）

- 合并证据索引：`vllm-ascend-quant-hust/docs/tech-outcomes-20261117/evidence-index-0.4.2-0.5.0.md`
- 分工、版本与适用范围：`vllm-ascend-quant-hust/docs/tech-outcomes-20261117/contributions.md`
- 0.5.0 PyPI
  发布记录：`vllm-ascend-quant-hust/docs/releases/ascend-quant-toolkit-0.5.0-pypi-release-20261112.md`
- MOD 仓本地 docs commit：`7bf54a11ae047a6b25bc24450f8bde40d2ed04d5`（分支 `docs/tech-outcomes-20261117`）；
  push 后替换为 GitHub 固定链接。
- 三人核对表：本目录 [`VERIFICATION.md`](VERIFICATION.md)。

## 6. 缺项与替代表述

| 缺项                               | 原因                                                                                                                                                                                                          | 替代表述 / 后续                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 同 KV 预算的 W4A8 vs W8A8 正式对照 | 18.85 GiB 正式组已完整（C16 205.58 vs 203.39）；19.22 GiB 附录已补测（W8A8 auto 19.22 GiB / W4A8 pin 19.22 GiB，均 820,136 tokens；C2 −4.6%，C4–C16 +3.2～+4.3%，方向不一致）；33.49 GiB 容量组不能与它们比较 | 正式对照用 18.85 GiB；33.49 GiB 单独呈现；19.22 GiB 仅作 W8A8 绝对上限附录，不作加速结论                         |
| 长上下文稳定性（完整 262,144）     | 满 262,144 capacity smoke 两臂均通过（262,143 prompt + 1 token）；任务级 NIAH 261K、长生成 260.8K 已测；260.8K 长生成 C3 触发（W4A8 2/5 vs W8A8 5/5，n=5，3 条截断；260K 方向相反）                           | 容量结论可写“262,144 两臂均通过”；任务级质量只写 260K 无可见退化 + 260.8K C3 触发/高方差，不声称稳定退化或无退化 |
| gcy #547 矩阵                      | 已有实验，不重跑；小结果已进 `data/results/gcy-547/`；大原始在 `controlled-location.json`/Release                                                                                                             | P0/P1/P2 结果与 KV 见 §3.7；不声称 int4 加速/降精度                                                              |
| 0.5.0 PyPI/Release 原始记录        | PyPI 0.5.0 已找到：wheel `ad3b320f...` 48,923 B、sdist `ea065055...` 66,144 B（2026-11-12）；无 GitHub Release；本地 dist wheel `9b55d3d1...` 不是 PyPI 字节                                                  | 引用 PyPI 固定链接/哈希；不写 GitHub Release；本地 wheel 只作本地构建                                            |
| 0.4.2 运行期 E2E 原件              | 旧 repo 的两份 E2E 文档不在当前 checkout                                                                                                                                                                      | 只链旧 Release/固定提交，不复制、不重跑；本报告不引用其数值                                                      |
| issue #664 验证总表逐项对齐        | 已取得 #664 总表；按 `evidence-map.csv` 逐行映射                                                                                                                                                              | 提交回帖附 PR + Release + 缺项                                                                                   |
| HLE 正式验收                       | `formal_acceptance=false`，自评分、单 trial                                                                                                                                                                   | 引用时带“非官方 leaderboard/自评分”限定                                                                          |
| 任务分数与吞吐不同设备             | 任务分数 TP2，吞吐 TP1                                                                                                                                                                                        | 表格中设备总量分列；不合并成“端到端单卡”结论                                                                     |

## 7. 一句话提交用结论（备用）

> 单卡 910B3、同一 18.85 GiB KV pin 下，W8A8 与 W4A8 专家 + GDN W8A8 的 C16 窗口吞吐为 205.58 vs 203.39 output
> tok/s（差 1.1%，无加速结论）；W4A8 的收益是 权重从 66.99 GiB 降到 22.38 GiB，单卡可把 KV 开到 33.49 GiB，容量组 C16 达 404.81
> output tok/s（不可与 18.85 GiB 组直接比较）；单卡 PPL：W8A8 7.1785 vs W4A8 7.4009（+3.10%）；双卡 TP2 的 +10.3%
> 只作双卡参考；长上下文任务级 8K/32K/128K + 260K 无可见退化（单针 100%，多针 90–100%，长生成 32K 80% vs 100%、 128K 均 60%、260K
> 80% vs 100%）；BF16 锚点 PPL 显示 W4A8 比 BF16 +2.1%～+2.8%、 比 W8A8 +1.95%/+2.14%/+4.38%；完整 262,144
> 任务级未测（capacity smoke 已通过）。 MMLU-Pro 12,032 题：主制品 79.17%，vs BF16 −0.69 pp（p=0.0082），vs 官方 W8A8
> −0.32 pp（p=0.238），vs gcy W4A8 −0.57 pp（p=0.0287）。 19.22 GiB 同 KV：C1 90.02 vs 89.63、C2 144.19 vs
> 151.14、C4 208.55 vs 201.33、 C8 320.17 vs 307.03、C16 212.66 vs 206.13；方向不一致，无加速结论。 满 262,144：两臂
> capacity smoke 均通过（262,143 prompt + 1 token）； 261K NIAH C1/C2 未触发；260.8K 长生成 C3 触发（W4A8 2/5 vs
> W8A8 5/5，n=5， 3 条截断；260K 方向相反），不声称稳定退化或无退化。 gcy W4A8 线（#547 P0/P1/P2，已有实验）：18.85 GiB C16 194.80 vs
> 200.81； MMLU-Pro 79.50% vs 79.75%（p=0.35）；逐层归因见 §3.7；长上下文 8K/32K/128K 无一致差异；不声称 int4 加速或降精度。
