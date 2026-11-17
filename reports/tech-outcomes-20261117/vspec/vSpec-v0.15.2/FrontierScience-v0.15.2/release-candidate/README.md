# FrontierScience：vSpec 0.15.2 两轮公开候选证据

本目录只公开冻结题号与衍生结果，不包含具体题目、逐题提示词、图像、请求正文或回答正文；完整原始数据仅留本地私有证据库。

数据集来源：https://huggingface.co/datasets/openai/frontierscience。读者须按上游条款自行取得原始数据，本包不再分发上游题目或媒体文件。

- 冻结题号及顺序：[题号.txt](题号.txt)，共 160 条；选择算法与来源哈希见 `params/selection.json`。
- 运行配置见 `params/`，客户端源码见 `client_source/`，编排/评分脚本见 `scripts/`；这些脚本保留本机路径，迁移机器时须重设环境、模型和数据源。
- 每臂 `solver-receipt-public.json` 保留总量、计时、配置哈希、以及未公开逐题响应的 SHA-256；`score-receipt.json`、`summary.json`、`per_case.csv` 仅含汇总与按题号的评分。
- `trace/` 是独立诊断轮，非正式配对轮；其决策按调度批次记录。
- `code/v0.15.2-local-compat.patch` 记录本地未提交修正。本轮不能被称为公开 wheel 或固定 SHA 的原样复现。

## 原始输出吞吐

单位为输出 token/s；计时不含服务启动。自然停止造成输出 token 数不同，吞吐比不等于同工作量端到端加速比。

| 轮次 | 臂 | 输出 token | 输出 token/s |
| --- | --- | ---: | ---: |
| round1 | MTP2 | 1,200,150 | 751.99 |
| round1 | MTP4 | 1,196,590 | 952.30 |
| round1 | MTP6 | 1,257,180 | 1022.58 |
| round1 | MTP8 | 1,271,431 | 1017.43 |
| round1 | vSpec | 1,246,992 | 1246.93 |
| round2 | MTP2 | 1,268,979 | 759.20 |
| round2 | MTP4 | 1,328,731 | 960.18 |
| round2 | MTP6 | 1,244,931 | 1015.88 |
| round2 | MTP8 | 1,272,149 | 1014.51 |
| round2 | vSpec | 1,268,953 | 1254.67 |

同轮输出吞吐比：

| 轮次 | 原生臂 | vSpec / 原生 |
| --- | --- | ---: |
| round1 | MTP2 | 1.658× |
| round1 | MTP4 | 1.309× |
| round1 | MTP6 | 1.219× |
| round1 | MTP8 | 1.226× |
| round2 | MTP2 | 1.653× |
| round2 | MTP4 | 1.307× |
| round2 | MTP6 | 1.235× |
| round2 | MTP8 | 1.237× |

## 质量判分

独立 judge 基于本地保存的回答判分，未决不视为已证实答错。本包不转发回答正文。

| 轮次 | 臂 | Olympiad 正确 | Research 达标 |
| --- | --- | ---: | ---: |
| round1 | MTP2 | 63/100 (未决 5) | 4/60 (未决 2) |
| round1 | MTP4 | 65/100 (未决 3) | 3/60 (未决 4) |
| round1 | MTP6 | 62/100 (未决 2) | 2/60 (未决 3) |
| round1 | MTP8 | 58/100 (未决 2) | 1/60 (未决 2) |
| round1 | vSpec | 60/100 (未决 3) | 5/60 (未决 2) |
| round2 | MTP2 | 65/100 (未决 1) | 2/60 (未决 6) |
| round2 | MTP4 | 66/100 (未决 1) | 3/60 (未决 4) |
| round2 | MTP6 | 59/100 (未决 4) | 2/60 (未决 3) |
| round2 | MTP8 | 64/100 (未决 3) | 3/60 (未决 4) |
| round2 | vSpec | 58/100 (未决 5) | 3/60 (未决 4) |

vSpec 的 Olympiad 正确数两轮分别比同轮 MTP4 少 5 和 8 题，不能声称质量无损。 两轮重复同一冻结题单，不是独立抽样。

## gamma 动态切换

独立诊断轮从 8 档起步，记录 18,167 次调度决策、18 次切换、9 条桥接提交。`trace/trace-decisions.jsonl` 与 `trace/trace-switches.json` 保留逐步证据。正式吞吐轮未开逐步 trace，诊断轮不纳入两轮吞吐表。

## 复现边界

测试模型为 Qwen3.5-35B-A3B，Ascend 910B2 两卡、TP=2、EP、BF16；并发上限 16、T=0、seed=17、每题最多输出 16384 token、自动 KV、无额外客户端预热，vSpec 候选 2/4/6/8 且从 8 开始、桥接开关为 1。本地安装版本报告为 0.15.2.post1，并含额外未提交兼容修正；本地源码基线 SHA 为 `ca9611d3f7494563069d7eda4330be2439f52da7`，补丁 SHA-256 为 `fbf3901dd1756de221bd79bc4acda339610252bbaea8036cbd99c94abc3e0f15`；原始 wheel SHA-256 为 `62631e85bfd6572d7110ede300a068313f360a3716fe912f092c4b5c2de8fd30`。算子与后端环境需要按所附配置自行配置。不同物理卡映射与旧模型权重哈希未完全锁定，不能将结果表述为公开 wheel 的原样跨机复现。

当前 `MANIFEST.sha256` 仅校验本候选包内文件；题目与回答原件不在包内，哈希只能对持有合法原件的人提供核验。
