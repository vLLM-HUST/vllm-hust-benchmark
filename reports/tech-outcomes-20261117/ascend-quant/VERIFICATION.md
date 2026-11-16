# 三人核对与证据表（2026-11-17）

状态：**AproAkk 已核对**本表所列文件/哈希/数值；**aly16-k、jxd1111 待本人签字**。
本表不代替本人确认；未签字项在 benchmark 报告中保留“待本人确认”。

## 1. 按线的核对表

| 线 / 制品 | 配方（sha256） | 制品 | 吞吐出处 | 精度出处 | KV 预算 | owner | AproAkk 核对 | 本人签字 |
|---|---|---|---|---|---|---|---|---|
| **gcy W4A8**（GDN bf16） | `qwen3_5_35b_a3b_w4a8_experts.yaml`（`3dd1102e…`） | `/data/gcy/Qwen3.5-35B-A3B-W4A8`，25,099,795,178 B | `swe_results/ab-*`；`data/results/perf/*` | `mmlu-pro-accuracy/`；`p1-2-layer-error/`；`p1-3-dequant-probe/`；`p2-1-routing-coupling/`；`p2-2-long-context/`；`p2-3-long-context-stability/` | 18.85 GiB；32.56/33.49 GiB | AproAkk | 已核对 | ✅ AproAkk |
| **GDN-W8A8**（主制品） | `qwen35-35b-a3b-w4a8-gdnw8a8.yaml`（`0bcef0f5…`） | `/data/jxd/Qwen3.5-35B-A3B-W8A8-NoRot`，24,096,662,520 B | `data/results/perf/{w8a8,w4a8gdn}-18.85g/`、`*-19.22g/`、`*-33.49g/` | `data/results/ppl/`；`data/results/mmlu/`；`data/results/hle/`；`data/results/frontierscience/`；`data/results/longctx*` | 18.85 / 19.22 / 33.49 / 8 GiB | jxd1111 | 已核对 | ⏳ 待签字 |
| **官方 W8A8** | `qwen3_5_35b_a3b_w8a8.yaml`（`b18286e8…`） | `/data/jxd/Qwen3.5-35B-A3B-W8A8`，39,810,890,505 B | `data/results/perf/w8a8-*` | `data/results/ppl/`；`data/results/mmlu/` | 18.85 GiB | TBD（aly16-k?） | 已核对 | ⏳ 待签字 |
| **BF16 基线** | n/a（原始模型） | `/data/jxd/Qwen3.5-35B-A3B`，71,927,122,971 B | `data/results/ppl/ppl_bf16_*` | `data/results/mmlu/bf16/` | n/a（MMLU/PPL） | TBD（aly16-k?） | 已核对 | ⏳ 待签字 |
| **toolkit 0.4.2** | `qwen25-w8a8`（`9c9385da…`） | 无 Qwen3.5 checkpoint | 发布/安装验证 | PyPI/Release 字节 | n/a | jxd1111 | 已核对 | ⏳ 待签字 |
| **toolkit 0.5.0** | `qwen25-w8a8`；`qwen35-35b-a3b-w4a8-gdnw8a8`（`0bcef0f5…`） | GDN-W8A8，6 分片 24,047,130,272 B | `data/results/perf/*` | `data/results/ppl/`、`mmlu/`、`hle/`、`frontierscience/` | 18.85 / 19.22 / 33.49 GiB | jxd1111 | 已核对 | ⏳ 待签字 |

## 2. KV 预算口径（引用时必须同时给出）

- **18.85 GiB**（20,242,376,704 B）：W8A8 vs W4A8/GDN 正式同 KV；gcy P0-1。
- **19.22 GiB**（20,637,317,857 B）：W8A8 绝对上限附录；两臂 820,136 tokens；不作加速结论。
- **33.49 GiB**（35,956,596,497 B）：GDN-W8A8 容量组；不可与 18.85 GiB 直接比较。
- **32.56 GiB**：gcy W4A8 容量组（ab-w4a8max）；不可与 18.85 GiB 直接比较。
- **8 GiB**（8,589,934,592 B）：P2-3/260K/261K/260.8K/262,144 capacity smoke。

## 3. 签字

- [x] **AproAkk**：已核对 benchmark 包文件、SHA256、数值、KV 预算与 MOD 证据索引；
  已核验 0.4.2/0.5.0 PyPI 字节和 0.4.2 tag 配方哈希。
- [ ] **aly16-k**：待确认分工/配方/制品/吞吐/精度出处。
- [ ] **jxd1111**：待确认 GDN-W8A8 制品、toolkit 版本与 runtime counterpart。

> 截止 2026-11-17 18:00。未签字项不冒充已核对。
