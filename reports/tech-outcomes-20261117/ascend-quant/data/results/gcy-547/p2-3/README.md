# P2-3 长上下文稳定性：NIAH + 长生成退化

回应 Issue #547 的第三条要求（「除吞吐/时延外，必须补精度与长上下文稳定性验证」）。
P2-2 已测 teacher-forcing PPL（分布级）；本轮补两个**任务级**稳定性指标：

1. **检索**：Needle-in-a-Haystack（NIAH），单针 + 多针，8K / 32K / 128K；
2. **长生成**：32K / 128K 上下文下生成 512 token 的复现率与退化指标。

## 预注册（写于 2026-11-05，跑之前，不事后改）

| 编号 | 判据 | 判读 |
|---|---|---|
| C1 | 单针 NIAH，按长度汇总（各深度合并）的 exact-match 准确率；\*\*\|W4A8 − W8A8\| ≥ 5 pp 且两者 Wilson 95% CI 不重叠\*\* | 判为"存在可检测的稳定性差异" |
| C2 | 多针 NIAH（3 针全中率）同 C1 的阈值 | 同上；n 小，必须同时报 CI |
| C3 | 长生成：复现率差 ≥ 10 pp，或 distinct-4 相对下降 ≥ 20% | 判为"长生成退化" |
| C4 | 无效输出（空串、被 stop 截断、请求失败）单独计数，不静默丢弃；单臂单长度 > 2% 则该格作废 | 防止用丢样本制造好看的数字 |
| C5 | 不以 BF16 为锚点（检索/复现有绝对真值）；不做未测长度的外推 | 边界 |
| C6 | 两个臂各自单卡、同 KV 8 GiB、同 bt 4096、关前缀缓存、客户端串行 | 口径 |

**同时报出的量**（不允许只报其中一个）：每长度的 n、命中数、准确率、Wilson 95% CI；
每深度的准确率；无效输出数；长生成的复现率、distinct-4、重复 10-gram 占比、输出长度中位数。

## 口径

| 项 | W8A8 | W4A8 |
|---|---|---|
| 卡 | **2** | **3** |
| 拓扑 | TP1 | TP1 |
| KV 预算 | 8 GiB | 8 GiB |
| prefill 分块 | `--max-num-batched-tokens 4096` | 同左 |
| 前缀缓存 | 关 | 关 |
| 采样 | temperature 0（greedy） | 同左 |
| 请求 | 串行 | 串行 |

- 两个臂**并发**跑在不同卡上（P1-1 有先例：W8A8 卡 7 / W4A8 卡 5 同日并行）。
  每臂仍是单卡、同模型、同负载、同 KV 预算，满足负责人的正式对照口径。
- **跨卡控制**：主实验结束后，在**对方卡**上重跑 8K 单针的一个 20 例子集；
  若同一臂两卡准确率差 > 5 pp，则标记"卡间效应"，主结果降级。
- 语料：wikitext-2 测试集（`../logs/wikitext2_chunks_1000.json`，297,053 token），
  P2-2 也用它；**不是** MMLU-Pro，不构成评测集泄漏。
- 128K 提示词 + 64 生成 token < `--max-model-len 133120`；长生成 131072 + 512 也在上限内。

## NIAH 设计

- 针：`The secret passcode for project <随机项目名> is <4位数字-3字母>.`
  每个 case 的项目名与口令唯一，避免跨 case 记忆。
- 单针深度：0%、25%、50%、75%、100%（针尾对齐上下文末尾）。
- 每格样本：8K 20 例、32K 20 例、128K 5 例（128K 单请求约 25 s，控制成本）。
- 多针：32K / 128K 各 10 例 × 3 针（深度 10% / 50% / 90%），问三个口令。
- 提问：`What is the secret passcode for project X? Answer with the passcode only.`
- 判对：规范化（小写、去标点）后输出包含该口令；多针要求三个全部包含。
- 生成：`max_tokens=64`、`temperature=0`、**不设 stop**。

> **协议修正（2026-11-06，首轮开跑后 10 分钟内发现，两臂同时适用）**
> 原设计为 `max_tokens=16` + `stop=["\n"]`。实测该模型在答案前固定输出
> `\n\n<think>\n\n</think>\n\n`，因此 stop 在第一个换行处即截断，245/245 条全部得到空串
> （记为 `out/niah_*.stopNL-invalid.jsonl`，保留备查）。去掉 stop、`max_tokens` 提到 64 后，
> 单条探针正常返回 `5738-TBT`（12 生成 token）。此项只影响「是否能取到输出」，不影响两臂对比；
> 两臂使用完全相同的客户端参数，首轮结果整体作废并重跑。

## 长生成设计

- 32K 10 例、128K 5 例。
- 在上下文末尾放一段 200-token 的目标文本，指令 `Repeat the passage verbatim, then stop.`
- 生成 512 token，temperature 0。
- 指标：目标文本的 exact substring 复现率、字符级 ROUGE-L、distinct-4、重复 10-gram 占比、
  输出长度中位数、提前停止/格式崩溃率。

## 产物

| 文件 | 内容 |
|---|---|
| `out/niah_cases.jsonl` | 预生成的用例（token id 提示词 + 真值） |
| `out/niah_<arm>.jsonl` | 逐 case 输出、命中、耗时、错误 |
| `out/longgen_<arm>.jsonl` | 逐 case 输出与指标 |
| `out/stability.json` / `out/stability.csv` | 汇总 + Wilson CI + bootstrap 差值 CI |

## 跑法

```bash
# 1) 预生成用例（只依赖 tokenizer，不占卡）
/data/jxd/envs/vllm-hust-v1-cann91/bin/python -I build_niah.py

# 2) 起服务。卡号在开跑前现查（npu-smi info -t usages），不要写死：
#    某卡 HBM 已用 > 20% 时 run_server.sh 会直接拒绝启动（防止挤占他人任务）。
#    有卡时优先两卡并行（下面例子用卡 2 / 卡 3，仅为示例）：
LABEL=w8a8 ASCEND_RT_VISIBLE_DEVICES=<卡A> PORT=8230 ./run_server.sh
LABEL=w4a8 ASCEND_RT_VISIBLE_DEVICES=<卡B> PORT=8231 ./run_server.sh
#    只有一张卡时空闲时，两臂在同一张卡上顺序跑（先全跑完一臂再换），
#    这样卡间差异和背景负载变化都被剔掉，可比性更好；两臂仍需同一 KV 8 GiB、bt 4096。

# 3) 抓结果（每个服务内部串行；两臂分别在两张卡上时可并行跑）
/data/jxd/envs/vllm-hust-v1-cann91/bin/python -I run_niah.py \
  --base-url http://127.0.0.1:8230 --model qwen35-a3b-w8a8 --arm w8a8
/data/jxd/envs/vllm-hust-v1-cann91/bin/python -I run_niah.py \
  --base-url http://127.0.0.1:8231 --model qwen35-a3b-w4a8 --arm w4a8

/data/jxd/envs/vllm-hust-v1-cann91/bin/python -I run_longgen.py \
  --base-url http://127.0.0.1:8230 --model qwen35-a3b-w8a8 --arm w8a8
/data/jxd/envs/vllm-hust-v1-cann91/bin/python -I run_longgen.py \
  --base-url http://127.0.0.1:8231 --model qwen35-a3b-w4a8 --arm w4a8

# 4) 汇总
/data/jxd/envs/vllm-hust-v1-cann91/bin/python -I compare_stability.py

# 5) 跨卡控制（主实验后，可选）
#    在一张不同的卡上重起某一臂，用 --limit 只跑各长度前 N 条做子集复核：
#      CARD=<另一张卡> LABEL=w8a8 PORT=8232 ./run_server.sh
#      run_niah.py --base-url http://127.0.0.1:8232 --model qwen35-a3b-w8a8 \
#        --arm w8a8 --limit 5 --out out/niah_w8a8_crosscard.jsonl
```

## 成本

| 阶段 | 估计 |
|---|---|
| 服务加载 ×2（并发） | 10–15 min |
| 单针 NIAH / 臂 | 约 20 min（8K 100 例 + 32K 100 例 + 128K 25 例） |
| 多针 NIAH / 臂 | 约 5 min |
| 长生成 / 臂 | 约 10–15 min |
| 跨卡控制 | 约 15 min |
| **合计** | **约 45–60 min**（两臂并行） |

## 边界

- 128K 只有 5 例/深度，Wilson CI 很宽；只能发现大差异，不能声称"等价"。
- NIAH 是合成任务，不等同于真实文档问答；长生成只测复现与退化，不测事实性。
- 跨卡控制是子集（8K、20 例），只能给卡间效应一个上界，不是完整控制。
- 不覆盖 1M 级上下文，不覆盖 MTP/投机解码。
