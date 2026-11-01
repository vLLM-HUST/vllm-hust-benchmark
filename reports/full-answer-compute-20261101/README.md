# 完整回答批次的有效算量复算

2026-11-01，复算 `joint-b0-r2` / `joint-b1-r4` 的全部 200×2 份记录。模型为 Qwen3.5-35B-A3B，BF16、TP2、同两张 910B2，固定
MMLU-Pro 200 题、输出上限 8192、自然结束。B1 包含 BetterScale E24/R36、逐请求重算和 FA Microflow，主机 State 为 4 GiB/rank。

| 本次可复算的量                         |                 B0 |                B1 |
| -------------------------------------- | -----------------: | ----------------: |
| 输入 token                             |             282265 |            282265 |
| 输出 token                             |             706796 |            661040 |
| 必要 target 位置 N                     |             988861 |            943105 |
| 因果 attention 对数 A                  |         3063790102 |        2763050179 |
| 主干矩阵算量 F_mat（FLOPs）            |   6040002578841600 |  5721201823580160 |
| 同轮客户端时间 T_client（秒）          | 1425.4634465482086 | 699.9029783103615 |
| F_mat/(2×T_client×10¹²)（TFLOPS/芯片） |       2.1186101241 |      4.0871392185 |

这两项是主干矩阵小计的处理速率，尚未包括 GDN 状态运算、基础标量运算和必要 MTP 依赖算量。大纲的完整 C、参考时间和 η 需要补测。已有日志不能恢复缺失的逐步接纳和分块状态。

## 计算式

每请求输入为 I_j、输出为 O_j。最后发出的 token 不再作为下一步输入：

```text
n_j = I_j + O_j − 1
N = Σ n_j
A = Σ n_j(n_j+1)/2
O = Σ O_j
F_mat = 4,873,420,800 N + 163,840 A + 1,017,118,720 O
C_mat,client = F_mat / (2 × T_client × 10¹²)
```

采用原计量约定：矩阵乘法为 2MNK，MoE 只计激活的 8 个专家和共享专家，TP2
全局工作量计一次，排除图填充和无输出依赖的草稿工作。模型配置取自当次冻结源码包；公开[配置](model-config.json)仅补充文件末尾换行，JSON 内容不变。原件 SHA256 为
`5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`。

投影系数分为三项：

| 分项                           | 每位置 FLOPs | 配置展开                           |
| ------------------------------ | -----------: | ---------------------------------- |
| 10 层完整 attention 投影       |    545259520 | 10×2×2048×(3×16×256+2×2×256)       |
| 30 层 GDN 输入与输出投影       |   2021130240 | 30×2×2048×(2×16×128+3×32×128+2×32) |
| 40 层专家、router 与共享门投影 |   2307031040 | 40×2×2048×[3×(8×512+512)+256+1]    |

QK/AV 系数为 `4×10×16×256=163840`；必要 target 词表投影系数为 `2×2048×248320=1017118720`。每个输出决策包含一次 target
词表验证；不把每个 prompt 位置或被拒绝的额外验证位置都计入词表小计。GDN 的 Q/K/V/gate 输入投影在矩阵小计内，其状态更新、分块计算另计。

两臂自然回答长度不同，各自计算 F_mat。不得用 B0 的算量除以 B1 的时间，也不得按 token 比例放大旧计数轮的总 F 或参考时间。

## 大纲指标与旧计数轮

```text
C = F / (k×T×10¹²)
C_ref = F / (k×T_ref×10¹²)
η = 100×C/C_ref = 100×T_ref/T
```

同一次 η 计算的 F、k 和任务边界相同。大纲中期门槛为 η≥50%，最终门槛为 η≥90%。B0 相对吞吐提升 90.4811% 属于吞吐比较，不代表第一项达标。

旧 out256 计数轮的 B0/B1 C 分别为 3.8407325729 / 7.1042207644 TFLOPS/芯片。旧 B1 的
F=1812905151088034、T=127.59352582134306 秒、T_ref=47.503348961177785 秒，得到模型比值
η=37.2302188966%。这些数字仍属于原 out256 轮次，不代表本次完整回答组合的总 C 或 η。

## 补测需要记录什么

1. 在预先固定的同一工况中保存逐请求 MTP 接纳根及必要 KV/hidden 依赖闭包、GDN prefill 的实际分块和初始状态，补齐总 F。
1. 配对记录同步起止边界及匹配的无插桩时间；所有测量开销保留，不估算扣减。
1. 用独立标定、必要执行依赖和资源约束构造本批次参考模型，检查长输出、B0/B1 各自路径的适用域，计算各自 T_ref 和 η。

当前容器核查只见物理 3/7 卡，原 5/6 卡及 `/models/Qwen3.5-35B-A3B`
挂载不在。恢复原资源或确定新的固定配对环境后，执行一次计数与无插桩配对；新运行使用新标识，保留原运行。此次只读复算，没有启动模型服务或加速卡实验。

## 复现与证据

```bash
python scripts/recompute_full_answer_compute.py --check
```

- [逐题标量](../full-answer-joint-20261030/joint-summary.json)：原 400 份响应的长度、时间与哈希。
- [原件核对](source-audit.json)：400 份请求逐项匹配，B0/B1 分别核验 416/427 个原始文件 SHA256。
- [派生结果](summary.json)：逐分项算量、客户端速率及明确的缺失字段。
- [复算程序](../../scripts/recompute_full_answer_compute.py)：从配置和逐题长度重算，拒绝输入不一致、重复请求、失败批次及无效时间。
- [旧计数轮复核](../acceptance-completion-20261029/reference-reassessment-20261029.json)：保留原数值与适用范围。

The current complete-answer pair yields target-matrix subtotals of 2.1186101241 and 4.0871392185
TFLOPS/chip over the original client intervals. These are not total effective compute or outline
efficiency. Total work requires same-run MTP dependency and GDN chunk ledgers; acceptance also
requires matched synchronized timing and independently calibrated reference times for the
full-answer workloads. No old reference time or throughput ratio is substituted.
