# DiffSpec / native Eagle：当前有效证据与未完成范围

2026-10-27，ascend87。**这不是完成的三臂280题实验，也不是对DiffSpec完整检索算法的否证。** 35B/vSpec的完整主实验另见
[results-qwen35.md](results-qwen35.md)。

## 评语

**性能收益不合格，需要重新评估项目可行性。**

此评语依据当前有效运行点的性能表现；保留下文样本量、检索未生效及未完成范围的限定，不将其扩大为对完整方法的否证，也不自动追加实验。

## 有效的短对照：B16、固定56题、每臂一轮

在原大缓存路径出现退化后，三臂统一使用memory0.4（C04）、TP4、B16、相同固定perf56、temperature0、max_tokens4000。原生Eagle与DiffSpec均使用gamma3。每臂新engine，按原生→Eagle→DiffSpec串行运行，计时排除加载/编译/预热。三臂均无连续token0退化、无解析失败、**抢占重算增量0**。

| 臂                   |  正确 | 截断 | 输出tokens | generation秒 | 输出tokens/s |
| -------------------- | ----: | ---: | ---------: | -----------: | -----------: |
| 原生 target-only     | 44/56 |   14 |     86,748 |      373.716 |        232.1 |
| 原生 Eagle3          | 46/56 |    9 |     73,219 |      766.069 |         95.6 |
| DiffSpec auto/linear | 46/56 |   12 |     84,884 |      929.285 |         91.3 |

原生Eagle接受24,661/145,623（约16.9%）；DiffSpec接受25,927/176,796（约14.7%）。单轮输出吞吐分别约为原生的41%和39%。DiffSpec相对Eagle的wall更长，但输出也更多；按输出吞吐仅低约4.4%，**没有重复实验，不将其声称为稳定的4.4%回退**。

此运行点没有显示加速，但不能由56题的44/46/46判定精度优劣；各臂截断和输出长度不同，样本小。也不泛化到其他target/draft配对、长上下文负载或充分优化的部署。当前目标为本地Qwen3.8-27B、草稿为作者指定的Qwen3.5-27B-EAGLE3-v2；不把目标权重的本地名称当作已核验的上游不可变revision。

## 实际运行的DiffSpec机制边界

四rank witness确认实际使用`AscendDiffSpecEagleProposer`及其cache，不是插件名注册了、实际仍跑plain
Eagle。可是当前默认auto解析为linear，线性路径没有proposed
tree，`proposed_token_ids is None`被传作`is_prefill=True`，使检索入口提前返回；未修改算法门禁以制造另一个方法。执行原源码方法体的CPU测试和设备计数共同支持此限制。

本轮warm/post witness的`retrieval_count=0`、`retrieval_cycle=0`、`selected_slots=0`；不声称完整attention-score
retrieval生效。滚动尾部cache路径仍可能工作，不能将“没有检索”扩大成“没有任何cache操作”。因此这组数据评价的是**当前实际执行的默认artifact**，不是完整DiffSpec检索思路的上限。

## 失败证据与修订，不能混进有效算法比较

1. **原memory0.7/B32**：native
   Eagle和DiffSpec短检查共享19/56题连续`!`（token0）退化。原生Eagle280题94/280、145解析失败的结果受污染，排除算法精度推断。原生target-only的235/280不能与这些坏输出组成有效对照。
1. eager、B4、async都未改变那19题。反转题序后坏题跟着位置变化；取消14题warmup后只坏最后5题，均从累计第52个请求开始。raw target
   logits检测到NaN；严格greedy采样审计未发现与参考不符。不要把warmup后的request编号直接当测试行号。
1. **memory0.4绕行**：65个单层cache
   group共享大buffer；较小预算的实际num_blocks=5508。56×16及56×128检查不再退化。大缓存寻址/状态边界是待定位方向，未证实具体算子、整数溢出或修复根因。共享vLLM/Ascend源码未改。
1. **C04/B32质量轮**：原生238/280；Eagle233/280、44截断、0解析失败，但Eagle最终计数器显示**1488次抢占重算**，generation3348.201秒。普通日志没出现preempt文字不代表没有抢占。这个耗时不能全归因于算法。随后DiffSpec
   B32质量轮在初始化期被停止，未产生有效280题结果。
1. 因此新短性能对照统一改B16，并加完成后抢占计数门。上表三臂均通过，不与B32质量轮混成同一配置。

所有停止仅作用于经argv/PGID核对的本任务进程组；错误日志、停止receipt、已完成及部分输出保留。诊断wrapper曾出现tuple接口和RPC序列化错误，已单独记录，不能作为被测算法故障。

## 投资与当前状态

- 已完成：上述无抢占的B16三臂56题首轮、兼容/容量诊断与默认检索门禁审计。
- 未完成：DiffSpec有效三臂280题、B16性能第2/3轮、全量12,032题。
- 完整B16三臂280题按本轮generation外推约2.9小时，另有加载/预热开销，实际可能变化。后两轮性能预计另约一至两小时。**Fletcher于2026-10-27明确决定停止追加实验、整理现有证据。**
  本轮到此收口，不再跑上述280题或第2/3轮；只有获得新的明确授权才恢复。

## 证据

- [协议及修订](plan.md)、[过程、错误和纠正](progress.md)、[结构化汇总](evidence/screen-analysis.json)。
- `evidence/screen-performance-c04b16-*-r1-{config,result,exit}.json`：上表原始三臂配置、逐题输出、完整指标及退出状态。
- `evidence/screen-performance-c04b16-diffspec-r1-*-witness.json`：运行机制与检索计数。
- `evidence/diag-native-eagle-*.json`、对应logs：容量/排序/预热/采样诊断；这些同步诊断耗时不作性能证据。
- `test_diffspec_linear_gate.py`：直接执行固定源码方法体，验证默认线性检索门禁。
