# 原生 gamma4 对照：未观察到 vSpec 额外吞吐收益

2026-10-27，Fletcher要求纠正原生baseline深度后的有界复测。原生Qwen3.5-35B-A3B MTP固定gamma4，对比原vSpec
adaptive配置；后三轮日志均initial_gamma4、gamma_switches0。两臂均使用原生norm，未安装GenNova修正，也未启用逐步分叉/采样同步探针。

## 冻结协议与验证

- 原56题性能子集（顺序不变），同一14题×16token预热，生成上限4000，temp0/top_p1；TP2+EP、BF16、B16、context16384、maxbatched8192、memory0.9，APC/async/chunkedprefill/FULL_AND_PIECEWISE开启。
- 两臂均传入原candidate48项capture列表。宿主原生gamma4调整为20项PIECEWISE候选（5至105）；candidate保留48项（1至112）。**两rank、六轮、计时前后均确认gamma4
  target FULL keys相同：tokens=5×requests，requests1..16。**
  不是全部graph/runtime实现完全相同；candidate还保留其他宽度及refill/兼容支持。
- 三次fresh-engine重复，顺序：native→candidate，candidate→native，native→candidate。物理0/1独占本任务，2–5上有他人负载，未触碰。共享宿主的CPU/内存/IO干扰仍可能存在；不与此前独占计时拼成新对照。
- 计时不含加载、编译、capture、warmup或前后metadata RPC；包含generate和原harness的逐请求完成落盘。未改变拒绝接受规则，confidence
  margin仍禁用。
- 六轮全部exit0、56题完整、答案重新解析一致、invalid0、preemptions0；无连续token0退化。候选三轮均gamma_switches0。完成后本任务driver退出、物理0/1无运行进程。

## 三轮原始结果

| 轮次 | 配置            | generation秒 | 输出tokens | output tokens/s | 正确/56 | 截断 | draft接受率 |
| ---- | --------------- | -----------: | ---------: | --------------: | ------: | ---: | ----------: |
| 1    | 原生gamma4      |      192.175 |     158681 |          825.71 |      40 |   26 |      66.50% |
| 1    | vSpec实际gamma4 |      186.704 |     148853 |          797.27 |      39 |   20 |      67.13% |
| 2    | 原生gamma4      |      193.026 |     156554 |          811.05 |      34 |   25 |      66.56% |
| 2    | vSpec实际gamma4 |      194.376 |     154511 |          794.91 |      35 |   22 |      66.64% |
| 3    | 原生gamma4      |      186.860 |     148790 |          796.26 |      41 |   22 |      66.42% |
| 3    | vSpec实际gamma4 |      187.615 |     149266 |          795.60 |      41 |   22 |      66.14% |

| 指标                  | 原生gamma4 | vSpec实际gamma4 | 解读                                                               |
| --------------------- | ---------: | --------------: | ------------------------------------------------------------------ |
| output tokens/s中位数 |     811.05 |          795.60 | candidate/native=0.98095，即-1.905%                                |
| generation秒中位数    |    192.175 |         187.615 | native/candidate=1.02431，但输出长度不同，不能据此宣称等工作量提速 |

按同轮配对，candidate output
tokens/s分别比原生低3.444%、1.990%、0.083%。3轮不能建立小幅减速的统计保证，但**本次没有测到额外吞吐收益**。第一轮candidate耗时更短同时少输出9828tokens；不能只取wall比忽略生成长度。

## 可以与不可以得出的结论

- 可以：在本机、该56题离线burst、B16、原生norm、匹配有效gamma4与target FULL
  buckets的配置下，没有证据支持vSpec在原生MTP4之上带来额外吞吐收益。此前相对原生MTP2约1.17×的整体耗时比，在这个更匹配的基线下不再表现为吞吐优势。
- 不可以：断言原17%全部由gamma深度造成。本次还扩充了baseline buckets，并重跑于共享宿主；没有逐项拆解各自贡献。
- 不可以：把中位数-1.9%写成已证明算法必然更慢，或泛化到全部并发、上下文、模型和线上流量。
- 不可以：把56题性能重复当作新的280题质量结论。得分native40/34/41，candidate39/35/41；每轮配对得/失为5/6、8/7、4/4，仍存在较多截断和序列差异。重复同一56题不是168道独立题。
- 实际选用意见：**当前证据不支持为这组35B/B16负载引入vSpec；先前的收益主张需要以原生gamma4这一更匹配基线重新评估。**
  这不是完整动态自适应算法的否定实验，因为被测candidate没有切换gamma。

## 证据与复现

- [冻结协议](PLAN.md)、[离线复算结果](results.json)、[复算脚本](analyze.py)。运行
  `python3 research/speculative-mmlu-pro/gamma4/analyze.py`，重新检查题目顺序、答案解析、退出码、抢占、gamma日志及两rank图键。
- 配置 `../gamma4-configs.json`，环境沿用 `../q35-environments.json`；producer
  `../screen_gamma4.py`、`../campaign_gamma4.py`、`../launch_gamma4.py`、`../gamma4_witness.py` 与
  `../gamma4-guard/sitecustomize.py`。
- 六轮原件在
  `../evidence/gamma4-performance-q35-{mtp4,adaptive}-r{1,2,3}-*`；campaign状态complete与日志一并保存。原35B结果不覆盖。
- 报告状态：本次证据与gamma未切换事实已纳入更新后的15页总报告；旧MTP2数据保留为历史对照。benchmark归档同时保留原件和复算脚本。
