# 1344条solver已关闭：质量评分尚未完成

本快照只读取已关闭的两臂各672条solver和10个独立审计receipt，所有audit真实子wait返回码0。原SSE、原请求绑定、原MMLU
extractor、指标及零复用已在受控环境逐条核验；公共此处提供原stage数值汇总和受控SHA，不冒充公开重新读取原SSE。完整逐请求数字胶囊与四份HLE/Frontier质量汇总在共同judge收口后追加。

| 数据集   |   B0 tok/s |   B1 tok/s | 吞吐变化 | length B0/B1 |
| -------- | ---------: | ---------: | -------: | -----------: |
| mmlu     | 614.568863 | 629.912506 | +2.4967% |        20/20 |
| hle      | 793.275630 | 794.147654 | +0.1099% |      124/121 |
| frontier | 761.031051 | 762.812493 | +0.2341% |        36/43 |
| swe      |  36.151535 |  35.544568 | -1.6790% |          2/2 |
| terminal |  85.392668 |  87.390692 | +2.3398% |          2/4 |

吞吐为accepted输出token/完整batch首提交至末响应wall，保留EOS、原预算、失败及原分母。SWE的负变化保留，不做跨数据集挑选或不明权重汇总。MMLU原extractor为144/200→149/200，截断20/20；配对不确定性待完整数字记录汇总，不能把5题变化升级为总体等价/显著结论。HLE/Frontier质量仍待原grader，length
stop不是解题错误率；SWE/Terminal没有任务解决率。

**整个full solver阶段Candidate实际启用，但两臂calls/decisions均0。**
正变化不能归因于BidKV机制，不能据此申请2x或项目四目标通过。只有一次固定顺序配对，环境/时序漂移保留。额外HLE接口诊断、初始化/内存和720计划judge另计，本snapshot明确formal_complete=false。

原字节数值与10个审计wait见[solver-progress.json](solver-progress.json)。C8真实生命周期另见[c8-owned-lifecycle.json](../c8-results-v2/c8-owned-lifecycle.json)，其中未持久捕获的client
PID/精确起止和API spawn时间保留null，不能用请求时钟倒填。
