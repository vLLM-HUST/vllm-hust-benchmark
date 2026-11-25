# 已闭合压力诊断，独立于正式五项

此处整理此前已经停止并释放模型的诊断原件/数字，不启动新模型、不继续调参。正式五项仍为默认16GiB
KV/rank的672请求/臂，不能用此目录改变其工作负载、质量分母或吞吐。诊断显式缩小KV到1600MiB/rank，并使用历史
`utility_preempt_weight=5.0`（非默认0.3）；这种组合变化及额外冷启动费用不能归因成默认BidKV收益。

## EvoScientist first12，四次ABBA

运行前冻结的配置/执行源见[原v2配置](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/9930a0b772b79c92a487509767991a9a476f110d/reports/tech-outcomes-20261117/bidkv/evoscientist-pressure02-v2)。共计划48、实际48请求，保留原12条提示与逐条预算、TP2、4096上下文、并发12、4req/s、EOS保留；每阶段独立冷启动、MTP2和共同默认优化。48条SSE重放和原件hash/ID绑定、实际返回码、退出释放在原闭合导出中核对通过。

| 顺序/阶段    | 输出tok/s | accepted输出token |     整批秒 | 原生抢占 | BidKV调用 | 改变内置victim |
| ------------ | --------: | ----------------: | ---------: | -------: | --------: | -------------: |
| 1 Native-r01 | 83.327487 |             24148 | 289.796329 |      303 |         0 |              0 |
| 2 BidKV-r01  | 83.513424 |             24148 | 289.151118 |      304 |       304 |              5 |
| 3 BidKV-r02  | 76.809099 |             24148 | 314.389836 |      316 |       316 |              8 |
| 4 Native-r02 | 86.093902 |             24148 | 280.484441 |      302 |         0 |              0 |

两组配对吞吐变化分别+0.223140%与−10.784507%，全部保留，不选最佳轮或混为独立重复的稳定收益。每阶段12条都最终length
stop；保留EOS不等于本次出现自然提前结束。Evo没有原任务质量评分，生成长度相同不证明质量正确或等价。

第二配对有明确共享CPU干扰：[更正记录](evo02/external-cpu-overlap.json)显示，在BidKV-r02模型生成期间，原agent错误安排了同机CPU
pytest，实际重叠16.930567502975464秒。CPU耗时对吞吐的影响幅度没有测得，不能据此减去16.93秒“修正”吞吐、也不能将−10.784507%全归因于MOD；该组不算无干扰复测。原导出
[study.json](evo02/study.json) 的 `fully_validated_pair=true`
只保留为当时输入/策略/原件审计结果，不覆盖这项运行环境更正。[requests.json](evo02/requests.json)保留48条数字，[controlled-index.json](evo02/controlled-index.json)保留原件hash，[actual-export-receipt.json](evo02/actual-export-receipt.json)是原有闭合导出的实际CPU返回码0收据；本次整理未重新执行Evo模型实验。

## MMLU原前8题，有预先质量门槛的小试

运行前冻结的配置/执行源见[原MMLU8配置](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/9930a0b772b79c92a487509767991a9a476f110d/reports/tech-outcomes-20261117/bidkv/mmlu8-pressure03-v1)。固定原200题的前8（biology前缀），预算8192、保留EOS、上下文65536、maxseq16；计划四阶段ABBA共32条，首个AB共16条完成后按预先门槛判断是否继续。

| 已执行阶段 | 输出tok/s | accepted输出token |     整批秒 | 原评分正确/计划 | 原生抢占 | 策略调用 | 实际改变内置victim |
| ---------- | --------: | ----------------: | ---------: | --------------: | -------: | -------: | -----------------: |
| Native-r01 | 90.091000 |             12743 | 141.445872 |             8/8 |       96 |        0 |                  0 |
| BidKV-r01  | 96.849445 |             15665 | 161.745893 |             6/8 |       95 |       95 |                  1 |

吞吐+7.501799%，同时输出token增加、整批耗时增加20.300021秒、正确数8→6；因此不是同等质量下完成相同工作更快。预先规则返回
`prospective_quality_triage_not_met`、`run_reverse_confirmation=false`，剩余BA16条未执行。已执行16条均原SSE/输入/评分核对通过，无实际HTTP失败；未执行16条不写成16条实际失败。

[study.json](mmlu8/study.json)逐项保留四阶段、32条计划（包含未执行占位）、质量、策略事件、采样、实际wait/生命周期与原件hash。旧导出器对未执行阶段的summary使用
`failed=8` 表示计划分母中未成功项；应按同一阶段明确的 `execution_status=not_executed` 和 `recorded=0`
解读，不能冒充返回错误。反向吞吐为null、共同四阶段几何验证为null，均不补零。[本次实际CPU导出收据](mmlu8/actual-export-receipt.json)返回0，在本次81新审计目录执行原冻结导出器，未进行新的模型生成。

以上诊断说明BidKV在真实抢占时可以被调用并合法改变victim；它们没有建立稳定提速和任务质量无损，更不能替代正式五项或四项项目验收。
