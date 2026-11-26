# 数字导出与独立复算工具v2

此处6个冻结成员由`analysis-manifest.json`逐字节守卫；原说明以`original-README.md.source`保留，公共本页独立说明范围。`.py.source`是自行编写的分析工具，不改v4运行packet，不包含私有Engine代码、原题、答案、对话或SSE。

完整五项导出契约见[public-export-contract.json](public-export-contract.json)：正式solver数值、原grader四份分层/轨道汇总、judge实际HTTP/未决分母、原件索引、独立SSE核验、初始化/内存以及额外诊断开销分开。CPU
fixture隐私与计算检查只证明分析合同，不作为模型成绩。

原HLE接口资格每臂red/blue各1条synthetic请求，两臂共4条计划诊断，max_tokens32、保留EOS，不计672正式solver或720计划judge。该既有源的会计边界在完整五项开始前已补充至[Issue664追溯说明](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664#issuecomment-6850808300)；实际attempted/失败/usage与时钟在`diagnostic-overheads.json`另计，缺失最终记录不假定0token或编造状态。
