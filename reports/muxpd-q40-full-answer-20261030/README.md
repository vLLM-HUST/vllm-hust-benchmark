原batch7八题在q40下全部自然 `stop`，原评分器判定 **8/8正确**。07338回答H、3292 tokens，历史外部prefix
q4曾在8192处截断；全部答案延续已确认的短窗口前缀。

**这是新进程的后验八题诊断。历史对照含前七批上下文；没有匹配的新进程q4、完整200题、性能或实际运行时候选通过声明。PR #27仍为Draft。**

- [逐题结果、调度、失败谱系与边界](REPORT-v1.md)
- [主分析](analysis-final-v1.json.gz)、[独立复核](independent-terminal-v1.json)、[完整token子报告](run-v2.json.gz)
- [冻结源码包](source-capsule-v1.tar.gz)、[发布映射](publication-map-v1.json)
- [v1启动失败](failure-lineage-v1.json)与[独立CPU后处理中断](cpu-postprocess-interruption-v1.json)均保留。

本包不含既有短logits NPY，不宣称原始张量已公开读回。
