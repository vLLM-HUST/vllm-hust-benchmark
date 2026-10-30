07338 的首个完整 logits 分歧已定位到输出位置 **183**，早于第 207 个输出 token 的分歧。 位置 1–182 的完整词表向量全部逐字节相同。答案退化尚未修复，原
full200 质量门仍 FAIL。

- [结果与边界](REPORT-v1.md)
- [主分析](analysis-final-v1.json.gz)与[独立复核](independent-pair-v1.json)
- [冻结采集源码包](source-capsule-v1.tar.gz)与[发布映射](publication-map-v1.json)

两次 TP4 运行各自复现旧的 80 份完整向量和八条保留前缀。原始 NPY 与完整采样 JSONL 仍在服务器；源码包不包含这些数据或全部历史依赖。
