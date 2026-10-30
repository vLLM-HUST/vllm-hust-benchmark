目标题 07338 的首次 prefill 观测已完成：前四层 120 对张量、40 层最终 MoE 输出 160 对张量、prefill 后保存的 320
对状态均数值及字节一致。这是缩小故障范围的阴性结果，答案退化尚未修复。

原 full200 答案不退化门继续 FAIL，运行时 PR27 保持 Draft。本轮没有新的质量或性能收益结论。

- [复核报告](REPORT-v1.md)：条件、结果、失败保留及下一步 decode 定位。
- [独立状态复核](state-independent-v1.json)与[主分析](state-primary-v1.json.gz)。
- [源码及预检包](state-source-v1.tar.gz)：39 个成员，包含冻结的采集、控制和分析代码；不包含原始 NPY。
- [状态发布映射](state-publication-map-v1.json)与[此前观测来源](local-source-map-v1.json)：原文件及压缩文件的 SHA256。

原始设备张量仍在服务器，未发布完整张量包。源码包需要原服务器依赖及历史对照，不能作为独立可运行的完整实验包。
