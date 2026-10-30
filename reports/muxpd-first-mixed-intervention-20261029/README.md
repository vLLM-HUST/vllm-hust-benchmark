本轮首次混合批次干预未修复 07338，答案不退化门继续 FAIL，运行时 PR #27 保持 Draft。

[中文报告](REPORT-v1.md)保留六次设备尝试与两次分析失败。指定 seed 的归约干预确实生效，但七个 peer 的完整保存前缀没有改变。没有完整答案或性能改善声明。

[证据 Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-first-mixed-intervention-20261029)
提供完整归档、外部引用索引和原始报告。505,248,445 字节归档含 2,172 个成员（含 MANIFEST），本地逐成员 SHA
核验通过；公开读回状态单独记录，未完成前不声称公开整包核验通过。

原始报告末尾的“仅在本地准备”描述归档生成时状态；本目录记录后续发布。仓库 Markdown 经格式化；Release 与归档保留原始字节。归档重建与校验见
[README-v1.md](README-v1.md)。本地包装器的 7 项 CPU 测试通过，不能替代质量评测。

后续已转向 07338 自身的 prefill 层观测。最终仍须验证有效运行时修复、原有退化题和完整 200 题，逐题列出修复与新增退化。

公开读回结果为 **INCOMPLETE**：[原始记录](public-readback/final-status-v1.json)。8 个小附件的匿名 HTTPS、大小及 SHA
全部通过；大归档单次 GET 在 301.54 秒被对端关闭（curl 18），收到 25,649,664 / 505,248,445 字节。部分文件原样保留，没有重试；公开整包成员及 gzip
校验未执行。此前本地全包校验通过不能替代此次公开读回失败。
