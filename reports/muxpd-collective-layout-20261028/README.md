# MuxPD TP4 通信布局短程对照

[完整报告](LAYOUT-v1.md)记录五个新进程、三种布局和观察器对照。前置补零在这条轨迹上恢复 120/120 捕获层张量与七个 P 请求的 126/126 logits；同形状复制保持
raw，尾部补零不恢复。seed、全批历史和完整答案不等价，原200题严格门仍为 **FAIL**，没有新增性能资格。

[公开原件入口](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-collective-layout-20261028)包含主归档和后完成的独立
suffix 审计补包；当前公开读回状态为
**INCOMPLETE**。本地成员验证已通过，尚不能替代公开下载复核。需要结合[前置层定位](../muxpd-layer-localization-20261028/README.md)的对照原件重建分析。本集合不包含随后独立的完整答案
quality-runs。

公开匿名读回已保留为
**INCOMPLETE**，具体已验证的小附件/区段、失败和未完成范围见[公开读回状态](public-readback-status-v1.json)。本地完整校验、上传完成和公开完整下载核验是不同结论，未将前两者写成第三者通过。
