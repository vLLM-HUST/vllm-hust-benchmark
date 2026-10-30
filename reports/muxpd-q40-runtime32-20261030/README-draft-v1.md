**内置 q40 posterior32 为29/32，01741新增答错（E→C，5551 tokens，自然stop），“不新增错误”失败；full200未启动。**
07338恢复H/3292和此前局部8题成功不能代表整体修复。

外置同源码、同32题历史对照 **PENDING**；当前目录只固定已终态的builtin证据，未读取外置live结果。此为本地staging，不表示已发布。

- [报告草稿与32题表](REPORT-v1.md)
- [机器可读32题表](posterior32-table-v1.json)
- [完整token子报告](builtin-run-v1.json.gz)、[原调度](builtin-scheduler-v1.jsonl.gz)
- [主分析](builtin-analysis-v1.json.gz)、[独立复核](builtin-independent-v1.json.gz)
- [源码包](builtin-source-capsule-v1.tar.gz)、[发布映射](publication-map-v1.json)、[外置PENDING状态](external-status-v1.json)

不包含logits NPY，不声称性能、Goodput或完整200题验收通过。
