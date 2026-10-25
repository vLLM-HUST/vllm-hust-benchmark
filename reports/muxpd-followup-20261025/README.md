# MuxPD TP4 后续数值研究：终态证据 v2

**同一源码、观察器、HCCL 配置和编译后端下，raw/group 的 30/30 条输出序列一致，240/240 对 rank-zero 完整 logits 文件逐元素和 SHA256
均一致。** 这是一项有限正确性诊断，不是语义质量、吞吐或 goodput 资格；Issues #3、#11、#23 与 draft PR #10 仍保留后续验收要求。

[当前报告](REPORT-v2.md)追加已结束的 group 实验；[v1 报告原字节](originals/PRELIMINARY-REPORT-v1.md.source)和旧归档保持不变。
[数值摘要](analysis/raw-vs-group-composite-summary-v2.json)区分数组值相等与完整 `.npy` 文件 SHA 相等。 四个 rank 均有采样
token 回执，但只有 rank zero 保存完整词表向量，不能扩展成四个 rank 的完整 logits 比对。

| 证据                             | 本轮观察                                                 | 边界                                                     |
| -------------------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| Native 默认配置重复性            | 两轮 warm 各 13/15 序列一致，跨进程 11/15                | 不能用原生变化免除插件检查                               |
| Native `HCCL_DETERMINISTIC=true` | warm 与跨进程各 15/15；对应 120/120 logits 文件 SHA 一致 | 有限输入上的观察，不是所有通信/算子的确定性证明          |
| 同源码 native/raw composite      | 30/30 序列一致；147/147 可比上下文向量一致               | 其余 93 个位置不作同批次数值等价推论                     |
| 同源码 raw/group composite       | 30/30 序列、240/240 可比向量一致                         | 不代表通用模型质量或性能提升                             |
| Group 元数据                     | 每 rank 捕获 128 次；已捕获事件的 v2 检查无失败          | 15 个 P cohort 中仅 1–12 完整，13 部分，14/15 未捕获回调 |

运行硬件来自实际 910B2 清单，模型为 Qwen3.5-35B-A3B、BF16、TP4/EP4。 原件保留两次失败：首次 native 请求 ID 验证错误和首次 raw worker
选择器验证失败。 无请求的压力池布局发现也作为独立证据保留，不代表已经验证压力、抢占或取消。

## 订正记录

[v1 host audit](analysis/composite-fusion-host-audit-v1.json)错误期待 `.mlp` 结尾的 MoE 名称；
[v2 host audit](analysis/composite-fusion-host-audit-v2.json)按原生 `.mlp.experts` 注册名修正，设备日志未改变。
[覆盖订正原文](originals/COMPOSITE-OBSERVATION-COVERAGE-CORRECTION.md.source)明确：12 个 driver phases 实际产生 15
个 P cohorts， 完整覆盖需每 rank 150 次回调。128 次上限覆盖前 12 个 cohort、cohort 13 的 groups 0–7；不补写缺失观测。
[独立复验](analysis/metadata-reverification-v2.json)重新核对原件 SHA 和这组覆盖数字。

## 归档与下载状态

[v2 归档索引](archive-index-v2.json)包含 11 个终态实验、13 个包；其中两个 support 包携带源码、环境、模型权重 SHA 清单、分析代码和订正记录。 共
1,882 个外层原件成员、1,792,926,073 bytes 原始内容，压缩包共 725,675,722 bytes。 七份源码由固定 commit 的 `git archive`
导出，各自还有嵌套成员 SHA 清单，不包含完整 Git 历史或模型权重。 完整 `.npy`、日志和编译缓存图在 Release 包内，未复制到 Git。

[准备阶段资产清单](release-assets-v2.json)给出各包文件名、大小、SHA、计划下载 URL、逐文件清单和本地复验回执；
[v1 索引](archive-index-v1.json)及 [v1 校验和](SHA256SUMS-v1)继续保留。 资产已上传至
[evidence-muxpd-followup-20261025](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-followup-20261025)。
**2026-10-25 23:13 UTC 的独立匿名公开下载复验已通过：21 个附件的大小与 SHA256 一致，13 个包的 1,882 个成员全部核验。**
[公开复验报告](PUBLIC-READBACK-v2.md)链接下载前后 API 快照、逐附件及逐成员验证回执。 旧准备清单和原报告保持原字节，其历史 `prepared_not_uploaded`
状态由新回执补充，不倒改历史记录。

这里的清单只接纳已结束并完成清理的显式运行；当前质量实验及后续压力、性能实验不在 v2 中。 新增结果必须追加新 capsule/index/report 版本，不能覆盖 v1/v2
或将后续结果改标为本轮结果。

## 离线复核

```sh
python reports/muxpd-followup-20261025/verify_report.py
python reports/muxpd-followup-20261025/verify_public_readback.py
python reports/muxpd-followup-20261025/tools/build_capsule.py verify \
  --archive /downloaded/capsule.tar.gz \
  --manifest reports/muxpd-followup-20261025/custody/CAPSULE.manifest.json \
  --archive-sha256 EXPECTED_SHA256
```

这些 Python 3.11+ 工具不启动模型，不访问 GitHub。`verify_report.py` 核验 Git 中原字节副本、manifest/receipt/资产关联、终态清单和结论边界；
归档工具逐成员读取下载包，校验完整成员集合、大小与 SHA，且不解包执行内容。 [复制身份清单](canonical-copy-map.json)保留执行端路径到 canonical
文件的映射；分析器内的原始绝对路径未被改写。 唯一未格式化的原始数值分析代码按
[`.py.source` 原件](originals/analyze_fixed_token_numerics.py.source)保存，以保持已发布分析中的 producer SHA。

后续仍需完成同条件质量测试的 no-new-wrong 与无截断门禁、实际 P/D 混合覆盖、allocator 压力与取消 ACK/物理块复用、安装路径和关闭检查，
以及无观察器的配对时延、吞吐、goodput 实验。本轮没有新的收益数字，也不改变历史收益的版本归属。
