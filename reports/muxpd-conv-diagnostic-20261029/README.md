# MuxPD：首个混合批 GDN 卷积诊断

两组v2运行通过历史前缀复现和observed/plain桥，四rank的A/B/C输出与更新后state均逐字节相同。本次首层、首个seed D卷积对照没有发现混合几何或模式切换引起的差异。原
full200 的答案不退化门仍为 **FAIL**：两次 native
均为165/200，prefix候选为167/200，但新增07338错误且有11次length；见[完整质量报告](../muxpd-prefix-quality-20261028/FULL200-v1.md)。本诊断没有新增答案质量、吞吐或Goodput结论。

v1 模型诊断在预热后的源码/库身份检查失败，发生在 observer 安装及请求入队之前；失败原件保留。后续独立单卡零输入 smoke 复现了“两个不同库具有相同basename”导致的误拒绝。v2
修复库身份检查与诊断RPC失败处理，保持数值采集代码及原batch计划字节不变；两组模型运行均正常终态。

- [结果报告](REPORT-v1.zh-CN.md)：实际读数、失败沿革、修复范围和解释边界。
- [完整v2分析](analysis/analysis-v2.json)、[plain终态controller](plain-v2/controller.json)。四rank各比较2048个输出元素与6144个state元素，A↔B、B↔C差异均为0，mode0/mode1重复稳定。
- [独立raw审计](independent-audit/summary-v2.json)通过：直接核验64对raw/NPY、32对历史前缀及8对observed/plain前缀。两次executor关闭日志均有grace期后SIGTERM、无SIGKILL；controller没有要求额外清理，不能描述为全部worker自然退出。
- [v1终态失败](failed-v1/controller.json)、[原冻结清单](failed-v1/freeze-manifest-v1.json)：强制清理与失败状态不会改写成成功。
- [smoke首次导入失败](loading-smoke/initial-import-failure.json)、[后续零输入conv结果](loading-smoke/zero-conv-result-v2.json)：独立加载诊断，不是模型质量样本。
- [v2冻结](protocol-v2/freeze-manifest-v1.json)、[库身份合同](protocol-v2/operator-contract-v1.json)、[失败依赖](protocol-v2/prior-failure-reference-v2.json)、[root CPU验证](protocol-v2/root-verification-v2.json)。
- [观测组终态controller](observed-v2/controller.json)、[仅生命周期摘要](observed-v2/lifecycle-summary-v1.json)：四rank各一次采集、四次私有重放、恢复三个hooks；这不是observed/plain桥结果。
- [复制来源](copied-sources-preparation-v1.json)：本地小型原件逐文件SHA；未复制运行中的v2 child或日志。
- [原归档准备计划](publication/archive-plan-v1.json)保留创建时`not-built-not-uploaded`状态；后续[归档完成回执](publication/archive-record-v1.json)、[逐成员清单](publication/archive-manifest-v1.json)、[本地完整读回](publication/local-verification-v1.json)记录本地包已完成。归档为15,190,758字节，SHA256
  `90780d770a6714f1b6c9aa4155dfc16a21fc01f23a637cfc023be839c6e15b8a`，包含326个payload与一份嵌入manifest，全部327成员及gzip
  EOF/CRC本地核验通过。
- [精确外部依赖](publication/external-dependencies-v1.json)列35项历史文件，并指向既有发布依赖清单；新包不自带所有历史完整run、模型或运行时。发布与匿名完整读回状态见下文；原本地记录保留其创建时状态。

本目录是后验诊断报告，不进入正式榜单。该负结果不能推广为所有GDN层/步骤等价，也没有观测实际prefix graph内部输入；Issue #23 与实验运行时 PR #27
的答案质量边界不因本次工具修复改变。

公开发布：
[9个Release附件](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-conv-diagnostic-20261029)均已匿名HTTP200下载，TLS、大小及SHA核验通过；15,190,758字节归档的327成员和gzip完整性再次通过。见[最终公开回执](public-readback/final-status-v1.json)、[下载脚本原件](public-readback/readback.py.source)及[API快照](publication/release-api-v1.json)。此前归档计划和本地记录中的未上传字段保留创建时语义。本次公开读回PASS只属于本轮新包，不改变旧layer/layout/new-loss大包的INCOMPLETE状态。
