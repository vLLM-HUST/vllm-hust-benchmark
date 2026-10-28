# MuxPD 新增退化 07338：batch7 有界诊断

本轮重现 target 第 207 token 的非平局贪心分叉，并通过四次运行的历史前缀复现与两组 observed/plain
桥。它是后验数值诊断，**不构成完整答案质量、性能或唯一根因认证**。此前 full200 仍为 167/200、1 个新增错误、11 个 length，原严格门仍
FAIL；见[完整质量报告](../muxpd-prefix-quality-20261028/FULL200-v1.md)。

- [中文报告](REPORT-v1.zh-CN.md)：实际读数、来源、分析修正与解释边界。
- [Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-new-loss-batch7-20261028)：完整
  producer、四次终态运行和 160 个向量在附件中；本目录只收录小型回执与分析原件。
- [归档身份](publication/archive-record-v1.json)、[逐成员 manifest](publication/archive-manifest-v1.json)、[完整本地核验](publication/local-verification-v1.json)。tar
  为 84,817,778 字节，SHA256 `19fb56eaf5201c30e38bcf9ced5736ccae09dfc1d943aa4b18291bb90d0af9bf`，含 324
  个原件与一份内嵌 manifest。
- [原 v1 FAIL](analysis/original-v1-failure.json)、[v2 分析](analysis/v2-analysis.json)、[v2 创建时 manifest](analysis/v2-manifest.json)；[独立摘要](independent-audit/summary-v1.json)、[数据审计](independent-audit/audit-v1.json)、[源码格式审计](independent-audit/source-schema-audit-v1.json)。
- 历史依赖须合看[原引用清单](publication/external-references-v1.json)与
  [raw 补充清单](publication/external-references-addendum-v1.json)：完整重跑 analyzer 需要两次 native、一次 raw 和一次
  prefix full-run 原件；本归档没有附带这些历史完整 run。

发布状态：已发布；[14 个附件的 API 回执](publication/release-inventory-v1.json)记录 uploaded、size/digest
与本地匹配。**匿名完整读回
INCOMPLETE**：[最终回执](public-readback/final-status-v1.json)记录13个小附件HTTP200/大小/SHA通过；大归档单次GET在180秒后超时，仅收到26,606,208/84,817,778字节。完整SHA/gzip/325成员尚未公开验证，不重试、不分段，也不将API匹配当作匿名完整读回PASS。
小型原件保留创建时的 `local_verified_not_uploaded`、`remote_verified:false` 或
`analysis_executed:false`，不追改历史字段；本段说明后续状态。

[复制来源清单](copied-sources-v1.json)逐文件绑定原件绝对路径、大小与
SHA；[本地检查](local-checks-v1.json)记录复制和链接核验，[SHA256SUMS](SHA256SUMS)绑定本目录文件。公开可用性不能由这些本地检查推断。
