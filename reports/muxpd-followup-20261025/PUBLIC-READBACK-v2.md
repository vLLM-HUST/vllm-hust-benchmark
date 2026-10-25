# MuxPD v2 公开附件下载复验

2026-10-25 23:13
UTC，独立匿名下载核验完成：[公开 Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-followup-20261025)
的 **21 个附件、725,765,919 bytes** 全部与冻结上传清单的大小和 SHA256 一致。 其中 **13 个归档、1,882 个外层原件成员**
通过完整成员集合、类型、大小和摘要检查。 没有将归档成员解包到文件系统或执行其中代码。

这次追加确认公开可用性和文件完整性。原数值诊断的覆盖限制、失败记录和验收边界保持不变； 质量实验、后续压力与性能实验不包含在这批附件中，也没有新增吞吐或 goodput 收益结论。

- [总回执](public-readback-v2/public-readback-receipt-v2.json)：全部 21 个下载 URL、大小和 SHA256，以及 13 个归档的成员数。
- [逐附件 HTTP/摘要回执](public-readback-v2/receipts/)与[完整下载汇总](public-readback-v2/download-summary.json)。
- [逐归档成员验证回执](public-readback-v2/archive-verification/)与[执行日志](public-readback-v2/verification.log)。
- [下载前 API 原件](public-readback-v2/release-api.json)、[下载后 API 原件](public-readback-v2/release-api-after.json)
  和[清单稳定性回执](public-readback-v2/inventory-stability.json)：两次均为同一个公开、非 draft Release，恰有 21
  个上传完成的附件，身份与内容摘要未变化。
- [原始采集器字节](public-readback-v2/verify_public_readback.py.source)与[复制身份清单](public-readback-v2/copy-map.json)：只把回执、日志和采集器原件加入
  Git，下载的压缩包仍在 Release。

成员验证调用已冻结的 [build_capsule.py](tools/build_capsule.py)，其 SHA256 为
`e1c63334e87118c06e4ef5f75c2aa1dee3b4c8cda3a6a0602ce68f0a412e9fbd`。 各本地冻结 manifest 的字节身份先与下载的公开
index 核对，再由该工具读取下载归档的全部成员及内嵌 manifest。 总回执 SHA256 为
`fe789aa44cb5658b1687bbd5c18eb7744c2ce5369335646196371d290efa8c7b`。

[v1](archive-index-v1.json)/[v2](archive-index-v2.json) 准备阶段清单、[原资产清单](release-assets-v2.json)
和[原报告](REPORT-v2.md)继续保留原字节及当时的 `prepared_not_uploaded` 状态。 当前公开状态由本次追加回执说明，不能倒改旧清单来伪造更早的上传或核验时间。

[关联入口索引](../../docs/data-entrypoints.json)也已按仓库流程显式刷新 Release 快照，收录这 21 个新附件。 固定网站 commit
与网站输入保持不变，没有修改网站代码或执行部署；导航元数据与上面的下载内容证明分别保留。

离线复核当前 Git 内的回执关联：

```sh
python reports/muxpd-followup-20261025/verify_public_readback.py
```

该命令不访问网络、不运行模型，也不把带时间的下载快照解释为永久可用性保证。
