# 已发布证据资产与目录来源

原始证据预发行
[evidence-muxpd-ecpa-20261025](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-ecpa-20261025)
绑定已合并证据提交 `19081ff39a25c3c8ac821ae7885688ea24a03e90`，与插件的软件预发行分开。2026-10-25 20:49:26 UTC
的[下载回验收据](evidence-release-download-verification.json)记录：四个资产均为 uploaded，下载大小、SHA256、API digest
和冻结的本地 v4 字节一致；SHA256SUMS 检查返回 0。

[下载后的逐成员检查](evidence-release-member-verification.json)再次核对全部 347 个成员，归档 SHA256 为
`5c16db263554c1d1538ca773790b91b4780ec4b9fe5cc6c5a30003f186aaebbb`。[API 原始快照](evidence-release-api.json)与回验收据中的摘要相符。此目录中的三份
JSON 是原始回执的逐字节副本；回执保留采集时的原文件名。

MuxPD
目录的功能资格链接固定到[已合并报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/19081ff39a25c3c8ac821ae7885688ea24a03e90/reports/muxpd-ecpa-20261025/RELEASE-QUALIFICATION.md)，不是将目录自身提交作为实验来源。目录保持
experimental alpha 与受限 TP4 功能范围，明确最终自然文本 6/8 一致、未证明逐 token/文本等价，不添加当前包性能测量。

本文件记录资产发布与目录输入；网站上线仍需后续固定目录合并 SHA、Pages 部署和实际页面/数据身份检查，不把本次上传直接记为网站上线。
