# v3 公开下载回验：保留失败尝试的完成回执

**2026-10-26 00:47:19 UTC，独立匿名回验完成：16 个新增资产的大小与 SHA256 一致，10 个 capsule 的 786 个外层成员全部核验。** source98
嵌套源码归档的 99 个文件和 9 个目录也通过完整集合与字节检查。新增资产共 65,585,167 bytes；不执行归档代码，不启动模型。

[完成回执](public-readback-v3/attempt-03/public-readback-completion-v3.json)绑定此前已验证的下载内容和后续独立 API 收尾请求，
SHA256 为 `1b9af89ceb35f63e5ec46c6555d628b221fcda0cb4183a19dd267e6935c8c07f`。
这是一份新的完成记录，未修改或改标较早的失败尝试：

| 尝试                                                                   | 保留状态 | 实际边界                                                                                    |
| ---------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------- |
| [01](public-readback-v3/attempt-01/public-readback-receipt-v3.json)    | FAIL     | 直接下载部分资产超时；API 快照成功                                                          |
| [02](public-readback-v3/attempt-02/public-readback-receipt-v3.json)    | FAIL     | 16 个资产、786 个外层成员和 99 个源码文件已验证；最后的共享代理 API 请求返回 403 rate-limit |
| [03](public-readback-v3/attempt-03/public-readback-completion-v3.json) | PASS     | 保留 01/02 原件，以稍后的直接匿名 API 请求完成同一批内容的稳定清单核对                      |

## 公开身份与时间边界

下载前的[00:43:14 API 快照](public-readback-v3/attempt-02/release-api-before.json)与
完成时的[00:47:19 API 快照](public-readback-v3/attempt-03/release-api-after.json)均包含 39 个资产， 且原始响应字节
SHA256 均为 `00ab21bffeec9b8cee5d34961005ed48283f7f7c715fb49f2a59bf613420d0f0`。 验证器另外比较稳定身份字段，包括 asset
ID、名称、状态、大小、digest、时间和下载 URL；不把下载计数作为不可变身份。

此前[23 资产 API 快照](public-readback-v3/previous23-release-api.json)中的稳定身份与本次一致，新增集合恰为本次 16 个资产。
**本次没有重新下载旧 23 个资产的全部字节**；这只是两个时间点的 API 身份比较，不能宣称期间持续不变。 旧 21 资产 v2 下载回验及其后的 2 资产质量协议回验保留原时间与原字节。

[逐资产下载回执](public-readback-v3/attempt-02/download-summary.json)保存 HTTP 请求、大小和 SHA；
[10 个 capsule 的成员证明](public-readback-v3/attempt-03/public-readback-completion-v3.json)逐一链接其完整验证记录。
[源码嵌套验证](public-readback-v3/attempt-02/nested-source98-verification.json)绑定 source98 的提交、tree、嵌套 tar
SHA 和完整成员集合。 786 为外层原件数，另有每个 capsule 自带的 manifest；嵌套源码的 99 个文件不重复计入外层数量。

## 原件保存与复核

[复制清单](public-readback-v3/copy-map.json)保存执行端路径、原大小、SHA 和 canonical 路径；
[显式小文件选择](public-readback-v3/small-copy-selection-v3.json)选中 67 个 metadata/source 文件、762,914
bytes，未把下载的归档 payload 加入 Git。 原始 Python 和 Markdown 分别以 `.py.source`、`.md.source` 保存，避免格式化改写。
[验证过程说明原文](public-readback-v3/READBACK.md.source)与三个验证器/冻结身份均保留在同一目录。

上传前的 [v3 准备清单](release-assets-v3.json)、[归档索引](archive-index-v3.json)、[报告](REPORT-v3.md) 仍保留当时的
`prepared_not_uploaded` / `local_verified_not_uploaded` 状态。本回执追加其后发生的上传与公开下载事实，不倒改历史状态。 原有 117
个报告/协议文件，以及已冻结的 v3 原件和准备元数据保持不变。

```sh
python reports/muxpd-followup-20261025/device-evidence-v3/verify_public_device_v3.py
```

离线检查只核对已保存回执与原字节的关系，不执行新下载，也不代表当前网络始终可用。 公开可下载不等同于模型正确性、成对质量、吞吐或 goodput 通过；v3
的九个运行范围和所有失败、退出码及结论限制均不变。
