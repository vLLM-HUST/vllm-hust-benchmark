# v4 公开回验完成：当前公开状态

**2026-10-26 02:05:57 UTC，独立匿名回验首次尝试通过：18个新增资产、96,990,295 bytes 的大小与 SHA256 全部一致。**
13个归档中的1181个原件文件及13份嵌入 manifest 均逐字节核验；support 的122个文件包含在1181个总数内。没有提取或执行归档代码。

[最终回执](public-readback-v4/attempt-01/public-readback-receipt-v4.json)的 SHA256 为
`30c334995098ba4f136956ea8634bafcd06b995fa9a30baf02dc30b80201368d`。[逐资产 HTTP 回执](public-readback-v4/attempt-01/download-summary.json)、13份成员证明和精确
manifest 保存在同一目录；[原过程说明](public-readback-v4/READBACK.md.snapshot)记录冻结的验证器、10项内存fixture检查和独立源码复核。

下载[前](public-readback-v4/attempt-01/release-api-before.json)/[后](public-readback-v4/attempt-01/release-api-after.json)的公开
API 快照均有57个资产，原始响应均为89,968 bytes、SHA256
`7ff2bb6b719eee3014fd7aaf331a4e33c1d183aee771ab282e490ea2461d6c21`。旧39资产的稳定身份同时匹配[此前v3公开快照](public-readback-v4/attempt-01/previous39-v3-public-api.json)和[本次上传前快照](public-readback-v4/attempt-01/previous39-preupload-api.json)。**旧39资产的字节未重新下载**；这不是期间持续不变的证明。下载计数和uploader不属于稳定字段。

[显式小文件选择](public-readback-v4/small-copy-selection-v4.json)及[复制清单](public-readback-v4/copy-map.json)保留66个文件、1,368,633
bytes，包括API、HTTP/成员回执、manifest、五个下载后的小型发布文件、验证器及检查记录。Python/Markdown仅改为`.snapshot`后缀，字节不变；不把tar
payload放进Git。原248文件及准备阶段81个v4文件均保持原字节，只有新的导航README更新当前状态。

上传前的[报告](REPORT-v4.md)、[归档索引](archive-index-v4.json)、[资产准备清单](release-assets-v4.json)及[上传选择](upload-selection-v4.json)仍保留历史准备状态；本回执追加随后发生的公开事实。此前v3两次失败尝试也未改变；本次v4没有失败尝试。

```sh
python reports/muxpd-followup-20261025/performance-lifecycle-v4/verify_v4.py
python reports/muxpd-followup-20261025/performance-lifecycle-v4/verify_public_v4.py
```

这些离线命令核对已保存的身份关系，不执行新下载或模型任务。公开可用不等于性能普遍领先、成对质量通过或物理重叠成立；v4固定十二项运行、所有失败及默认原生v2/实际profiler/成对质量待完成的边界均未改变。
