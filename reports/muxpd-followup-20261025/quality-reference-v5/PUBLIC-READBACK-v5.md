# v5 公开读回：13 个新资产完整核验

独立匿名 GET 于 **2026-10-26 04:52:10 UTC** 完成，13 个新资产共 209,021,252 字节的大小和 SHA256 全部匹配。八个胶囊的完整成员集合、2,463
个原始成员及八份嵌入 manifest 全部核验；其中支持胶囊为 1,166 个成员。没有执行或解压包内代码，没有新模型实验。

[完成回执](public-readback-v5/attempt-01/public-readback-receipt-v5.json)绑定了
[50 个小文件的明确复制清单](public-readback-v5/small-copy-selection-v5.json)、每次 HTTP 响应、每个成员和前后 API 清单；共
3,424,951 字节的回执副本保持原字节。压缩包本身不进入 Git。[原读回说明](public-readback-v5/READBACK.md)和
[producer 冻结身份](public-readback-v5/producer-freeze-v1.json)完整保留。

读回前后 Release 均有 70 个资产，稳定身份字段一致；旧 57 个资产与此前公开快照及本次上传前快照的稳定 API 字段均相同。旧资产未重新下载，不声称两次快照之间连续不变；下载次数与
uploader 等动态字段排除在身份比较外。

本次准备时的 `prepared_not_uploaded` 和 `local_verified_not_uploaded`
字段保持历史原字节，以本页的后续回执补充发布状态。公开可下载性不是质量接受：原 strict 与 supplementary 答案门槛 FAIL、吞吐和 TTFT 代价、增加的 profiler
kernel launch 数都不变。

上传后才到达的 [mixed-profile 小文件复核](analysis/postselection/mixed-profile-independent-review-v1.json)与
[native-reference 小文件复核](analysis/postselection/native-default-independent-review-v1.json)仅作为
canonical 追加，**不在已上传的 support 胶囊中**。它们没有重新读取原始 token 数组或 trace，也没有新增验收门槛。

```sh
python reports/muxpd-followup-20261025/quality-reference-v5/verify_public_v5.py
```

该检查只复核已保存回执和原件身份，不执行新的网络访问或实验。
