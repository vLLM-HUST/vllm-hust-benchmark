# 24题后验诊断发布暂存

状态：`local_verified_not_uploaded`。只包含已终态 `prefix-quality-losses-01`、
其冻结生产者/输入、选定 CPU 回执和独立审计。未读取或打包 full200 live child；
未写 GitHub、未操作设备、未修改源码。由 root 决定上传。

入口：[中文报告草稿](REPORT-DRAFT-v1.zh-CN.md)、[归档记录](archive-record-v1.json)、
[逐成员 manifest](muxpd-prefix-quality-losses-01-20261028-v1.manifest.json)、
[完整本地验证](local-verification-v1.json)、[外部历史引用](historical-controls-reference-v1.json)。

归档含45个payload文件与1份嵌入manifest；28个candidate文件、8个CPU回执、
8个独立审计文件、1个历史引用索引。原始来源总计12,237,498字节；
压缩包636,414字节。原run、assessment、producer、输入、审计原字节均保持。
不含compiler-cache/pycache，也不含历史native/raw完整run。

候选22/24，与两native22/24相同，旧raw19/24；三旧失分恢复，0新增错误、
0候选截断。完整token仅5/24与每个native相同。保持原8192预算；
这是后验子集，`full_frozen_plan_passed=null`。

`build_archive_v1.py`固定终态源目录，只在该目录内遍历且提前剪枝缓存，
不会遍历其他run；失败时不覆盖既有产物。`verify_archive_v1.py`逐成员读取、
验证安全路径/普通文件/精确成员集合/SHA/大小/嵌入manifest，不解压、不执行。
本地构建已额外逐成员与原文件重哈希比较。公开读者使用不带
`--compare-sources`的验证命令；该选项只适用于原生产机器。

`SHA256SUMS`涵盖待交付文件。新包尚无公开下载回执，不能把本地验证报告当成
公开可下载证明；历史URL也仅引用已保存的canonical v3/v5回验，无本次网络请求。
