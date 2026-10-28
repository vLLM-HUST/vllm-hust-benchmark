# 全200候选发布暂存

状态：`local_verified_not_uploaded`。由root负责后续上传和Issue更新；本任务未写GitHub、未访问设备或current-main live child。

[中文报告草稿](REPORT-DRAFT-v1.zh-CN.md)、[归档记录](archive-record-v1.json)、[逐成员manifest](muxpd-prefix-quality-full-01-20261028-v1.manifest.json)、[完整本地验证](local-verification-v1.json)、[历史external引用](historical-controls-reference-v1.json)。

**生命周期和证据审计PASS；原严格质量FAIL。** native1/native2/raw/prefix为165/165/166/167；prefix有1个new wrong07338和11个length。原三loss修复，旧四gain保留2、丢失2；完整token与每次native仅89/200相同。201–207窗口全部8成员仍在，scheduler保存序不证明runner内部行序。原8192、题集、评分和逐题门不变。

归档 `muxpd-prefix-quality-full-01-20261028-v1.tar.gz` 含57个payload文件及1个嵌入manifest，原来源共97,653,052字节，压缩后2,313,081字节。全部payload与明确来源逐字节SHA相同。candidate28、CPU/冻结回执9、全量audit10、loss context9、历史引用1；历史完整native/raw run不在包内。上下文保留选定历史scheduler行，报告如实披露。排除compiler-cache、pycache、pytest/ruff缓存。

builder仅遍历已终态full候选和两份明确审计目录，CPU回执使用固定清单；不枚举兄弟run目录。verifier读取每个tar member，检查安全路径、精确集合、普通文件、完整SHA/大小和嵌入manifest；不解压或执行归档。已完成一次完整本地source对照，无需重复运行。

SHA256SUMS绑定全部交付侧文件。报告为旁附草稿；新包尚无公开下载回执。历史controls引用原冻结索引，不新下载；不声称本次检查了其当前远端可用性。
