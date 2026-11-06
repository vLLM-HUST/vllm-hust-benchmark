# RIDE 控制面证据迁存（2026-11-06）

本目录按组织证据归档规范，保全 RIDE PR #2 和 #4 已公开的来源、软件验证、一键部署摘要及规划/路线图历史快照。没有新增实验，不发布私有原始日志或凭据，不进入正式榜单或网站。

[manifest.json](manifest.json)记录每个文件的原仓库、完整提交、原路径、字节数和
SHA256；`source/`逐字节复制原件。规划中的目标不是已测收益，软件检查不是硬件资格；原路线图的失败、Unknown 和未验收项不因迁存改变。

- [SDK 来源及验证记录](source/docs/evidence/)
- [路线图历史证据和负结果](source/docs/architecture/CONTROLLER_ROADMAP.md)
- [规划及候选验收目标快照](source/docs/PROJECT_PLAN.md)
- [显式机器部署摘要](source/evidence/machine-deployment-20261106/summary.json)

当前开发计划继续在 RIDE 维护，本目录快照不作为第二份活动计划。机器部署摘要中的硬件源码版本保持原值，不能提升为合并后主干的硬件验收。私有归档仅保留原有公开哈希，不提供或上传其内容。

复核：沿 manifest 的固定来源链接取回原字节，再比较 SHA256；本次迁存已在本地从原 Git 对象逐项核验。目录索引只说明可发现性，不表示实验通过。
