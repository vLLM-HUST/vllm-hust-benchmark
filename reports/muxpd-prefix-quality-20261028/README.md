# MuxPD prefix 布局质量复测：后验24题与原全200题

**原全200题质量门 FAIL。**
两native、旧raw、prefix正确数为165/165/166/167；prefix有1个新增错误07338和11个length。此前按旧loss选出的24题后验复测为22/24、无new
wrong/候选截断，但不能替代全量门。生命周期成功、质量通过和公开可用性是三项独立判定。

| 集合                          | 质量结果                                                                            | 生命周期 | 公开原件/读回                                                                                                                                                                            |
| ----------------------------- | ----------------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [后验24题](POSTERIOR24-v1.md) | native22/22、raw19、prefix22；full gate=null；完整token仅5/24与native相同           | PASS     | [Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-prefix-quality-losses-20261028)；[匿名读回PASS](public-readback/posterior24/final-status-v1.json) |
| [原全200题](FULL200-v1.md)    | 165/165/166/167；1 new wrong、11 length；strict FAIL；完整token仅89/200与native相同 | PASS     | [Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-prefix-quality-full-20261028)；[匿名读回PASS](public-readback/full200/final-status-v1.json)       |

全200修复原03766、10672、01741三个loss，但旧raw四个gain只保留00882、08107，06074、06741回到length；新增07338在第207个输出token对native分叉。原200题、25个batch、seed先1token放7peer、8192预算、评分extractor和逐题不退化规则不变；native原9个length也不改写。

[新loss上下文](new-loss-context/README.md)核对四个终态run：目标201–207及扩展190–220窗口都仍有原8个成员调度/退役；没有邻近退出。seed在raw/prefix多领先9
tokens但到目标435才结束。收据未保存runner物理行序、graph bucket、KVslot或logits，不能据此宣称原因已闭环。

证据入口：

- [24题独立审计](independent-audit/posterior24/README.md)、[全200独立审计](independent-audit/full200/README.md)，含独立逐题rows和完整输出SHA。
- [24题归档记录](publication/posterior24/archive-record-v1.json)、[全200归档记录](publication/full200/archive-record-v1.json)，含归档/manifest/原run/assessment
  SHA。
- [历史controls固定external引用](publication/full200/historical-controls-reference-v1.json)，两新包均不附带历史完整native/raw
  run；派生评分与选定历史scheduler片段明确披露。
- [冻结producer来源](quality-producer/source-manifest-v1.json)、[冻结文件身份](quality-producer/freeze-manifest-v1.json)、[复制原字节清单](copied-sources-v1.json)、[当前发布状态](publication-status-v1.json)。

本目录为canonical分析与小型custody证据，最大文件是逐题派生rows；大型候选run、完整scheduler流和压缩原件保留在Release，不重复进Git。所有复制文件保持原字节，包括publication子目录中的暂存README/草稿；其中历史“未上传”状态由顶层canonical报告和新public-readback回执补充，绝不回写旧记录。public读回PASS只针对本次指定quality资产，不改变先前大体积layer/layout读回INCOMPLETE。

来源为冻结6c插件加外部诊断prefix模块、原core/Ascend、实测910B2四卡0/2/3/4。旧f7已有20个runtime文件字节等价，producer/module和物理卡差异显式记录，未伪装原compare_provenance。这里没有current-main、async/group/SP、安装包或新增性能资格。后续main
bridge不在本集合。

本目录只新增报告，不更改raw、历史报告、官方leaderboard或验收口径。索引由仓库整合者在stage本目录后统一生成；本次作者按目录ownership未修改索引。
