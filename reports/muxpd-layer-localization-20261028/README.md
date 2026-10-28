# MuxPD TP4 层内数值定位

[完整报告](LAYER-LOCALIZATION-v1.md)保留六次实验，包括原声明PASS但退出生命周期审计FAIL的一次raw采集。有效v2/v3先通过同角色144
logits/8前缀桥，再将第一个已观测数值差异定位到TP collective输入与返回值之间；不唯一归因某种HCCL算法。

[公开原件入口](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-layer-localization-20261028)提供两片可重组归档，1,815个原件、全部1,820个成员本地校验通过。公开匿名读回状态目前
**INCOMPLETE**；旧同角色控制是显式[外部依赖](external-dependencies-v1.json)。后续[通信布局对照](../muxpd-collective-layout-20261028/README.md)是另一组实验。原200题质量门仍
**FAIL**，本集合不提供生产修复或新性能成绩。

公开匿名读回已保留为
**INCOMPLETE**，具体已验证的小附件/区段、失败和未完成范围见[公开读回状态](public-readback-status-v1.json)。本地完整校验、上传完成和公开完整下载核验是不同结论，未将前两者写成第三者通过。
