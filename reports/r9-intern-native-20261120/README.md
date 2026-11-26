# R9：书生模型原生集成验证

书生在同一双910B2容器完成MMLU200和C8连续4K/16K验证。按用户明确要求，对照网站此前发布的R8结果，不再重跑Qwen或正反配对。完整源码、SDK、三份原生运行包和实测证据见[R9正式Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261119-r9)。这组结果属于模型集成验证，不是Original018
B0/B1、固定五项或项目最终验收；本版相应位置保留缺项。

| 场景    | 书生吞吐token/s | 相对公开R8吞吐 | 平均首token延迟变化 | 平均TPOT变化 |
| ------- | --------------: | -------------: | ------------------: | -----------: |
| MMLU200 |        1460.406 |         +1.00% |              -4.41% |       -1.05% |
| C8 4K   |         369.684 |         -4.46% |             -14.24% |       +7.53% |
| C8 16K  |         158.040 |         -1.00% |              -7.24% |       +3.39% |

MMLU正确162/200，17题长度终止，失败及评分未解析均为0。4K首token延迟P99增加27.21%，16K首token延迟P95增加8.85%，全部正负结果与尾部统计保留在[原始摘要](native-source-summary.json.gz)。网站R8参考曾按当时已合并benchmark来源清单核验SHA及字节数；其身份和本轮书生分别列出。

继承R8的10项MOD不重复记增量。新增正式MOD为Intern-S2模型/词表/tokenizer适配和State
page-copy兼容；Qwen原worker及9份继承执行源码保持原字节。不根据组件数值或跨模型差异宣称某个MOD的独立加速。

最终冻结源码84cc5dcd02a6b0ac7b346ff3cd96fbb2765d301f；实测工具源码388a385c3012b86aa01c459ecd2bd818efbb57d5。SDK
    <!-- pragma: allowlist secret -->
SHA256为b56fc7e63e4db0981e2e43cb251e4fd7272f230dd4486a46f5755c0d528b0dd6。最终构建与实测SDK及3份运行包逐字节一致；公开4包已匿名下载、验哈希并在原容器新私有环境实际安装，最终书生启动参数与实测版本一致。打包阶段未启动新的推理。
    <!-- pragma: allowlist secret -->

私有Release的`r9-native-replay-capsule-20261120-v1.tar.gz`保存2031个原始文件、控制器、分析脚本、失败筛选证明及安装回执；SHA256为e3eb913187fa4a9b3f6be7f45ffed3e302709f39c37325866afd2e93661a080b。上传后已重新下载，解压到新目录并用`python3 replay.py /absolute/path/to/NEW-output-directory`逐项核验2031份原件、复现全部指标；不需要模型、NPU、网络或凭据。模型权重和供应商CANN/Torch环境单独配置，未声称一并恢复。
    <!-- pragma: allowlist secret -->

先前失败、局部退化及用户叫停的测试保留原身份，不改写成新成绩。一次受停止影响的Qwen质量阶段未完成，不能当成完整模型准确率。当前单模型结果、独立组件数值、源码集成、完整包安装和项目验收分别记录。

网站发布使用[模型集成数据](../../website-publication/data/r9-intern-native-20261120/summary.json)及固定来源的科学绘图。[DeepLinkEngine稳定版适配](https://github.com/DeepLinkNext/DeepLinkEngine/pull/16)使新安装和新测试选择官方最新稳定版，整轮保存同一版本记录；其合并与公开安装检查单独记录。历史R8测试、参考算力和启动内存修复保留，不把旧效率或五项成绩借给R9。

冻结引擎标签仍为R9。PyPI安装包使用`2026.11.19.9.post1`补充Markdown说明元数据；原包保留，SDK、固定工具及三份运行包字节不变。

公开数据可从保留的原摘要独立复算：`python reports/r9-intern-native-20261120/reproduce_publication.py --check`。该入口不访问网络、不占用设备，也不读取模型权重。
