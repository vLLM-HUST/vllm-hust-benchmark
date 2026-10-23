# 四个 MOD 剩余集合：原件与消费者

本批次归档公开 Git 原字节，不改变历史数值、标签、失败、模型输出差异或配对关系。 固定 source commit、成员路径/SHA256/长度和下载入口见各
manifest。归档不代表原实验通过。

| 集合                                   | 范围                                                                           | 消费者迁移边界                                                                                                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [KV Arrival](kv-arrival/manifest.json) | paper experiments/results 全部剩余 formal、M1/M2 等记录，docs/g0、docs/wavemat | 原始报告和失败保留；WaveMat 从外部归档读取原 provenance 和 summaries，再用既有 benchmark Release 原始 trace 重算；G0 既有验证链不放宽                                                              |
| [PegaFlow](pegaflow/manifest.json)     | Issue23 handoff/runtime、handoff/、docs/analysis Issue28 等历史材料            | 冻结 runtime code/overlay 仍供分析器导入，保留代码；数据移出。校验器保留在代码仓库，读取外部原清单，校验包内未改动的原校验器，不重签历史。Issue28 输出 gate 的失败不是迁移失败，也不改成正确性通过 |
| [MuxPD](muxpd/manifest.json)           | evidence/ 的全部剩余 TP4/TP8、goodput/groupfx 索引与报告                       | 当前流式归档校验器移入 scripts；测试及 CI 从外部读取原索引，继续回验既有 benchmark Release，不删除有读取关系的 Release 资产                                                                        |
| [Profiler](profiler/manifest.json)     | .benchmarks/results 的全部剩余采集、诊断、回放、合成及受阻记录                 | 保留生成器与合成单元用例。离线分析和论文导出改用外部输入；第一批 Issue19 和第二批 overhead 继续各用独立 capsule，不能把缺项静默置空                                                                |

下载到 checkout 外的新目录，不执行下载包中的代码：

PegaFlow Issue28 的实现摘要绑定 22 个文件，不能拿当前源码代替。 [补充源码 capsule](pegaflow-issue28-source/manifest.json)
来自公开合并提交 `f3cfb19`， 逐项满足原 validation 的 LF-normalized SHA256；原始字节另按 capsule SHA256 保留。 记录中的 `37d2119`
已从远端取回，但尚不含两份后补文档；公开合并提交同时包含这些文档和原实现。 不把当前版本的代码格式变化伪装为原摘要，不修改原 validation。

```sh
python scripts/fetch_mod_consumer_evidence.py \
  --manifest archive/mod-collections-20261023/profiler/manifest.json \
  --destination /absolute/new/external/collection
```

各 MOD 的 `BENCHMARK_COLLECTION.json` 固定本批 manifest，与此前 `BENCHMARK_EVIDENCE.json` 和
`BENCHMARK_HISTORY.json` 分开。旧 absolute/local paths 是原件身份，不自动变成可访问链接； 按成员清单定位，不把完整设备 trace
未归档、历史受阻或部分结果写成已恢复/已测通过。 安装制品、硬件/审核门禁以及其他 MOD 不在此批清理范围。

复查记录：#326 指定集合迁移 | 公开固定 Git、远端成员核验及 CPU 消费者 | E04/E37/E42/E43 | 不重签原件，不删语义断言，双索引同步，未启动补测。
