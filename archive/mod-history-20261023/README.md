# 四个 MOD：历史目录及混合文档归档

本批次延续 [消费者首批迁移](../mod-consumers-20261023/README.md)，不改变原测量、 历史基线名称、失败或配对关系，不启动设备测试。每份 manifest
给出公开源提交、 原路径、字节数、Git blob、SHA256 及可下载的 Release 包。

| 集合                                   | 原目录 / 文档                                                                                                                              | 消费者与身份                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| [KV Arrival](kv-arrival/manifest.json) | `.benchmarks/results/`；`docs/NPU_LIVE_VALIDATION_20260910.md`                                                                             | 工作负载生成 smoke 不是模型性能；旧 NPU 验证报告原文保留，但所称 ignored live 日志未在两批清单中找到，不声称恢复 |
| [PegaFlow](pegaflow/manifest.json)     | `results/issue23/20261012-attention-diagnostic/`；`docs/experiments/issue23-pilot-20261012.md`                                             | 四臂 attention-transfer diagnostic，非完整 P/D 正确性验收；CI 的 457 文件审计改读外部归档                        |
| [MuxPD](muxpd/manifest.json)           | `evidence/2026-10-10-tp4-optimization/`；`docs/performance-tp4.md`                                                                         | 历史 eager/graph、正确性差异、负收益、状态复用诊断均保留；冻结审计脚本不是可直接重跑的完整 profiler 输入         |
| [Profiler](profiler/manifest.json)     | `npu6_existing_server_probe_blocked/`、`npu6_existing_server_trace_overhead_smoke/`（位于 `.benchmarks/results/`）；`docs/claim_ledger.md` | 保留受阻记录；客户端 proxy 开销不等于内部 runtime overhead；论文导出读取外部归档，其他 claim 暂留源仓库          |

## 读取与复核

包内布局为 `snapshot/<原仓库路径>`。README、LICENSE 和列出的混合文档作为语境冻结， 不代表这些上下文文件均从源仓库删除。下载器不执行归档代码；输出必须是 checkout
外的新目录。

```sh
python scripts/fetch_mod_consumer_evidence.py \
  --manifest archive/mod-history-20261023/pegaflow/manifest.json \
  --destination /absolute/new/external/directory
```

PegaFlow 审计仍验证完整请求、SSE、四臂重算和运行源码身份；不通过删除断言解除消费者。 MuxPD 原报告记录完整 profiler capture 未包含在该历史 Git
目录中，迁移没有恢复那些输入。 旧文档原字节在包内，源仓库保留方法/读取入口。引用原路径表示归档成员，不表示本地仍有副本。

## 尚未收口

- KV Arrival formal/WaveMat/G0 等目录及其 custody 元数据消费者。
- PegaFlow 其他性能目录、Issue28 混合诊断、handoff 和 runtime 冻结源码/数据清单。
- MuxPD 其他 TP4/TP8 日期，TP4 goodput/groupfx 本地索引及验证器；既有 Release 安装/读取消费者尚未据此解除。
- Profiler 其他采集/回放/合成、派生论文表格与混合 claim；不得把这些统一视为已测性能。
- Triton 审核 runner、LMCache-Ascend 设备检查和 SGLang 审批门禁仍按 #326 的实际状态保留；本归档不豁免门禁。

工作复查：本轮为公开历史数据迁移；依据固定源 Git 和消费者引用；复查 E04/E37/E42/E43， 保留失败与原字节、同时更新双索引、不放宽测试、不恢复旧 probe 追索或已取消补测。
