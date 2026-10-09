# 浦江指定数据集：35B B0/B1 资产与缺口审计

审计日期：2026-10-09

## 范围

浦江指定的数据集名称固定为：MMLU-Pro、HLE-Verified、SWE-bench-Pro、FrontierScience、Terminal-Bench
2.1。名称确认不代表版本合同已经冻结；不得用 MMLU、SWE-bench Verified 或旧 Terminal-Bench 等近似名称替代。

## 当前状态

五项源数据快照已经落地并通过字节级校验，但“资产已冻结”不等于“完整评测合同可执行”。统一入口为
`/root/vllm-hust-eval-data/pujiang-five`：`MANIFEST.json` SHA-256 为
`2666f482a6f6c974cde19b6549f72ee37df82a93c9dc032754005af3a839299b`，`SHA256SUMS` SHA-256 为
`c02431d400b89146fd1b65dec8d505c40a981066ff8d4615eebe12ba0fa7effc`。2026-10-09 已执行全部 1135 项文件校验，全部通过，且快照中没有 `.git` 目录。

| 数据集 | 资产状态 | 冻结 revision / split | 样本或任务数 | 仍未解除的正式执行门禁 |
|---|---|---|---:|---|
| MMLU-Pro | `ASSET_FROZEN` | `b189ec765aa7ed75c8acfea42df31fdae71f97be`；validation/test Parquet | 待从快照提取并冻结任务 manifest；旧服务压测的 200 prompts 不是数据集基数 | 冻结任务 ID、抽样规则、scorer 和执行环境，并补任务准确率；现有 B0 仍只是服务遥测 |
| HLE-Verified | `ASSET_FROZEN_CONTRACT_BLOCKED` | `b705e0fb541c025a1532ce0d60d70ae2f53b00e0`；Gold/Revision/Uncertain | 2500（668/1143/689），与上游 Git LFS SHA-256 一致 | 上游未声明数据许可；须先完成许可核验，再冻结评测子集、judge 模型/提示词、scorer 与镜像 |
| SWE-bench-Pro | `ASSET_FROZEN_CONTRACT_BLOCKED` | `2d52cb3df914a3fcf80c7f66738b3a88ae37fc50`；default/hard/v1 test Parquet | 待从快照提取任务 ID | 上游数据卡未声明许可；须冻结代码仓库 revision、fresh-sandbox 镜像 digest、agent scaffold、工具策略、预算与 regrader |
| FrontierScience | `ASSET_FROZEN` | `25ed67db7da8f4591484e764008ff585544f5a30`；olympiad/test、research/test | 160（100/60） | 两个赛道分别冻结任务 manifest、答案提取、scorer 与执行环境，不得静默合并 |
| Terminal-Bench 2.1 | `ASSET_FROZEN_RUNTIME_BLOCKED` | `7131e4375048a0e408a8fb404b5f499d726b695b`；89 个任务定义 | 89 | 尚未预拉取约 40 GB Harbor 运行镜像；须冻结 image digest、agent scaffold、工具策略、预算、scorer 与任务级第三方权利核验 |

源快照不得改写。正式执行必须另建结果目录，并保存实际模型、server argv、client contract、镜像 digest、scorer、到达序列和资源释放证据。

## 历史 B0 边界

历史 B0 工作簿包含 21 个结果集，与浦江指定范围分开管理。五项中只有 MMLU-Pro
出现在该工作簿中，而且记录的是服务吞吐与时延遥测，不是任务准确率。HLE-Verified、SWE-bench-Pro、FrontierScience、Terminal-Bench 2.1
均没有被历史 B0 覆盖。

## 准入规则

`ASSET_FROZEN` 只表示源快照具备精确 revision、统一 manifest 和逐文件 SHA-256。只有再具备不可变任务 manifest、样本数及抽样规则、固定 scorer、执行镜像、许可证记录和依赖清单时，状态才能提升为“已落地且可执行”。在此之前不得发布正式 B0/B1 任务质量结果，也不得换用名称相近但合同不同的数据集。
