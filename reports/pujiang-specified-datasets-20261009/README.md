# 浦江指定数据集：35B B0/B1 资产与缺口审计

审计日期：2026-10-09

## 范围

浦江指定的数据集名称固定为：MMLU-Pro、HLE-Verified、SWE-bench-Pro、FrontierScience、Terminal-Bench 2.1。名称确认不代表版本合同已经冻结；不得用 MMLU、SWE-bench Verified 或旧 Terminal-Bench 等近似名称替代。

## 当前状态

| 数据集 | 状态 | 实际材料路径 | revision / split | 样本数 | manifest / SHA-256 | scorer | 运行依赖与主要缺口 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| MMLU-Pro | 已有材料但版本未冻结 | `leaderboard-data/dataset-validation/dataset_validation_qwen35_tp2_ep_ctx32k_apcoff_inf_out256.json`；本容器观察到 `/root/b0-validation-input/MMLU-PRO/准备操作.md` 与两张截图 | 未冻结 | 200（旧服务压测命令的 `--num-prompts`，不是数据集基数） | 数据集 manifest 缺失；规范 B0 artifact SHA-256 `f75a86fb802d933e55ab12738fae198233b350f94d573bd16a5fbdcfb397f019`；准备说明 SHA-256 `b0984321c69037ca9d0645441f3c55c6c84eea49b0598d96b6c78b12f23a0a87`；截图 SHA-256 `1bb66c18688891f12a30a38a6379162eb27e0573d4ca5bb531675cef571fd1c7`、`84d4280e10c57bd9d588485b013d15099f96c20a804e3584f083f4cac8d77b09` | 缺失 | 旧材料只描述 Parquet 读取与 `vllm bench serve`；需冻结 revision、split、任务 ID、抽样、scorer、镜像、许可证与 manifest 哈希，并补任务准确率。 |
| HLE-Verified | 缺失 | 无本地数据、manifest 或 runner | 未冻结 | 未知 | 缺失 | 缺失 | 需取得并审查精确版本，明确 full verified 或命名 verified-gold 子集，并冻结 scorer、镜像与哈希。 |
| SWE-bench-Pro | 缺失 | 无本地数据、manifest、沙箱镜像或 runner | 未冻结 | 未知 | 缺失 | 缺失 | 需取得 SWE-bench-Pro 精确版本；不得用 SWE-bench Verified 替代。需冻结 task IDs、fresh-sandbox 镜像、agent scaffold、工具策略、预算与 regrader。 |
| FrontierScience | 缺失 | 无本地数据、manifest 或 runner | 未冻结 | 未知 | 缺失 | 缺失 | 需取得精确版本，并分别冻结 Olympiad 与 Research 的任务清单、scorer、执行环境和哈希。 |
| Terminal-Bench 2.1 | 缺失 | 无本地 2.1 数据、manifest、Harbor 环境或 runner | 未冻结 | 未知 | 缺失 | 缺失 | 需取得 2.1 精确发布；不得用旧版 Terminal-Bench 替代。需冻结 Harbor 镜像、agent scaffold、工具策略、预算、scorer 与哈希。 |

## 历史 B0 边界

历史 B0 工作簿包含 21 个结果集，与浦江指定范围分开管理。五项中只有 MMLU-Pro 出现在该工作簿中，而且记录的是服务吞吐与时延遥测，不是任务准确率。HLE-Verified、SWE-bench-Pro、FrontierScience、Terminal-Bench 2.1 均没有被历史 B0 覆盖。

## 准入规则

只有同时具备精确 revision、split、不可变任务 manifest 与 SHA-256、样本数及抽样规则、固定 scorer、执行镜像、许可证记录和依赖清单时，状态才能提升为“已落地且可执行”。在此之前不得发布正式 B0/B1 任务质量结果，也不得换用名称相近但合同不同的数据集。
