# GenNova MOD 单项与组合测试证据归档（2026-11-05）

本目录补齐此前未进入 `vllm-hust-benchmark` 的
[`GenNova-HUST/inference-engine-project`](https://github.com/GenNova-HUST/inference-engine-project)
MOD 单项、组合与负结果索引。来源固定在提交
[`c8416aad2c9e161abe2dde5e01690edc48e756e3`](https://github.com/GenNova-HUST/inference-engine-project/commit/c8416aad2c9e161abe2dde5e01690edc48e756e3)，
不是对源仓库当前分支的浮动引用。

## 已归档内容

| 文件                                                                                           | 内容                                |                      记录数 |
| ---------------------------------------------------------------------------------------------- | ----------------------------------- | --------------------------: |
| [`snapshots/results-catalog.json`](snapshots/results-catalog.json)                             | 结果、契约和冻结归档的总目录        |                         107 |
| [`snapshots/gennova-positive-integrations.json`](snapshots/gennova-positive-integrations.json) | 有正向证据的 MOD 接入记录及组合影响 |    55 项接入、29 项组合影响 |
| [`snapshots/combination-plan.json`](snapshots/combination-plan.json)                           | 组合测试计划、已测组合和待办登记    | 5 项已测组合、22 项待办登记 |
| [`snapshots/negative-results.json`](snapshots/negative-results.json)                           | 带适用边界和重开条件的负结果        |                          90 |

[`manifest.json`](manifest.json) 为 107 个结果目录补充固定提交 URL，并记录四个来源文件的 Git blob、 源文件 SHA-256、归档
SHA-256、大小和统计值；[`SHA256SUMS`](SHA256SUMS) 可校验本目录的机器可读归档。 四个 snapshot 保留源 JSON 数据；其中
`gennova-positive-integrations.json` 仅将 CRLF 行尾规范化为 LF， manifest 仍保留原始 Git blob 和源文件摘要供逐字节复核。

## 如何使用

先按 MOD 或组合名称检索摘要，再沿 `evidence` 或目录项中的固定提交链接查看源证据。例如：

```bash
# 单项 MOD 接入记录（正向、负向、边界和证据路径一起读）
jq '.entries[] | select(.id == "graph-expert-sparsity")' \
  snapshots/gennova-positive-integrations.json

# 已整理的组合影响
jq '.combination_effects[] | {id, combination, effect, quality, qualification, evidence}' \
  snapshots/gennova-positive-integrations.json

# 失败或无收益结果；do_not_repeat 与 reopen_requires 不能省略
jq '.entries[] | select(.id == "gdn-serial64-r22-mixed-no-stable-benefit")' \
  snapshots/negative-results.json

# 查看全部结果目录及其固定来源链接
jq '.catalog_entries[] | select(.tasks | index("gennova-7"))' manifest.json
```

这些记录同时保留正向与负向信息。局部 kernel 加速、有限配对、单次启动、小样本、截断输出、质量未评分或 成本口径不完整，都不能外推为整机稳定收益。组合结果也不能拆分归因给其中某个
MOD；具体边界以记录内的 `boundary`、`qualification`、`quality_scope`、`do_not_repeat` 和 `reopen_requires` 为准。

## 范围与排除项

本次是已有证据的发现性归档，没有运行新实验，也没有把记录晋升为正式榜单提交。源仓库仍是原始运行文件的 权威位置；本目录没有复制大体量日志和全部 raw
payload，也排除了协调文档中的私有端点、主机地址、临时路径、 凭据和无法由文件证据支持的会话结论。本归档不进入 `website-publication/`，因此不会让 GenNova 或这些
MOD 重新出现在 vLLM-HUST 网站插件页。

生成物可复核：

```bash
python scripts/build_gennova_mod_evidence_import.py --check
(cd reports/gennova-mod-evidence-20261105 && sha256sum -c SHA256SUMS)
```
