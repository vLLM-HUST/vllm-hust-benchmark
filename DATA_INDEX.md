# 数据导航与 Agent 读写指南

先读 [agent.md](agent.md)，再用本页定位数据。索引只是导航，不证明实验通过、数据可比或 原件已恢复。不要因为文件位于 benchmark
就把失败、诊断、估算、历史数据当作正式成绩。

## 从哪里读、允许怎么改

| 想做的事                                 | 权威入口                                                                                                              | 写入规则                                                            |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 查一次运行的原始输入、输出、环境和校验和 | [submissions/](submissions/)：每轮目录中的 manifest、原始结果、日志、checksums                                        | 新运行建新目录；不手改旧指标或覆盖原件                              |
| 查专题实验、分析及验收边界               | [reports/](reports/)：先 README，再报告引用的原件/manifest                                                            | 原始证据保持不变；订正另存新版本并说明 supersedes，不抹掉失败       |
| 查正式榜单、模型配置、数据集验证         | [leaderboard-data/README.md](leaderboard-data/README.md)、[dataset-validation/](leaderboard-data/dataset-validation/) | 先验证准入，再通过已有生产流程更新；不要把所有 submissions 自动入榜 |
| 改网站展示数据                           | [website-publication/README.md](website-publication/README.md)、[data/](website-publication/data/)                    | 在 benchmark 修改来源，通过校验后以固定提交发布；不先改网站副本     |
| 查历史 MOD、迁移数据、Release 附件       | [archive/](archive/)：对应 README、manifest、release.json                                                             | 冻结原件只追加；Release/嵌套包按 manifest 核验，不执行其中代码      |
| 改实验规格、验收口径、数据结构           | [docs/](docs/)、[schemas/](schemas/)、[包内契约](src/vllm_hust_benchmark/data/)                                       | 修改契约及其测试/生成物；不能用改口径掩盖实验失败                   |
| 改生产、校验、迁移工具                   | [scripts/](scripts/) 和对应 tests                                                                                     | 先查现有工具，不将新实验结果写入测试 fixture                        |
| 查测试用例                               | [tests/fixtures/](tests/fixtures/)                                                                                    | 测试数据不是实测数据，不可用于网站成绩                              |

Archive 与 reports 内也可能同时包含代码、报告和原始数据；上述规则是安全默认值，不意味着 每个 JSON 都是原始测量。修改前读该集合的契约和
provenance。涉及原件订正时保留旧字节及说明， 不要仅重新计算 SHA256 来掩盖内容改变。

## 常用问题入口

- **中期 B0/B1、质量、长上下文、资源成本、M01：** [中期报告目录](reports/vllm-hust-midterm-20261019/README.md)，
  [B1 候选审计](reports/vllm-hust-midterm-20261019/B1-CANDIDATE-AUDIT.md)，以及该目录的
  `m01-*`、`quality*`、`long*` 文件/子目录。按当前报告的链接找证据，不凭文件名日期选“最新有效”。
- **Frontier 原始并发曲线：** [56-run 索引](reports/frontier-raw-recovery-20261020/README.md)。
- **迁入 benchmark 的 MOD 数据：** [2026-10-21](archive/mod-git-20261021/)、
  [2026-10-22](archive/mod-git-20261022/)；每个 capsule 的 manifest 记录原仓库、固定提交、路径和 SHA256。
- **大文件/Release：** [Frontier 原件](archive/frontier-recovered-20261021/README.md)、
  [源码和校准附件](archive/frontier-capsules-20261021/README.md)、
  [terminal history](archive/terminal-history-20261021/README.md)。下载固定资产后先核验包摘要，再核验成员； 本索引不联网，不枚举
  GitHub 当前所有资产，也不证明远端链接仍可下载。
- **Split-Batch 缺失原件：**
  [可用性审计](archive/mod-git-20261022/split-probe-recovery/local-reference-audit.json)。 33
  处旧引用仍不可回验，停止重复追索；新实验不能回填成旧原件恢复。恢复/补测是否继续以
  [#326 的最新决定](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/326)为准，旧计划不是执行授权。
- **当前验收标准：** [README 的 Current Delivery Contract](README.md#current-delivery-contract)，
  [公开目标](docs/OFFICIAL_TARGETS.zh-CN.md)；历史规格不能覆盖当前权威契约。

## 按路径查询（不加载大数据，不运行实验）

机器索引：[docs/data-index.json](docs/data-index.json)。包含集合、文件数、格式、入口和写入规则。 入口最多显示 12
个；`entrypoints_truncated` 表示还有其他入口。需要完整文件列表时用 `--files`。

```bash
python scripts/data_index.py --query m01
python scripts/data_index.py --query 'midterm b0' --files --limit 100
python scripts/data_index.py --query 'gdn manifest' --files
python scripts/data_index.py --query 'website-publication dataset' --files
python scripts/data_index.py --query 'archive split' --limit 100
```

查询是大小写不敏感的**路径关键词 AND**，不是 JSON 正文搜索。`matched` 是全部匹配数， `truncated` 表示输出被 `--limit`
截断；零匹配不代表实验不存在。先定位集合，再查 README、 manifest 和报告中的字段。文件数只统计 Git tracked 路径（包括已暂存新增文件），不包含未跟踪文件、
归档包内部成员或远端 Release 成员；它不是“实验数”。

## 增添、改写和验证的最短流程

1. 明确要改的是原件、分析、契约还是展示；检查 worktree，保留他人改动。
1. 新实验写入独立目录，记录来源版本、模型/精度、真实芯片、参数、请求、开始结束时间、失败和局限。 测试脚本不能伪造结果；设备执行需遵守原有资源与审批规则。
1. 已有报告的结论需要订正时，追加订正/新分析并链接旧证据。可编辑当前展示和导航，但不能倒改原始测量。
1. 沿原有生产链更新派生文件、manifest、校验和及对应测试；字段语义不明确时先读 schema/生产脚本。 `.publication.json` 与
   `.publication.part-*` 是压缩发布包，不能直接把分片当 JSON 改；使用 [发布工具和格式说明](website-publication/README.md)。
1. 按任务运行下列已有校验器。它们的通过不代表硬件实验或模型正确性通过。
1. 只暂存本轮路径，然后更新导航并检查；提交审查后再发布固定版本链接。

```bash
# 根据改动范围选择；网站发布另按 website-publication/README.md 执行
python scripts/verify_submission_checksums.py --root submissions
python scripts/validate_public_leaderboard_snapshots.py
PYTHONPATH=src python -m vllm_hust_benchmark.dataset_validation --root leaderboard-data/dataset-validation
PYTHONPATH=src python -m vllm_hust_benchmark.official_targets --check --repo-root .

# 先 git add <明确的新/删路径>；只读内容变化不改变路径索引
python scripts/data_index.py --write
git add docs/data-index.json
python scripts/data_index.py --check
python -m pytest -q tests/test_data_index.py
```

索引完全离线、按路径确定性生成，不扫描未跟踪目录，不读取证据正文，不下载或解包资产。 生成器只写 `docs/data-index.json`，不要手工修改它。CI 会检查其是否与 tracked
路径一致。
