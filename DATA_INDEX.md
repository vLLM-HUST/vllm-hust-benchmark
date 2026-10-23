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
  [2026-10-23 后续盘点](archive/mod-followup-20261023/README.md)列出尚待分类的 MOD 数据、BiDKV 消费者迁移、
  旧安装资产保留依据及硬件/审核阻塞；路径命中不等于可删除实测原件。
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

### 网站引用与 Release 附件入口

[V0.72 上线与索引复核回执](docs/data-entrypoints-v072-audit.json)记录 2026-10-23 的线上字节校验、
发布身份订正及附件状态；它是带时间的验收快照，不替代下述当前索引，也不代表 M01 已通过。

[关联入口索引](docs/data-entrypoints.json) 补足目录索引不包含的三类信息：

- `publication_bindings`：网站 `data/` 路径到 benchmark 原文件或压缩描述符/分片的映射，包含还原后 SHA256。
- `references`：固定网站提交的前端 HTML/JS/MJS 字面引用，以及 benchmark 发布数据中的链接、结构化证据路径；保留引用来源，区分已映射、动态模板、未映射和外部链接。
- `releases`：GitHub 分页 API 返回的全部非 draft Release 及附件名称、下载 URL、大小、资产 ID、API digest、上传状态。附件通过逐 Release
  分页端点读取，避免列表内嵌清单漏项。仅 `state=uploaded` 表示上传完成； `starter`
  是未完成上传，保留失败元数据但不可当作可下载证据。包含历史附件，不将它们自动认定为当前验收证据。

[采集快照](docs/data-entrypoints-sources.json) 固定网站 commit、输入文件摘要、Release 元数据和采集时间。
发布清单优先采自线上；若获取失败，明确记录错误并标明使用的是网站 Git 中的历史副本，不能当作线上当前版本。 网站部署时会重新生成数据，网站代码提交与线上发布数据身份不是同一个概念。

```bash
python scripts/data_entrypoints.py --query midterm
python scripts/data_entrypoints.py --kind publication_bindings --query dataset
python scripts/data_entrypoints.py --kind releases --query m01 --limit 20
python scripts/data_entrypoints.py --kind assets --query profiler --limit 20
python scripts/data_entrypoints.py --query unmapped-frontend-reference

# 本地数据/描述符内容变化：离线重建并检查
python scripts/data_entrypoints.py --write
python scripts/data_entrypoints.py --check

# 网站或 Release 更新：显式联网重新采集，不修改网站，不下载 Release 附件
python scripts/data_entrypoints.py --refresh \
  --website /path/to/vllm-hust-website --revision <完整网站提交>
```

`--check` 会发现本地发布内容、分片及映射的过期情况，但不会联网发现新增/删除的远端 Release。 远端更新后应执行 `--refresh` 并审查快照差异。索引不执行
JavaScript，不保证动态链接已解析，也不证明 外部链接可下载、附件 SHA256 已回验或页面已经完成浏览器验收。`unmapped-embedded-path` 可能是原始机器路径，
不是自动判定网站断链；`snapshot_publication_status` 仅比较与采集清单的字节身份。 Release 中 manifest/inventory/SHA256SUMS
等附件也是可直接下载的入口，包内部成员仍由这些清单描述。

若既有快照包含公开 Release，而刷新返回空清单，工具会拒绝覆盖；先独立核对 API 和可见性， 不能把一次空响应当作所有附件已删除。线上清单获取失败时的历史副本回退也必须保留错误标记。

### 数据维护步骤

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

目录索引 `data_index.py` 完全离线、按路径确定性生成，不读取证据正文，只写 `docs/data-index.json`。 关联入口生成器 `data_entrypoints.py`
另行读取发布数据、核验并还原发布分片；只有显式 `--refresh` 才联网， 从固定网站 Git 对象、线上发布清单及 Release API 更新采集快照。不下载或解包 Release
附件，不执行网站代码。 不要手改生成索引；CI 检查目录路径和关联入口是否与本地来源及采集快照一致。
