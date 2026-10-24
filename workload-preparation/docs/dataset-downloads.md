# 数据获取

先运行 `python -m vllm_hust_workload_data.catalog download-plan SOURCE_ID` 查发布方、版本及获取说明；用
`catalog source SOURCE_ID` 查目录身份。登记并不表示文件已经取得或许可已经复核。

公开数据从发布方渠道取得并遵守许可证；gated 数据由获授权使用者取得。企业原始数据只放受控存储，不提交仓库。模型、任务环境和评分器需另外安装。

新取得文件使用 `pipeline bind` 固定文件 SHA
与版本；绑定仅证明本地文件身份，不冒充发布方认证。完整用法见[组件说明](../README.md)，逐来源进度见[验证清单](../../docs/workload-data/DATASET_CHECKLIST.zh-CN.md)。

## 已有获取与数据校验能力的迁移入口

以下工具复用研究仓原有的数据工具，不运行模型、下载的程序或研究分析。它们是Git源码/sdist入口，不是wheel自动安装的命令；只安装wheel的用户需要取得相同固定提交的源码。相对脚本路径从仓库根目录执行。数据载荷留在忽略的本地缓存或外部授权目录，回执应放私有工作目录。

```sh
# 默认只查询发布方文件元数据；不会下载载荷或自动接受访问条款。
python workload-preparation/source_tools/download_dataset.py GSM8K
# 只有明确指定--download才下载；必须给出独立回执，不修改研究仓的回执注册表。
python workload-preparation/source_tools/download_dataset.py GSM8K --download --files main/test-00000-of-00001.parquet --revision REVIEWED_REVISION --output /private/receipts/gsm8k-acquisition.json
# 既有固定release元数据来源；按原字节数及SHA校验，不执行其内容。
python workload-preparation/source_tools/acquire_release_datasets.py --source BURSTGPT --data-root /private/datasets --output /private/receipts/burstgpt-release.json
# 原固定文件逐文件适配校验；省略limit-per-file为全量，而非只核对文件摘要。
python workload-preparation/source_tools/validate_source_adapters.py --source GSM8K --data-root /private/gsm8k --limit-per-file 3 --output /private/receipts/gsm8k-adapter.json
# 只检查Parquet内嵌图像字节存在；不宣称解码、视觉质量或模型推理通过。
python workload-preparation/source_tools/validate_embedded_media.py CHARTQA --data-root /private/chartqa --output /private/receipts/chartqa-media.json
```

`download_dataset.py`复用原HF文件获取路径，仅处理明确选择的JSON/JSONL/Parquet；非HF来源、旧revision归档、媒体归档等仍使用发布方专用路径。`acquire_release_datasets.py`仅复用既有BURSTGPT/TraceLab固定选择，不是新增数据集或任意URL下载器。发布方HEAD变化、gate、字节/摘要不一致时不自动替换旧文件。

本轮获取命令在测试服务器执行，release工具显式使用仓库外的`--data-root`；实际文件位于`DATA_ROOT/SOURCE_ID/TAG/`。回执的`path_relative_to=explicit_data_root`说明其中路径相对此根目录。省略参数仅保留旧缓存调用兼容，不用于本轮数据交付。已有文件先核验并复用，摘要不符时拒绝覆盖。

原三份公共工具`verify_public_sources.py`、`materialize_public_datasets.py`、`validate_paio_assets.py`保留在同一`source_tools/`目录。企业校验只针对用户明确提供的授权根目录，报告为聚合/脱敏数据完整性核验，不迁入科研观测或机制分析。动态源冻结改用`pipeline bind`与显式manifest，不把研究`results/`注册表作为公共依赖。
