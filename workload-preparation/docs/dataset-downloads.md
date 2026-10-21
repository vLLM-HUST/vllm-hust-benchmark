# 数据获取

先运行 `python -m vllm_hust_workload_data.catalog download-plan SOURCE_ID` 查发布方、版本及获取说明；用
`catalog source SOURCE_ID` 查目录身份。登记并不表示文件已经取得或许可已经复核。

公开数据从发布方渠道取得并遵守许可证；gated 数据由获授权使用者取得。企业原始数据只放受控存储，不提交仓库。模型、任务环境和评分器需另外安装。

新取得文件使用 `pipeline bind` 固定文件 SHA
与版本；绑定仅证明本地文件身份，不冒充发布方认证。完整用法见[组件说明](../README.md)，逐来源进度见[验证清单](../../docs/workload-data/DATASET_CHECKLIST.zh-CN.md)。
