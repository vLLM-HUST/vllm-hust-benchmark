# 旧 Native 包版本恢复

本目录是派生元数据，不替换或修改原始归档。MMLU-Pro、FrontierScience、HLE-Verified 的五份版本回执均记录：

- vLLM `0.28.1.post1.dev143+gf18cf803c.empty`
- vLLM-Ascend `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909`

这些记录属于开发运行栈，不是指定的 `0.18/0.18` 原生基线。恢复身份不等于补齐指定基线，也不证明源码没有修改、质量配对成立或性能运行独占。映射范围仅限列出的版本回执及对应运行。

`version-mapping.json` 记录仓库相对路径、精确归档成员、归档和成员 SHA256。审计器只读取指定 `versions.txt`
成员，不提取问答记录，不导出版本文件内其他环境信息。原归档允许分散在同一仓库的多个检出目录；输出不包含容器绝对路径。

复核：

```bash
python3 audit.py --repo /path/to/benchmark --check version-mapping.json
```

若原始证据分散在不同检出目录，重复传入 `--repo`。在仓库根目录运行测试：

```bash
python3 -m pytest tests/test_native_version_recovery.py -q
```

测试使用当前原始归档实读验证，并检查缺失、重复包身份和冒充 0.18 的情况；不调用 NPU，不启动服务。
