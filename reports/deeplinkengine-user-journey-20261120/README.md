# DeepLinkEngine 安装与流程验证（2026-11-20）

补齐独立安装、模型加载、API 调用及退出清理的实机证据，保留六次未完成流程的原始失败。这里只归档已有记录，没有新增设备实验，也不提供 R9/R10 成绩或完整验收结论。

[汇总与原件索引](publication-v1.json) ·
[原始存档与恢复工具](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/deeplinkengine-user-journey-20261120-v2)
·
[匿名下载及全量恢复核验](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/deeplinkengine-user-journey-20261120-v2/public-recovery-verification-v1.json)

下载同一 Release 的 `recover_archive.py` 和存档后执行：

```sh
python recover_archive.py --archive deeplinkengine-user-journey-20261120-v2.tar.gz --out recovered-user-journey
```

共 203 份原始记录、7 个完整源码快照，恢复 4,936 个文件；逐文件 SHA-256 与源码 Git 字节身份均已核验。恢复仅使用 Python
标准库，不安装依赖或启动模型。运行时、模型权重和生成缓存不在包内，原版本、依赖及权重标识随记录保留。
