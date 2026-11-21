# 官方 0.23 C8 原件与 CPU 复算

本归档保留已完成的合成连续负载：4K 正式 267/0、16K 正式 137/0（请求/失败），预热单列。原整轮在后续阶段失败，原状态保留；固定五项、真实 32K、四租户及新 F/T 不由本归档认证。

原设备执行源码：`7ffd13b7fe9f5c9d0cdf010d2ef6a977706c34c6`；本次 CPU
报告源码：`12cf909492e63d5015cb5e548d7e3e83641db9ef`。两者分别记录，CPU 重建没有新模型请求。原件中的 npu-smi 工具版本为
26.0.rc1，驱动版本未记录。实际图模式为 FULL_DECODE_ONLY。

从 DeepLinkEngine 获取上述 CPU 报告源码，保留本附件目录中的配置和归档，执行：

```sh
python run_all.py --official-baselines --report-only --config /附件目录/report-only.example.json
```

无需 SDK、模型权重或 NPU；输出目录必须是新的目录。`source_sha256` 为归档 SHA256
`d4e1e0fe13b0300909735cb691a9cc2991f5bfad5da94a223637b422dd7d616b`。命令成功只表示 CPU
核验与重建完成。原始版本/阶段封存、SSE、相关源码字节、原启动记录及公开源码核验收据均保留；归档源码不执行。

[下载原件与 CPU 复算归档](https://github.com/DeepLinkNext/DeepLinkEngine/releases/download/official-b0-0.23-c8-20261121-v1/portable-c8.tar.gz)；摘要与各附件见
[归档清单](archive-manifest.json)。

| 阶段                             | 请求 / 失败 | token/s（含排空） | TTFT / ms   | TPOT / ms |
| -------------------------------- | ----------- | ----------------- | ----------- | --------- |
| 4K / C8 路 continuous synthetic  | 267 / 0     | 225.504618        | 1198.376138 | 30.780033 |
| 16K / C8 路 continuous synthetic | 137 / 0     | 115.897037        | 3303.719687 | 56.185415 |
