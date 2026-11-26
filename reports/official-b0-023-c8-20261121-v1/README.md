# 官方 0.23 C8 原件与 CPU 复算

保留合成连续负载的 4K 正式 267/0、16K 正式 137/0（请求/失败），预热单列。原整轮后续阶段失败，F/T 与第三方认证仍未成立。

设备执行源码：[7ffd13b7](https://github.com/DeepLinkNext/DeepLinkEngine/tree/7ffd13b7fe9f5c9d0cdf010d2ef6a977706c34c6)；CPU
报告源码：[b7df8fda](https://github.com/DeepLinkNext/DeepLinkEngine/tree/b7df8fda9bbbb788b099cdb01d96c5754665812b)。本次复算没有新模型请求。npu-smi
工具版本为 26.0.rc1，驱动未记录；实际图模式为 FULL_DECODE_ONLY。

公开包仅附完整 C8、父轮身份和相关源码；其余 20 个原文件只保留摘要，未运行的真实上下文输入不分发。初次目录核验检查了完整原封存；便携冷复算只核验所附闭包，见
`cold-replay.json`。

获取上述 CPU 报告源码，保留本附件目录的配置与归档，执行：

```sh
python run_all.py --official-baselines --report-only --config /附件目录/report-only.example.json
```

无需 SDK、权重或 NPU，输出目录须为新目录。配置同时固定归档摘要
`26123ca7151fe25533c5661635b264df98a4e43a2fb4f52a83fb3670c503bd93` 和原版本封存摘要
`721d6dff52613510a3e9ff33b2d0d794c7ff5c286551572decc014261d26cc44`；归档源码不执行。命令成功表示 CPU 重建完成，不代表全套通过。

[下载 C8 原件与复算归档](https://github.com/DeepLinkNext/DeepLinkEngine/releases/download/official-b0-0.23-c8-20261121-v1/portable-c8.tar.gz)；附件摘要见
[归档清单](archive-manifest.json)。

| 阶段                             | 请求 / 失败 | token/s（含排空） | TTFT / ms   | TPOT / ms |
| -------------------------------- | ----------- | ----------------- | ----------- | --------- |
| 4K / C8 路 continuous synthetic  | 267 / 0     | 225.504618        | 1198.376138 | 30.780033 |
| 16K / C8 路 continuous synthetic | 137 / 0     | 115.897037        | 3303.719687 | 56.185415 |
