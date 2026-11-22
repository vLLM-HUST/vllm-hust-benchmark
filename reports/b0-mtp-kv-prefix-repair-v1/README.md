# B0 0.23 / 0.30 的 MTP KV 补丁复测

Qwen3.5-35B-A3B，Ascend 910B2，TP2，MTP2。每版先做 200 请求的算量适用性核验，再做 3 轮独立原生计时，每轮 200 请求、每请求输出 256 tokens。

| 补丁版 | 必要算量 F 区间（FLOP）           | 三轮合计 T（秒） | 有效算力（TFLOP/s/卡） |
| ------ | --------------------------------- | ---------------: | ---------------------: |
| 0.23.0 | 5412404567406489–5441441524836282 |    343.153977552 |      7.886262–7.928571 |
| 0.30.0 | 5412225380549900–5441434216427102 |    315.084118686 |      8.588540–8.634891 |

算力按 F / (T × 2) 聚合。证据范围为打补丁后的固定负载，正式 B0/B1 配对、官方原版 F/T 核验及五项、上下文、四租户指标沿用各自报告。
发布包提供 CPU 复算，实验尚无独立第三方见证。

- [机器可读报告](../../website-publication/data/b0-mtp-kv-prefix-repair-v1/report.json)
- [补丁、SHA-256 与复算脚本](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/b0-0.23-0.30-mtp-kv-prefix-repair-v1)
- [官方 0.23 原测](../official-b0-023-full-report-20261122-v1/README.md)
- [官方 0.30 原测](../official-b0-030-full-report-20261122-v1/README.md)

从 Release 下载所有附件后运行 `python verify_release.py`。Benchmark 报告与 Release 附件逐字节一致，SHA-256 为 `d9a53879647e68302736b77d765aae23e0c4ebbf19ee7ee0f9d463a67f327591`。<!-- pragma: allowlist secret -->
