# R10 与两个补丁版 B0 的原生 F/T 对照

Qwen3.5-35B-A3B，BF16，Ascend 910B2 TP2。三臂使用同序 200 条输入、15 组预热和相同的采样设置。 两版 B0 复用各自封存的三轮原生计时与独立算量证据。R10
稳定包新测三轮，每轮 200 条、每条输出 256 tokens。 两个对照共用这一组 R10 原生时间。各臂的时钟来自各自的双卡环境。B0 使用 24 个执行席位，R10 使用发行包的
E36-R54。

| 版本                   |           必要算量 F 区间（FLOP） | 三轮合计 T（秒） |    TFLOP/s/卡 | R10 / B0 有效算力 |
| ---------------------- | --------------------------------: | ---------------: | ------------: | ----------------: |
| B0 0.23.0 补丁         | 5412404567406489–5441441524836282 |          343.154 |   7.886–7.929 |      2.308–2.333× |
| B0 0.30.0 补丁         | 5412225380549900–5441434216427102 |          315.084 |   8.589–8.635 |      2.119–2.142× |
| B1 R10 `2026.11.21.10` | 5412097110213408–5441172863529042 |          147.867 | 18.301–18.399 |                 — |

有效算力按三轮算量之和除以三轮时间之和与卡数。对照区间用每臂各自的必要算量上下界。 独立人工见证待安排。原参考分母和固定五项结果记录在各自的报告中。

- [机器可读报告](../../website-publication/data/r10-stable-patched-b0-ft-20261123-v1/report.json)
- [公开 F/T 算术复算包](https://github.com/DeepLinkNext/DeepLinkEngine/releases/tag/r10-native-patched-b0-ft-20261123-v1)
- [两个 B0 补丁版的归档](../b0-mtp-kv-prefix-repair-v1/README.md)
- [R10 固定五项报告](../r10-contiguous-restore-fixed-five-20261121/README.md)
- [R10 正式发行包](https://pypi.org/project/vllm-hust-engine/2026.11.21.10/)

## 源码资格复算

[正式汇总生成脚本](qualify_r10_ft.py.source)仅给 SHA-256 常量添加了扫描白名单注释。执行原件的 SHA-256 为
`98089c98a839f55136412b0df0b1d5966d457d423f48c19d8d0b2ed165d03dbb`。脚本核对 R10 稳定包及实际的原生 gated RMS
配置，从新三轮原始输出重算 F，核验每轮时钟，再用原完整参考和两版 B0 封存报告计算结果。旧 HTTP companion 只用于源码适用性核验。
[GDN 源码复核回执](source-audit.json)记录稳定配置下原活跃源码路径的比较结果。

拥有原始封存读取权限时，可在原 10202 安装环境重新生成正式汇总，全程使用 CPU。

| 参数           | 封存输入                                                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `--kit`        | 正式一键入口为该安装生成的原生计时客户端，清单摘要 `77e18df51b200a564546fd316259684aeb373fd27502fa87ccf0699458cebce7`                     |
| `--native`     | 正式 `formal-v1` 原件，`--native-seal` 填 `1296bec274af033628542942df39fbe81c47ff3c90db247ba9d30844d96d8f5f`                              |
| `--deployment` | 该轮实际安装的 R10 SDK，按原件记录的路径读取                                                                                              |
| `--companion`  | 独立 HTTP companion 的封存读回，终态摘要 `aebd5c5ea84ed0435ba7f348a868b33eb09f34216f99b0a4deb813cba1a071bd`                               |
| `--reference`  | 已从原完整证据复核的 `qualified-full-reference-statistics.json` 所在目录                                                                  |
| `--b0`         | [两版 B0 补丁封存报告](https://github.com/DeepLinkNext/DeepLinkEngine/releases/download/b0-0.23-0.30-mtp-kv-prefix-repair-v1/report.json) |
| `--out`        | 新的汇总 JSON 路径                                                                                                                        |

以 Python 执行 `qualify_r10_ft.py.source`
并传入上述参数。汇总应与[已发布的 comparison.json](https://github.com/DeepLinkNext/DeepLinkEngine/releases/download/r10-native-patched-b0-ft-20261123-v1/comparison.json)
字节一致，SHA-256
`923291f560f27503947c3660df829752bd424c47075259cb9d5e51c7de0a1837`。没有原始访问权限时，可用公开复算包独立复算算量与时间。
