# 35B 历史配对性能：2026-10-30 冻结实现
这组结果属于 `1294c95` 加冻结本地补丁；它不是 #11 合并版重新测量。已取回原生 linear Eagle3 与 DiffSpec 的同 target/draft、K4、负载和计时窗口原件。
| 场景 | 方法 | 输出 tok/s 中位数 | 接受率 | TTFT P50 秒 | 请求均值 TPOT P50 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| context-2048-output-4096 | Target-only | 60.10 | — | 1.278 | 66.278 |
| context-2048-output-4096 | Eagle3 | 105.35 | 38.85% | 1.398 | 34.938 |
| context-2048-output-4096 | DiffSpec | 111.01 | 44.95% | 1.440 | 35.101 |
| context-32768-output-2048 | Target-only | 51.77 | — | 14.349 | 69.873 |
| context-32768-output-2048 | Eagle3 | 58.89 | 16.53% | 15.737 | 57.400 |
| context-32768-output-2048 | DiffSpec | 80.90 | 42.20% | 16.456 | 40.746 |
| context-32768-output-8192 | Target-only | 56.75 | — | 14.925 | 68.088 |
| context-32768-output-8192 | Eagle3 | 65.75 | 16.49% | 15.947 | 55.315 |
| context-32768-output-8192 | DiffSpec | 97.35 | 41.92% | 16.609 | 38.668 |

每场景每方法三次独立引擎启动；两个较短场景共用一组引擎，8192 输出另用三次。合计 18 次独立启动、108 个固定长度请求，启动/请求失败与抢占为零。吞吐计时包含 prefill 和 decode，排除加载、编译、热身。九格中位数已从原始 summary.json 独立复算一致。

2K→4096 相对 Eagle3 为 1.05×，范围重叠且 TPOT 未改善；32K→2048 为 1.37×，32K→8192 为 1.48×。长输出 TTFT 同时增加，不把解码收益写成所有延迟改善。

每种方法的自然 EOS 小型检查为 60/60 响应通过，来自十道题的重复；完整长文本 token 全等未通过，也没有标准 MMLU-Pro 准确率结论。每格请求 P99 为 12 个样本。完整延迟、失败、输出等价和输入来源见原始 [报告](historical/REPORT-20261030.md)与 [results.json](historical/results.json)。

原始大包 `diffspec-review-20261030.tar.gz` 的 SHA256 与清理前索引一致，内嵌 `archive/reference-measurements.tar.gz` 保存两个短场景的三轮原始记录。新功能运行不产生新的加速数字。
