# DiffSpec 本轮补证：合并版功能回归与历史性能配对

日期：2026-11-15（北京时间）。对应
[DiffSpec #1 本轮要求](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec/issues/1#issuecomment-6846020813)及
[补证总表 #664](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/664)。

## 可引用结论

> 在固定 #11 合并版 `48b1828`、Qwen3.5-35B-A3B BF16、TP4/draftTP4、K4、PIECEWISE 下，两条真实请求各完成 256 token。在真实
> target 快照上受控增加一条 dummy 行后，21 个两请求检索周期均排除 dummy 和拒绝后缀；四 rank 在全部 24 个检索周期中所选 chunk 与 token
> 摘要一致。该结论是指定功能边界回归，不代表完整数值无损或性能验收。

35B 加速数字继续引用 **2026-10-30 冻结实现**的原生 linear Eagle3 配对：32K→2048 与 32K→8192 的端到端输出吞吐中位数分别为 Eagle3 的
1.37×、1.48×，包含 prefill。此次未重跑性能，不把历史数字改记为 #11 合并版新测结果。

## 本轮任务与完成材料

| 要求                                   | 本次材料                                                                                                                                                              | 状态                                             |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 固定 #11 及实际环境                    | [31 个源码 Git blob 清单](functional/pr11-source-manifest.json)、[Core/Ascend 身份](functional/backend-identity.json)、[预检日志](functional/attempt01/preflight.log) | 完成；20 个 backend 文件与冻结实测摘要一致       |
| 两真实请求加 dummy                     | [指定组合汇总](functional/attempt02/functional-summary.json)、[四 rank 原件](functional/attempt02/artifacts/traces/)、[构造及日志说明](FUNCTIONAL.md)                 | 完成；21 个周期，dummy 为受控填充                |
| 拒绝后缀排除                           | 同一批原件中的实际 valid counts、kernel 前缀长度、所选 token 最大值                                                                                                   | 完成；21 个指定周期均覆盖拒绝后缀                |
| TP4 四 rank 所选块一致                 | 同一批原件中的 selected_chunks 和 selected_tokens_sha256                                                                                                              | 完成；24/24 检索周期一致，四 rank 是同一事件副本 |
| 合并版已有边界用例                     | [CPU 原始输出](functional/attempt01/cpu.log)及 [命令](functional/attempt01/cpu-command.json)                                                                          | 36 passed、33 deselected                         |
| 35B 原生 Eagle3 配对、延迟、失败和质量 | [历史性能说明](PERFORMANCE.md)、[原始机器结果](historical/results.json)、[冻结报告](historical/REPORT-20261030.md)                                                    | 原件已恢复；保留冻结版本归属                     |
| 可核验归档及提交                       | [附件清单](release-manifest.json)、[离线复核](verify_evidence.py)、SHA256SUMS                                                                                         | 附件已发布，清单含固定下载链接及下载回验记录     |

## 证据范围

本轮新增 36 项聚焦 CPU 用例和两次功能模型启动：第一次自然运行两请求，未出现额外 dummy；第二次在两请求真实快照上受控增加 padding，覆盖指定分支。两轮各完成 512
token，均保留原件。第一轮汇总的 `passed=false` 表示未覆盖指定 dummy 场景，不是请求执行失败。

本次材料只支持上述指定回归结论；完整输出等价、HTTP serving 及完整准确率验收不在本次结论范围内。

## 复核与附件

取得 [release-manifest.json](release-manifest.json) 中两份固定附件后：

```bash
python verify_evidence.py --assets /path/to/downloaded-assets
```

复核器只读取文件、日志和冻结 JSON，核验附件 SHA256、四 rank 边界和九格历史中位数，不启动设备，不执行冻结包里的代码。旧报告及指标字节保持不变，新的说明另行追加。

原始 DiffSpec 方法归属外部团队；本地贡献为兼容修复、Ascend 实现与专项验证。新功能记录与历史性能分别归档，不进入统一 native-MTP2/APC/async Frontier
对照。
