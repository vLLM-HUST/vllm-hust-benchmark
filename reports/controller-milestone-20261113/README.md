# 国产推理控制面里程碑实测

毛言粲 · 2026-11-13

## Request Throttling Controller

在昇腾上实现多轮任务放行与跨轮缓存复用。在 Qwen3.5-35B-A3B 双卡 910B2 同引擎回放中，接入原版 ThunderAgent 调度后，112 个任务、336 次模型调用的整批耗时由 353.74 秒降至 270.80 秒，缩短 23.45%。

- [双卡 B1 结果与方法](../controller-b1-discovery-20261104/README.md) · [机器可读结果](../controller-b1-discovery-20261104/summary.json)。B 和 C 使用相同缓存配置，各完成 336 次调用。缓存输入占比从 34.73% 升至 57.02%，实际计算的输入词元由 3,911,106 降至 2,575,715。单次回放，任务质量和独立重复仍待验证。
- [四卡比较表](request-throttling/four-card-comparison.csv) · [直连逐轮数据](request-throttling/vllm-direct-service-rounds.jsonl) · [限流逐轮数据](request-throttling/thunderagent-program-throttling-rounds.jsonl)。2026-10-07，Qwen2.5-14B，每秒到达一个任务。两组各完成 200 个任务、6,294 次调用和 639,700 个输出词元，逐轮输入摘要一致。整批耗时为 5,199.2993 与 2,717.078 秒，缩短 47.74%。限流组同时关闭了后端连接复用，结果归属于这组联合配置。输入为固定轨迹回放。

四卡证据来自 `vllm-workflow-aware-serving/data/results/workload/` 的 `four-card-200-programs-fresh-connections-random-seed-11-20261007` 及同名 `-rate-1` 目录。比较表保留归档内的原始运行路径。

## Instance Controller

在昇腾上实现推理副本就绪后接流、排空后缩容。在 Azure Code 已知到达时刻的回放中，提前批量扩容至四副本，使目标峰段达标请求相对历史固定两副本从 58 条增至 191 条，首词元时延 P95 从 4.76 秒降至 1.26 秒。

| 观测项 | 历史固定两副本 | 计划扩缩 1–4 副本 |
| --- | ---: | ---: |
| 目标峰段完成请求 | 334 | 334 |
| 目标峰段达标请求 | 58 | 191 |
| 目标峰段首词元时延 P95（秒） | 4.7635 | 1.2563 |
| 全窗口完成请求 | 4,132 | 4,132 |
| 全窗口达标请求 | 1,649 | 1,213 |

达标指首词元时延 ≤ 2 秒且每输出词元时延 ≤ 100 毫秒。目标峰段按到达时间 930–960 秒统计。动态组于第 800.78 秒提交三个作业，第 894.48 秒首次观测到四副本就绪。第 3、4 副本在该峰段实际处理 89、78 条请求。全窗口为 1,320 秒。该轮使用已知到达时刻，历史对照未控制运行顺序和物理卡位。全窗口未通过收益门槛。

- [原始分析回执](instance-scaling/scheduled-gates-result.json) · [逐段统计](instance-scaling/interval-attribution.json)。源归档为 Instance Controller 的 `myc-instance-schedule-r2`，历史固定组为 `myc-instance-capacity-two-r2`。JSON 保留源文件摘要及峰段路由回执。
- [归档文件校验和](SHA256SUMS)。这些文件用于里程碑专题复核。
