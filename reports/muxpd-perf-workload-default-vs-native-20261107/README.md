# MuxPD default mode vs native on real-text derived workloads, TP2/TP4 (2026-11-07)

[REPORT-v1.md](REPORT-v1.md)（中文）：用真实文本（LongBench v1 qasper、OASST1）构造的派生负载上，对 MuxPD 默认模式与 native
vLLM-Ascend 在 TP2 / TP4 做的单轮性能对比，沿用 #36 的 SLO 和指标。到达过程和固定输出长度是测试设置，不代表原生任务执行或
生产流量。前序报告：`reports/muxpd-perf-default-vs-native-20261107`。跟踪 issue：
vLLM-HUST/vllm-ascend-MuxPD-hust#36。SHA256SUMS-v1 覆盖本目录全部文件。
