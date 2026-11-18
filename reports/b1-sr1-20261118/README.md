# B1-sr1 experiment evidence

B1-sr1 paired against frozen B1-r8 on physical cards 0 and 1, two Ascend 910B2 devices, TP2, Qwen3.5-35B-A3B BF16, E36/R54. There was one paired run in each workload. B1-sr1 regressed in throughput and latency and remains experimental.

| Workload | B1-r8 output tok/s | B1-sr1 output tok/s | B1-r8 end-to-end P95/P99 (s) | B1-sr1 end-to-end P95/P99 (s) |
| --- | ---: | ---: | ---: | ---: |
| 4K continuous C8 | 386.69 | 95.68 | 5.83 / 6.31 | 23.07 / 23.42 |
| 16K continuous C8 | 159.69 | 69.36 | 14.19 / 15.38 | 32.90 / 35.02 |
| MMLU-Pro 200 | 1,499.31 | 131.86 | 438.11 / 465.07 | 5,128.76 / 5,365.80 |

MMLU-Pro accuracy was 150/200 (75.0%) for B1-r8 and 151/200 (75.5%) for B1-sr1. All 200 requests per arm succeeded. Generation reached the token cap on 21 and 26 requests, respectively. MMLU-Pro submits all 200 requests together, so TTFT and end-to-end latency include queueing. Continuous throughput counts output tokens over the submission and drain window. The earlier B0/B1-r8 publication is a separate comparison.

[Summary JSON](summary.json) includes TTFT and TPOT tails, HBM, effective target KV capacity, draft KV activity, preflights and data coverage. [Request metrics CSV](request-metrics.csv) includes all 1,251 measured responses from the completed workloads, without case content or answer text. The raw streams and diagnostic evidence are accessible to authorized collaborators in the [private evidence archive](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/b1-sr1-evidence-20261118).

HLE, FrontierScience, SWE-bench-Pro and Terminal-Bench 2.1 had no completed pair when this snapshot was taken. HLE image handling and host State restore passed hardware preflights. Those checks are reported separately from scored workload results.
