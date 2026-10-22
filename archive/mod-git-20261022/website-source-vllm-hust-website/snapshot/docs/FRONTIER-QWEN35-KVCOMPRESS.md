# Qwen3.5-35B KVCompress Frontier results

KVCompress was launched through ECPA and compared with the single existing Qwen3.5-35B unified
Native series. Each cell is one real-online 900-second observation. Values are output token/s.

| Concurrency | Native | KVCompress |  Change |
| ----------: | -----: | ---------: | ------: |
|          C1 |  91.55 |      94.57 |  +3.31% |
|          C2 | 152.72 |     136.71 | -10.49% |
|          C4 | 217.48 |     206.60 |  -5.00% |
|          C8 | 291.11 |     286.10 |  -1.72% |
|         C16 | 359.75 |     347.75 |  -3.33% |

Geometric mean throughput change: **-3.55%**, dynamically derived from the five candidate/Native
ratios by the website data model. This is a valid negative result, not an optimization claim.

Fixed controls: Qwen3.5-35B-A3B BF16, TP2/PP1, 262144-token context, APC, aligned Mamba cache,
asynchronous scheduling, native MTP2, FULL_AND_PIECEWISE graph mode, 16 maximum sequences, 4096
maximum batched tokens, and 26038239232 KV-cache bytes per chip.

The run recorded 434 successful compression scheduler commits. Every commit received exactly one TP0
and one TP1 acknowledgement. Every window had positive APC and native-MTP counter deltas, zero
preemptions and zero failed requests, and the service released both selected NPUs after the series.

Plugin revision:
[`2ca0f9335399a698342285870df1514e9c519bd4`](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/commit/2ca0f9335399a698342285870df1514e9c519bd4).
The immutable
[formal evidence bundle](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/a020c164ec68cf3a7c53a40c0129b37a14ed1a05/handoff/2026-09-28-qwen35-frontier-formal/README.md)
contains the clean qualification, raw request/token timing streams, service logs, manager lifecycle,
compression commit/ack audit, NPU boundaries and SHA-256 manifests.
