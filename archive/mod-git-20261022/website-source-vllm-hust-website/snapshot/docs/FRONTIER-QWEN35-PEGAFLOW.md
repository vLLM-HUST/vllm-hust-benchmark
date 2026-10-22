# Qwen3.5-35B PegaFlow Frontier results

PegaFlow and `PegaKVConnector` were launched through Extension Manager and compared with the single
existing `swe-unified-native-20260927` series. Each candidate cell is one real-online 900-second
observation; no Native point was rerun or replaced.

| Concurrency | Native output tok/s | PegaFlow output tok/s |  Change | P90 decode tok/s/user |
| ----------: | ------------------: | --------------------: | ------: | --------------------: |
|          C1 |   91.54555555555555 |     65.52333333333333 | -28.43% |     113.9060417414391 |
|          C2 |   152.7211111111111 |    167.14333333333335 |  +9.44% |    109.48681228877716 |
|          C4 |  217.47555555555556 |    237.96777777777777 |  +9.42% |     87.30566240630844 |
|          C8 |   291.1066666666667 |     361.4411111111111 | +24.16% |     73.03849278224763 |
|         C16 |              359.75 |    459.71555555555557 | +27.79% |     46.37622765934938 |

The website data model dynamically computes the five-point geometric-mean throughput change as
**+6.342297886583825%** from the unrounded candidate/Native ratios. C1 regresses even though the
aggregate is positive; both observations are retained.

Fixed controls: Qwen3.5-35B-A3B BF16, Ascend 910B2, TP2/PP1/DP1, expert parallel disabled,
262144-token context, APC, aligned Mamba cache, asynchronous scheduling, native MTP2, thinking
enabled, temperature zero, FULL_AND_PIECEWISE graph mode, 16 maximum sequences, 4096 maximum batched
tokens, and 26038239232 KV-cache bytes per serving chip. PegaFlow used a 30 GiB pinned host pool and
`PegaKVConnector` in `read_write` mode.

All five windows were valid and completed with zero failures. The 1,957 request records agree on
token IDs, usage completion tokens, expected output tokens and streamed chunk counts. Across the
five windows, the real control path recorded 3,386 successful loads and 3,050 successful saves, zero
load/save failures, 177,294,016,512 bytes loaded and 83,560,382,464 bytes saved. Native MTP accepted
794,265 of 803,298 draft tokens. The owned services stopped after measurement and all four visible
NPUs were released.

The tested PegaFlow revision is
[`cd64ecc283ff856a44437a9a25659929ef3a0653`](https://github.com/vLLM-HUST/pegaflow-hust/tree/cd64ecc283ff856a44437a9a25659929ef3a0653).
The immutable
[formal evidence bundle](https://github.com/vLLM-HUST/pegaflow-hust/blob/64ef9d0b5c4f11589d296f8500dc1664ffc346dc/handoff/2026-09-29-qwen35-frontier-formal/README.md)
contains the clean qualification, raw request and token timing streams, vLLM and PegaFlow metric
snapshots, service logs, Extension Manager lifecycle, NPU boundaries, recomputation script and
SHA-256 manifests.
