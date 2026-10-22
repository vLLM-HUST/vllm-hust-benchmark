# Qwen3.5-35B unified MOD results

All candidates use the same five-point Native series. Values are output token/s; parentheses show
the change from Native at the same concurrency.

| MOD                      |             C1 |              C2 |              C4 |              C8 |             C16 | Geometric mean vs Native |
| ------------------------ | -------------: | --------------: | --------------: | --------------: | --------------: | -----------------------: |
| Native                   |          91.55 |          152.72 |          217.48 |          291.11 |          359.75 |                    0.00% |
| mooncake-vllm-connectors | 90.82 (-0.79%) | 151.70 (-0.67%) | 216.22 (-0.58%) | 290.27 (-0.29%) | 352.89 (-1.91%) |                   -0.85% |
| kv-tiering-migration     | 91.03 (-0.56%) | 150.45 (-1.49%) | 215.93 (-0.71%) | 286.12 (-1.71%) | 341.63 (-5.04%) |                   -1.91% |
| bidkv                    | 92.13 (+0.64%) | 153.43 (+0.46%) | 216.70 (-0.36%) | 293.45 (+0.80%) | 358.71 (-0.29%) |                   +0.25% |
| dla                      | 91.85 (+0.33%) | 152.61 (-0.07%) | 215.56 (-0.88%) | 288.15 (-1.02%) | 358.20 (-0.43%) |                   -0.42% |

Fixed controls: Qwen3.5-35B-A3B BF16, TP2/PP1, context 262144, APC, natural MTP2, async scheduling,
FULL_AND_PIECEWISE graph capture, max sequences 16, batch tokens 4096, and 26038239232 device KV
bytes per chip.

Each point in the TP2 SWE series is one real-online 900-second observation with zero failed
requests, successful retrieval and prefix-reuse gates, and verified device release.

The 25 TP2 observations share the existing Qwen3.5-35B-A3B BF16 / SWE15-minute chart at the owner's
request; checkpoint/source differences are retained per point, not promoted into a duplicate model
button. The original campaign cohort contract is retained in `archived_cohorts`, and moved points
record `evidence.original_cohort_id`. This presentation merge does not pool repeats, alter metrics,
or replace the campaign's own matched Native controls for MOD speedup attribution. The prepared
workload variant/tokenizer equivalence was already recorded in the shared SWE contract. Rotation
depths continue to own independent Pareto frontiers within the combined chart.

## BidKV under KV pressure: separate TP4 observation

On 2026-09-29, BidKV was also measured with Qwen3.5-35B-A3B on machine 91, using vLLM-HUST TP4 and
`FULL_AND_PIECEWISE` graph capture. The workload was the first 12 requests of the EvoScientist
subset in `vllm-hust-benchmark`: concurrency 12, request rate 4, maximum model length 4096, and 512
MiB device KV cache per NPU. Baseline and BidKV each completed all 12 requests and generated the
same 24,148 output tokens in each repeat. BidKV used `BIDKV_UTILITY_PREEMPT_WEIGHT=5.0`.

**Observed single-cell TP4 C12 output-throughput uplift: +13.88%**, the arithmetic mean of the two
paired percentage changes below.

| Run order      | Native output tok/s | BidKV output tok/s |  Change | Native / BidKV preemptions | BidKV active selections / failures |
| -------------- | ------------------: | -----------------: | ------: | -------------------------: | ---------------------------------: |
| Native → BidKV |              35.370 |             41.422 | +17.11% |                  850 / 786 |                             10 / 0 |
| BidKV → Native |              35.966 |             39.795 | +10.65% |                  850 / 711 |                              7 / 0 |

Unlike the TP2 SWE series above, both TP4 repeats exercised BidKV's victim selector. This is one
KV-constrained workload cell with two pairs, not a five-concurrency aggregate or a confidence
interval. The TP2 +0.25% value above remains the result for its original SWE protocol. Other
attempted settings gave mixed or negative results; they are included in the
[full report and raw evidence](evidence/qwen35-bidkv-tp4-20260929/README.md). The second BidKV
server logged an `EngineDeadError` during shutdown after its benchmark result and metrics had been
saved; its 12 requests completed, and the policy failure counter was zero.
