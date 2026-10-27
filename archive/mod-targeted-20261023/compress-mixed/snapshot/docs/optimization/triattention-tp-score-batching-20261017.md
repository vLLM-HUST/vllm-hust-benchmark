# TriAttention TP score batching and SWE configuration search

English | [简体中文](triattention-tp-score-batching-20261017.zh.md) |
[Optimization index](README.md)

Date: 2026-10-17. Scope: the 0.10.0 source implementation on the pinned
Qwen3.5-35B-A3B BF16, TP=2 Ascend 910B2 host. This is an engineering record,
not a claim about every model, host, or request mix.

## Problem and implementation

The [0.9.0 SWE pair](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261008-v090-swe-pair.json)
raised per-request decode P90 by 1.05% but reduced output tokens/s/chip by
2.71%. In the TP=2 scoring path, each sampled attention layer previously
performed its own HCCL MAX reduction. With stride 2 on Qwen3.5, five sampled
layers meant five collectives per compression selection.

The change allocates a persistent contiguous score workspace, normalizes
each sampled layer into its own workspace row, performs one HCCL MAX over the
whole workspace, then reduces the layer axis. MAX acts independently on each
layer/token entry, so batching preserves the per-layer cross-rank result.
The Ascend normalization kernel uses the same minimum variance as the Torch
fallback. TriAttention's calibrated scoring, V3 protected windows, segment
quotas, and KV materialization for all ten full-attention layers remain in
place. See [method.py](../../src/vllm_ascend_kvcompress/methods/triattention/method.py)
and [kernels.py](../../src/vllm_ascend_kvcompress/methods/triattention/kernels.py).

This code change reduces synchronization work during a compression event.
The service result also depends on **when** that event occurs. The candidate
search therefore kept the same optimized source and changed the physical KV
budget and requested-output guard. The guard checks the request's configured
`max_tokens`; a 2,048-token guard bypasses requests with smaller limits.

## Frozen test and correctness checks

All four arms used the same optimized method and kernel source, model revision
`712cf74392b05026a6db2bf213d343747d1f6d45`, prepared SWE workload hash
`8044561ffa1bb430bea8f778ef814d96649321e1a92654b95f64263b996d5e85`,
host source, two 910B2 NPUs, C4, and separate 900-second windows. The
configured context was 262,144 tokens; the longest observed prompt was
52,505. B0 loaded the plugin with compression unreachable
(`kv_budget=262656`, `recompute_window=1024`). B1 variants used stride 2
and a 1,024-token recompute window. APC, real MTP2, async scheduling,
`mamba_cache_mode=align`, and FULL_AND_PIECEWISE were retained.

The matched host suite passed 210 tests. A TP=2 numerical test checked
cross-rank agreement for batched scoring. Each service arm passed its separate
60-second check and completed the 900-second window with zero failed
requests. The selected candidate passed four deterministic 32K retrieval
checks with compression. Its service log recorded 20 lifetime scheduler
commits and 40 TP-rank acknowledgements. These checks do not establish SWE
semantic answer quality.

## Service measurements

| Arm | KV budget | Requested-output guard | Output tokens/s/chip | Per-request decode P90 tokens/s | TTFT P95 ms | In-window completions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0, compression off | 262656 | 512 | 74.426 | 47.843 | 1561 | 218 |
| B1, first setting | 8064 | 512 | 73.154 (−1.71%) | 48.560 (+1.50%) | 1828 (+17.15%) | 214 |
| B1, smaller budget | 4608 | 1024 | 74.098 (−0.44%) | 48.426 (+1.22%) | 1827 (+17.06%) | 217 |
| B1, selected | 4608 | 2048 | **75.418 (+1.33%)** | **48.616 (+1.62%)** | 1696 (+8.65%) | 218 |

Percentages compare each candidate with B0. The selected budget is four
1,152-token attention pages. The first two candidate settings missed the
two-metric objective despite their decode gains. The selected setting improved
both target metrics in one matched pair, while TTFT P95 still regressed.
The report is an observation from one workload and one pair, without a
repeatability or significance estimate. The high requested-output guard
changes which requests compress, so its benefit cannot be attributed solely
to the HCCL change.

The [aggregate evidence](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261017-v010-swe-pair.json)
contains source and host identities, exact values, run IDs, and SHA-256 hashes
of the original reports. The
[benchmark contract](../benchmark/frontier-benchmarking.md) defines the
metric calculations and qualification checks.
