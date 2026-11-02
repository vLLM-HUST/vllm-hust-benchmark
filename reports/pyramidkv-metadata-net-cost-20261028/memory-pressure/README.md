# PyramidKV under a fixed 2 GiB KV budget

The preregistered primary cell, **24576 input tokens / concurrency 4**, changes pooled output
throughput by **+9.27%** with compression ON. Independent process-pair changes are **+12.70% /
+6.02%**. Interpret this together with preemptions, latency and device HBM below; this separate
exploratory scenario does not replace the [negative 8 GiB result](../batched/README.md).

The repeated positive primary result accompanies elimination of baseline preemptions, consistent
with recycled KV blocks avoiding recomputation under this budget. It is a bounded service-throughput
benefit, not a universal acceleration claim. The 8K/C4 result is mixed across pairs; both long-input
C1 cells regress. At equal reservation, ON still uses about 270 MiB more peak HBM per device.

## Frozen comparison

The
[protocol was published before the first ON measurement](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/blob/30dfdc59b6bfc03f96ce286846096f80e590d4c9/docs/benchmark/pyramidkv-memory-pressure.md).
Exactly one budget was selected: **2 GiB KV per worker in both arms**. There was no budget search,
reduced-context fallback or eager-mode substitution. Physical 910B2 devices 2/5, Qwen3.5-35B-A3B
BF16/TP2, 32768 context, four sequences, 4096 batched tokens, MTP2, APC, async, FULL_AND_PIECEWISE,
1368/beta-2 and selected-page materialization match the separate 8 GiB series. Host CPU limits are
eight CPUs and 64 GiB. This imposes a KV allocation budget; it does not mean the physical 64 GiB
devices are otherwise full.

Both arms load source `f33785844be5bff7716c9464dd8444ef0b4b05e9` from
[shared PR #33](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/33), with
`VLLM_ASCEND_KVCOMPRESS_BATCHED_METADATA=1`. Only compression activation differs within the series.
The batched adapter remains opt-in. These results cannot isolate its speedup against the old
adapter: that requires a separate same-budget source comparison. No local diagnostic instrumentation
runs inside the formal service processes.

Order is OFF/ON/ON/OFF with four fresh servers. Each receives the same 149 cases: 99 measured
synthetic requests, 38 predefined exclusions (33 warmups and five repeatability probes), and 12 QA
smoke cases. Synthetic outputs are forced to 128 tokens. All **596 requests succeed**. Both ON
processes contain 103 compression transactions with matching acknowledgements from both TP ranks;
there are no repeated compression lifecycles. OFF processes contain none. Private QA prompts and
references remain private; original input IDs/hashes and all generated outputs are preserved.

## Throughput and latency

| Input / concurrency | Native OFF output tok/s | PyramidKV ON output tok/s | ON vs OFF | Independent process pairs |
| ------------------- | ----------------------: | ------------------------: | --------: | ------------------------- |
| 1024-c1             |                  80.459 |                    76.663 |    -4.72% | -3.36% / -6.04%           |
| 1024-c4             |                 204.836 |                   196.251 |    -4.19% | -3.96% / -4.41%           |
| 8192-c1             |                  52.943 |                    49.882 |    -5.78% | -6.05% / -5.51%           |
| 8192-c4             |                  65.872 |                    65.990 |    +0.18% | +2.22% / -1.78%           |
| 24576-c1            |                  32.424 |                    31.052 |    -4.23% | -2.12% / -6.28%           |
| 24576-c4            |                  23.041 |                    25.177 |    +9.27% | +12.70% / +6.02%          |

The 1K cells are uncompressed controls and cannot demonstrate a compression benefit. Every cell and
all per-process/cohort values are retained in `analysis.json`, including negative observations. Only
two independent processes per arm are available; within-process cohorts are not independent process
replications. These are descriptive measurements, not a statistical-significance claim.

Primary-cell latency means over measured requests (excluding warmups):

| Process | Output tok/s | TTFT seconds | TPOT milliseconds | End-to-end seconds |
| ------- | -----------: | -----------: | ----------------: | -----------------: |
| off-1   |       22.955 |       10.893 |            82.793 |             21.408 |
| on-1    |       25.871 |        8.289 |            87.923 |             19.456 |
| on-2    |       24.519 |        8.845 |            92.144 |             20.549 |
| off-2   |       23.127 |       10.759 |            82.175 |             21.196 |

Across equally sized primary-cell request sets, mean TTFT falls 20.87% and mean end-to-end latency
falls 6.10%, while TPOT increases 9.15%. The throughput improvement is therefore not evidence of
faster per-token decoding. The preemption explanation is supported by the observed counters, without
claiming an isolated causal decomposition of all scheduler, prefill and decode costs.

All other cells' latency means and individual request timings remain in the capsule. The finite
cohort throughput includes complete request execution; this is not a sustained online arrival-rate
or tail-latency SLO test.

## Resource and recomputation evidence

| Process | Sampled HBM peak MiB, NPU 2 / 5 | Peak KV-pool usage | Preemptions | API reported cached tokens | MTP accepted / drafted |
| ------- | ------------------------------: | -----------------: | ----------: | -------------------------: | ---------------------: |
| off-1   |                   41557 / 41559 |           96.6667% |          33 |                      34816 |                92.376% |
| on-1    |                   41827 / 41829 |           87.7778% |           0 |                          0 |                90.081% |
| on-2    |                   41827 / 41829 |           87.7778% |           0 |                          0 |                90.179% |
| off-2   |                   41558 / 41559 |           96.6667% |          34 |                      40960 |                92.344% |

These resource/counter observations cover the complete process request set, including warmups and
QA; they are not the measured-throughput subset. All raw telemetry phases and before/after metrics
are preserved. KV occupancy is relative to this smaller pool, so its percentage is not directly
comparable to the 8 GiB pool percentage. Freeing request-owned blocks does not release the reserved
pool from device HBM. Lower HBM after reducing the configured reservation is a configuration change,
not equal-reservation HBM savings caused by PyramidKV. This experiment does not measure maximum
concurrency or maximum request capacity; both arms complete the tested requests.

Each of the 149 requests has a distinct API cache salt. The native baseline reports nonzero cached
tokens in long C4 cases alongside preemptions. The pinned host's cache hashing includes the salt in
the first block and chains subsequent hashes through their parent. Thus these observations are
consistent with the same request retaining cached work during resumption, rather than inter-request
prefix sharing. That attribution is an inference from source/counters, not a per-preemption trace.
API cached-token totals are not assumed to equal the total number of recomputed tokens. The first
baseline audit, all later raw counters/case IDs, and the pinned host source/license are included;
this must not be described as an entirely uncached pressure run.

## Quality, implementation and public evidence

Both ON processes match all 12 frozen stock-ON QA outputs. This checks the adapter's small
regression smoke, not quality equivalence with native OFF. The existing 300-example LongBench result
remains: development/holdout F1 dropped 0.59/0.26 points under compression, as recorded in the
[original review](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1#issuecomment-6446726027).
The new series does not replace that evaluation or issue #8's separate judge/calibration contract.

The [related batched capsule](../batched/README.md) holds the exact runtime source's 229-passed /
one-skipped local CPU suite, 72-case / 792-layer native NPU slot oracle, and live graph/MTP
diagnostic. Those code checks do not make this an official benchmark score or a public-wheel
acceptance receipt. Model config/tokenizer/template/index hashes and shard sizes/mtimes are frozen;
the 66.97 GiB weights were not fully rehashed for this series. Frozen host, package and command
identities are included.

[PyramidKV PR #26](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/pull/26) exposes the same
byte-valued CLI budget through `PYRAMIDKV_KV_CACHE_MEMORY_BYTES`, retaining the original default.
Its environment-variable entrypoint is qualified by argv-only checks; the NPU series directly sets
that CLI argument in the archived `serve-profile.sh`. Both launcher sources and the argv receipt are
preserved, so the new entrypoint is not misrepresented as a separately measured server build.

Run `python reports/pyramidkv-metadata-net-cost-20261028/memory-pressure/verify.py` from this
checkout. The verifier checks archive/member hashes, source/input identities, actual worker
reservation logs, all 596 SSE/usage records, both TP acknowledgements, throughput/latency, HBM/KV,
cached-token and preemption counters. It checks stock-ON output identity against the immutable
parent diagnostic and binds the related batched capsule. It does not extract or execute archived
code. `analysis.json` is a newline-formatted outer view; the original bytes remain in the archive.
Checksums bind each published file, and the complete compressed original is split into parts below
the Git size limit.
