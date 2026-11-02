# Batched PyramidKV metadata: independent 8 GiB service series

**The batched metadata candidate still does not provide a long-input service speedup or lower device
HBM with the original reservation.** All four 8K/24K cells regress in both independent process
pairs. Do not promote the large local host-call reduction into a service-acceleration claim.

This follows the [correction-cache-only negative result](../README.md), but uses four new process
lifecycles; no timing samples are shared between series.
[Shared PR #33](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/33) contains the
implementation. Executed source is `f33785844be5bff7716c9464dd8444ef0b4b05e9`, with
`VLLM_ASCEND_KVCOMPRESS_BATCHED_METADATA=1` in both arms. Only compression activation differs. The
flag remains opt-in; these are experimental BF16/TP2 results, not official scores, public-wheel
qualification or shared Agent work.

## Uninstrumented throughput

Order: OFF/ON/ON/OFF, each with a fresh server. Each process executes the same 149 requests (99
measured synthetic, 38 predefined exclusions, 12 QA smoke). All **596 requests succeed**, every
formal prompt reports zero cached tokens, and no preemptions occur. Each ON process has **103
transactions** with both TP acknowledgements; OFF has none. Raw SSE, result metadata, source/input
hashes, transaction records and resource samples pass the independent offline verifier.

| Input / concurrency | Native OFF output tok/s | Batched ON output tok/s | ON vs OFF | Individual process pairs |
| ------------------- | ----------------------: | ----------------------: | --------: | ------------------------ |
| 1024-c1             |                  80.886 |                  77.744 |    -3.88% | -4.08% / -3.68%          |
| 1024-c4             |                 190.986 |                 195.162 |    +2.19% | +1.39% / +3.03%          |
| 8192-c1             |                  52.925 |                  50.626 |    -4.34% | -3.92% / -4.76%          |
| 8192-c4             |                  66.167 |                  64.870 |    -1.96% | -2.15% / -1.77%          |
| 24576-c1            |                  32.303 |                  30.963 |    -4.15% | -5.60% / -2.67%          |
| 24576-c4            |                  25.792 |                  24.851 |    -3.65% | -6.15% / -1.06%          |

The positive 1K/C4 cell is an uncompressed control: these prompts never reach the compression
threshold. It does not demonstrate a PyramidKV compression benefit. Only two independent processes
per arm are available; cohorts within one process are not independent process replications. These
are descriptive results, not statistical significance or a general online-performance guarantee.

## Resource observations

| Process | Sampled HBM peak MiB, physical NPU 2 / 5 | Peak KV-pool usage | Preemptions | MTP accepted / drafted |
| ------- | ---------------------------------------: | -----------------: | ----------: | ---------------------: |
| off-1   |                            47678 / 47679 |           29.1209% |           0 |                92.524% |
| on-1    |                            47948 / 47949 |           21.7033% |           0 |                89.776% |
| on-2    |                            47947 / 47949 |           21.7033% |           0 |                90.076% |
| off-2   |                            47677 / 47683 |           29.1209% |           0 |                92.119% |

Lower logical KV ownership again does not become lower device HBM: the fixed 8 GiB reservation per
worker remains allocated. The ON peaks are roughly 270 MiB higher. Successful C4 requests do not
establish a higher maximum concurrency. The tighter-budget scenario is separately preregistered and
must not overwrite this default-budget result.

## Local diagnosis and correctness

The new path batches expanded-table slot mapping and sequence-length corrections, while retaining
persistent graph slot addresses. It follows the actual metadata device and allocates independent
length storage per call. Selection, retained KV tokens, method budget and selected-page
materialization are unchanged.

In the separate instrumented diagnostic, active metadata median host-call times are **1.309 / 1.249
ms** versus stock **6.855 / 6.896 ms** on the two workers, about **81% lower**. Inactive-call
medians are 0.903 / 0.863 ms. These durations include host dispatch and incidental synchronization,
with no inserted device synchronization; they are not isolated kernel time. The diagnostic marker
confirms that batched corrections were created; its cached-state marker must not be interpreted as
an independent trace of every later fast-path decision.

All 20 diagnostic outputs match stock ON exactly. Both formal ON processes also match all 12
stock-ON QA smoke outputs. These original development examples are not a replacement for the
existing 300-example quality record. The diagnostic inherits the original missing API cache-salt
limitation (two 8K probes reuse 4096 tokens); it is not the uncached service comparison above.

The qualified local CPU suite passes **229 tests, one skipped**. A separate synthetic NPU integer
oracle checks **72 combinations / 792 layer comparisons** against the pinned host's actual native
block-table kernel, including CPU or NPU sequence metadata, mixed active/plain requests, reordered
rows, unequal query counts, multiple physical pages, padding, stable addresses and MTP corrections.
The metadata dataclasses in that oracle are fixtures; live FULL_AND_PIECEWISE/ MTP2 requests provide
the separate serving check.

## Preserved unsuccessful attempt and limits

The first batched candidate (`4aac2a8`) allocated length storage on NPU while the qualified host
supplies CPU lengths. Its guards safely selected the native path: 20 requests succeeded, but the
optimization never activated. No four-process service series was started for that candidate. The
complete fallback diagnostic, source, tests, original NPU-only-length oracle and unused-plan
explanation are preserved under `fallback-v1/`. The corrected candidate adds the CPU-length case and
keeps prior length views immutable. No failed activation is counted as a win.

This run uses the original Qwen3.5-35B-A3B BF16/TP2 host: physical 910B2 devices 2/5, 64 GiB CPU
cgroup, eight-CPU quota, 32768 context, four sequences, 4096 batched tokens, MTP2, APC, async,
FULL_AND_PIECEWISE, and 1368/beta-2 compression. Frozen package/host/model identity and exact
commands are inside the capsule. The configuration/tokenizer/template/index hashes and shard
size/mtime inventory are preserved; no new full-content hash of the 66.97 GiB weights was made.

The archive contains 1479 public members in three checksum-bound parts. Private QA prompts and
reference answers are omitted; original IDs/hashes and all generated outputs remain. Synthetic
token-ID inputs are complete. The stock diagnostic is bound to the immutable parent capsule at
benchmark commit `8d4d9a23020570613a13ed928eaaf2b14deedd53`. The original 8 GiB correction-cache
series remains intact. No explicit-prefix recovery was needed for this series; the parser still
preserves that narrowly defined handling for possible log interleaving.

Run `python reports/pyramidkv-metadata-net-cost-20261028/batched/verify.py`. It checks
archive/member hashes, frozen source/input identities, all 596 SSE and usage records, both TP
acknowledgements, pooled throughput, HBM/KV samples, preemption/MTP counters, and the
oracle/diagnostic receipts without extracting or executing archived code. `analysis.json` is the
outer, newline-formatted analysis view; the byte-exact original is also retained inside the capsule.
