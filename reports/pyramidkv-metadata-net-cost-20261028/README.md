# PyramidKV metadata net-cost review — 2026-10-28

**The correction-cache candidate still fails to produce a service speedup or HBM saving.** All six
matched cells regress. This report supports the teacher's
[issue #1 review](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1#issuecomment-6446726027),
rather than treating lower logical KV ownership as a performance win.

Implementation: [shared PR #33](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/33).
These are new experimental Qwen3.5 BF16/TP2 results, not official v0.18 scores, public-wheel
acceptance, or the shared #403 Agent campaign. The later batched candidate is a separate experiment
and cannot replace these originals.

## Uninstrumented service result

Four fresh processes in OFF/ON/ON/OFF order use identical candidate source and serving parameters;
only compression activation differs. Each process executes 149 requests: 99 measured synthetic
requests, 38 preregistered exclusions and 12 QA smoke examples. All 596 requests succeed; every
prompt reports zero cached tokens. There are 103 scheduler transactions in each ON process, each
confirmed by both TP workers, and none in OFF. No preemptions were observed.

The primary metric pools output tokens over cohort wall time. Each cell has three cohorts per
process, but only two independent processes per arm. Percentages are descriptive; they are not
confidence bounds or a general speedup claim.

| Input / concurrency | Native OFF output tok/s | Candidate ON output tok/s | ON vs OFF | Individual process pairs |
| ------------------- | ----------------------: | ------------------------: | --------: | ------------------------ |
| 1024-c1             |                  80.232 |                    76.747 |    -4.34% | -3.33% / -5.36%          |
| 1024-c4             |                 200.147 |                   187.500 |    -6.32% | -4.93% / -7.70%          |
| 8192-c1             |                  53.082 |                    46.825 |   -11.79% | -11.37% / -12.22%        |
| 8192-c4             |                  65.964 |                    62.306 |    -5.55% | -3.74% / -7.36%          |
| 24576-c1            |                  32.303 |                    29.400 |    -8.99% | -7.55% / -10.43%         |
| 24576-c4            |                  25.484 |                    24.449 |    -4.06% | -1.98% / -6.16%          |

The sampled peak KV-pool usage falls from **29.1209% to 21.7033%**. Sampled device HBM peaks instead
increase from **47,678 / 47,679 MiB** to **47,948 / 47,949 MiB** on physical NPUs 2 / 5 (**+270 MiB
per device**), consistently in both processes. The fixed 8 GiB reservation per worker is unchanged.
Returning request-owned blocks does not release that preallocated pool to the device allocator. C4
success is not a measured maximum-capacity improvement.

## Diagnosis and correctness boundary

A separate instrumented stock-ON/candidate-ON diagnostic times host calls without inserting device
synchronization. Active per-layer metadata median call durations are 6.855 / 6.896 ms on stock and
5.120 / 5.075 ms with cached corrections, about 25% lower. These include host dispatch and
incidental synchronization; they are not isolated kernel durations. The diagnostic confirms a local
improvement, while the uninstrumented table above still fails the service-benefit test.

All 20 diagnostic outputs match exactly. The 12 QA smoke outputs in both formal ON processes also
match stock ON exactly. These examples are the first four original development examples in each of
three LongBench tasks; this is not a new 300-example quality acceptance. The prior
150-development/150-holdout quality record remains separate and is not replaced by favorable smoke
scores.

The diagnostic client originally omitted API `cache_salt`: two measured 8K probes reuse 4096 cached
tokens after warmup. Both diagnostic implementations retain that same behavior. Their latency must
not be presented as the uncached service result. The formal client sends a unique API salt for each
case, reuses exact input IDs/salts across fresh processes, and checks zero cached prompt tokens.

## Provenance, failures and reproduction

- Hardware: two locally allocated 910B2 devices, physical 2 / 5, logical 0 / 1; 64 GiB CPU cgroup
  and eight-CPU quota. NPU identity is recorded in actual telemetry.
- Model: local Qwen3.5-35B-A3B, BF16, TP2, FULL_AND_PIECEWISE, MTP2, APC and async scheduling;
  context 32768, four sequences, 4096 batched tokens, 8 GiB KV per worker. Model
  configuration/tokenizer/template/index hashes and shard size/mtime inventory are preserved; no new
  full-content hash of 66.97 GiB weights was made.
- Host pins, packages and exact commands are inside `environment.json`,
  `runtime-package-versions.json`, `serve.sh` and `paired-plan/manifest.json`. Candidate source is
  commit `b593d84` on shared base `3ddff1c2afee1eccc3d1b65def61167752298f5f`; the complete executed
  source is archived. Pyramid selection remains 1368/beta-2 with the original threshold and
  selected-page materialization. No new token-selection tuning was performed.
- All raw requests/SSE, server logs, metrics, per-cohort results, source snapshots, plans, resource
  samples and analysis scripts are in the capsule. QA prompts and reference answers are omitted;
  their input IDs/hashes and generated outputs remain. All synthetic token-ID inputs are present.
  See the private-input boundary record for the fixed LongBench dataset/source revisions.
- The original environment test invocation failed six tests (old Manager plus missing custody
  originals); the corrected qualified environment passes **227 tests, one skipped**. The initial
  launch-directory collision happened before any server process. Preserve
  `preparation-failures.json`; do not turn these into successful runs.
- The initial analysis parser rejects one ON-2 warmup acknowledgement because an API access-log line
  splits its explicit TP0 prefix from the message. The raw log at lines 451–452 and initial
  `analysis-check.log` are retained. Analysis v2 recovers only this adjacent explicit-prefix
  pattern, records the original text, and still requires one matching acknowledgement from each
  rank. It does not infer a missing rank from counts or change any raw bytes. No failed request was
  discarded.
- Original diagnostic contracts were annotated after the initial stock diagnostic with candidate
  identity and the discovered cache omission. The formal manifest was frozen before all four formal
  process lifecycles; distinguish these histories.

Run `python reports/pyramidkv-metadata-net-cost-20261028/verify.py` from a checkout. The offline
verifier reads the three archive parts in order, checks whole-archive and every member's
SHA-256/size, checks source hashes and public input metadata, joins SSE/results/usage and both TP
acknowledgements, and recomputes throughput. It neither extracts nor executes archived code.
`chunks.json` describes the exact immutable archive; `members.json` indexes 1442 public members.
Splitting the archive keeps every Git file below the repository's 1000 KiB limit without changing
its bytes.

`analysis.json` is the versioned analysis view; the byte-exact copy used by the run is also inside
the capsule. Trailing-newline formatting of this outer view does not alter the archived original.
`profile-analysis.json` contains diagnostic call summaries and output comparisons. These results
justify continued metadata work, not advertising PyramidKV as faster or lower-HBM on this
model/configuration.

## Separate batched follow-up

The [batched-metadata series](batched/README.md) preserves four new process lifecycles, the
unsuccessful first activation and the corrected native-kernel/device tests. It does not overwrite
the correction-cache-only observations above.

## Separate fixed-budget follow-up

The [preregistered 2 GiB pressure comparison](memory-pressure/README.md) uses four new process
lifecycles and the batched candidate. It reports the complete six-cell matrix, conditional
primary-cell service benefit, remaining regressions, preemption/cache evidence and equal-reservation
HBM cost. It does not replace either negative 8 GiB series.
