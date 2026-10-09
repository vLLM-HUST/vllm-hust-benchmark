# Selected-page PyramidKV: four fresh-process serving arms

This is a descriptive comparison of two **compression-enabled** PyramidKV implementations. It is not
compression-off B0 versus compression-on B1, an official leaderboard submission, or a new task
quality score. **No serving acceleration was observed:** pooled throughput for the four
compression-exercising cells is 0.24–0.45% lower with the selected-page implementation. This small
sample does not establish a general regression either. The original negative end-to-end off/on
observations remain unchanged.

The original method is `47580977a2dca108054470b3132811bd690020a6`; selected-page code is
`508a4eadb5e1b3f148975d362e13bf78e688b6da`. The serving profile is Qwen3.5-35B-A3B, BF16, TP2 on two
Ascend 910B2 devices, FULL_AND_PIECEWISE graph mode and MTP2. Both use the same 1368/beta-2
configuration, seed 17, four maximum sequences, 32K context and 8 GiB KV pool per worker.
Source/model revisions, Manager wheel identity and actual Manager configuration are recorded in the
provenance and raw archive. This experimental v0.23 source stack is separate from the public v0.18
FP16 baseline.

## Design and validation

The predeclared order is original-1, selected-1, selected-2, original-2. Every arm starts a fresh
server process and sends the identical 137-request plan: five excluded repeatability controls, 33
excluded warmup requests and 99 measured requests. The six measured input/concurrency cells are
1K/8K/24K at concurrency one/four, with three cohorts in each cell. Each output is forced to 128
tokens. Unique cache salts per case are identical across fresh arms; all observed cached-prompt
counts are zero. These synthetic prompts and forced completions are timing diagnostics.

All 548 requests succeeded. The analysis verifies frozen prompt hashes/lengths, every per-request
JSON against the aggregate, all SSE text and usage, group membership, two-worker compression
transactions and telemetry errors. Raw failures would be retained; none of these requests failed.
The exact per-arm resource and compression summaries are in `comparison.json`.

## Observed serving throughput

Pooled throughput is the sum of measured output tokens divided by the sum of cohort wall times
across the two processes of the same implementation. Pair percentages compare selected-1 with
original-1 and selected-2 with original-2. Three cohorts inside a process are not independent
process replications. There are only two fresh processes per implementation: these numbers do not
establish statistical significance, general acceleration, or a device-memory reduction.

| Prompt tokens / concurrency | Original output tokens/s | Selected output tokens/s | Pooled change | Pair 1 / pair 2 changes |
| --------------------------- | -----------------------: | -----------------------: | ------------: | ----------------------: |
| 1024-c1                     |                   77.489 |                   77.040 |        -0.58% |         -1.01% / -0.14% |
| 1024-c4                     |                  190.021 |                  191.933 |        +1.01% |         +0.88% / +1.13% |
| 8192-c1                     |                   46.550 |                   46.441 |        -0.24% |         -0.71% / +0.25% |
| 8192-c4                     |                   63.743 |                   63.454 |        -0.45% |         -0.88% / -0.02% |
| 24576-c1                    |                   29.717 |                   29.582 |        -0.45% |         -0.60% / -0.31% |
| 24576-c4                    |                   24.942 |                   24.878 |        -0.25% |         -0.69% / +0.20% |

The 1K cells are below the 4,096-token admission threshold and do not exercise compression. They are
control observations, not evidence of a PyramidKV optimization. `comparison.json` also retains
per-process and per-cohort throughput, TTFT, E2E and time per output token. HBM/KV observations
cover the complete request arm, including excluded controls/warmups; the fixed allocated pool must
not be confused with live block occupancy.

## Output repeatability and correctness boundary

- `original-1__original-2`: 35 of 137 complete outputs differ.
- `selected-1__selected-2`: 34 of 137 complete outputs differ.
- `original-1__selected-1`: 41 of 137 complete outputs differ.
- `original-2__selected-2`: 28 of 137 complete outputs differ.

All differing case IDs are retained in `comparison.json`; none are dropped to improve timing or
agreement statistics. Same-implementation differences show why cross-process text mismatches alone
cannot identify a cache-copy defect. They do not prove the numerical cause. The synthetic forced
outputs are not scored for correctness.

- `original-1`: 1 distinct output(s) across five repetitions; 93 compression transactions.
- `selected-1`: 1 distinct output(s) across five repetitions; 93 compression transactions.
- `selected-2`: 1 distinct output(s) across five repetitions; 93 compression transactions.
- `original-2`: 1 distinct output(s) across five repetitions; 93 compression transactions.

Separate
[immutable quality and live-cache evidence](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/tree/b0da8416f87925e902594dbe18cd6f9fa2e50aa1/evidence/current/2026-10-09-selected-page-serving)
contains 300 sampled quality outputs with identical F1/full text across implementations, and 20 live
requests / 40 worker checks / 880 K/V tensor comparisons with bitwise-equal selected pages. Its
earlier sequential timing pair had competing CPU work and remains diagnostic only. The isolated
copy/materialization operator's approximately 36% lower latency is a different measurement from this
serving comparison; it must not be relabeled as an end-to-end gain.

## Resource monitoring and disclosed deviation

Known competing test/lint/audit jobs were sampled throughout each request arm. None were recorded.
This process-name scanner cannot detect every possible competing process. A separately recorded
Harbor constructor check began during selected-2 startup and ended approximately 1.73 seconds into
its first **already excluded** repeatability request. It performed no model call or sandbox setup.
The first measured telemetry sample occurs 109.564 seconds after that check ended; all five controls
and 33 warmups were excluded before collection. The deviation and timestamps are preserved, not
hidden behind the scanner's zero count. No inference is made that the entire shared host was
otherwise perfectly idle.

## Reproduction and integrity

Verify `SHA256SUMS`; concatenate `raw.tar.gz.part-*` in filename order and verify the combined hash
against `archive.json` before extraction. `members.json.gz` records every member's SHA-256 and size.
The archive includes the actual salted token-ID plan, SSE/raw results, telemetry, host-load samples,
server/startup logs, commands, configuration, both method sources and analysis/publication scripts.
The final server remains running; its captured log also contains a later local Harbor chat API
diagnostic after the measured series. That request is outside the 548-request plan and statistics.
Paths in the captured scripts are from this host; rebind them explicitly on another machine while
preserving the frozen source/environment identity and input hashes. The series runner refuses an
existing state file, and the analysis refuses to overwrite its output.

Public shared adapter 0.9.0 is not assumed equivalent to the qualified source backport. The
[public-artifact audit](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/tree/42bf07f8379c148d0325fa5b00e4a8bf7c1b83b8/evidence/current/2026-10-09-public-package-audit)
records the release gap; this experiment uses the qualified source pin. Releasing and qualifying a
corrected public artifact remains separate work.
