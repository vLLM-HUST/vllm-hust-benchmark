# Qwen3.5-35B-A3B EAGLE3 Dynamic Gamma C1-C16

## Test setup

- Target: `/workspace/models/Qwen3.5-35B-A3B`
- EAGLE3: `/workspace/models/Qwen3.5-35B-A3B-Eagle3-Specforge`
- Dataset: GSM8K materialized benchmark set
- Requests: 64
- Output length: 256
- Tensor parallelism: TP2
- Execution: graph (`FULL_AND_PIECEWISE`), asynchronous scheduling
- Dynamic gamma candidates: 1-6
- Refill: cohort-boundary admission, with no fixed refill size or hold count
- Baseline: the fixed target-only C1-C16 results from `20260929-c1-c16`; it was not rerun

## Results

| Concurrency | Target-only (tok/s) | EAGLE3 dynamic (tok/s) | Speedup | Mean TPOT (ms) | Acceptance length |
|---:|---:|---:|---:|---:|---:|
| 1 | 58.45 | 84.10 | 1.439x | 9.99 | 3.56 |
| 2 | 99.17 | 133.75 | 1.349x | 10.99 | 3.85 |
| 3 | 133.77 | 198.71 | 1.485x | 11.02 | 3.78 |
| 4 | 185.47 | 269.16 | 1.451x | 11.18 | 3.71 |
| 5 | 208.23 | 298.22 | 1.432x | 11.87 | 3.59 |
| 6 | 244.50 | 344.80 | 1.410x | 12.38 | 3.53 |
| 7 | 268.51 | 375.27 | 1.398x | 12.58 | 3.79 |
| 8 | 329.73 | 441.92 | 1.340x | 12.82 | 3.73 |
| 9 | 319.05 | 478.87 | 1.501x | 12.94 | 3.74 |
| 10 | 351.74 | 474.71 | 1.350x | 13.68 | 3.74 |
| 11 | 409.08 | 531.47 | 1.299x | 14.03 | 3.66 |
| 12 | 416.15 | 524.17 | 1.260x | 14.40 | 3.64 |
| 13 | 472.43 | 593.05 | 1.255x | 14.85 | 3.68 |
| 14 | 431.25 | 617.36 | 1.432x | 14.98 | 3.72 |
| 15 | 473.71 | 587.14 | 1.239x | 15.68 | 3.59 |
| 16 | 554.15 | 717.98 | 1.296x | 15.34 | 3.73 |

All tested concurrency points exceed the 1.2x throughput requirement. The
minimum is 1.239x at C15; the maximum is 1.501x at C9.

## Controller behavior

The refill controller no longer waits for a fixed running-request threshold.
It preserves each active decode cohort and admits queued requests when that
cohort drains. This avoids mixing prefill work into a recurrent EAGLE3 decode
cohort and avoids repeated graph-width and GDN-state transitions. The initial
cohort is released by an arrival quiet-period/max-wait rule rather than a fixed
batch size.

Gamma remains independently adaptive over 1-6. Across this C16-to-C1 run, the
controller selected gamma 3/4/5/6 and performed 51 gamma switches. The final
batch-bucket choices were gamma 4 for B1/B2/B4, gamma 3 for B8, and gamma 5 for
B16. Candidate gamma 1-6 were all launched during online exploration.

The tradeoff is increased queueing/TTFT under burst load because a new request
does not enter a partially drained cohort. The acceptance criterion in this
test is output throughput relative to target-only, not TTFT or SLO goodput.

## Artifacts

- EAGLE3 results: `benchmark_results/qwen35_eagle3_gsm8k/20260930-adaptive-cohort-gated/`
- Target-only baseline: `benchmark_results/qwen35_frontier_mtp2_gsm8k/20260929-c1-c16/baseline/`
- Server log: `benchmark_results/qwen35_eagle3_gsm8k/20260930-adaptive-cohort-gated-server.log`
- Runtime config: `configs/qwen35-35b-a3b-eagle3-adaptive.toml`

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
