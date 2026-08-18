# Issue 214 P1 historical hardening

This internal report records the completed three-checkpoint historical matrix after applying the
same-spec and effective-behavior checks defined for issue 214.

- all 27 workload/checkpoint cells retain three independent successful invocations;
- every selected representative is the median of the predeclared primary metric, never the best
  invocation;
- all 27 cells are grade A in the generated stable-trend evidence audit;
- online cells use TTFT as the primary stability metric, except the two throughput workloads, which
  use total token throughput; random latency uses latency;
- throughput decreases are accepted only within the 1% noise band, and latency increases only within
  the 5% noise band;
- the one failed C3 random-online invocation remains under the submission's `diagnostics/` tree and
  is not counted as a successful repeat.

The nine primary-metric series, listed as C1 / C2 / C3, are:

| Workload | Metric | C1 | C2 | C3 | C1 to C2 | C2 to C3 | | --- | --- | ---: | ---: | ---: | ---: |
---: | | agent-research-online | TTFT ms | 3147.526 | 3109.841 | 3031.676 | -1.197% | -2.513% | |
instructcoder-online | TTFT ms | 114.893 | 114.994 | 114.144 | +0.088% | -0.739% | |
prefix-repetition-online | TTFT ms | 279.129 | 259.331 | 261.991 | -7.093% | +1.026% | |
random-latency | latency ms | 4909.350 | 4907.898 | 4946.433 | -0.030% | +0.785% | | random-online |
TTFT ms | 234.514 | 230.557 | 231.664 | -1.687% | +0.480% | | sharegpt-online | TTFT ms | 128.695 |
131.942 | 128.029 | +2.523% | -2.966% | | sharegpt-throughput | total TPS | 2054.498 | 2041.982 |
2042.348 | -0.609% | +0.018% | | sonnet-throughput | total TPS | 3654.776 | 3621.894 | 3634.784 |
-0.900% | +0.356% | | visionarena-online | TTFT ms | 431.601 | 420.048 | 430.758 | -2.677% | +2.550%
|

All transitions remain inside their applicable noise bands. `summary.json` lists the 21 cells
hardened in this wave, their selected repeat, metric, and submission checksum. VisionArena and
Sonnet already had strict three-invocation suites and are included in the regenerated 27-cell audit.
