# Single-card high-TTFT outlier audit (2026-07-05)

## Summary

A left-to-right audit of the single-card online workload lines found that several historical
`sharegpt-online` and `instructcoder-online` points were not stable performance measurements. Their
same-spec parameters matched neighboring rows, but their TTFT was tens to hundreds of seconds while
reruns on the same official specs restored normal sub-second TTFT.

All reruns were executed on NPU0 only, using the isolated backfill repositories and container. No
global proxy was enabled and no remote downloads were routed through the local VPN.

## Corrections imported

| Workload             | Historical point                              | Old throughput |     Old TTFT | Corrected throughput | Corrected TTFT | Result                      |
| -------------------- | --------------------------------------------- | -------------: | -----------: | -------------------: | -------------: | --------------------------- |
| sharegpt-online      | pr77 / ceec19abb0 + 51e577b17b                |         142.99 |  22094.54 ms |               191.57 |      199.96 ms | imported                    |
| sharegpt-online      | pr69 / ec4847981f + 51e577b17b                |         137.55 |  28465.52 ms |               191.44 |      198.77 ms | imported                    |
| sharegpt-online      | align defaults / 2fb7859dd0 + 51e577b17b      |         156.29 |  15664.18 ms |               191.75 |      198.51 ms | imported                    |
| sharegpt-online      | align baseline spec / dcc06b18f3 + 51e577b17b |         154.10 |  15490.62 ms |               193.70 |      197.80 ms | imported                    |
| sharegpt-online      | ascend pr70 / ceec19abb0 + 312ca80a90         |         151.72 |  18389.53 ms |               191.25 |      201.37 ms | imported                    |
| sharegpt-online      | ascend pr66 / ceec19abb0 + e0686f12d1         |         157.73 |  14902.95 ms |               188.14 |      197.00 ms | imported                    |
| instructcoder-online | pr77 / ceec19abb0 + 51e577b17b                |         180.25 |  45775.09 ms |               169.13 |      260.91 ms | imported                    |
| instructcoder-online | ascend pr70 / ceec19abb0 + 312ca80a90         |         172.68 | 114943.90 ms |               167.36 |      263.32 ms | imported                    |
| instructcoder-online | align defaults / 2fb7859dd0 + 51e577b17b      |         158.98 |  32166.16 ms |               167.42 |      287.62 ms | imported                    |
| instructcoder-online | ascend pr66 / ceec19abb0 + e0686f12d1         |         164.64 | 159812.43 ms |               166.37 |      299.42 ms | imported after second retry |

## Failed retry excluded

The first NPU0 retry for `instructcoder-online` at `ascend pr66 / ceec19abb0 + e0686f12d1` failed
242/2048 requests (`error_rate=0.1181640625`) and was not imported. The second retry completed with
zero failures and is the row used in `submissions/`.

## Data-quality rule

Rows with high TTFT outliers are not kept in `submissions/` when a same-spec rerun proves them
unstable. Failed reruns are also excluded from `submissions/`; they remain only as audit evidence
under the backfill artifact directory and archive notes.

## Final line audit

After restoring corrected reruns to their original public `github_ref` labels, the default
single-card online matrix has 14 x-axis versions. The six main online workload lines each have a
point at every x-axis version:

| Workload                 | Points |
| ------------------------ | -----: |
| random-online            |  14/14 |
| prefix-repetition-online |  14/14 |
| agent-research-online    |  14/14 |
| sharegpt-online          |  14/14 |
| visionarena-online       |  14/14 |
| instructcoder-online     |  14/14 |

Residual audit flags are not missing points: the prefix-repetition official baseline has a high TTFT
and 0.5% error rate, the visionarena baseline and one historical PR row have small nonzero error
rates, and `agent-research-online` at `pr49-kv-offload-worker` is a positive throughput jump rather
than a dropped point.
