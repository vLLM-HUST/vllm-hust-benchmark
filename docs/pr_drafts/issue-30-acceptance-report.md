# Issue #30 Acceptance Report

## Run Count Check

- Current runs: 3 (required >= 3)
- Baseline runs: 3 (required >= 3)

- Run-count gate: PASS

## Same-spec Check

- Current spec gate: PASS (29e7cca1f96eb32bb5ebd186931ca002aebdac9de46f8137b46e0ca39476d259)
- Baseline spec gate: PASS (29e7cca1f96eb32bb5ebd186931ca002aebdac9de46f8137b46e0ca39476d259)
- Cross-group same-spec gate: PASS

## Per-run Metrics

| group | run_id | spec_hash | throughput_req_s | throughput_tok_s | mean_ttft_ms | mean_tpot_ms | mean_itl_ms | slo_attainment | total_preemptions | consecutive_preempt_ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | compare-unoptimized-temp0-conc2-100-goodput-20260525T113352Z/current | 29e7cca1f96eb32bb5ebd186931ca002aebdac9de46f8137b46e0ca39476d259 | 0.0752 | 19.2591 | 303.8355 | 102.8986 | 102.8986 | 0.4300 | 0 | 0.0000 |
| current | compare-unoptimized-temp0-conc2-100-goodput-20260525T122739Z/current | 29e7cca1f96eb32bb5ebd186931ca002aebdac9de46f8137b46e0ca39476d259 | 0.0757 | 19.3908 | 301.2428 | 102.2015 | 102.2015 | 0.6600 | 0 | 0.0000 |
| current | compare-unoptimized-temp0-conc2-100-goodput-20260525T132133Z/current | 29e7cca1f96eb32bb5ebd186931ca002aebdac9de46f8137b46e0ca39476d259 | 0.0755 | 19.3272 | 302.3828 | 102.5412 | 102.5412 | 0.5100 | 0 | 0.0000 |
| baseline | compare-unoptimized-temp0-conc2-100-goodput-20260525T113352Z/baseline | 29e7cca1f96eb32bb5ebd186931ca002aebdac9de46f8137b46e0ca39476d259 | 0.0592 | 15.1632 | 387.3399 | 130.7341 | 130.7341 | 0.0000 | 0 | 0.0000 |
| baseline | compare-unoptimized-temp0-conc2-100-goodput-20260525T122739Z/baseline | 29e7cca1f96eb32bb5ebd186931ca002aebdac9de46f8137b46e0ca39476d259 | 0.0589 | 15.0723 | 391.4880 | 131.5108 | 131.5108 | 0.0000 | 0 | 0.0000 |
| baseline | compare-unoptimized-temp0-conc2-100-goodput-20260525T132133Z/baseline | 29e7cca1f96eb32bb5ebd186931ca002aebdac9de46f8137b46e0ca39476d259 | 0.0591 | 15.1217 | 389.4846 | 131.0891 | 131.0891 | 0.0000 | 0 | 0.0000 |

## Aggregated Statistics

| metric | current_mean | current_std | current_var | baseline_mean | baseline_std | baseline_var |
|---|---:|---:|---:|---:|---:|---:|
| request_throughput | 0.0755 | 0.0003 | 0.0000 | 0.0591 | 0.0002 | 0.0000 |
| output_throughput | 19.3257 | 0.0659 | 0.0043 | 15.1190 | 0.0455 | 0.0021 |
| mean_ttft_ms | 302.4870 | 1.2995 | 1.6887 | 389.4375 | 2.0745 | 4.3034 |
| mean_tpot_ms | 102.5471 | 0.3486 | 0.1215 | 131.1113 | 0.3888 | 0.1512 |
| mean_itl_ms | 102.5471 | 0.3486 | 0.1215 | 131.1113 | 0.3888 | 0.1512 |
| slo_attainment | 0.5333 | 0.1168 | 0.0136 | 0.0000 | 0.0000 | 0.0000 |
| total_preemptions | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| consecutive_preempt_ratio | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Required Metric Coverage

| metric | current_coverage | baseline_coverage |
|---|---:|---:|
| request_throughput | 3/3 | 3/3 |
| output_throughput | 3/3 | 3/3 |
| mean_ttft_ms | 3/3 | 3/3 |
| mean_tpot_ms | 3/3 | 3/3 |
| mean_itl_ms | 3/3 | 3/3 |
| slo_attainment | 3/3 | 3/3 |
| total_preemptions | 3/3 | 3/3 |
| consecutive_preempt_ratio | 3/3 | 3/3 |

## Acceptance Summary

- Run-count gate: PASS
- Same-spec gate: PASS
- Metric-coverage gate: PASS
