# Prefix Repetition Single-Card Data Audit (2026-07-05)

## Summary

The published single-card `prefix-repetition-online` vLLM-HUST backfill points were collected with prefix caching disabled or unrecorded. That configuration is not comparable with the official repeated-prefix workload intent: the benchmark sends many requests sharing long prefixes, so disabling prefix caching inflates TTFT and underreports throughput.

The affected canonical submissions were moved to `archive/suspect/prefix-repetition-single-no-prefix-cache-20260705/` and should not be used for public trend lines.

## Evidence

A controlled NPU0 rerun of `main#2206f1f7b7` + `vllm-ascend-hust#bf2984e34a` changed only the prefix-cache setting while keeping graph mode, `max_model_len=32768`, `max_num_seqs=16`, `gpu_memory_utilization=0.90`, no chunked prefill, and the official `prefix_repetition` client spec (`prefix_len=3840`, `suffix_len=256`, `output_len=256`, `request_rate=1`, `num_prompts=200`).

| Target | Old throughput | Old mean TTFT | Corrected throughput | Corrected mean TTFT |
| --- | ---: | ---: | ---: | ---: |
| main-current `83cf83ff20` + `7803bc8d0f` | 153.70 tok/s | 7,228.62 ms | 233.46 tok/s | 546.07 ms |
| main `7a63f81e86` + `d40e28c348` | 149.71 tok/s | 12,328.70 ms | 234.16 tok/s | 548.68 ms |
| main `2206f1f7b7` + `bf2984e34a` | 160.56 tok/s | 10,747.44 ms | 236.12 tok/s | 625.00 ms |

All corrected reruns completed 200/200 requests on NPU0 with `error_rate=0.0`.


## Historical PR Corrections

The same configuration issue affected older historical PR backfills. These rows were rerun on NPU0 with prefix caching enabled and the same official repeated-prefix client spec. The old no-prefix-cache rows remain archived under `archive/suspect/prefix-repetition-single-no-prefix-cache-20260705/`; the corrected rows use `data_quality=corrected-prefix-cache-enabled` and record the superseded archived submission.

| Historical target | Old throughput | Old mean TTFT | Corrected throughput | Corrected mean TTFT |
| --- | ---: | ---: | ---: | ---: |
| align-ascend-benchmark-perfgate-defaults `2fb7859dd0` + `51e577b17b` | 110.07 tok/s | 45,937.65 ms | 232.76 tok/s | 538.55 ms |
| align-perfgate-baseline-spec-source `dcc06b18f3` + `51e577b17b` | 115.99 tok/s | 58,443.69 ms | 235.31 tok/s | 545.72 ms |
| ascend-pr53-model-runner-platform `7a63f81e86` + `bf2984e34a` | 156.91 tok/s | 11,476.17 ms | 235.51 tok/s | 527.46 ms |
| ascend-pr66-simllm-kv-manager `ceec19abb0` + `e0686f12d1` | 119.98 tok/s | 51,814.23 ms | 234.46 tok/s | 531.80 ms |
| ascend-pr70-simllm-baseline-validation `ceec19abb0` + `312ca80a90` | 117.44 tok/s | 54,438.59 ms | 234.33 tok/s | 529.71 ms |
| pr-77-ceec19abb0 `ceec19abb0` + `51e577b17b` | 122.03 tok/s | 55,135.82 ms | 237.43 tok/s | 459.17 ms |
| pr77-perfgate-l2-scenario-registry `ceec19abb0` + `51e577b17b` | 123.27 tok/s | 49,407.68 ms | 237.43 tok/s | 459.17 ms |
| pr41-v1-attention-boundary `51621c35bc` + `d40e28c348` | 145.60 tok/s | 5,612.74 ms | 233.43 tok/s | 582.17 ms |
| pr49-kv-offload-worker `f273f9c5e2` + `d40e28c348` | 146.09 tok/s | 10,710.22 ms | 233.88 tok/s | 506.43 ms |
| pr69-perfgate-l2-targeted-verification `ec4847981f` + `51e577b17b` | 124.18 tok/s | 50,618.85 ms | 235.25 tok/s | 562.26 ms |

`pr-77-ceec19abb0` and `pr77-perfgate-l2-scenario-registry` share the same `ceec19abb0` core commit and `51e577b17b` plugin commit. One unique rerun covers both historical labels; both corrected submissions point at the same measured result with separate run IDs so the website does not leave either historical x-axis slot empty.

## Publication Rule

For `prefix-repetition-online`, public same-spec data must either enable prefix caching or explicitly label itself as a no-prefix-cache diagnostic. No-prefix-cache diagnostic runs must stay out of canonical public submissions unless the website surfaces them as a separate series.
