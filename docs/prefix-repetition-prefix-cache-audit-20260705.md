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

## Publication Rule

For `prefix-repetition-online`, public same-spec data must either enable prefix caching or explicitly label itself as a no-prefix-cache diagnostic. No-prefix-cache diagnostic runs must stay out of canonical public submissions unless the website surfaces them as a separate series.
