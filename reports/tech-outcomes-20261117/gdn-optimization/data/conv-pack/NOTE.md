# gdn_conv_pack — round-2 status

Re-tested on 2026-11-15 in eager mode (the caliber the original evidence used), 32 prompts,
concurrency 8, 512 in / 128 out, one ABBA block. Result: parity.

The local L0 replication does NOT reproduce the original 308.62 us per layer: eager gives 4.48 ->
0.10 us and graph-packed gives 0.56 -> 0.66 us, so the local saving is strongly caliber-dependent.
The original figure is kept as a historical record only.

Original evidence (fixed SHA):
https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/4b4c794dcb5146d78bd419333a6bbedd0db4e5f5/ops/gdn_conv_pack/REPORT.md
and evidence.json

Not retained: the original per-round raw JSON and any op_statistic export.
