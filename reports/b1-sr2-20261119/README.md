# B1-sr2 PyramidKV experimental freeze

B1-sr2 changes real Qwen3.5 scheduler page ownership, releases complete allocation blocks and projects target and draft Wave page tables independently. Both ranks exercised page reuse and cross-page MTP rollback on Ascend 910B2.

The v3 C8 repeat on physical cards 0 and 1 recovered 16K throughput from the v2 regression. Compared with the sealed same-card B1-r8, it measured 159.69 → 160.80 token/s, 14.19 → 14.44 s P95 and 15.38 → 16.44 s P99 end-to-end. Each arm ran once. v4 stopped before completing the formal 16K window when the host cgroup reserve crossed the 4 GiB guard.

The v2 five-profile round includes matched-card MMLU-Pro 200, from 150/200 to 144/200. HLE and Frontier used equivalent protocols with different physical card positions. SWE and Terminal are trace replays with measured latency and throughput. Their reference arms also used different card positions. B1-r8 remains the default. All comparisons reuse sealed controls.

[SR1 and SR2 scalar publication](../../website-publication/data/sr-development-results-v1.json) includes throughput, TTFT, TPOT, end-to-end P95/P99, scoring where available and physical card positions. [Private B1-sr2 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/b1-sr2-20261119) contains source overlays, raw evidence, run receipts and checksums.
