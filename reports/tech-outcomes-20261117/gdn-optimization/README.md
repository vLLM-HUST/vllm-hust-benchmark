# GDN fusion / weight pre-pack evidence (2026-11-17 round)

Tracking issue: vLLM-HUST/vllm-ascend-split-batch-hust#59. Evidence binding PR: vllm-ascend-split-batch-hust#61 (`809d8f71abd4986012767f1e2e3b7573a7960a18`). Full report: [REPORT-v1.md](REPORT-v1.md). SHA256SUMS-v1 covers every file in this directory.

Four quotable conclusions:

1. Output-norm fusion: original cohort +2.66% is NOT reproduced. Eight interleaved runs give +0.44% overall and -0.98% after dropping the cold first run; the fused kernel is proven to execute. The figure is kept as history only.
2. GDN split operator group: -39.7% is a same-window target-group self-ratio (8560 -> 5163 ms); the all-kernel-denominator figures are internal only. Service side is at parity (512-in +2.62%, 2048-in -1.23% after the cold run).
3. Conv weight pre-pack: serving parity in eager mode (-0.42% over four runs); the original 9.26 ms local saving is not reproduced and stays historical.
4. Gate fp32 hoist: local Cast removal is solid; service side is +3.07% over four runs but -0.13% once the cold run is dropped, consistent with the recorded within_noise, and the gate stays default-off.

Method note: the first server start of a session is 6-12% slower than later starts on this box. Because the original +2.66% cohort ran off,off,on,on, that bias alone can produce a multi-percent apparent gain.
