# gate_fp32 — no new serving measurement in this round

The service-side verdict is `within_noise` and the gate stays default-off; the
existing evidence already carries three arms with 2 runs each in ABBA order.
This round added the auxiliary-role label, the two L0 calibers, and one fresh
ABBA block (random 512/256 @16, FULL_AND_PIECEWISE): +3.07% over four runs but
-0.13% once the cold first run is dropped.

New in this archive: `abba-random-4runs.json`, `l0-gate-40layers.json`.

Original evidence (fixed SHA): https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/809d8f71abd4986012767f1e2e3b7573a7960a18/ops/gate_fp32/REPORT.md and evidence.json
