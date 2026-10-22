# LeapQuant 576-head vectorized component gate (M2)

- Evidence label: `component-gate / negative / no-integration`
- Parent source: initial private engine-integration snapshot
- Parent commit: `25bbdc413a1deeb7d9a11a33c820de337f09a28d`
- Branch/worktree: `feature/leapquant-all-head-batched` at `/workspace/stateaxis-leapquant-all-head`
- Source state: dirty experimental prototype; complete diff is `vectorized-prototype.patch`
- Patch SHA-256: `3c142d6e74abfddceee0a0ae7db5639c30d8d85a7aab020efdfed62d32ed9745`
- Hardware: Ascend 910B2, physical NPU 6 / container device 0; NPU 7 was idle and unused
- Runtime: CANN 9.1.0, `ascend910b2`, Release build, `/usr/bin/cmake` 3.22, GNU C++ 11.4
- Model binding: Qwen3.5-35B GDN recurrent state, TP=4, 48 GDN layers, 12 local value heads/layer
- Tested geometry: 576 x 128 x 128 FP32 recurrent heads/rank
- Graph mode: standalone AscendC component probe, not an online model run
- Entry command: `LD_LIBRARY_PATH=<build>/lib:<CANN>/aarch64-linux/lib64 <build>/statecentric_leapquant_all_head_runtime_probe 0`
- Repeats: five independent processes; each process uses 3 warmups and 11 timed samples, reporting its median

## Correctness and lifecycle

- Quantized code mismatches: 0 / 9,437,184 in every run
- Maximum smoothing error: 5.96046e-08
- Maximum scale error: 4.76837e-07
- Relative RMS reconstruction error: 0.00797512
- Maximum dense reconstruction error: 0.0100098
- Guard regions: passed for every device allocation
- Invalid admission: rejected head counts 0 and 577 and a null input
- Allocation, stream synchronization, destruction, device reset, and ACL finalization: passed in every process

## Outer medians across five processes

- smoothing: 5,627.53 us
- quantize excluding smoothing: 7,368.77 us
- encode total: 12,974.6 us
- compressed D2D copy: 49.891 us
- dequantize: 5,393.55 us
- measured codec pipeline: 18,370.5 us
- raw FP32 D2D copy: 61.35 us

The earlier copy-only M0 probe gives the more generous theoretical savings budget: raw 74.35 us versus planned FP16-compensator payload 33.561 us, or 40.789 us saved. M2's measured codec cost excluding rank fitting is already 18,370.5 us, 450.38x that generous budget. Rank fitting is not included in M2; because it can only add non-negative work, the complete pipeline cannot pass the no-harm gate. This is a strict lower-bound rejection, not a performance claim for an online workload.

## Decision

`FAIL_NO_HARM`. Do not integrate this prototype into StateAxis runtime and do not open or merge a source PR. Preserve the patch and negative evidence for future architecture-specific redesign. A future attempt needs a representation produced incrementally during the GDN update (or hardware matrix-engine rank fitting), not a boundary-time scan and refit of all 576 dense states.
