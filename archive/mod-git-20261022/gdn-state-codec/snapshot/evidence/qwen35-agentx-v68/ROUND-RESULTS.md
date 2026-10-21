# Qwen3.5-35B LeapQuant runtime round results

Date: 2026-10-07 UTC

Evidence label: unqualified experiment. These measurements are matched eager,
single-request, 34-completion-token probes on two Ascend 910B2 devices. They do
not establish AgentX or online end-to-end benefit.

## Fixed baseline

- Fresh matched baseline last-five median: 7.527670 seconds (attempt 39).
- Prompt: `Explain why the sky is blue in several concise sentences.`
- Prompt tokens: 22; completion tokens: 34; temperature: 0.

## Candidates

| Candidate | Last-five median | Delta vs baseline | Correctness |
| --- | ---: | ---: | --- |
| p16 Cube, low-rank boundary | 8.449839 s | +7.99% | shadow passed |
| p16 Cube, quant-only fast rebase | 8.179676 s | +4.53% | max shadow RMS 0.010601984 |
| p48 Cube | 8.081017 s | +3.27% | max component RMS 0.0041417; shadow passed |
| p16 single-kernel BF16 | 8.263351 s | +5.60% | max shadow RMS 0.010602134 |
| p16 single-kernel BF16, in-kernel exp | 8.257959 s | +5.53% | max shadow RMS 0.010602134 |
| p16 transposed INT8 rows + UB vector reduction | 8.025475 s | +2.56% vs the earlier 7.824954 s baseline | max shadow RMS 0.010602133 |
| p16 paired projection v18, first run | 7.852164 s | +0.35% vs the earlier 7.824954 s baseline | max shadow RMS 0.010602133 |
| p16 paired projection v18, fresh interleaved rerun | 8.041425 s | +6.82% vs fresh 7.527670 s baseline | 8/8 HTTP 200; identical output |
| p16 v20 two shards + vector record projection | 8.138226 s | +8.11% vs fresh 7.527670 s baseline | shadow max RMS 0.010602133; 8/8 HTTP 200 |
| p16 v23 parallel fixed-smoothing boundary | 8.005255 s | +6.34% vs fresh 7.527670 s baseline | shadow max RMS 0.01135432; 8/8 HTTP 200 |

All eight responses in each final replace timing set returned HTTP 200 and the
same 34-token output. None of the end-to-end candidates passes the no-harm
performance gate. The fresh baseline/candidate interleave rejects the apparent
near-no-harm v18 result as run-order noise.

The subsequent component-only optimization round produced these 34-step,
32-head results. They are not end-to-end claims:

| Component candidate | Non-boundary median | Max relative RMS | Result |
| --- | ---: | ---: | --- |
| v18 paired projection | ~400.0 us | 0.010602134 on model shadow | reference |
| v19 two 64-column shards | 392.624 us | 0.009178324 | correctness pass; small gain |
| v20 shards + vector record projection | 342.684 us | 0.009178324 | current best component |
| v21 metadata UB prefetch | 361.425 us | 0.009178324 | negative result; reverted |
| v22 fast boundary before pipeline fix | 2.09--2.21 ms boundary step | 0.924121 | negative result; record-buffer hazard |
| v23 fast boundary after pipeline fix | 2.23--2.37 ms boundary step; 355.344 us non-boundary | 0.008901 | correctness pass; current candidate |
| v25 production-matched vector control | 331.760 us; 2.326x batched | 0.00765053 | correctness pass; no-harm fail |
| v26 Cube with log-decay fix | 366.250 us; 2.552x batched | 0.00765259 | correctness pass; no-harm fail |
| v27 boundary record-key prefetch | 365.360 us; 2.527x batched | 0.00765259 | correctness pass; boundary improved |
| v29 Qwen GQA record reuse | 373.530 us; 2.630x batched | 0.0077116 | negative performance result; reverted |
| v30 boundary Axpy | 355.709 us; 2.387x batched | 0.00765259 | correctness pass; boundary improved |
| v31 boundary vector divide/reduce | 360.669 us Cube; 2.263x batched | 0.00765259 | correctness pass; boundary improved further |
| v32 vector one-head/one-task | 334.048 us; 2.184x batched | 0.00765053 | correctness pass; no-harm fail |
| v33 vector record-key bulk prefetch | 328.009 us; 2.198x batched | 0.00765053 | correctness pass; current source candidate |
| v34 correction Axpy | 303.539 us | 1.11223 | negative correctness result; rejected |
| v35 correction Muls+Add | 303.104 us | 1.23141 | negative correctness result; rejected |
| v36 correction events | 300.293 us | 1.16755 | negative correctness result; rejected |
| v37 correction prefetch, scalar accumulation | 324.154 us; 2.112x batched | 0.00765053 | correctness pass; GM-to-UB path isolated as sound |
| v38 fetched events + correction Axpy | 304.899 us; 2.132x batched | 0.00765053 | correctness pass; hard-coded event ID was invalid |
| v39 FP16 boundary dot, rerun | 300.663 us; 2.100x batched | 0.00769923 | correctness pass |
| v40 four column shards | 304.714 us; 2.139x batched | 0.00769923 | negative performance result; reverted |
| v41 paired FP16 boundary dots | 289.474 us; 2.076x batched | 0.00769923 | correctness pass; current source candidate |
| v42 transposed-A Cube boundary, missing DMA reuse fence | 293.598 us; 2.193x batched | 0.691794 | negative correctness result; scaled-record buffer reuse hazard |
| v43 explicit GM-scalar correction pack | 295.108 us; 2.226x batched | 0.853165 | negative correctness result; incomplete/corrupt pack |
| v45 fenced scale + UB correction pack | 296.188 us; 2.215x batched | 0.00767021 | correctness restored; UB scalar transpose caused 2.70/5.25 ms boundaries |
| v46 fenced scale + native Cube transpose-A | 303.014 us; 2.187x batched | 0.00767021 | correctness pass; removes explicit pack, no-harm fail |
| v47 FP16 record accumulation | 294.379 us; 2.158x batched | 0.00767597 | correctness pass; small batched gain |
| v48 bulk residual prefetch | 297.474 us; 2.178x batched | 0.00767597 | negative performance result; reverted |
| v49 shard-only correction prefetch | 302.324 us; 2.188x batched | 0.00767597 | negative performance result; reverted |
| v50 metadata UB prefetch | 297.964 us; 2.164x batched | 0.00767597 | correctness pass; no material gain |
| v51 paired FP32 record Mul | 299.254 us; 2.041x batched | 1.0 | negative correctness result; invalid repeat-stride layout |
| v52 separate Mul + paired record reduction | 289.298 us; 2.125x batched | 0.00767597 | correctness pass; current source candidate, no-harm fail |
| v53 bulk output writes | 304.488 us; 2.223x batched | 0.00767597 | negative performance result; reverted |
| v54 preallocated boundary scratch | 305.678 us; 2.151x batched | 0.00767597 | correctness pass; boundary allocation moved to slot creation |
| v55 preinitialized boundary runtime, rerun | 292.368 us; 2.015x batched | 0.00767597 | correctness pass; boundary 1.50/1.35 ms, no-harm fail |
| v56 dynamic zero-compensator branch | 297.333 us; 2.062x batched | 0.00767597 | correctness pass; no material gain |
| v57 compiled quant-only specialization | 294.639 us; 2.038x batched | 0.00767597 | correctness pass; steady ratio 1.408x, current source candidate |
| v62 vectorized boundary/input/final writes | 291.7 us host steady; 37.36 us device (32 states) | 0.007748 | correctness pass; host no-harm initially failed |
| v64 cached stream and stable contiguous pointers | 118.05 us host steady (32 states) | 0.007748 | 4/4 correctness/no-harm pass; host overhead removed |
| v65 TP2 dynamic block count | 19.62 us device (16 states) | 0.008021 | 4/4 correctness/no-harm pass |
| v66 eight-column residual DMA/cast tiles | 13.06 us device (16 states) | 0.008021 | MTE2 reduced from 6.80 to 1.15 us |
| v67 scale UB prefetch | 13.34 us device (16 states) | 0.008021 | negative performance result; reverted |
| v68 batched eight-column projection/reduction | 11.22 us device vs 10.80 us native | 0.008021 | current source; clean-build gate pass |

The early matched native single-layer probe measured the stock recurrent GDN
op at 162.493 us median and v20 at 382.724 us median (2.36x), with max output
RMS 0.009052. The corrected v25-v36 harness moves metadata construction out of
the timed call and consistently measures the production native op near
200--213 us per synchronized call. This is the relevant no-harm comparison;
the older custom FP32 reference kernel is only a correctness oracle and not the
production baseline. Early v24/v25 files without `fixed` in their names used
different native/compressed cache slots and are retained as invalid controls,
not performance or correctness evidence.

## Build identity for the workload-qualified candidate

- Source base: `e74e9ece60e32a7a6e6a93f9f35c0e9ec4434596`
- Worktree: `/workspace/stateaxis-leapquant-cube`
- Build: `/workspace/stateaxis-build-leapquant-runtime-v68-final/lib`
- Window kernel SHA-256: `ff8220f08b904e49208d9f7ee0ca0deb7ac7c55e75429d84d143baf61fb27254`
- Decode runtime SHA-256: `a26b5bc46b795d1421691b6f030b13a77cab90e57a2105b91a7b5da0754af266`
- Boundary kernel SHA-256: `42b185687b79fecd6bfa8b5183de3b417c249b319b7d48836931e8bb51348ef5`
- Boundary Cube kernel SHA-256: `2f14f3d36bfec9343fa39d5c551cc3963226b366a3cabd5dabc56c345c2dd7d3`
- Boundary runtime SHA-256: `e342caa4640b253197504c060644b0de84f76ad89d5eb3153b6d7a1d9729bfa2`
- Runtime-declared manifest SHA-256 during the recorded E2E runs:
  `2af719ca923015eb4e46502112e9dff29393ec256504835a9eb76a816b4b8bb6`
  (the evidence/results fields were added to the manifest after measurement).
- Initializer: quant-only; boundary iterations: 0; window: 16; rank: 4.

## Bottleneck conclusion

The CANN v31 profile quantifies the three-launch Cube path at 24.542 us for
prepare, 16.002 us for INT8 Cube matmul, and 78.920 us average for finish. The
finish stage grows from 37.681 to 139.882 us with window position and is the
dominant Cube bottleneck. The v32 single-kernel vector profile averages 92.411
us on device versus 14.871 us for production `RecurrentGatedDeltaRule`.
The three-launch Cube path also pays prepare/Cube/finish launch overhead. The
single-kernel path removes those launches, and transposed residual rows plus UB
vector reduction materially reduce its cost, but the fresh v18 end-to-end rerun
still regresses by 6.82%, while v20 regresses by 8.11%. The parallel quant-only
boundary cuts its component cost by roughly 75%, but v23 still regresses by
6.34% end to end. Splitting each state
over two AIV cores alone helps
little; vectorizing record projections lowers the component median from about
400 us to 343 us. Bulk-prefetching record keys reaches 328 us. Vectorizing
record-correction accumulation initially appeared to reach about 300 us, but
v34-v36 failed correctness because a hard-coded event ID did not establish the
required scalar/vector dependency. Fetching real event IDs restores correctness
in v38. FP16 boundary dot products and pairing the two reductions reach 289 us
in v41. The next material target is the p=16 boundary: its record merge is still
a 128x16x128 operation expressed as per-column Axpy and should be evaluated as
a batched Cube GEMM. No 35B/AgentX end-to-end run is admitted before the
single-layer no-harm gate passes.
The v42-v46 round established that the transposed-A Cube GEMM is numerically
sound; the initial failure came from reusing a one-buffer scaled-key UB region
without an MTE3-to-MTE2 dependency. An isolated random-tensor check after the
fence measured 0.0002517 relative RMS for scaled keys and 0.0002512 for the
FP16-input/FP32-output Cube merge. Explicit correction packing was both slower
and unreliable, so v46 uses the Cube transpose-A contract directly. The
boundary remains too expensive (about 1.35--1.50 ms warm). Preallocating all
boundary scratch and creating Cube tiling at slot initialization lowers the
best batched ratio from 2.125x to 2.015x. A compile-time quant-only kernel
removes provably zero rank-4 compensator work. Subsequent host-path caching,
dynamic TP-local block counts, eight-column residual DMA/cast tiles, and a
16-way batched boundary reduction reduce the TP2 single-layer device median
from 19.62 us at v65 to 11.22 us at v68, versus 10.80 us for production
`RecurrentGatedDeltaRule`. The v68 clean build passes the matched correctness
and no-harm gates with max relative RMS 0.008021.

## Matched Qwen3.5-35B / AgentX end-to-end result

The admitted workload is deliberately narrow: Ascend 910B2 TP2, eager mode,
one real-online request at a time, the first AgentX conversation/turn, 649
prompt tokens and 128 generated tokens, temperature 0 and seed 0. All 18
formal measurements returned the same 128-token content SHA-256
`2035b0e9d6d6de9c4fcc6a4adeb0574e198ddbfb66c3cc93c257377659e465ee`.

| Arm/lifecycle | Measurements | Total median | Run-level ITL median |
| --- | ---: | ---: | ---: |
| v68 candidate A | 6 | 28.4146 s | 219.9409 ms |
| native baseline B | 6 | 28.6807 s | 222.3608 ms |
| v68 candidate A2 | 6 | 28.4343 s | 219.8913 ms |
| combined v68 | 12 | 28.4164 s | 219.9409 ms |

Against the contemporaneous native arm, combined v68 is 0.922% faster in
total time and 1.088% lower in run-level median ITL. A deterministic 20,000
resample percentile bootstrap gives total-time delta interval
[-1.623%, -0.175%] and ITL interval [-1.818%, -0.602%]. This is evidence of a
small repeatable benefit for this workload only; it is not a general online,
multi-request, graph-mode, or capacity claim. Earlier 34-token and v65
128-token neutral/negative results remain retained.

The current integration retains the provider's dense state alongside compressed
slots and therefore cannot claim capacity or memory savings.

## Validation

- Core parser/admission tests: 17 passed.
- Ascend runtime initialization tests: 3 passed.
- v68 clean-build matched TP2 gate: correctness pass, no-harm pass, max
  relative RMS 0.008021, steady ratio 0.565, batched ratio 0.487.
- Full repository conftest was not loaded because this isolated environment
  lacks optional `tblib`; the two narrow suites pass with `--noconftest`.
- Failed server attempts 29 and 30 are retained. They stopped during MoE
  warmup because the wrong custom OPP library was inherited and did not execute
  LeapQuant. Attempts 31-34 used the matching `/vllm-workspace` custom OPP.
