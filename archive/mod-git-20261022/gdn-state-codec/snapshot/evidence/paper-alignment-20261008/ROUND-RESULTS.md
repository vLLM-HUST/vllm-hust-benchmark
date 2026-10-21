# LeapQuant paper-alignment round

Date: 2026-10-08  
Evidence label: `independent-paper-reimplementation`,
`model-derived-component-negative`

## Claim boundary

This round independently maps the method in LeapQuant arXiv:2609.38166v1 to
Ascend 910B2. It is not a reproduction of the paper's NVIDIA throughput and is
not an online end-to-end speedup result.

Matched method choices are an INT8 residual, four FP16 Compensator Tokens,
FP32 smoothing scales, a 16-token window, and a power-iteration refit at every
boundary. The paper does not disclose its iteration count or smoothing floor.
This implementation chose one iteration from a measured 1/2/4/8 scan.

The paper's 2.52x Qwen3.5-9B kernel ablation is measured at batch 256. It also
states that fitting up to four compensators is overlapped with state reads in
that setting. This round gates one active Qwen3.5-35B-A3B TP2 request; the
current Ascend integration issues the boundary stages serially and therefore
does not inherit the paper result.

## Reproducibility boundary

- experimental source commit:
  `1995614325bbc409219f024fc8efa8ea75446155`
- source parent: `6814215e8c2702e0023740dfba070ab1f32a3704`
- model revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`
- AgentX conversation: `4b88101f5ed7507b73138a73dfbf55f972de`
- fixture: first-turn layer-0 state plus 16 actual decode updates
- hardware: Ascend 910B2; TP2 model shape; logical device 0 for component runs
- runtime: CANN 9.1.0, eager component path
- full local evidence archive:
  `/root/stateaxis-evidence/leapquant/2026-10-08-paper-alignment/`

## Power-iteration scan

| Iterations | Boundary | Reconstruction relative RMS | Probe cycle speedup vs FP32 oracle |
| ---: | ---: | ---: | ---: |
| 1 | 4.179 ms | 7.22781e-05 | 2.463x |
| 2 | 5.149 ms | 6.78963e-05 | 2.363x |
| 4 | 7.153 ms | 6.37440e-05 | 2.164x |
| 8 | 11.049 ms | 5.64476e-05 | 1.863x |

The standalone FP32 probe is a correctness oracle, not the production timing
baseline. One iteration is the measured Pareto point for this fixture.

## Matched production-operator gate

The retained 82-step result uses 16 TP-local states and crosses five window
boundaries:

| Metric | Result |
| --- | ---: |
| Compressed steady median | 116.922 us |
| Native steady median | 206.637 us |
| Steady time ratio | 0.5658x |
| Warm boundary steps | 2.835--2.870 ms |
| Maximum output relative RMS | 0.008844586 |
| Batched 82-step compressed total | 16.342 ms |
| Batched 82-step native total | 10.324 ms |
| Total time ratio | 1.5829x |

Correctness passes the unchanged 2% gate. Performance fails the no-harm gate.
No replace-mode online run was admitted after this failure.

## Bottleneck and rejected attempt

The captured msprof statistics attribute average device time to residual
quantization (`1193 us`), residual materialization (`1123 us`), and smoothing
(`316 us`). An INT8 Cube matmul averages only `15.8 us`, so tensor-core compute
is not the present limiter.

A continuous row-write experiment reduced the 32-state boundary to about
`3.50 ms`, but its vector conversion raised reconstruction RMS to roughly
`0.017`. It was rejected. Exact scalar rounding restored reconstruction RMS to
`7.23e-05`; its cache-line-safe row layout improved the 16-state warm boundary
from roughly `3.65 ms` to `2.85 ms`, but remained negative after amortization.

## Artifact bindings

- full round summary:
  `6096479ca23c82efd1c35c07b82f4a69cc410975bcf5f544095edd0128c5ad52`
- retained 82-step component JSON:
  `3b3db8ab56fa8b4e35bebec5fbbece62b7fa32118096e586bbf12944bfc74891`
- msprof operator statistics:
  `95546a022d2c688be03d033578dcd5a173c4ddd0a3473d04581e4189932f485c`
- fixture metadata:
  `f79e0a50dc379dc1b45e2509a10e8d5859704aaedcba96b88df216bbd7ae1942`

All raw fixture and result hashes are retained in the archive's
`SHA256SUMS`. Failed and performance-negative artifacts remain preserved.

## Decision

Keep the implementation on its experimental source branch. Do not enable it,
merge it into the retained mod runtime, or broaden the v68 workload claim.
The next implementation target is a fused boundary with a batched four-vector
subspace iteration and explicit concurrency scaling, so that fitting can be
overlapped with state reads rather than serialized into many small launches.

