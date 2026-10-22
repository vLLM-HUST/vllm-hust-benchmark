# Batched provider-state initialization

Date: 2026-10-09
Evidence label: `model-shaped-component-candidate`

## Bottleneck

The Qwen3.5-35B integration creates a compressed capsule lazily on the first
decode token of each of its 30 GDN layers. The retained PyTorch quant-only
initializer measured a synchronized median of `1072.90 us` per TP-local
16-state layer (`20` repetitions, minimum `1035.96 us`). This explains a
material part of the consistently worse candidate TTFT in the expanded AgentX
round.

## Candidate

Two AscendC ABI-v2 launches consume the provider-owned FP32
`[state,value,key]` tensor directly:

1. accumulate mean absolute values as 128-element UB vectors and form the
   per-key square-root smoothing vector;
2. normalize and reduce each value row as vectors, then emit the transposed
   INT8 residual layout and per-value scales expected by the decode kernel.

This removes the provider-state transpose, the PyTorch elementwise launch
chain, and intermediate normalized FP32 tensors. It does not alter the v1 ABI.

## NPU result

Both visible Ascend 910B2 devices passed the same deterministic 16-state,
128x128 probe:

| device | median two-launch time | reconstruction relative RMS |
| ---: | ---: | ---: |
| 0 | 67.990 us | 0.00738162 |
| 1 | 67.761 us | 0.00738162 |

The component launch pair is about `15.8x` lower latency than the prior
Python initializer measurement. The comparison is intentionally conservative
about scope: the Python number includes tensor allocations, while the native
probe uses preallocated outputs. StateAxis integration and matched online TTFT
must still pass before this is an end-to-end result.

The vector conversion differs from the scalar CPU rounding oracle at 4,915 of
262,144 INT8 elements because it converts through FP16, but reconstruction RMS
is below the unchanged 2% gate. The legacy v1 probe remains exact at the INT8
level (`0` mismatches) and passes with reconstruction RMS `0.0135034`.

## Decision

Admit the candidate to StateAxis shadow and matched online gates. Do not claim
an online improvement from this component result alone.

## Combined host ABI follow-up

ABI v3 adds a host-only wrapper that launches the unchanged v2 smoothing and
quantization kernels on the caller's stream. It does not claim device-kernel
fusion. A fresh default-configured build now selects `Release`, because the
legacy CANN merge helper fails when `CMAKE_BUILD_TYPE` is empty.

The extended alternating-order probe passed on both visible devices, rejected
zero states, and preserved the same `0.00738162` reconstruction relative RMS:

| device | split median | combined median |
| ---: | ---: | ---: |
| 0 | 68.011 us | 68.041 us |
| 1 | 67.930 us | 68.011 us |

As expected, synchronized device time is unchanged. For the model-shaped
16-state tensors, 200 Python `ctypes` submissions on device 0 reduced median
host enqueue time from `21.2805 us` to `17.3800 us` (`18.33%`). This is about
`3.90 us` per initialized GDN layer and is too small on its own to establish
an online improvement.

A separate attempt to pack same-dtype Python workspaces into contiguous slabs
was rejected and reverted: full synchronized initialization regressed from
`537.811 us` to `623.188 us` per layer (`+15.88%`). The Python slice/view
construction cost exceeded allocator savings. `RESULTS.json` records both the
retained ABI result and this negative result with artifact hashes.
