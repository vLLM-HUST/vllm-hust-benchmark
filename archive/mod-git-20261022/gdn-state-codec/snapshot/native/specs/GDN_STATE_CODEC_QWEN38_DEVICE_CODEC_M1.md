# GDN State Codec Qwen GDN device codec M1

> **Historical component only.** This standalone boundary codec does not
> implement the paper's fused `p=16` windowed decode path. Its timings must not
> be presented as a reproduction or refutation of LeapQuant. See
> `GDN_STATE_CODEC_WINDOWED_GDN_M2.md` for the corrected scope and matched gate.

## Boundary

This is a default-off Ascend 910B2 component prototype for one Qwen GDN
`128 x 128` recurrent-state head.  It converts an already fitted residual to
row-smoothed, per-value-channel symmetric INT8 and reconstructs FP32 state
with supplied rank-4 compensators.  The measured quantize path includes the
smoothing kernel.  The device screen deliberately excludes compensator
fitting, the 16-token update window, model execution, graph capture, runtime
integration, and transfer time.

The prototype must not be described as a model or serving speedup.  It is
`performance_qualified=false`, has no admission flag, and cannot replace the
unchanged FP state path.  Its purpose is to establish device exactness and a
cost floor before a batched-head kernel and matched end-to-end gate.

## Frozen mapping and layout

The provider owns state as `[value,key]`; the capsule owns its transpose as
`[key,value]`.  For provider order decay, read, update, output, the adapter is:

- `decay=exp(g)`;
- write key `beta*k`;
- read vector `exp(g)*k`;
- value `v`;
- query `scale*q`.

The CPU adapter test uses non-square `key=3,value=2` geometry and checks both
state and output equivalence so equal production dimensions cannot hide a
transpose error.

## Kernel shape

Smoothing launches one block per key row.  Quantization launches one block per
value column and uses padded UB gathers for the strided FP32 column and INT8
scatter.  Dequantization launches one block per key row.  All arithmetic is
FP32 except the stored residual codes.  The ABI is fixed to `128 x 128`, INT8,
and rank 4 for this screen.

The probe checks every smoothing value, every value scale, and all 16,384
INT8 codes against a deterministic host oracle.  It then reports relative RMS
against residual plus the supplied rank-4 product.  This RMS is quantization
error, not model-quality evidence.

## Hardware screen (2026-10-06)

Five independent probe processes ran on an idle physical Ascend 910B2 NPU 6,
exposed as device 0, with CANN 9.1.0.  Every run reported zero code mismatch
and relative RMS `0.0135034`.  The median of the five per-process medians was:

| Path | M0 single-core UB | M1 row/column parallel UB | M1/M0 |
| --- | ---: | ---: | ---: |
| smoothing + quantize | 1424.88 us/head | 90.481 us/head | 15.75x faster |
| dequantize + rank-4 reconstruct | 1165.09 us/head | 49.181 us/head | 23.69x faster |

These timings do not yet clear the no-harm gate.  The existing copy-only
screen moved the complete 576-head raw state in about 71.991 us, so a
sequential per-head codec is not a viable checkpoint-copy replacement.  A
single all-head batched kernel must amortize launch/synchronization and prove
that codec plus compressed transfer is cheaper than raw transfer.  Rank-4
fitting cost must then be added before any runtime admission decision.

Evidence is archived outside the repository at
`/root/stateaxis-evidence/gdn_state_codec/2026-10-06-qwen-gdn-device-codec-m0` and
`/root/stateaxis-evidence/gdn_state_codec/2026-10-06-qwen-gdn-device-codec-m1`.
