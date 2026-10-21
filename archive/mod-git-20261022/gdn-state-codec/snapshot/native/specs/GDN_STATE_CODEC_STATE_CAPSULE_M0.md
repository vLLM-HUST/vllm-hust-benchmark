# GDN State Codec state capsule M0 reference

## Status and claim boundary

This document defines a default-off CPU reference contract for applying GDN
State Codec to StateAxis recurrent state. It is derived independently from the
formulas in [LeapQuant v1](https://arxiv.org/abs/2609.38166) (CC BY 4.0). No
author implementation or repository was linked from the paper as of
2026-10-06, and no author code was used here.

The reference is not a serving integration, an Ascend kernel, a performance
result, or evidence that Qwen3.5/Qwen3.8 quality is preserved.  It remains
`performance_qualified=false` and has no runtime admission flag.  The paper's
NVIDIA GPU kernel and end-to-end results must not be transferred to Ascend.

The first research parameters are fixed to the paper's 8-bit setting:

- window length `p=16`;
- compensator rank `r=4`;
- symmetric INT8 residual;
- FP32 smoothing and per-value-channel scales in the reference;
- FP32 update and compensator records in the CPU oracle.  A later device ABI
  may use FP16 records only after matching this oracle.

## Audited StateAxis integration points

| Concern | Current integration point | M0 decision |
| --- | --- | --- |
| Qwen GDN state | `native/src/qwen38_geometry_normalized_gdn_provider.cpp`; FP32 `gdn.recurrent`, slot/head/value/key geometry | no provider mutation in M0 |
| StateFork identity | `StateForkRequest` schema v3 and runtime-owned `StateForkLease(owner_id,generation)` | capsule carries positive generation and epoch |
| StateFork publication | `Scheduler._publish_state_fork_children` and `KVCacheCoordinator.fork_cached_prefixes` | future adapter must publish capsule and KV ownership atomically |
| Recurrent copy | `mods/stateaxis-ascend/vllm_ascend/patch/worker/patch_mamba_utils.py` | future data-plane hook; do not change in M0 |
| Existing state snapshot | `hybrid_state_snapshot.py` writes immutable recurrent/KV/logit artifacts with optional lease | candidate checkpoint envelope for a future capsule payload |
| Hibernation | `hybrid_hibernation.py` currently records pause/resume transitions only | no resumable payload exists yet; future restore must use the capsule decoder |
| Feedback plane | `state_feedback.py` is a bounded typed drop-newest ring | future metrics only; no hot-path logging in M0 |
| V17/V18/V19 | eager parent/follower submission and decode-tail topology | orthogonal to state representation; never couple admission flags |

## Capsule ABI v1

The serialized capsule is canonical little-endian bytes followed by a
lowercase ASCII SHA-256 over every preceding byte.  It contains:

1. magic `LQCAPS01` and schema version;
2. `p`, `r`, bit width, and smoothing floor;
3. runtime generation and private slot epoch;
4. key/value geometry and current window position;
5. a 64-character model-geometry SHA-256;
6. row-major INT8 residual `Z`;
7. positive key-row smoothing vector `C`;
8. per-value-channel symmetric INT8 scale vector `B`;
9. `r` high-precision Compensator Token pairs `K,U`;
10. the branch-local ordered FP update records `(alpha,k,u)`;
11. the capsule checksum.

Restore requires the caller's expected schema/parameters, model-geometry
digest, dimensions, generation, epoch, and window position.  A checksum,
length, identity, geometry, phase, or parameter mismatch fails closed.  The
fallback is the existing unquantized path; mismatched bytes are never
reinterpreted or partially restored.

## Numerical semantics

For each token, the reference computes

`u = v - S^T beta` and `S' = diag(alpha) S + k u^T`,

then computes the output from `S'`.  During a window the immutable quantized
boundary is not rewritten.  Each output is reconstructed from that boundary,
the `r` compensator records, and the ordered FP branch-local records.

At the boundary, deterministic power iteration fits `r` rank-one components.
The residual is smoothed with

`c_i = sqrt(max(mean_j(abs(R_ij)), smoothing_floor))`.

`C^-1 R` is symmetrically quantized per value channel to INT8.  The next
window begins from `C * (Z/127) * B + K U^T`; the old records are discarded.
This implements per-window quantization, high-precision buffered updates,
Compensator Tokens, and residual smoothing together. Naive per-token INT8 is
not an admissible implementation of this codec.

The oracle deliberately fixes details not specified by the paper: sinusoidal
power-iteration initialization, 32 iterations per rank, sequential deflation,
the singular value folded into `K`, half-away-from-zero rounding, the
`[-127,127]` code range, and scale zero for an all-zero value channel.  These
are local ABI choices, not claims about an unpublished author implementation.
The fixed-output gate hashes values rounded to four decimal places so the full
trajectory remains stable across Debug and optimized floating-point builds;
the exact build-specific byte hash is recorded separately.

The Qwen GDN adapter is frozen by a non-square basis/layout oracle.  The
provider state is `[value,key]`; capsule storage is its transpose in
key-major `[key,value]` order.  Because the provider decays state before its
read, the exact token mapping is `decay=exp(g)`, write key `beta*k`, read
vector `exp(g)*k`, value `v`, and query `scale*q`.  The adapter test compares
both the updated non-square state and output against the provider recurrence.

An experimental Ascend 910B2 boundary codec implements residual smoothing,
per-value INT8 quantization, and rank-4 reconstruction for the fixed
`128 x 128` geometry.  It is a component screen only: rank fitting, window
updates, model quality, graph capture, batching across heads, and serving
integration remain outside its measured boundary.  It remains default-off
and `performance_qualified=false`.

## Ownership and lifecycle

The boundary object is immutable and reference-counted.  A fork shares that
object and copies only the records already accumulated inside the current
window.  Subsequent records are branch-private.  Reaching a window boundary
creates a new immutable boundary for that branch.

Each branch captures `(generation,epoch)` from its slot.  Commit succeeds only
when both still match and advances the epoch.  Cancel makes the branch
inactive without publishing.  A late completion, second commit, commit after
cancel, stale generation, or stale epoch is rejected.  Replacing a slot
generation requires a strictly greater generation with unchanged model
geometry and quantization parameters.  A future StateFork
adapter must additionally retain the existing owner ID and rank-complete lease
checks; a token count or window position is never a generation identity.
Epoch rollover fails closed and consumes the branch without publishing.
Commit prepares a complete next capsule before its no-throw move publication;
an allocation failure cannot partially mutate the slot or the branch.

`GdnStateCodecCapsuleSlot` is a scheduler-owned, single-threaded reference object;
it does not provide internal synchronization.  A future asynchronous adapter
must serialize slot mutation on the existing scheduler ownership domain (or
add an external lock).  The generation/epoch fences prevent stale publication
but do not themselves make concurrent C++ access data-race-free.

Pause/resume serializes the boundary plus the in-window records.  Resume must
validate the complete envelope before allocating or publishing destination
state.  Cross-worker transfer uses the same bytes and checksum.  Partial
decode, corrupt metadata, foreign geometry, parameter drift, and an unexpected
window phase fall back to the FP state path.

## Memory model

For one `128 x 128` state head, FP32 state is 65,536 bytes.  The planned device
boundary with INT8 residual, FP32 `C/B`, and FP16 `r=4` compensators is 19,456
bytes (1.1875 bytes/element, 3.37x smaller).  The CPU reference deliberately
stores compensators in FP32 and reports 21,504 bytes (3.05x smaller).  Its full
16-record FP32 update buffer is 24,576 bytes for the generic
`(alpha[128],k[128],u[128])` representation.  These are representation
estimates, not measured HBM traffic or throughput.

No-harm admission must compare state-copy bytes avoided against quantize,
dequantize, compensator-fit, record-read, and synchronization costs.  It is
eligible only for sufficiently wide forks; a missing cost estimate, narrow
fork, unsupported geometry, graph incompatibility, or negative matched screen
selects the unchanged FP path.

## Reference test matrix

| Gate | Required case |
| --- | --- |
| fixed oracle | deterministic 256-token trajectory, frozen numeric anchors, fixed four-decimal full-output SHA-256, and recorded exact build-specific SHA-256 |
| error | final-state and all-output relative RMS against an independent FP32 trajectory |
| window | 16 boundaries for 256 tokens; position returns to zero |
| fork | before window, inside window, and exactly at a boundary |
| isolation | divergent branch inputs produce divergent states without changing the shared boundary |
| lifecycle | successful commit, epoch advance, second-commit/cancel rejection, late completion, foreign capsule, and stale-generation rejection |
| resume | byte-identical serialize/restore at an in-window position |
| fail closed | corrupt/overflowing payload, foreign generation, model digest, quantization parameters, and window phase; boundary failure is atomic |
| accounting | FP baseline, compressed-boundary, shared-boundary, and branch-private bytes |

The M0 test uses small tensors so it is deterministic and fast.  Before an
Ascend prototype, the same matrix must cover Qwen `128 x 128` heads, long
trajectories, multiple seeds, fork-relative-to-each-branch FP baselines, and
resource release.  Device qualification then adds matched C-A-C-A-C or ABBA
measurements.  Kernel speedup alone cannot qualify an end-to-end claim.

## Future typed telemetry

The feedback-plane extension should carry bounded counters/records for FP
baseline bytes, compressed bytes, quantize/dequantize time, window position,
rank, saturation/outlier statistics, shared and private fork bytes, fallback
reason, and exactness/error digest.  It is an observation dependency, not a
separate performance mod, and must not allocate or synchronize in the hot path.
