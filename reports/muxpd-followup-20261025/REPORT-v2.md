# MuxPD TP4 follow-up: matched raw/group composite, v2

This append-only update adds the terminal `group-fusion-composite-hccl-true-01` run to the ten runs
in v1. V1 archives, manifests, index, failures and report remain unchanged. The active quality
experiment and future pressure/performance runs are excluded. This remains a draft correctness
study; Issues #3, #11 and #23 and draft PR #10 are not declared resolved by these observations.

## Matched numerical result

Raw and group use source `6c606f789450a45ef35afee398b0a3e516caa1a1`, the same post-warmup numerical
observer, exact production worker selector, TP4/EP4 on measured 910B2, Qwen3.5-35B-A3B BF16,
HCCL_DETERMINISTIC=true, and explicit `enable_npugraph_ex=false, enable_static_kernel=false`. The
engine difference is the explicit `compile_prefill_groups` boolean. Each process runs the normal
family followed by seed-first phases, including three D1/P384 staggered phases. The earlier normal
native46/raw9f observer boundary remains separate evidence.

| Comparison       | Exact output sequences | Comparable vectors | Elementwise equal | Literal `.npy` SHA equal |
| ---------------- | ---------------------: | -----------------: | ----------------: | -----------------------: |
| Raw warm pairs   |                  30/30 |            240/240 |           240/240 |                  240/240 |
| Group warm pairs |                  30/30 |            240/240 |           240/240 |                  240/240 |
| Raw versus group |                  30/30 |            240/240 |           240/240 |                  240/240 |

All four ranks agree on sampled token receipts. Full-vocabulary vectors are captured on rank zero,
so this is not a claim of four-rank full-vector capture. The independent v2 summary rehashes source
files and pairs literal `.npy` identities; its bytes were reproduced exactly during publication
preparation. The older analyzer field named `logits_byte_equal` means elementwise NumPy equality;
the separate file-SHA check establishes literal identity here.

V1's same-source native/raw composite result remains 30/30 equal output sequences, with 147/240
positions meeting observed same-batch/context checks and all 147 vectors exactly equal. Those 147
comparisons must not be combined with the 240 raw/group comparisons to claim a new native/group
same-batch result. Captured equality on these fixed inputs does not establish broad semantic
quality, all hidden/KV state equivalence, or performance benefit.

## Metadata correction and coverage limit

The composite has 12 driver phases but **15 actual P cohorts**: six normal, three seed-first single,
and six from three seed-first four-lane phases. Each cohort has ten four-layer group returns, so
full callback coverage would require 150 events per rank. The immutable observer capped this at 128.

Each of the four ranks retained exactly 128 group-return callbacks:

- Cohorts 1–12: all groups 0–9 observed.
- Cohort 13, the second mixed repeat's P384 peer cohort: groups 0–7 observed; groups 8–9 are outside
  the cap.
- Cohorts 14 and 15, the third mixed repeat's seed P128 and peer P384: no group-return callbacks
  observed.

All 15 serving P execute entries and all 30 request/240 output samples are present. The cap limits
group metadata, not the full-vocabulary collection. The captured events satisfy MoE index
`4*(group_index+1)`, pre-checkpoint layer count `4*group_index`, P lane/task identity, final-group
flag, and the complete ordered 40-entry native MoE registry. Unobserved events are not counted as
passing, and host metadata does not inspect recurrent/KV tensor contents.

The retained v1 host audit incorrectly expected registry names ending `.mlp`. Pinned native
registration names end `.mlp.experts`, so its 128 name-check failures per rank were validator false
positives. The separately versioned v2 validator checks the qualified namespace and reports zero
captured-event check failures. The device code and recorded events were not changed. The earlier
incorrect 12-cohort expectation is explicitly superseded by the accompanying coverage correction
note; no missing observations were fabricated or backfilled.

Native/raw compiler tables contain 30 scalar entries per rank; group tables contain 60. Every entry
in all three arms has the same scalar payload hash
`e4527977521582a72db5e7126357ae7aca09f4ccb00e24725851d1d28632c5c9`. The table identities differ in
entry count/indexing. This does not prove identical lowered kernels, graph references or runtime
state.

## Custody and publication state

V2 adds one complete group run capsule and one support capsule containing the original v1/v2 host
audits, both validator sources, the coverage correction, full raw/group derived analysis and both
the original and independently reproduced v2 summary. V1 capsules are reused by exact identities.
Every original member and compressed archive is hashed and archive members are read back locally.
Source6c is the exact tree already exported in v1; no Git history or model-weight payload is added.
The historical Ascend dirty-build/binary provenance limit from v1 remains.

The canonical benchmark report will link planned assets under
`https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-followup-20261025/`.
These URLs are prepared publication locations, not evidence that upload or remote download
verification has occurred. Root owns those operations.

Remaining gates include full or explicitly bounded group metadata coverage, the frozen 200-case
paired quality test with no-new-wrong veto and real mixed coverage, actual allocator
pressure/preemption and partial/full cancellation ACK/reuse, installed-path configuration/cleanup,
and ordinary-worker unprofiled latency/throughput/goodput comparisons. No quality or performance
result is inferred from this numerical study, and native instability is not used to waive
candidate-specific failures.
