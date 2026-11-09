# Reorganized MOD qualification campaign (2026-11-07)

This campaign requalifies the three MOD families created during the catalog reorganization:

- `vLLM-HUST/vllm-hust-gdn-optimizations`
- `vLLM-HUST/vllm-hust-runtime-micro-optimizations`
- `vLLM-HUST/vllm-ascend-attention-optimizations`

The measurements imported before the reorganization are source records only. They are not
performance qualification for the new repositories or for a new combination of their candidates.

## Preflight result

The first preflight ran in the assigned four-card Ascend 910B2 environment. All cards were healthy
and idle. Eight retained candidate packages built as wheels. The GDN static-decay lifecycle suite
passed 21/21 tests, and the Native GDN base contract/lifecycle suites passed 22/22 tests.

Formal NPU performance execution did not start during this first preflight because the current host
failed candidate admission:

- Native GDN expects `_get_mamba_spec_for_layer`; the current Core host no longer exposes that
  helper.
- Attention adapters reject the current Ascend host at the pinned source hash gate.
- Empty-padding and query-start-location reuse still target older host file identities and require a
  reviewed current-host binding.
- Residual-scale targets an older layer-norm implementation and needs a new equivalence review.
- The old RoPE argument repair is already represented by the current host call shape and is not
  admitted as an independent current MOD.
- GDN static decay has valid lifecycle tests but lacks a contract bound to the current host
  revision.

This is a fail-closed compatibility result, not a performance result. No old percentage is
transferred to the reorganized repositories.

## Runtime micro-optimizations formal result

The runtime family was subsequently ported, reviewed, and committed in its canonical repository.
Candidate commit `210038bc250d2558c039f6c0874f403c4d89588e` was measured against unchanged Native on
Core `c4df37dfcc7f72ca41cd43801a468526ecf99a2b` and Ascend Host
`ab05682845beb9f54a9522122f48f628c3f5b649`. Both TP ranks emitted install and first-exercise
receipts for empty-padding, query-start-location reuse, and residual-scale in every candidate
process.

The complete frozen campaign used Qwen3.5-35B-A3B BF16 TP2/EP on Ascend 910B2. It completed three
alternating-order Native/candidate pairs with fresh processes. Every process ran one warm-up plus
three measured 200-request windows. Every measured window used 282,265 prompt tokens and 51,200
output tokens, with zero failed requests.

Pair throughput changes were +0.841%, -0.494%, and -0.213%. The geometric pair aggregate was +0.043%
(arithmetic nine-window aggregate +0.041%). Geometric pair TTFT and TPOT changes were -0.152% and
-0.122%, respectively. These sub-percent, mixed-direction changes do not establish a stable speedup.

The 70/2048 quality overlap completed in every process. Across the three pairs, all 210 candidate
outputs and reasoning traces had exactly the same hashes as Native; transitions were 129
correct-to-correct and 81 incorrect-to-incorrect. The result is therefore **measured, quality
equivalent in this cell, and without demonstrated stable performance benefit**. It is not
`retest_pending`, but it must not be advertised as a positive speedup.

Machine-readable aggregation is in [`runtime-formal-summary.json`](runtime-formal-summary.json). The
complete 17 MiB custody bundle is attached to the
[`runtime-micro-formal-retest-20261107` release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/runtime-micro-formal-retest-20261107),
with SHA-256 `9f2cb812f3d593e84b9c450510d35efa24ace44da3a7a7fd831b610a75fee972`.

## GDN static-decay formal result

Static decay was ported and committed in the canonical GDN repository. Candidate commit
`807d1b0d2295a438ec696fdfdc8c2ebd97b6b448` was measured against unchanged Native on the same Core
and Ascend Host commits used for the runtime campaign. Every candidate process emitted
`cache-prepared` and `candidate-exercised` receipts from both TP ranks for 30 expected layers and 16
heads per rank.

The frozen Qwen3.5-35B-A3B BF16 TP2/EP Ascend 910B2 campaign completed three alternating-order
Native/candidate pairs with fresh processes. Every measured window completed 200 requests, 282,265
prompt tokens, and 51,200 output tokens with zero failures. Pair throughput changes were +0.292%,
+0.214%, and -0.516%. The geometric pair aggregate was -0.004% (arithmetic nine-window aggregate
-0.0036%); geometric TTFT and TPOT changes were +0.030% and -0.053%.

All three 70/2048 quality overlaps completed. All 210 candidate content and reasoning hashes were
exactly equal to Native, with 132 correct-to-correct and 78 incorrect-to-incorrect transitions. The
static-decay candidate is therefore **measured and quality equivalent in this cell, but it does not
demonstrate a stable performance benefit**. The card is no longer `retest_pending`, and the mixed,
sub-percent measurements must not be advertised as a speedup.

Machine-readable aggregation is in [`gdn-formal-summary.json`](gdn-formal-summary.json). The
complete 17 MiB custody bundle is attached to the
[`gdn-static-decay-formal-retest-20261107` release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/gdn-static-decay-formal-retest-20261107),
with SHA-256 `6085e887b1f129b163f2255d2de0c349d87ffee778dc0fdfdd69b48ceb6ec74a`.

## Frozen retest protocol

Each port must first be committed to its canonical reorganized repository and must remain disabled
by default. The formal campaign then uses:

1. exact candidate, `vllm-hust`, and `vllm-ascend-hust` commit identities;
1. a specialty Qwen3.5-35B-A3B BF16 TP2 target on Ascend 910B2, because these mechanisms do not
   apply to the public 14B fixed target;
1. an unchanged Native arm and one independently enabled candidate arm;
1. three alternating-order Native/candidate pairs with fresh processes;
1. one warm-up window and three measured windows per process;
1. exact activation receipts and mechanism counters from both ranks;
1. MMLU-Pro 200/256 performance work plus a 70/2048 quality overlap, with complete output comparison
   and no quality regression;
1. environment manifest, resolved same-spec, raw outputs, process logs, per-run leaderboard
   artifacts, and SHA-256 custody for every retained file;
1. separate records for single candidates and combinations; individual gains are never added or
   transferred to a combination.

Any source mismatch, inactive mechanism, output mismatch, process failure, graph fallback, or
missing receipt rejects the run before publication.

Machine-readable observations are in [`preflight.json`](preflight.json).
