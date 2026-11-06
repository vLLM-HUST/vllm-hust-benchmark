# Reorganized MOD qualification campaign (2026-11-07)

This campaign requalifies the three MOD families created during the catalog
reorganization:

- `vLLM-HUST/vllm-hust-gdn-optimizations`
- `vLLM-HUST/vllm-hust-runtime-micro-optimizations`
- `vLLM-HUST/vllm-ascend-attention-optimizations`

The measurements imported before the reorganization are source records only.
They are not performance qualification for the new repositories or for a new
combination of their candidates.

## Preflight result

The first preflight ran in the assigned four-card Ascend 910B2 environment.
All cards were healthy and idle. Eight retained candidate packages built as
wheels. The GDN static-decay lifecycle suite passed 21/21 tests, and the Native
GDN base contract/lifecycle suites passed 22/22 tests.

Formal NPU performance execution did not start because the current host failed
candidate admission:

- Native GDN expects `_get_mamba_spec_for_layer`; the current Core host no
  longer exposes that helper.
- Attention adapters reject the current Ascend host at the pinned source hash
  gate.
- Empty-padding and query-start-location reuse still target older host file
  identities and require a reviewed current-host binding.
- Residual-scale targets an older layer-norm implementation and needs a new
  equivalence review.
- The old RoPE argument repair is already represented by the current host call
  shape and is not admitted as an independent current MOD.
- GDN static decay has valid lifecycle tests but lacks a contract bound to the
  current host revision.

This is a fail-closed compatibility result, not a performance result. No old
percentage is transferred to the reorganized repositories.

## Frozen retest protocol

Each port must first be committed to its canonical reorganized repository and
must remain disabled by default. The formal campaign then uses:

1. exact candidate, `vllm-hust`, and `vllm-ascend-hust` commit identities;
1. a specialty Qwen3.5-35B-A3B BF16 TP2 target on Ascend 910B2, because these
   mechanisms do not apply to the public 14B fixed target;
1. an unchanged Native arm and one independently enabled candidate arm;
1. three alternating-order Native/candidate pairs with fresh processes;
1. one warm-up window and three measured windows per process;
1. exact activation receipts and mechanism counters from both ranks;
1. MMLU-Pro 200/256 performance work plus a 70/2048 quality overlap, with
   complete output comparison and no quality regression;
1. environment manifest, resolved same-spec, raw outputs, process logs,
   per-run leaderboard artifacts, and SHA-256 custody for every retained file;
1. separate records for single candidates and combinations; individual gains
   are never added or transferred to a combination.

Any source mismatch, inactive mechanism, output mismatch, process failure,
graph fallback, or missing receipt rejects the run before publication.

Machine-readable observations are in [`preflight.json`](preflight.json).
