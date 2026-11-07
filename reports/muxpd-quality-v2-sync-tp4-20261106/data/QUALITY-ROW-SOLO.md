# #23 acceptance (redefined by the maintainer 2026-11-07): ROW == native single-request, TP2 and TP4 (frozen before any run)

All earlier records stay unchanged, including every QUALITY-V2 FAIL (QUALITY-V2-SYNC-TP4.md part A raw/group
endpoint 4, S = 03766/10672/01741; QUALITY-V2-SYNC-TP4-ROW.md endpoint 4, S = 5 cases) and all GATE*.md results.
Old standard: answer non-regression against batched native (QUALITY-V2 endpoints). New standard (this file): with
ROW enabled, MuxPD output equals the same-configuration native single-request output. Default mode is no longer
required to match batched native per case or per token; ROW stays default-off and no ROW performance is measured.

## Gate
For every one of the 200 frozen-v2 cases, the complete output token_ids array of each ROW run equals that of each
native single-request run. Pass = 200/200 for every ROW x native pair. Text and finish_reason agreement are reported.
If repeated native single-request runs differ from each other, all runs are reported, none is selected, and the
first differing token of each mismatch is analysed before any further experiment. Multi-device reduction is a
hypothesis to test, not an assumed cause.

## Part A - TP4 (gating)
- Code: MuxPD 43fcaf4 (clean), Core d0f22d2, Ascend c182a9e: identical to sv2-w1/sv2-w2.
- Inputs: frozen-v2 plan dd3e1f30... (200 cases, same prompt token ids), sampling temperature 0, seed 42,
  max_tokens 8192, ignore_eos false. HCCL_DETERMINISTIC=true, VLLM_BATCH_INVARIANT unset, devices 0-3.
- Native engine args: root/native-quality-fusion-engine-args.json unchanged (TP4+EP, max-num-seqs 8,
  FULL_DECODE_ONLY [1,2,4,8], chunked prefill on), the same file used by sv2-n1..n4.
- Driver: quality/run_quality_solo.py imports the frozen run_quality.py (389ab14e..., verified at runtime) and reuses
  its validate/enqueue/receipt/identity code; the only change is admission: one case enqueued, drained, then the
  next, with at most one request in the scheduler asserted on every step. Controller root/run_quality_controlled_solo.py
  differs from run_quality_controlled_v3.py only in driver name, explicit bounds and device argument (diff archived).
- Bounds (fixed now, from TP2 solo decode rate 0.0126 s/token and 211k output tokens per arm, ~45 min expected):
  collection wall 9000 s, controller deadline 9300 s, engine steps 1,700,000 (>= 200 x 8193, not binding by
  construction). Not raised after the fact; a run exceeding them is reported with its partial outputs.
- Runs: native single-request ns4-1, ns4-2 sequential. A run failing before output is relaunched once, failure kept.
- ROW reused: sv2-w1, sv2-w2 (same code, args file raw-row-fusion-engine-args.json, devices, environment, plan;
  token-identical to each other). They were collected with batched seed-first admission (8 per batch).
- Known differences recorded: native uses chunked prefill (validated requirement of the frozen driver) and MuxPD does
  not; prompts never exceed max_num_batched_tokens 16384, so native prefill is one step per request.

## Part B - TP2 token-level recheck (gating; existing TP2 evidence qf-row == native-solo200 is text-only, max_tokens 2048)
Same plan, sampling and environment; devices 0,1; engine args = the TP4 files with tensor_parallel_size 2
(root/*-tp2-engine-args.json). The frozen plan pins runtime tensor_parallel_size 4, so the driver takes
--tp-override 2, applied after the plan digest check and visible in run.json engine_args. Runs after part A:
ns2-1 native single-request (solo admission) and rw2-1 ROW with --admission batched, which calls the frozen
run_quality.run unchanged (seed-first batches of 8, as sv2-w1). Same bounds as part A.
The old text-only evidence is kept and reported; it is not converted into a token result.

Revision note (2026-11-07, before any run of this file): the first draft ran TP2 dry-runs through the unmodified
frozen driver, which rejected tensor_parallel_size 2; part B was changed to the override above and all hashes redone.

## Closure
#23 is closed only if A and B both pass. Otherwise it stays open with the first differing cases/positions listed.
