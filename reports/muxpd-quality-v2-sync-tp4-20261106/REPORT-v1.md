# MuxPD QUALITY-V2 on the issue #23 TP4 synchronous protocol, and final-main checks (2026-11-06)

**Verdict: the default synchronous raw recipe FAILS QUALITY-V2 endpoint 4 (systematic losses 3 > 2).
Issue #23 stays open on this evidence.** Thresholds and the closure rule were frozen before any run
(`data/QUALITY-V2-SYNC-TP4.md`, hash in `data/QUALITY-V2-SYNC-TP4.sha256`) and are not changed. All
runs are kept. The opt-in ROW mitigation run on this protocol is pre-registered separately in
`data/QUALITY-V2-SYNC-TP4-ROW.md` (frozen before its runs) and is reported in a later version of
this report.

Previous report: `reports/muxpd-quality-v2-20261106` (QUALITY-V2 at TP2/TP4, asynchronous default
mode).

## Identity

MuxPD main `43fcaf4` (after refactor PR #35, clean tree), Core `d0f22d2`, Ascend `c182a9e`,
Qwen3.5-35B-A3B BF16, 910B2. Part A ran on devices 0-3 (TP4+EP), sequentially, with
`HCCL_DETERMINISTIC=true` in every arm, as the 2026-10-25 protocol did. The 2026-10-25 runs used
Ascend `03766ac`; their outputs are not mixed into these arms.

## Part A: TP4 synchronous raw/group protocol under QUALITY-V2 (gating)

Unchanged archived protocol `muxpd-quality-protocol-v1` frozen-v2 (200 MMLU-Pro cases, 25 batches of
8, seed-first InprocClient admission, temperature 0, seed 42, 8192-token cap), unchanged driver and
scorer, engine args in `data/*-engine-args.json`. Order N1 R1 G1 R2 N2 N3 R3 G2 R4 N4; all 10 runs
completed on first launch. Primary correct = frozen scorer: complete natural stop with a valid
letter equal to gold; truncation, invalid and missing count wrong.

| Endpoint            | Raw (4 runs)        | Native (4 runs)  | Delta   | Threshold   | Result   |
| ------------------- | ------------------- | ---------------- | ------- | ----------- | -------- |
| 1 Accuracy          | 83.0% (166 each)    | 82.5% (165 each) | +0.5 pp | >= -2.0 pp  | PASS     |
| 2 Truncation        | 3.5%                | 4.5%             | -1.0 pp | \<= +2.0 pp | PASS     |
| 3 Failures          | 0                   | -                | -       | 0           | PASS     |
| 4 Systematic losses | 03766, 10672, 01741 | -                | -       | \<= 2       | **FAIL** |

Group (opt-in, 2 runs): identical counts and the same three losses; FAIL.

- Every arm is token-identical across its own fresh runs (native 4/4, raw 4/4, group 2/2, 200/200
  cases), as anticipated in the frozen plan. Repetition therefore gives no noise filtering and S
  equals the per-case loss count; this was stated before collection and the threshold was not
  adjusted.
- Raw and native complete token arrays are equal on 11/200 cases.
- The three losses and the four gains (06074, 06741, 00882, 08107) are exactly those of the
  2026-10-25 protocol: the earlier result reproduces on final main.
- Raw/group mixed P/D coverage: all 25 batches covered.
- For information only (not this round's rule): the strict v1 veto also fails on the same three
  cases.

Per-run, per-case outcomes with token-array digests: `data/sv2-*.outcomes.json`; full score output:
`data/sv2-score.txt`.

## Part B: final-main checks at TP2 (regression checks of the refactor)

These replace PR #35's device smoke test, which covered the intermediate commit `dc64420` only.

- B1, default asynchronous mode (`scripts/serve-qwen35-muxwise.sh`), 2 fresh runs vs the 5
  QUALITY-V2 native runs: accuracy -0.8 pp, truncation -0.3 pp, 0 transport failures, S = {04748}.
  **PASS.** Two-level bootstrap 95% [-3.10, +1.50] pp (a loss above 2 pp is not excluded).
  `data/qf-b1-score.txt`.
- B2, ROW opt-in (`scripts/serve-qwen35-muxwise-row.sh`): text and finish reason equal native
  single-request output on 200/200 cases; 04748 correct. **PASS.** `data/qf-row-check.txt`. Quality
  only; no ROW performance measured.

## Status per item

| Item                                     | Default mode under QUALITY-V2                      | Verified opt-in mitigation       | Remaining limitation          |
| ---------------------------------------- | -------------------------------------------------- | -------------------------------- | ----------------------------- |
| TP4 synchronous raw/group (#23 protocol) | FAIL (endpoint 4)                                  | Not yet (ROW run pre-registered) | 3 reproducible losses         |
| TP2 04748                                | Within tolerance (S \<= 2)                         | ROW (TP2)                        | Default-mode loss persists    |
| Asynchronous default mode TP2/TP4        | PASS (earlier report; TP2 rechecked on final main) | -                                | Loss up to ~2 pp not excluded |
| Batch-dependent numerics                 | Present                                            | ROW removes it where verified    | Kernel-level, also in native  |

## Performance

No new performance runs. No plugin code changed in this round; refactor PR #35 changes no
default-mode numeric or stream path. Earlier reports remain the performance reference.
