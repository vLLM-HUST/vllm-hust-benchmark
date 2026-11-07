# QUALITY-V2 applied to the issue #23 TP4 synchronous protocol, plus final-main checks (frozen 2026-11-06, before any run)

Earlier records are unchanged: the v1/v2 strict and supplementary answer-nonregression gates of the
2026-10-25 synchronous protocol stay FAIL; GATE*.md, QUALITY-V2.md and QUALITY-V2-TP4.md stay as recorded.
This file is written and hashed (QUALITY-V2-SYNC-TP4.sha256) before any run listed here starts.

Code under test: MuxPD main 43fcaf4 (after refactor PR #35), Core d0f22d2, Ascend c182a9e (the Ascend
revision used by all QUALITY-V2 runs; the 2026-10-25 protocol ran on Ascend 03766ac, so new runs are not
byte-comparable to it and old outputs are never mixed into new arms).

## Part A - TP4 synchronous raw/group protocol under QUALITY-V2 (gating for #23)

Inputs, admission and runtime: exactly the archived protocol `muxpd-quality-protocol-v1`, `frozen-v2`
(plan dd3e1f30..., 200 MMLU-Pro cases, 25 fixed batches of 8, seed-first InprocClient admission,
temperature 0, seed 42, max_tokens 8192), unchanged driver run_quality.py (389ab14e...) and scorer
score_quality.py (157b52ff...), controller run_quality_controlled_v3.py, engine args
native-quality-fusion / raw-fusion / group-fusion (TP4+EP, max-num-seqs 8, FULL_DECODE_ONLY [1,2,4,8],
npugraph_ex/static kernel off, synchronous), HCCL_DETERMINISTIC=true in every arm (as 2026-10-25).
Raw = default synchronous raw-P q4 isolated-overlap recipe; group = opt-in compile_prefill_groups.

Runs (pre-registered, sequential on devices 0-3): N1 R1 G1 R2 N2 N3 R3 G2 R4 N4
(native 4, raw 4, group 2). A run that fails before producing output is relaunched once and the failure kept;
any run that produced outputs is counted as collected. No case removal, retry or cap change.

Scoring: primary correct = the frozen scorer's `correct` (complete natural stop, valid in-range letter equal to
gold; truncation/invalid/error = wrong). Endpoints and thresholds are QUALITY-V2's, unchanged:
1. acc(raw) - acc(native) >= -2.0 pp
2. trunc(raw) - trunc(native) <= +2.0 pp
3. 0 errors / incomplete collections in raw runs
4. S = cases wrong in all raw runs and correct in all native runs, S <= 2 (every case listed)
Raw is gating. Group is evaluated with the same four endpoints using its 2 runs (S over 2 group runs is a weaker
filter; stated with the result) and is reported separately; it is an opt-in experiment, not the default.
Reported, not gating: per-run counts, cross-run token equality (determinism), the earlier losses
03766/10672/01741 and gains, mixed P/D coverage, strict v1 veto outcome (for information only; it is not the
acceptance rule of this round), two-level bootstrap and run-level Welch intervals as in QUALITY-V2.
Note stated before collection: under HCCL_DETERMINISTIC repeated fresh runs may be identical; then repetition
adds no noise filtering and S equals the per-case loss count. Thresholds are not changed because of this.

## Part B - final-main device checks (not #23 gates; regression checks of the refactor)

B1 Default asynchronous mode, TP2 (scripts/serve-qwen35-muxwise.sh at 43fcaf4, tools/quality_arm.sh settings,
  fixed200 c16): 2 fresh runs qf-c1@0,1, qf-c2@6,7(logical 2,3) in parallel. Compared with the 5 QUALITY-V2 native
  runs using the QUALITY-V2 endpoints and thresholds (S over 2 runs). Pass = all four endpoints pass.
  04748 is reported; it is expected to stay a loss in default mode.
B2 ROW opt-in, TP2 (scripts/serve-qwen35-muxwise-row.sh at 43fcaf4, as tools/row_recheck.sh): 1 run qf-row.
  Expectation recorded now: text identical to native-solo200 on 200/200 cases (as row-q1/q2 and rowfix-q1).
  Pass = 200/200 identical; any difference is reported with the first differing cases. Quality/function only,
  no ROW performance measurement.
B1 and B2 run before part A.

## Performance
No new performance runs. This round changes no plugin code unless a defect is found; refactor PR #35 changes
no default-mode numeric or stream path (independent review). Existing default-mode results remain the
reference (QUALITY-V2 / QUALITY-V2-TP4 reports, synchronous v4/v5 reports). If a code fix becomes necessary,
its performance impact is assessed separately before any rerun.

## Closure rule for #23 (fixed now)
#23 may be closed only if Part A raw passes all four endpoints, B1 passes and B2 passes. Otherwise #23 stays open
and the failing endpoint is the stated missing evidence. TP2 04748 stays documented as a reproducible default-mode
systematic loss within the accepted S <= 2 tolerance with ROW as its verified opt-in mitigation; batch-dependent
numerics stay documented as a default-mode limitation. Group failing does not block closure (opt-in); it is
reported.

## Result (recorded 2026-11-06 after collection; text above unchanged, sha256 in QUALITY-V2-SYNC-TP4.sha256)
Scored by tools/quality_v2_sync_score.py (quality/sv2-score.txt). All 10 part A runs completed on first launch
(controller passed, exit 0), MuxPD 43fcaf4 clean, HCCL_DETERMINISTIC=true, devices 0-3.

Part A, raw (gating), 4 raw vs 4 native runs:
| Endpoint | Raw | Native | Delta | Threshold | Result |
|---|---|---|---|---|---|
| 1 Accuracy | 83.0% (166 x4) | 82.5% (165 x4) | +0.5 pp | >= -2.0 pp | PASS |
| 2 Truncation | 3.5% | 4.5% | -1.0 pp | <= +2.0 pp | PASS |
| 3 Failures | 0 | - | - | 0 | PASS |
| 4 Systematic losses S | 3: 03766, 10672, 01741 | - | - | <= 2 | **FAIL** |
Group (opt-in, 2 runs): identical numbers and the same S; FAIL.
As stated before collection, every arm is token-identical across its fresh runs (native 4/4, raw 4/4, group 2/2,
200/200 each), so repetition gave no noise filtering and S equals the per-case loss count. Raw vs native complete
token arrays equal on 11/200 cases. The losses and the four gains (06074, 06741, 00882, 08107) are exactly those
of the 2026-10-25 protocol on Ascend 03766ac: the result reproduces on final main after #30/#32/#35.
Mixed P/D coverage: raw/group all 25 batches covered (native has no MuxPD receipts by design).
Part B (run earlier): B1 default TP2 at 43fcaf4 PASS (S = 04748 only), B2 ROW TP2 200/200 = native solo PASS.

**Verdict: Part A raw FAILS endpoint 4. Under the closure rule fixed above, #23 stays open.**
Thresholds are not changed after the fact.
