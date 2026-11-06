# QUALITY-V2 addendum: TP4 re-check (frozen 2026-11-06, before any TP4 quality request)

Applies QUALITY-V2.md unchanged (plan fixed200, harness/scorer, concurrency 16, primary rule = gold correct and valid
and transport ok and not length-limited, endpoints 1-5, verdict rule), except where stated below. TP2 results are
not used as TP4 evidence. Earlier gate records (GATE*.md, QUALITY-V2.md TP2 result) are unchanged.

## Configuration
Qwen3.5-35B-A3B BF16, **TP4+EP**, devices 0,1,2,3 (all four cards; runs are sequential), max-model-len 16384,
max-num-seqs 16, no prefix cache, FULL_DECODE_ONLY graphs [1,2,4,8,16], same configuration as the existing formal TP4
performance arms (runs/native-s16-tp4, runs/muxovl-s16-tp4).
- Candidate C: `scripts/serve-qwen35-muxwise.sh` (decode_attention_layout default tnd) at MuxPD b276a4e.
- Native N: tools/serve-native.sh, async scheduling, chunked prefill, max-num-batched-tokens 16384.
- Runner tools/quality_arm_tp4.sh (= tools/quality_arm.sh with TP 4 on devices 0,1,2,3).

## Runs (pre-registered)
4 candidate + 4 native fresh runs, order N1 C1 C2 N2 N3 C3 C4 N4 (balances drift over time). No existing TP4 native
quality run exists, so all native runs are fresh. Start failure before any request: relaunch once, failure kept.
A run that fails after requests are sent counts as collected.

## Endpoints (same thresholds as QUALITY-V2)
1. acc(C) - acc(N) >= -2.0 pp. 2. trunc(C) - trunc(N) <= +2.0 pp. 3. 0 candidate transport failures.
4. S (wrong in all 4 C runs, right in all 4 N runs) <= 2, every case listed.
5. Reported: paired case bootstrap 95% CI (4000 resamples, seed 0), majority right->wrong / wrong->right, per-run
   numbers, shared-scorer numbers, TP2-watched cases (04748, 06908, 11119, 00168).
The -2.0 pp margin and S <= 2 are kept from QUALITY-V2 (calibrated on TP2 native noise); TP4 native noise is
reported after collection as a check, and does not change the thresholds.

## Performance
Reuse the completed formal TP4 collections (not repeated): results/tp4/native-s16 and results/tp4/muxovl-s16, 3 passes
each, same days and cards, MuxPD 9f6bbe4. Difference from b276a4e on the default path: request-level rejection of
logprobs/structured-output requests (87c277c; the plans send none) and opt-in-only decode layouts; no change to the
measured computation or scheduling. Rule: tools/perf_v2_compare.py (every candidate pass better/worse than every
native pass).

## Scope
This re-checks the current asynchronous MuxPD candidate at TP4. It does not re-run the earlier TP4 synchronous
raw/group protocol of issue #23 (8 sequences, 8192 cap, other case set); conclusions are not transferred to it.

## Result (recorded 2026-11-06 after collection; text above unchanged, sha256 in QUALITY-V2-TP4.sha256)
Scored by tools/quality_v2_tp4_score.py (quality/qv2tp4-score.txt; same code as the TP2 scorer with TP4 run
lists). All 8 runs completed on first launch, MuxPD b276a4e clean, TP 4 on devices 0-3. Server-log tracebacks
(t4-c2, t4-c4, t4-n2) occur after tools/stop.sh SIGTERM.

| Endpoint | Candidate (4 runs) | Native (4 runs) | Delta | Threshold | Result |
|---|---|---|---|---|---|
| 1 Accuracy (primary rule) | 76.38% (150/152/154/155) | 76.62% (152/151/156/154) | -0.25 pp | >= -2.0 pp | PASS |
| 2 Truncation | 9.38% | 9.38% | 0.00 pp | <= +2.0 pp | PASS |
| 3 Transport failures | 0 | - | - | 0 | PASS |
| 4 Systematic losses S | 0 | - | - | <= 2 | PASS |

Reported: case bootstrap 95% CI [-1.75, +1.25] pp; majority right->wrong 3 (05094, 07288, 07775), wrong->right 1
(04871); shared scorer -0.25 pp. Watched cases: 04748 right 4/4 vs 4/4 (the TP2 systematic loss does not occur at
TP4); 11119 2/4 vs 1/4; 00168 0/4 vs 0/4; 06908 4/4 vs 4/4. TP4 native run SD 1.11 pp (TP2 calibration assumed 1.6).
Unlike TP2, TP4 candidate runs vary as much as native (pairwise flips 4-11 vs 2-9).

Post-hoc uncertainty (tools/quality_v2_uncertainty.py, quality/qv2-uncertainty.txt; added after the independent
statistics review, not gating): two-level bootstrap 95% [-2.62, +2.00] pp, share of resamples with delta < -2 pp = 7%; run-level
Welch t 95% [-2.17, +1.67] pp.

Performance (results/perf-v2-tp4-native-vs-default.txt; existing 3+3 passes, same days and cards, MuxPD 9f6bbe4):
57 gains, 15 regressions under the every-pass rule (expected false flags per direction with 3 vs 3 passes: 1/20 of
91 cells, about 5).

**Verdict: ACCEPTED under QUALITY-V2 (TP4).** Quality endpoints 1-4 pass; performance gains exist; regressions are
reported with them. Statistically this means no detectable accuracy change at TP4; a loss of up to about 2 pp is
not excluded with 95% confidence.

Wording note (2026-11-06, pre-merge review): "share of resamples below -2 pp" is a two-level
bootstrap proportion showing that a loss of that size is not excluded by the data; it is not a
probability that the true loss exceeds 2 pp and not a p-value. Numbers and verdict unchanged.
