# Quality/performance acceptance v2 (frozen 2026-11-06, before any v2 run)

Status of earlier gates: GATE.md, GATE-BSH.md and GATE-ROW.md and their PASS/FAIL results stay as recorded.
They are not re-scored or re-labelled under this standard.

## Why the standard changes
- The earlier strict endpoint ("no wrong answer on a case both fresh native runs answered correctly") treats native
  as ground truth. Native is a baseline, not an oracle: fresh native c16 runs on the same config disagree with
  each other (native-q3: 3 such losses vs native-q1/q2; native-det-q2: 2), and the root-cause work showed the
  differences come from batch-composition-dependent kernel numerics that native also has (ROOT-CAUSE.md,
  GATE-ROW.md). A single disagreeing case is therefore not evidence of a plugin defect.
- Token or text equality with native is not required.
- Main goal from now on: a performance gain at an acceptable, pre-declared quality change, measured against the
  gold answers, with repeated runs. Reproducible systematic losses are still investigated.

## Fixed inputs
Plan `muxwise-runs/plans/fixed200` (200 MMLU-Pro cases, temperature 0, seed 17, thinking off, max_tokens 2048),
harness/scorer as in GATE.md, concurrency 16, TP2+EP BF16, max-model-len 16384, max-num-seqs 16, no prefix cache,
FULL_DECODE_ONLY graphs [1,2,4,8,16]. No case removal, no retries, no cap change.

## Scoring
Primary correctness = gold-answer correct AND valid AND transport success AND NOT length-limited (truncation counts
wrong; tools/gate_score.py rule 'gate'). The shared scorer's own `correct` is reported alongside.

## Arms and runs (pre-registered)
- Candidate C: MuxPD default mode = `scripts/serve-qwen35-muxwise.sh` (decode_attention_layout tnd) at the
  committed branch head b276a4e (computation identical to 87c277c; b276a4e only adds opt-in code).
  4 fresh runs: qv2-c1..qv2-c4.
- Native N: tools/serve-native.sh, async scheduling, chunked prefill, max-num-batched-tokens 16384.
  2 fresh runs (qv2-n1, qv2-n2) + the 3 existing default-environment native c16 runs native-q1/q2/q3
  (same commits/config) = 5 native runs. native-det-* (HCCL_DETERMINISTIC) are context only.
- Schedule (device pairs alternate to spread card effects): round A c1@0,1 + n1@2,3; round B n2@0,1 + c2@2,3;
  round C c3@0,1 + c4@2,3. Every run is reported; a run that fails to start before any request is relaunched once
  and the failure kept; a run that fails after requests are sent counts as collected with its transport failures.
- Context only (not in the verdict): muxopt-q1/q2, bsh-q1/q2, row-q1/q2.

## Quality endpoints (all computed on the primary rule)
Let acc(X) = mean correct fraction over the runs of arm X, trunc(X) = mean length-limited fraction.
1. **Accuracy (gating):** acc(C) - acc(N) >= -2.0 percentage points (= -4 cases of 200).
2. **Truncation (gating):** trunc(C) - trunc(N) <= +2.0 pp.
3. **Transport (gating):** 0 transport failures in all candidate runs.
4. **Systematic loss (gating):** S = cases wrong in all 4 candidate runs and correct in all 5 native runs.
   S <= 2. Every case in S is listed; S > 2 fails and triggers investigation.
5. Reported, not gating: paired case bootstrap 95% CI of acc(C)-acc(N) (4000 resamples, seed 0); per-run
   correct/trunc/invalid; per-case right->wrong and wrong->right counts by majority vote; shared-scorer numbers;
   cases that were in S-like sets for the historical gates (04748, 06908, 11119, 00168).

### Calibration (native-only data, computed before any v2 run; no candidate data used)
- Default-env/det native c16 runs: 156, 150, 150, 156, 151 correct (mean 152.6, SD 3.1 cases = 1.6 pp per run).
  Pairwise native per-case flips: 8-10 cases.
- Expected SD of acc(C)-acc(N) from run noise with 4 vs 5 runs: about 1.6*sqrt(1/4+1/5) = 1.1 pp; a -2.0 pp
  margin is about 1.9 SD, i.e. a ~3% false-fail rate when there is no real difference.
- Native-only 2-vs-3 splits of these 5 runs: S (all of 2 wrong, all of 3 right) was 0-1 in all 10 splits; with
  4 vs 5 runs a null S is expected to be smaller still, so S <= 2 tolerates noise but flags a repeated pattern.
- The margin is a product decision: up to 2 pp lower gold accuracy is accepted in exchange for a measured
  performance gain. It is not tuned to any candidate result.

## Performance endpoints
Throughput and latency on the 13 frozen formal plans (plans/formal), same TP2 config. Existing, already completed
collections are reused (not repeated): native `results/tp2/native-s16` (3 passes) and candidate default mode
`results/tp2bsh/muxovl-s16-now` (2 passes, 87c277c computation). Reported per plan: output tok/s, goodput
fraction, TTFT p50/p99, TPOT p50, ITL p99, worst stall. A gain is claimed only for plan/metric pairs where every
candidate pass is better than every native pass; regressions are reported with the same rule. The two collections
were taken on different days; this is stated with every number.

## Verdict
Accepted = quality endpoints 1-4 all pass AND at least one pre-declared performance gain exists; every regression
is reported next to it. Scope: TP2 only. TP4/TP8 and other models are not covered.

## Result (recorded 2026-11-06 after collection; text above unchanged, sha256 in QUALITY-V2.sha256)
Scored by tools/quality_v2_score.py (output quality/qv2-score.txt). All 6 new runs completed first launch; candidate
and native runs used MuxPD b276a4e, clean tree. Server-log tracebacks only after tools/stop.sh SIGTERM.

| Endpoint | Candidate (4 runs) | Native (5 runs) | Delta | Threshold | Result |
|---|---|---|---|---|---|
| 1 Accuracy (primary rule) | 75.50% (150/151/152/151) | 76.30% (153/154/156/150/150) | -0.80 pp | >= -2.0 pp | PASS |
| 2 Truncation | 9.00% | 9.30% | -0.30 pp | <= +2.0 pp | PASS |
| 3 Transport failures | 0 | - | - | 0 | PASS |
| 4 Systematic losses S | 1 (mmlu-pro-04748) | - | - | <= 2 | PASS |

Reported: paired bootstrap 95% CI of the accuracy delta [-2.75, +1.05] pp (includes the margin; with 4 vs 5 runs
the estimate cannot exclude a loss of up to ~2.75 pp). Majority-vote right->wrong 2 (04748, 04966), wrong->right 2
(07288, 11251); no case all-candidate-right/all-native-wrong. Shared-scorer accuracy 78.38% vs 78.60% (-0.22 pp).
Watched cases: 04748 wrong in 4/4 candidate runs, right in 5/5 native (reproducible; root cause in ROOT-CAUSE.md,
removed only by the opt-in row layout); 11119 0/4 vs 2/5; 00168 0/4 vs 1/5; 06908 4/4 vs 5/5.

Performance (tools/perf_v2_compare.py, results/perf-v2-native-vs-default.txt; native 3 passes 2026-11-04 vs
candidate 2 passes 2026-11-05): 60 plan/metric gains and 19 regressions under the every-pass rule. Gains: goodput
fraction on all 13 plans, worst stall on 12, output throughput on the 4 short-r and mixed-r1.6-s2 overload plans and
(<1%) burst-s1/contend; TPOT/ITL p99/TTFT on overload plans. Regressions: TTFT p50 on contend and mixed-r1.0 plans,
TPOT p50 and ITL p99 on burst/contend, output throughput -0.1..-0.2% on mixed-r1.0-s1/s3.

**Verdict: ACCEPTED under QUALITY-V2 (TP2 only).** Endpoints 1-4 pass and pre-declared gains exist; the regressions
listed above are part of the result. One reproducible loss (04748) remains and stays under investigation.
Not covered: TP4/TP8, other models, the bsh/row layouts as performance presets.

## Independent statistics review (2026-11-06, after the result above; the frozen text and verdict are unchanged)
- Every endpoint was recomputed from score.json with independent code: all numbers match; perf output reproduced.
- The -2.0 pp margin works as a point-estimate threshold, not a non-inferiority test: with the observed run SD a
  true -2 pp loss would pass about 50% of the time, -3 pp about 13%.
- Better intervals for TP2: two-level bootstrap (runs, then cases) [-3.20, +1.48] pp with 15% of resamples below -2 pp;
  run-level Welch t [-2.39, +0.79] pp. Conclusion: no detectable accuracy change; a loss above 2 pp is not excluded.
- TP2 candidate runs are nearly deterministic (run SD 0.41 pp, 0-2 case flips between runs) while native varies
  (1.30 pp, 5-11 flips); 4 candidate runs act like about one, so the realistic null for S is 1-vs-4 native, where S=2
  already occurs (native-q3). S<=2 is about right but S=3 would not be strong evidence. 04748 is a real reproducible
  difference (candidate truncated at 2048, letter I). 04966 (wrong 4/4 vs right 3/5) is not caught by S.
- 3 of 5 native runs were collected two days before the candidates; against same-day natives only the delta is
  -1.25 pp (n=2), not distinguishable from noise.
- Performance rule: with 2 vs 3 passes the per-cell false-gain rate is 10%; ~9 false gains expected over 91 cells.
  60 observed gains indicate a real overall effect; sub-1% throughput cells (burst, contend, mixed-r1.0) are not
  meaningful in either direction; native and candidate passes were collected 40 h apart (day confounded with arm).
- Recommendation adopted for future rounds: report the two-level bootstrap and run-level interval alongside
  the point estimate; collect native and candidate on the same days in alternation.

Wording note (2026-11-06, pre-merge review): "share of resamples below -2 pp" is a two-level
bootstrap proportion showing that a loss of that size is not excluded by the data; it is not a
probability that the true loss exceeds 2 pp and not a p-value. Numbers and verdict unchanged.
