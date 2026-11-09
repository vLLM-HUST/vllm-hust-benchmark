# Opt-in ROW mitigation on the issue #23 TP4 synchronous protocol (frozen 2026-11-06, before any ROW run)

Earlier records are unchanged: QUALITY-V2-SYNC-TP4.md part A raw FAIL (endpoint 4, S = 03766, 10672, 01741) stays
as recorded, as do all earlier gates. ROW stays default-off; this validates an opt-in mitigation only. This file is
hashed (QUALITY-V2-SYNC-TP4-ROW.sha256) before any run listed here starts. No ROW performance is measured.

## Applicability (checked on CPU before freezing)
- Engine args root/raw-row-fusion-engine-args.json = raw-fusion-engine-args.json plus
  `decode_attention_layout: "row"` and `max_prefill_requests: 1` (the same two settings as the shipped TP2
  preset scripts/serve-qwen35-muxwise-row.sh). Everything else (synchronous decoupled, isolated overlap, q4 raw P,
  decode graph FULL_DECODE_ONLY [1,2,4,8], max-num-seqs 8, fusion backend) is unchanged.
- MuxPDConfig accepts it (row requires execution=decoupled); the frozen driver run_quality.py dry-run validates it
  as role raw. ROW patches only DecodeOnly attention calls, which is the D lane of this worker; P is passed through.
- ROW has not run at TP4 before; its applicability is established only by this run.

## Runs (pre-registered)
MuxPD main 43fcaf4, Core d0f22d2, Ascend c182a9e, HCCL_DETERMINISTIC=true, devices 0-3, same frozen-v2 plan,
driver, scorer and controller as part A, unchanged bounds (8192 cap, 10800 s collection bound, 11100 s controller).
Two fresh ROW runs, sequential: sv2-w1, sv2-w2. A run that fails before producing output is relaunched once and the
failure kept. A run that exceeds the frozen bounds or errors counts as collected with its failures (endpoint 3);
bounds are not raised.

## Native control reuse
The four part A native runs sv2-n1..sv2-n4 are reused, not repeated: same commits, engine args, environment and
devices, collected the same day, and token-identical to each other on 200/200 cases, so fresh native runs under
this deterministic configuration would add no information. No other native run is mixed in.

## Endpoints (QUALITY-V2, thresholds unchanged), scored by tools/quality_v2_sync_score.py
C = sv2-w1, sv2-w2; N = sv2-n1..sv2-n4.
1. acc(C) - acc(N) >= -2.0 pp  2. trunc(C) - trunc(N) <= +2.0 pp  3. 0 failures in C
4. S = wrong in both C runs and right in all 4 N runs, S <= 2, every case listed.
If the two ROW runs are token-identical, S has no noise filtering (as in part A); thresholds are not changed.
Reported, not gating: per-case outcome of the 3 part A losses (03766, 10672, 01741) and 4 gains (06074, 06741,
00882, 08107); ROW run-to-run token equality; ROW vs raw and vs native token equality; mixed P/D coverage;
both uncertainty intervals.

## Closure rule for #23 (fixed now)
If the ROW arm passes endpoints 1-4, #23 may be closed with the scope: "opt-in ROW mitigation verified at TP2 and
in the TP4 synchronous protocol; default synchronous raw/group still FAIL endpoint 4 and batch-dependent numerics
remain a documented default-mode limitation; TP2 04748 stays a default-mode loss within tolerance."
If any endpoint fails, #23 stays open; results are kept and the failing endpoint and cases are investigated.

## Result (recorded 2026-11-07 after collection; text above unchanged, sha256 in QUALITY-V2-SYNC-TP4-ROW.sha256)
Scored by tools/quality_v2_sync_score.py (quality/sv2-row-score.txt). sv2-w1, sv2-w2 completed on first launch, exit 0,
within the frozen bounds; MuxPD 43fcaf4, HCCL_DETERMINISTIC=true.

| Endpoint | ROW (2 runs) | Native (4 runs, reused) | Delta | Threshold | Result |
|---|---|---|---|---|---|
| 1 Accuracy | 83.0% (166 x2) | 82.5% (165 x4) | +0.5 pp | >= -2.0 pp | PASS |
| 2 Truncation | 4.5% | 4.5% | 0.0 pp | <= +2.0 pp | PASS |
| 3 Failures | 0 | - | - | 0 | PASS |
| 4 Systematic losses S | 5: 04069, 06620, 01310, 05393, 05579 | - | - | <= 2 | **FAIL** |

- ROW recovers all three part A raw losses (03766, 10672, 01741 correct in both runs) and keeps three of the four raw
  gains (06741, 00882, 08107; 06074 wrong as in native), but loses five other cases that native answers correctly.
- The two ROW runs are token-identical (200/200). ROW vs native token-identical 10/200; ROW vs raw 23/200.
- Native here is itself batched (8 sequences, seed-first admission), so ROW (single-request numerics for decode)
  is not expected to equal it; no TP4 native single-request reference exists, so whether ROW reproduces TP4
  single-request output (as verified at TP2) is untested.

**Verdict: the ROW arm FAILS endpoint 4. Under the closure rule fixed above, #23 stays open. Thresholds unchanged.**
