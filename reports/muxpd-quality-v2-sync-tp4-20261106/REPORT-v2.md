# v2 addendum: opt-in ROW on the #23 TP4 synchronous protocol (2026-11-07)

REPORT-v1.md is unchanged. Plan frozen before the runs: `data/QUALITY-V2-SYNC-TP4-ROW.md` (hash in
`data/QUALITY-V2-SYNC-TP4-ROW.sha256`). Engine args `data/raw-row-fusion-engine-args.json` = the raw
recipe plus `decode_attention_layout: row` and `max_prefill_requests: 1`. Two fresh runs; the four
v1 native runs reused.

**Verdict: ROW FAILS QUALITY-V2 endpoint 4 (5 systematic losses > 2). Issue #23 stays open.**

| Endpoint            | ROW (2 runs)                      | Native (4 runs) | Delta   | Threshold   | Result   |
| ------------------- | --------------------------------- | --------------- | ------- | ----------- | -------- |
| 1 Accuracy          | 83.0%                             | 82.5%           | +0.5 pp | >= -2.0 pp  | PASS     |
| 2 Truncation        | 4.5%                              | 4.5%            | 0.0 pp  | \<= +2.0 pp | PASS     |
| 3 Failures          | 0                                 | -               | -       | 0           | PASS     |
| 4 Systematic losses | 04069, 06620, 01310, 05393, 05579 | -               | -       | \<= 2       | **FAIL** |

- ROW answers the three raw losses (03766, 10672, 01741) correctly, but loses five different cases.
- ROW runs are token-identical to each other; ROW vs native 10/200, ROW vs raw 23/200 identical.
- The native control is batched, so a decode-row-invariant arm is not expected to match it. A TP4
  native single-request reference was not collected; whether ROW equals TP4 single-request output is
  untested.
- No ROW performance measured. Outcomes: `data/sv2-w*.outcomes.json`; score:
  `data/sv2-row-score.txt`.
