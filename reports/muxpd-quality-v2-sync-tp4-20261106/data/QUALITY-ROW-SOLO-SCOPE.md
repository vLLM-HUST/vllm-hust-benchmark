# Scope change to QUALITY-ROW-SOLO.md (maintainer decision, 2026-11-07, during part A ns4-2)

QUALITY-ROW-SOLO.md and its hashes are unchanged; this note records a scope change made after ns4-1 finished
(rc=0) and while ns4-2 was running. No result of part A had been compared with ROW when it was made.

- Part B (TP2 token-level recheck: ns2-1, rw2-1) is cancelled and never ran. The maintainer accepts the existing
  TP2 evidence as meeting the project quality requirement: qf-row (ROW, MuxPD 43fcaf4) equals native-solo200 in
  output text and finish reason on 200/200 cases (quality/qf-row-check.txt). That evidence is text-level only
  (max_tokens 2048, no token arrays retained); it is not reported as a token-level verification.
- Part A (TP4) is unchanged: ns4-1 and ns4-2 native single-request vs ROW sv2-w1, sv2-w2, complete token_ids
  arrays, every pair 200/200.
- Mechanism: run_row_solo.sh (already started and hash-checked) was not edited; directories runs/ns2-1,
  ns2-1-retry, rw2-1, rw2-1-retry were pre-created so the controller stops at mkdir before any device use.
  The resulting nonzero controller exits in logs-row-solo-driver.log are this cancellation, not run failures.
- #23 closure: TP2 by the accepted text/finish evidence, TP4 by this round's token-level result.
