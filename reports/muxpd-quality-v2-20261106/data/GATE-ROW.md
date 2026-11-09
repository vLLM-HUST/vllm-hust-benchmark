# Gate addendum: exact per-row decode attention candidate (frozen 2026-11-06, before any ROW 200-case request)

Applies GATE.md unchanged (plan fixed200, harness, concurrency 16, endpoints, no retries/removal/cap change).
Scoring: tools/gate_score.py, both rules reported; GATE rule (truncation counts wrong) is authoritative per
the 2026-11-04 correction.

- Candidate: MuxPD `perf/muxwise-async-scheduling` @ 87c277c plus the uncommitted opt-in change
  (src/vllm_muxpd/decode_attention.py with layout "row", config.py, worker.py; sha256 recorded in
  runs/row-q1/identity.txt + quality/row-sources.sha256), preset `tools/serve-mux-row.sh` =
  `scripts/serve-qwen35-muxwise.sh` policy + `"decode_attention_layout":"row","max_prefill_requests":1`,
  max-num-seqs 16, graphs [1,2,4,8,16]. Runner `tools/quality_arm_row.sh` (quality_arm_bsh.sh + row arm).
- Reference: existing fresh native-q1/native-q2.
- Runs: row-q1 (devices 0,1) and row-q2 (devices 2,3, VLLM_PORT=29700), fresh processes, launched together.
  Verdict uses row-q1; row-q2 is a pre-registered repeat reported alongside.

Evidence that motivated the candidate (diagnostic only, not acceptance evidence):
- inv-c16/inv-c1 (BSH + max_prefill_requests 1, graph): c16 == c1 text 200/200, but BSH != native M=1 numerics;
  04748 answers A in both.
- exact-c16 (eager, fenced, per-row TND via tools/fiasplit, max_prefill_requests 1): text == native-solo200
  200/200 at concurrency 16; gate rule 154 correct, 0 regressions. ctl-c16 (same without split): 1 regression.
- row layout (graph) 04748 probes K=0/1/9/15 and co-prefill K=3/7/15: all token-identical, == native solo text.

Known cost recorded before collection: per-row attention adds per-layer host updates in graph mode
(single-kernel probe: ~3.5 ms per 16-row layer update vs ~0.3 ms batched; 10 full-attention layers), and one
prompt per prefill forward. Performance is measured after the quality runs and reported regardless of outcome.

## Result (recorded 2026-11-06 after collection; gate text above unchanged)
Scored by tools/gate_score.py (quality/row-gate.txt) against native-q1/native-q2.
- row-q1, GATE rule (authoritative): correct 154, regressions 0. **Strict non-regression: PASS. Floor (>=150): PASS.**
- row-q1, shared scorer rule (reported): correct 157, 1 regression (00168, length-limited at 2048, letter C).
  Strict FAIL, floor (>=157) PASS. Native solo (native-solo200) has exactly the same 00168 output and outcome.
- row-q2 (pre-registered repeat): identical numbers under both rules.
- Exact text: row-q1 == row-q2 == native-solo200 on 200/200 cases. invalid 1, length-limited 18 (= native solo).
- Server-log tracebacks (1 per run) are logged after tools/stop.sh SIGTERM, after score.json was written.
Verdict: the ROW candidate PASSES this round's gate (authoritative rule). Under the shared-scorer rule it fails
strict by one case that the unbatched native run fails identically.

## Performance (recorded 2026-11-06, tools/perf_row.sh, results/perf_row.log, results/tp2bsh/aggregate_row.json)
Same 13 formal plans x 2 passes, cards 0,1, compared with muxovl-s16-now (TND) and muxbsh-s16 from 2026-11-05.
- Output throughput vs TND: burst -6%, mixed-r1.0 -2%..-8%, contend -28%, mixed-r1.6 -34%..-39%, short-r2/r4 -44%..-45%.
- TPOT p50: about 1.6x-2.3x TND (e.g. contend 0.021 -> 0.033 s, mixed-r1.6 0.043 -> 0.095 s).
- TTFT p50: burst 1.2 -> 3.7 s, mixed-r1.6 0.8 -> 25-43 s, short-r4 105 -> 212-218 s; goodput collapses under load.
- Errors 0 in all 52 runs.
The two cost sources (per-row attention host updates; one prompt per prefill forward) were not separated.
Conclusion: ROW passes the quality gate but is NOT acceptable as the formal performance configuration.

Post-run note (2026-11-06): after all ROW quality and perf runs, the decode_attention.py module docstring was extended to describe "row" (comment-only; sha256 now differs from quality/row-sources.sha256).

Correction (2026-11-06, after recomputation from aggregate_row.json; the measurements are unchanged): the
"Performance" section above understates two ranges. TPOT p50 is 1.6x-3.0x TND (mixed-r1.0-s2 3.0x), not
1.6x-2.3x; TTFT p50 is 1.2x-45x TND (contend 1.2x, burst 3.2x, mixed-r1.6 31-45x), and contend throughput is
-27% to -28%. The original lines are kept.

Version note (2026-11-06, pre-merge review): every ROW performance number in this file and in the
benchmark report was measured on b276a4e, before the 9b0641f padding fix. 9b0641f was re-checked
for quality only (rowfix-q1, 200/200 identical to native solo); its performance was not measured.
