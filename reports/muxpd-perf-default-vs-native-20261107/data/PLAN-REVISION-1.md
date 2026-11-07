# MuxPD#36 plan revision 1 (maintainer decision 2026-11-07 07:06 UTC, during TP2 MuxPD pass1)

PLAN.md and PLAN.sha256 are unchanged and kept; this file supersedes their arms and pass count.

- All four configurations are measured fresh in this round, one pass of the 13 frozen formal plans each:
  TP2 native, TP2 MuxPD default, TP4 native, TP4 MuxPD default. Historical native/MuxPD results are reference only.
- Single-pass comparison: no pass-to-pass variance is available; differences are reported as measured, not as
  significant gains/regressions. The report labels it a single-run comparison.
- Kept from the running round: TP2 MuxPD pass1 plans already completed (burst-s1, burst-s2) and the in-flight
  contend-s1 (allowed to finish on the same server). Remaining TP2 MuxPD pass1 plans are filled on the same server
  process (run_passes skips existing results). Passes 2-3 are cancelled and never ran.
- Same NPUs per TP, run one at a time with nothing else on them (keepalive stopped):
  TP2 cards 0,1 (MuxPD, then native); TP4 cards 0-3 (MuxPD, then native). Port 8300.
- Args: max-num-seqs 16, FULL_DECODE_ONLY graphs [1,2,4,8,16] mode 3, seed 0 for both arms.
  MuxPD: scripts/serve-qwen35-muxwise.sh (43fcaf4, clean; no chunking, full 16K budget by design).
  Native: tools/serve-native.sh --max-num-batched-tokens 16384 (chunked prefill and async scheduling at defaults).
  Difference from history: native TP2 now on cards 0,1 (was 2,3) with explicit mode 3 (was the default mode).
- SLO/metrics/loadgen/plans unchanged (tools/aggregate.py slo(), tools/loadgen.py, plans/formal SHA256SUMS).
- Results: results/perf36-tp{2,4}/{mux43-s16,native-s16}/pass1. A failed plan stays failed; an arm that fails to
  start is relaunched once.
- Earlier records kept: runs/mux43-tp2-aborted-keepalive (aborted before any plan, see NOTES.md).
