# MuxPD#36: default mode vs native, TP2 and TP4, performance only (frozen 2026-11-07, before any run)

No answer-quality gate. ROW not tested. No TP8 / other models.

## Load, SLO and metrics (unchanged existing definitions)
- Plans: plans/formal/*.json (13 plans, verified against plans/formal/SHA256SUMS), replayed by tools/loadgen.py via
  tools/run_passes.sh, unchanged.
- SLO (tools/aggregate.py slo()): TTFT <= 1.0 s (long requests 3.0 s), TPOT <= 0.1 s, max inter-token gap <= 0.5 s;
  Goodput = fraction of requests meeting all three. Not changed.
- Metrics: out_tok_s, goodput_frac, ttft p50/p99, tpot p50/p99, itl p50/p99, worst_stall, maxgap p90, errors.
- Rule (tools/perf_v2_compare.py): GAIN/REG only if every candidate pass beats/loses to every native pass; all cells
  reported.

## Arms
- Native, reused (no MuxPD code loaded; Core d0f22d2, Ascend c182a9e unchanged): results/tp2/native-s16 (3 passes,
  cards 2,3, 2026-11-04) and results/tp4/native-s16 (3 passes, cards 0-3, 2026-11-04). tools/serve-native.sh,
  max-num-seqs 16, max-num-batched-tokens 16384, graphs [1,2,4,8,16].
- MuxPD default, new, main 43fcaf4 (clean): scripts/serve-qwen35-muxwise.sh --served-model-name qwen3.5-35b --seed 0
  --max-num-seqs 16 --compilation-config graphs [1,2,4,8,16] (identical argv to the historical arms).
  - TP2: label mux43-s16, cards 0,1, port 8300, 3 passes -> results/perf36-tp2/mux43-s16
  - TP4: label mux43-s16, cards 0-3, port 8300, 3 passes -> results/perf36-tp4/mux43-s16
  Sequential, TP2 then TP4. Historical MuxPD arms (87c277c dirty TP2, 9f6bbe4 TP4) are superseded for this comparison
  because #32/#35 changed the default async path; they stay on disk and are reported for reference only.
- HCCL_DETERMINISTIC unset (as in all historical perf arms); NPU keepalive stopped during runs, restored after.

## Known differences recorded now
- Native TP2 ran on cards 2,3 possibly concurrently with a MuxPD arm on 0,1; the new TP2 MuxPD arm runs alone.
- Native data is 3 days older than the candidate (same software stack and host).
- A failed plan in a pass is kept as failed; no reruns except a whole arm that fails to start (relaunch once).
