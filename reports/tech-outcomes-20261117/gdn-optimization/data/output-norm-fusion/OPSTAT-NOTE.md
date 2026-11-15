# op_statistic.csv — new collection (2026-11-15), NOT the original file

The original `op_statistic.csv` mentioned in `gdn_split_fusion/REPORT.md`
was never committed and cannot be regenerated. The CSV files here are
**new measurements** taken in this round and are labelled as such:

- `op-statistic-20261115-8prompt-{off,on}.csv`: exported with
  `msprof --export=on --summary-format=csv` from the torch_npu profile
  trees used for the fused-kernel engagement check (rank 0, 8 prompts,
  512 in / 32 out). Small window; use only for operator-presence checks.
- `op-statistic-20261115-full-{off,on}.csv` (when present): same export
  from a full 128-prompt / 256-out / concurrency-32 run.

These files belong to the 2026-11-15 cohort. They must NOT be used to
recompute the historical 2.41% / 5.00% / 5.05% all-kernel-denominator
figures, whose original denominator was never stored.

A structured transcription of the historical values that *were* retained
is in `../split-fusion/op-statistic-reconstructed-from-evidence.csv`.
