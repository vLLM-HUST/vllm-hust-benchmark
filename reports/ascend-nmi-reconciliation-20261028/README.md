# Late-state timing reconciliation

Research evidence from executed Ascend 910B2 experiments on the user-owned evaluation container. These are BF16 research controls, not official leaderboard submissions. Original finite regressions, warmups, source identities and process logs are retained.

## Design

Twelve fresh processes cross the unmodified original and follow-up captured-input runners with 40 or 100 repetitions, in three frozen-order blocks. Both use API setting 0, the same late captured state, layer counts 1/28, one CPU math-library thread and three timing rounds. The follow-up additionally measures cached arguments and performs FP32 work outside timing. This is a common-contract comparison, not a pure single-factor isolation of all runner differences, and cannot reconstruct historical host activity.

## Complete update/replay results

Positive reductions favour BNSD. Each estimate summarizes within-process comparisons; ranges are not confidence intervals. Replay-only comparisons and every raw timing cell are also retained.

| Runner | Repetitions | Layers | BNSD reduction (%) | Range (%) |
|---|---:|---:|---:|---:|
| old | 40 | 1 | -7.523 | -9.255 to -3.272 |
| old | 40 | 28 | 9.251 | 8.937 to 9.617 |
| old | 100 | 1 | -4.360 | -4.420 to -2.935 |
| old | 100 | 28 | 9.010 | 5.081 to 9.329 |
| new | 40 | 1 | -1.883 | -2.304 to 3.304 |
| new | 40 | 28 | 8.709 | 5.890 to 10.974 |
| new | 100 | 1 | -3.807 | -4.753 to -3.390 |
| new | 100 | 28 | 7.089 | 6.568 to 8.377 |

The historical -6.7% late single-layer estimate and the earlier +0.74% follow-up remain separate records. New observations are not retroactively substituted for either. Original-runner checks compare captured outputs; follow-up checks also use CPU FP32. Successful completion means finite checks and complete records, not universal tolerance passage.

Captured-layer inputs execute independently; these are not complete-model or serving speedups. The large original captured tensors remain in the benchmark checkout's previously verified `local-artifacts/ascend-nmi-captured-inputs-20261028/` archive. The public manifest in `../ascend-nmi-bridge-20261028/captured-inputs.json` identifies them; it is not a public tensor release.

## Verify and reproduce

`manifest.json` binds every byte of the ordered base64 archive parts, all archive members and derived public files. The capsule contains frozen plans, source snapshots, raw result JSON, logs, warmups and executable analysis. Verification does not execute archive content.

```sh
python verify_archive.py --extract /tmp/ascend-nmi-reconciliation-20261028
python /tmp/ascend-nmi-reconciliation-20261028/campaign/analyze.py
```

The archived scripts retain original absolute runtime paths as provenance. New device execution requires the recorded model/runtime, a free device lease and explicitly rebased paths/hashes; it must write a fresh directory and never overwrite this evidence. No model weights are uploaded. This collection does not complete the public release of all original study data.

Analysis reproduction uses the Python standard library.
