# Complete-model layout and API-setting baseline comparison

Research evidence from executed Ascend 910B2 experiments on the user-owned evaluation container. These are BF16 research controls, not official leaderboard submissions. Original finite regressions, warmups, source identities and process logs are retained.

## Design and scope

Eighteen independent processes cross native TND, always-BNSD and the 4,096-token conditional selector with `inner_precise=0/1`, in three frozen-order blocks. Qwen2.5-7B-Instruct, BF16, TP1, FULL_DECODE_ONLY, 32 live requests, 32,768-token context/scheduling budget, memory utilization 0.55. Prefill is unchanged. All arms use the same setting hook, counters, buffered schedule observer and forced-next-token sampler. This is connected complete-model execution, including projections, MLPs and KV writes.

Each process completes three excluded warmup cohorts followed by three formal cohorts. Every generated sequence and the exact ordered scheduler history must match the archived trajectories. The analyzer recomputes schedule digests from raw steps, compares full outputs with the original frozen trajectories, checks all 18 independent leases and runtime identities, and verifies capture/workspace/update setting evidence.

`cohort_ms` includes the timed cohort loop. `pure_decode_ms` and `prefill_mixed_ms` sum classified engine-step durations; `non_step_ms` retains the remaining timed loop overhead. They are not uninstrumented HTTP latencies or natural-generation measurements. Ranges below span three paired process estimates, not confidence intervals.

## Primary stronger-baseline comparisons

Positive reductions favour the candidate. Every other arm, scope and regression remains in `summary.json` and `per-case.csv`.

| Cohort | Candidate | Baseline | Cohort reduction (%) | Range (%) |
|---|---|---|---:|---:|
| summary0_tail0 | conditional0 | native1 | -1.477 | -2.219 to -1.276 |
| summary0_tail0 | conditional1 | native1 | -0.574 | -1.062 to 0.530 |
| summary0_tail1 | conditional0 | native1 | -1.567 | -1.802 to -0.776 |
| summary0_tail1 | conditional1 | native1 | -1.262 | -1.845 to -0.820 |
| summary0_tail4 | conditional0 | native1 | -1.927 | -2.952 to -1.827 |
| summary0_tail4 | conditional1 | native1 | -1.784 | -1.881 to -1.588 |

## API semantics and replication limits

The parameter name does not establish a change in arithmetic bit width. Frozen upstream op-plugin documentation (commit a52bcd00bcbc0c08f1d8f699de6ff93257b96a86) states that the low bit does not distinguish precision/performance modes for BF16. It is not a version match to the installed CANN binary. Timing differences are reported as setting-associated observations; no reduced-precision mechanism, exact internal kernel identity or production-safety guarantee is inferred.

The capsule also includes a reanalysis of the original sustained-arrival summary in absolute units. It is the same stream data, not newly measured offers. Unfinished-request snapshots include waiting and running work. Paper drafts and editorial backups are excluded.

## Verify and reproduce

`manifest.json` binds every byte of the ordered base64 archive parts, all archive members and derived public files. The capsule contains frozen plans, source snapshots, raw result JSON, logs, warmups and executable analysis. Verification does not execute archive content.

```sh
python verify_archive.py --extract /tmp/ascend-nmi-baseline-20261028
python /tmp/ascend-nmi-baseline-20261028/campaign/analyze.py
```

The archived scripts retain original absolute runtime paths as provenance. New device execution requires the recorded model/runtime, a free device lease and explicitly rebased paths/hashes; it must write a fresh directory and never overwrite this evidence. No model weights are uploaded. This collection does not complete the public release of all original study data.

Analysis reproduction uses the Python standard library.
