# Remaining MOD and installation consumers — 2026-10-23

This receipt continues [issue 326](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/326). It
does not close organization-wide migration. No hardware experiment, approval label, runner
substitution or branch-protection bypass was used.

## Completed: SimLLM paired replay consumers

[SimLLM #14](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/pull/14) merged as
`28f2297c58cec2249576dff2ccdd251c662ad8d6` after all five final-head checks passed: trusted
organization policy, policy tests, Python 3.10 and 3.13 package checks, and remote original
readback. The selected local CPU tests passed (62 tests). Post-merge
[package checks](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/actions/runs/39959895815) and
[original readback](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/actions/runs/39959895368)
also passed.

The existing [manifest](../mod-git-20261022/website-source-vllm-ascend-simllm-hust/manifest.json) at
benchmark `71c2d4ad5f34536d73f66c577e1017da59dba085` has SHA256
`40cadc866fbfe2102ee352c5b5dd809108008f60d8144ffbd50a08994ac5d3ec`. All 13 originals (7,518,969
bytes, including LICENSE) were downloaded, length/SHA256 verified and compared byte-for-byte with
source commit `76e2685ffa359cb311bd333a6c1e65e67c069312` before removing 12 duplicate result/report
paths. LICENSE remains in the MOD. No original or historical Git object was rewritten; the copies
are recoverable from the fixed benchmark snapshot and source Git.

README, runtime-support documentation and MOD metadata now point to the canonical collection. The
downloader rejects unsafe paths, nonempty or repository-local destinations and checksum/length
mismatches. CI verifies the remote originals and rejects reintroduced tracked copies in the migrated
directory. No archived code is executed. Other SimLLM collections and historical claims retain their
existing identity; the tested replay is not promoted to formal task-quality acceptance.

## Public inventory and next consumers

[remaining-public-inventory.json](remaining-public-inventory.json) was freshly collected **after**
the SimLLM merge with central policy `fa129e64dca4841a27cb317344298682c17de232`: 66 public,
non-archived repositories, 1,183 candidate Git paths and 27 Release asset records. The corresponding
[organization audit](https://github.com/vLLM-HUST/.github/actions/runs/39960049111) passed both
policy tests and drift checks without baseline or detection-rule changes. Candidate paths include
legal implementation and synthetic fixtures; they are not measurement counts or deletion
authorization.

Compared with the V0.75 snapshot, changed candidate sets are PegaFlow (777 to 198), MuxPD (562 to
0), KV Arrival (1,399 to 2), Profiler (441 to 18) and SimLLM (145 to 133). Most reductions belong to
the previously merged batches, not this receipt. Asset metadata is unchanged. Zero heuristic paths
does not prove absence of embedded measurements.

| Queue             | Candidate paths | Next consumer boundary / responsible role                                                             |
| ----------------- | --------------: | ----------------------------------------------------------------------------------------------------- |
| SimLLM            |             133 | MOD maintainer: inference rollback, runtime, MMLU and older serving collections; distinguish fixtures |
| PyramidKV         |             123 | MOD maintainer: qualification and joint-launch consumers; preserve identities and failures            |
| KVCompress        |              73 | MOD maintainer: historical result documents and raw-result readers; preserve bilingual relationships  |
| vSpec             |              57 | MOD maintainer: benchmark logs and paper dependencies; match existing capsules                        |
| Extension Manager |              42 | MOD maintainer: host-contract methods versus measured qualification evidence                          |
| PegaFlow          |             198 | MOD maintainer: retained implementation and remaining document consumers, not automatically raw data  |
| Profiler          |              18 | MOD maintainer: presentation projections, fixtures and independent local preflight inputs             |

## Installation assets: retain

[remaining-consumers.json](remaining-consumers.json) independently paginates public releases and
each release's asset endpoint, and binds installation-document observations to full source commits.
The five asset-bearing repositories remain separate from raw-evidence migration:

| Repository              | Assets | Decision                                                                                                                           |
| ----------------------- | -----: | ---------------------------------------------------------------------------------------------------------------------------------- |
| Split-Batch             |     12 | Retain wheels, sdists and checksums: README documents PyPI installation; eight distributions match PyPI names, sizes and SHA256    |
| Extension Manager       |      2 | Wheel/sdist; source installation consumers exist, exact old-version consumers remain unproven                                      |
| Production Stack        |      6 | Helm distributions; default chart endpoint is upstream, use of these exact fork assets remains unproven                            |
| vLLM Metal              |      1 | Platform-specific wheel; installer documentation does not establish zero consumers of the old asset                                |
| Website State Scheduler |      6 | Historical packages and provenance/install attachments retained for offline research inspection; performance positioning withdrawn |

No package, checksum or Release asset was deleted. Metadata identity checks are not clean-install
tests, and general installation documentation is not proof that an exact old package is still used.
Nor does a lack of discovered consumers establish that deletion is safe. Private source repositories
referenced by public release descriptions were not fetched or republished.

## Old local data and external gates

- KV Arrival's archived September compatibility report still references ignored local live logs; its
  current source notice explicitly says recovery is not established. The report is preserved, not
  proof that its raw logs exist. It remains a 0.23 compatibility smoke, not fixed 0.18/0.18 B0.
- Profiler's independent `/root/kv-recovery-service-g4/runs` was absent at the exact path on the
  current machine and the authorized storage container. At profiler commit
  `ba5f6cfba3172411671a07c92e9b01d047082957`, the G1 consumer actually expects
  `.benchmarks/results/g1_cpu_controlled_trace_20260808` under its repository, and separately refers
  to `/root/vllm-hust-g1-default-off-f229ba7`. Both corresponding absolute paths were also absent on
  both machines; the collection manifest contains no G1-controlled-trace members. This bounded check
  is not an exhaustive backup search or a declaration of irreversible loss; the dependent validation
  is not reported as passed. MOD/runtime owners must supply matching existing inputs before it runs.
- The 33 old probes remain missing with search stopped at the user's request. The cancelled GDN
  retest was not resumed; new results cannot be backfilled as old originals.
- Triton #10 remains open: reviewer job cancelled, earlier title failure retained alongside later
  title success. Repository runner owners must restore/authorize the actual review capacity.
- LMCache-Ascend #1 remains open with three device checks pending; device/repository owners must
  provide real checks. SGLang #2 remains open without approval labels and with extra gates failing;
  repository maintainers own approval. CPU policy checks do not replace these gates.

## Failures and bounded recovery

The downloader's initial local lint found two overlong lines; these were repaired before the final
green CI. PR creation initially needed an explicit remote head. A later tool environment change lost
ephemeral working directories, so work resumed from pushed commits; no original was reconstructed
from invented values. A root-volume worktree checkout failed for lack of space and was recreated on
the larger workspace volume. No unrelated data was deleted. Default SSH authentication failed; the
existing dedicated storage identity initially had overly broad file permissions. Restricting that
key to mode 0600 restored the read-only connection. No key content was read or disclosed.

Skill review: current use is migration and consumer reconciliation, based on fixed manifests, actual
remote checks and public inventories; E04/E31/E34/E37/E42/E43/E45 require preserving original bytes,
updating the issue body and both indexes, retaining genuine gates, and not treating absence at one
path as global data loss. This receipt is not a report/site revision or hardware acceptance.
