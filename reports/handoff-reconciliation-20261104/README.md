# Remaining handoff reconciliation, 2026-11-04

[Source identities](sources.json) retain report files from #531, #540, #583 and #598. Raw evidence
is byte-preserved; the two intentional edits are recorded separately.

- TraceLoom (#531): retain structural observations, frozen reproduction tools and the historical
  public verification receipt. This is not an unprofiled backend speedup. Independent large-archive
  readback is tracked separately; retaining the historical receipt does not claim a fresh
  verification of every profiler database.
- Delivery V1.00 (#540): retain its original report as `中期测试报告-v100.md`, alongside its SWE, Terminal
  and reference-strength evidence. The V1.00 generator writes only this archived report and requires
  the frozen V0.99 input. Current V1.03 remains intact.
- W4A8 TP2 (#583): add one cohort, five points and five summary records without changing existing
  observations. Correct the missing report link and the claim that concurrency is the only changing
  variable: execution order and server history are uncontrolled. Summaries, request-file hashes and
  runtime counters are public; complete per-request files are not archived here, so P90 and
  generated outputs cannot be independently recomputed from this publication. Preserve the smoke
  observation scope.
- SimLLM (#598): all 172 checksums pass, and both analyses reproduce from original outputs. The
  constructed 16k workload has severe answer degradation; the original similarity replay does not
  establish a reproducible 10% gain. No quality promotion.

Five draft PRs (#283, #297, #320, #543, #544) remain drafts. Their missing task-quality or
activation evidence is not repairable by rewriting metadata. No cancelled experiment, runtime
installation, or new device workload is launched by this reconciliation.
