# Retained evidence integration, 2026-11-04

Reports from PRs #305, #329, #334 and #580 are retained against the current main navigation.
[Source identities](sources.json) pin each original commit and Git blob. The original PR branches
remain available.

- #305: five numeric throughput records, including both warmups. The three formal output-throughput
  values have median 882.093459 token/s. These are single-start serving measurements; private full
  responses are not publicly verified.
- #329: candidate DLA applicability analysis remains not-exercised and unqualified. Corrected the
  false inference that single concurrency rules out preemption. Allocation failure can preempt the
  current request; alternative eligible victims are required for a victim-ranking comparison. The
  report links immutable scheduler source and records the correction date. Its SHA256SUMS covers the
  corrected analysis.
- #334: diagnostic guarded retest only; no task-quality or reproduced speedup claim.
- #580: negative fresh-process q4/q40 result (6/8 versus 5/8). Compressed originals are preserved
  and checked against publication-map-v1.json, including decoded bytes.

Numeric telemetry and compressed evidence retain original Git blob identities. Only #305/#334
narratives are formatted; #329 analysis and its checksums are corrected. No leaderboard admission,
runtime activation, or change to published performance claims follows from retaining these reports.
