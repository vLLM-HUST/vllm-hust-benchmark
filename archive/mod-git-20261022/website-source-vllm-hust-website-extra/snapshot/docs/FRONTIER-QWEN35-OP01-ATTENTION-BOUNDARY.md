# OP01 attention boundary · Frontier PR evidence

This PR (#354) contains ten displayed points from the repeat-backed `20261008` campaign: OFF/ON at
C1, C2, C4, C8 and C16. All thirty formal runs remain in the evidence records, and the superseded
`20261003` points remain archived.

The measured host is vLLM 0.23.0+empty with vLLM-Ascend 0.23.0.post1, matching the operation record.
Every imported formal summary is valid and has zero failed requests.

## Follow-up measurement status

The separate `20261008` remeasurement is complete: three formal 900-second repeats were run for each
OFF/ON and C1/C2/C4/C8/C16 combination. All thirty formal windows are valid and have zero failed
requests. The pre-declared median output-throughput repeat is the displayed point for each mode and
concurrency; the other twenty valid repeats remain archived evidence.

For the follow-up campaign, ON is accepted only when
`runtime_effective mechanism=ascend_boundary_first_true_search phase=formal_request` is present in
the isolated formal server log; an installation event or a warmup/capture event alone is not
sufficient. OFF has no such formal-request event in any of its fifteen formal logs. The campaign is
submitted as a new campaign rather than relabelling the original points.

## Limitations

- The measured prepared file hash is dff300…; it is added to this local draft as a
  content-equivalent variant because metadata normalization reconstructs the canonical 804456… hash.
  This does not independently prove tokenizer identity.
- Server logs were rotated on each restart. The retained ON/OFF logs are campaign-level artifacts,
  not point-isolated activation logs. Performance attribution to OP01 is therefore not marked
  verified.
- The new campaign has three valid formal runs in each mode and concurrency. The displayed point
  selects the median output-throughput run. Raw artifacts for the ten selected runs are public in
  this repository; the other twenty repeats remain local supporting evidence.
- The benchmark snapshot used by the server matches public swe-prefix-reuse commit
  695dd8b1ab280145627a108b434f7a54cca05810: the remote pyproject, README, package initializer,
  runner, CLI, client and prepare source hashes match that commit.
- The installed MOD source is publicly available at
  https://github.com/xmdhb/vllm-hust-ascend-attention-boundary, pinned to package commit
  4a1843d1e2a81f0415a1cfa5141e9b17b3262835. The point-level mod_sources field records the MOD
  package repository commit separately.
- Raw config/summary/requests artifacts for the ten selected runs are published with this report.
  The twenty non-selected repeats remain local evidence and are not marked as public raw evidence.
