# Evaluation API #112 live roundtrip

This directory contains the accepted live Evaluation API roundtrip used by
benchmark issue #136.

- Job: `eval-20261006T212429Z-51ac8c7bedde`
- API status: `succeeded`
- Worker exit code: `0`
- Admission attestation: `VERIFIED`
- Repetitions: `3/3`, each `STATUS=OK`
- Assigned physical NPU: `2`
- Bundle SHA256: `00c2e0997a76f1a88025cd7be54fbb95d61602dacf8695edf6268e8d7a1fcaa1`
- Request SHA256: `e60d08e878583418ea3b03b47d9b516f0984dba168966cc846a5791acd98f027`
- Schedule SHA256: `ca399e3aac03db4d0e4dee3780127369c3884d2e4d08cc01af9dd0af4a43cdf6`

All three retained attempts record the requested canonical
`target_contract_id`. The sealed bundle digest independently matches the
bundle marker, terminal API record, and admission attestation.

`artifacts/` is the immutable API result bundle, including the three raw
benchmark results and server logs. The top-level request, schedule, terminal
record, attestation, API log, and post-run NPU/process/port snapshots preserve
the end-to-end control-plane and resource-release evidence. Volatile API state
databases, lock files, and duplicate runner working directories are excluded.

Run `sha256sum -c SHA256SUMS` from this directory to verify the archive. The
bundle itself is also verified by `artifacts/*/BUNDLE_SHA256`.
