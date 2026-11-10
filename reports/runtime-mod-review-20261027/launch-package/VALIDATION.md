# Packaging validation — 2026-11-10

No accelerator workload was started for this publication.

- All Python files parsed and shell entry passed `bash -n`.
- Frozen pilot/performance inputs and engine/environment JSONs matched the original capsule
  structurally.
- Performance IDs/order and dataset revision matched the original screen manifest; unused selection
  metadata is omitted.
- Two existing pure-Python MTP compatibility tests passed; dynamic draft-width checks retained valid
  IDs and rejected dropping real IDs.
- The included analyzer reproduced the original structured six-run results against an isolated copy
  of historical evidence.
- Saved plugin patch passed `git apply --check` against the pinned upstream source.
- Repository pre-commit checks passed; tar contents were compared directly against the readable
  package files.

These checks establish package consistency, not compatibility with a new host or fresh performance
qualification. Full host-dependent compatibility tests were not run on this publishing machine.
