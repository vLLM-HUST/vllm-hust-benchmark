# Matched ECPA process evidence — 2026-10-10

Label: `matched_process_mock_non_performance`.

This campaign validates activation and lifecycle wiring, not accelerator speed.
Each arm starts a fresh StateAxis process through vLLM-HUST Extension Manager.
The interleaved order is `OFF/ON`, `ON/OFF`, `OFF/ON`. Each run creates one
cold retained state, then carries four two-token exact continuations forward so
every replay performs one host-tier restore followed by one re-offload.

Results per run:

| Arm | host-tier restores | async submitted | async completed | async failed |
| --- | ---: | ---: | ---: | ---: |
| OFF | 4 | 0 | 0 | 0 |
| ON | 4 | 4 | 4 | 0 |

All outputs, reused-token counts, generation transitions, host-tier residency,
and bounded live-generation counts matched. ON published the exact MOD ID,
version, experiment flag, and research-manifest SHA-256; OFF published no MOD
identity. StateAxis, MOD source, and Manager tracked files were clean. All six
services stopped without forced termination.

The deterministic protocol worker does not execute accelerator kernels, so wall
times in these files must not be used as performance evidence. The historical
NPU result remains separately scoped in
`../ISSUE76_ASYNC_TRANSFER_LANE_CLOSURE_20260815.md`.

Preserved preliminary campaigns:

- retained-decision workload (only one restore per run):
  `/root/stateaxis-evidence/archives/async-kv-ecpa-prelim-retained-decision-20261010.tar.gz`,
  SHA-256 `d79da9c0474b187ada9e5cd754c06d03b5b76f7b709c7e0d40864cddf3350f54`;
- correct four-restore workload with an incorrect one-live-state assertion:
  `/root/stateaxis-evidence/archives/async-kv-ecpa-prelim-retained-generations-20261010.tar.gz`,
  SHA-256 `c683a563425ee34c3ad83049850b744b20efb3560f3c142e357704d6e9ec9fdd`;
- functional pass before source-dirty metadata separation:
  `/root/stateaxis-evidence/archives/async-kv-ecpa-functional-pass-pre-source-dirty-fix-20261010.tar.gz`,
  SHA-256 `fc8c6bf2301363cee7f1b1c86e6097ec7a358695af56dbb0dda881b8057f534f`.
