# Native-current NPU gate: dependency mismatch

Date: 2026-10-10

The native-current NPU release gate remains blocked before device execution.
This is a compatibility failure, not an ECPA activation failure.

## Exact source state

- vLLM-HUST current host after PR #52:
  `b44ea22917a0225c3bf0c1db7ebb88ed6e1dee18`
- merged upstream parent:
  `2d69083c4a09dfeb17d761ae96d2e571a628309c`
- upstream baseline: 243 commits after `v0.31.1rc0`
- vLLM-Ascend-HUST current main:
  `f769db916b53446de5779bced0b7d3d38e8a7540`

Host PR #52 corrected the package metadata to release line `0.31.1` and
upstream line `0.31.1rc0`. The host source requires `torch==2.13.0`; the
Ascend tree requires `torch==2.10.0` and `torch-npu==2.10.0.post4`. An empty
host wheel build in the assigned evaluation container stopped at dependency
validation with `wanted: torch==2.13.0` and `found: torch==2.10.0+cpu`.

## Decision

No NPU process was started. Replacing the assigned Ascend torch stack merely
to make the build continue would not be a supported current Host/Ascend pair
and could not qualify ECPA 0.3. The Manager must continue to fail closed on the
real host version, and the KV plugin must not broaden its range to hide the
dependency mismatch.

The platform-owner follow-up is
[vLLM-Ascend-HUST issue #45](https://github.com/vLLM-HUST/vllm-ascend-hust/issues/45).
It requests an explicit compatible Host/Ascend/torch/torch-npu matrix and a CI
contract. After that pair exists, the qualifying run must still produce live,
process-owned observer evidence and prove stop, rollback, process, port, and
device-resource release. Historical vLLM 0.23 adapter runs and the external
request-controller benchmark do not satisfy this gate.
