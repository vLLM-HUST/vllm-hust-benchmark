# Evidence index

This MOD is default-off. Its implementation and activation contract are reviewed
independently from the historical StateAxis result.

| Evidence | Label | Meaning |
| --- | --- | --- |
| `evidence/ISSUE76_ASYNC_TRANSFER_LANE_CLOSURE_20260815.md` | `matched-real-online-tested-cell` | Historical 3+3 Qwen2.5-14B NPU cell. It shows directional overlap and latency improvement only in that frozen cell. |
| `evidence/issue76-async-transfer-20260815-r5/` | `historical-matched-real-npu-component-positive` | Machine-readable R5 aggregate, all six arm results and run identities, hardware selection and artifact audits. `RESULT.md` states exact metrics and limits. |
| `evidence/on-off-preflight-20261010/RESULT.json` | `preflight-not-performance` | Repository/manifest inventory before active extraction. |
| `evidence/ecpa-matched-process-20261010/SUMMARY.json` | `matched-process-mock-non-performance` | Three interleaved OFF/ON pairs through Extension Manager and the StateAxis process; validates identity, counters, exactness, lifecycle, and fail-closed wiring, not accelerator performance. |

The historical result does not qualify version 0.2.0 automatically. A fresh
ECPA-launched matched ON/OFF campaign must bind the StateAxis commit, this MOD
commit, manifest identity, hardware, model pack, graph mode, command, dirty
state, exactness, lifecycle, recovery, and effect counters.
