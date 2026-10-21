# Evidence index

| Evidence | Label | Decision |
| --- | --- | --- |
| `evidence/ISSUE95_BOUNDED_APC_CLOSURE_20260815.md` | historical matched real-online no-harm, scoped | Retain bounded ownership design; no speedup claim because the accepted candidate recorded zero hits and 0.0256% throughput overhead. |
| `evidence/on-off-preflight-20261010/` | import-only admission snapshot | Superseded by version 0.2.0 active Manifest 0.3 implementation; retained as historical negative admission evidence. |
| `evidence/host-activation-20261010/` | host integration and ECPA activation, non-performance | StateAxis PR #484 merged; 397 host tests and Manager configure/enable/check/status/dry-run passed. The result deliberately makes no NPU or speedup claim. |
| `evidence/ecpa-matched-process-20261010/` | matched 3+3 ECPA process gate, non-performance | All six fresh-process arms passed. OFF stayed at zero; each ON arm recorded 8 lookups, 7 exact hits, 28 reused tokens, one insertion, one bounded resident entry, exact outputs, and graceful lifecycle release. |

The independent native crate and Manifest 0.3 activation contract require a new
matched ON/OFF campaign. Historical aggregate StateAxis results do not transfer
qualification to this repository.

The next real-NPU gate must use a fresh server for each arm and a stable cache
salt so candidate requests produce authoritative APC hits. The currently
available container has the raw Qwen2.5 checkpoints but no longer has the
custom Native Qwen2.5-14B weight packs or operator/full/decode oracles used by
the historical campaign. Those inputs must be regenerated and frozen before a
new device result can be admitted.
