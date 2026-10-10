# StateAxis MOD ON/OFF benchmark admission

- Generated: `2026-10-10T09:16:17+00:00`
- MODs audited: **23**
- Ready for real matched ON/OFF: **2**
- ON not runnable: **21**
- Performance pairs completed: **0**
- Exact catalog-pinned checkouts are required; mismatches fail with `PINNED_COMMIT_MISMATCH`.
- Hardware snapshot: `npu-smi.txt` (SHA-256 `42e5bdb6352129abb69d4ebb642e9b1f33500d8f1e9024a011babb109d5fee29`).

No performance delta is emitted when the ON arm cannot be activated. This is a
fail-closed benchmark result, not a zero-percent performance result.

| MOD                                                        | ON status                  | Blockers                                                                                                                                   |
| ---------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `org.vllm-hust.stateaxis-automatic-prefix-cache`           | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-async-kv-transfer`                | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-pipelined-weight-loading`         | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-host-tier-residency`              | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-session-lifecycle`                | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-program-aware-scheduling`         | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-shared-prefix-attention`          | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-identity-bound-int8-kv`           | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-multi-lora-residency`             | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-typed-hybrid-state-groups`        | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-length-aware-queue`               | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-workflow-disaggregated-residency` | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-one-step-scheduling`              | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-device-metadata-delta`            | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-aclgraph-dispatcher`              | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-speculative-decoding`             | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-chunked-prefill`                  | `READY_FOR_MATCHED_ON_OFF` | none                                                                                                                                       |
| `org.vllm-hust.stateaxis-incremental-eviction`             | `READY_FOR_MATCHED_ON_OFF` | none                                                                                                                                       |
| `org.vllm-hust.stateaxis-launch-fusion`                    | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-dynamic-microbatch`               | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-preemption-recompute`             | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-sleep-wake`                       | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
| `org.vllm-hust.stateaxis-on-device-sampling`               | `ON_NOT_RUNNABLE`          | PINNED_COMMIT_MISMATCH, IMPLEMENTATION_NOT_ACTIVE, NO_ACTIVATION_ENTRY_POINT, IMPLEMENTATION_NOT_EXTRACTED, ACTIVATION_CONTRACT_REFUSES_ON |
