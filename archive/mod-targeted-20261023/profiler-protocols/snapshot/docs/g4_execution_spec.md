# G4 Execution Spec (repo-defined modes, 2026-08-10)

> Mode semantics follow the repository's own definition in
> `scripts/verify_g0_pinned_pair.py` (`validate_config_candidate_data`) and
> `docs/pinned_pair_scheduler_compatibility.md`. Registry constants are copied
> verbatim from the active registry (v1.3.0).

## 1. Target declaration

- **declared target**: `specialty-target`
- Why not official: the official registry targets fix `gpu_memory_utilization=0.6`
  with no tiering; the #134 mechanism question (fixed-8-GiB tiering state
  machine) requires the mode-specific connector/spec variations below, which the
  official target does not cover. Independent spec keeps the official registry
  base (Qwen2.5-14B-Instruct, 910B2×1, FP16, `max_model_len=32768`, TP=1).
- Registry base: `registry_version=1.3.0`,
  `source_set_sha256=20e94aab3fdb09b47aa388d87bee1464e129b39bad97af34db810437a9d31c9a`.

## 2. Common parameters (ALL modes — repo-mandated)

- Model `Qwen/Qwen2.5-14B-Instruct` (HF cache), Ascend 910B2 ×1 (NPU6), FP16
- `--max-model-len 32768`, `--gpu-memory-utilization 0.6`, TP=1,
  `--shutdown-timeout 120`, `--trust-remote-code`
- `--kv-cache-memory-bytes 8589934592` (**exact 8 GiB device KV for every mode**)
- `--additional-config '{"kv_recovery_profile_enabled":true,"recompute_scheduler_enable":false,"SLO_limits_for_dynamic_batch":-1}'`
- default runtime scheduler (`scheduler_cls` unset)

## 3. Frozen mode definitions (repo `verify_g0_pinned_pair.py`)

| mode_id (repo) | connector / spec | kv_transfer_config | communication |
| --- | --- | --- | --- |
| `hbm_only_no_connector` | **none** (`connector_present=False`, factory must NOT be called) | `None` (no `--kv-transfer-config`) | `none` |
| `tiering_disabled` | OffloadingConnector + **`CPUOffloadingSpec`** (repo proposal; repo marks `OUTSIDE_CURRENT_PROFILE_OWNER_IMPLEMENTATION_DECISION` / proposed_non_executable) | `{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":8589934592}}` | `None` (future mode) |
| `tiering_enabled` | OffloadingConnector + **`TieringOffloadingSpec`** (module_path_policy=OMIT, device NPU spec forbidden) | `{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":8589934592,"secondary_tiers":[]}}` | `issue2:kv-recovery-v1alpha1` |

Repo status notes: the #134 candidate is `BLOCKED_INCOMPLETE_CONFIGURATION_CANDIDATE_REVIEWED`
with `BLOCKED_TIERING_DISABLED_SEMANTICS` in the repo verifier. Per AGENTS.md,
authority/approval gates are non-normative; the engineering mode semantics above
are normative and are what this spec executes. `tiering_disabled`'s
`CPUOffloadingSpec` executability must be boot-verified before the matrix.

## 4. Workload

Pressure profile (the official 1-RPS registry workloads are negative coverage at
8 GiB — first-round finding): 4 concurrent `/v1/completions`, 15000-token prompt
(seed 20260809), `max_tokens=8192`, temperature 0, streaming.

## 5. Matrix protocol (completed 2026-08-10)

- 3 modes × 3 independent lifecycles, alternating order
  (`disabled → enabled → hbm → …`). **Done** — see
  `docs/g4_execution_result.md` and
  `.benchmarks/results/g4_fixed_8gib_modes_20260810/run_summary.json`.
- Per lifecycle: one independent server process (start → warm → pressure client
  → drain); report every repetition + median/IQR per mode.
- Observer overhead: `tiering_enabled` ×3 tracing off vs the ×3 tracing-on rows.
  **Done** — +1.1 s median (~0.3%).

## 6. Evidence per run

`server.log`, `pressure_client.out`, trace shards, run_id, resolved config,
environment manifest (CANN 9.0.0, driver 26.0.rc1, torch 2.10.0, torch_npu
2.10.0). Artifacts root `/root/kv-recovery-service-g4/runs/<run_id>/`.
