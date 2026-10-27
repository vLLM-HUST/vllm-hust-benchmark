# KV-Recovery Runtime Call-Site Map

- Audit status: `complete_static_read_only`
- Evidence status: `NOT_M0_PROVEN`
- Runtime: `f229ba7cad21a4dba58681af6738a9fd947388e2`
- Device plugin: `cafad89a5e103f31ea517c1edb56130578c3cd56`
- Proposed profile: `rlp.kv-recovery/v1alpha1` (`owner_review_required`)

All paths and line numbers below refer to the exact Git blobs at those commits,
read with `git show`; they do not refer to the current checkout line numbers.
No runtime/device file was changed or checked out during this audit.

## 1. Runtime configuration-to-connector path

For the runtime-owned candidate family used by the historical tiering shape:

1. `vllm/engine/arg_utils.py:1523,2350` accepts and stores
   `--kv-transfer-config`.
2. `vllm/v1/core/sched/scheduler.py:132-148` constructs the configured KV
   connector.
3. `vllm/distributed/kv_transfer/kv_connector/factory.py:206-210` resolves
   `OffloadingConnector`.
4. `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:59-66`
   creates the offloading spec plus scheduler/worker connector halves.
5. `vllm/v1/kv_offload/factory.py:33-46,65-73` resolves
   `TieringOffloadingSpec` to the runtime-core implementation when no
   `spec_module_path` overrides it.
6. `vllm/v1/kv_offload/tiering/spec.py:65-75,141-215` builds the CPU primary
   tier and tiering manager.
7. `vllm/v1/kv_offload/cpu/worker_factory.py:10-30` selects
   `AscendCPUOffloadingWorker` when the active platform is NPU.
8. `vllm/v1/kv_offload/cpu/npu_worker.py:304-374,403-438` performs D2H/H2D,
   reports device-event bytes/time, and owns the synchronization boundary.

This path is distinct from the device-plugin classes
`vllm_ascend.kv_offload.npu.NPUOffloadingSpec` and
`NPUTieringOffloadingSpec`. A resolved configuration must name exactly one
family; class-name similarity is not evidence that they are interchangeable.

## 2. Exact seven-stage map

### 2.1 `preempt`

Runtime source:

- victim selection and accepted call:
  `vllm/v1/core/sched/scheduler.py:591-620`;
- committed transition: `Scheduler._preempt_request()` at `1201-1223`;
- output roster: `SchedulerOutput.preempted_req_ids` construction at
  `1148-1166`, with the next-set rollover at `1268-1272`.

The transition frees request blocks, frees encoder state, removes in-flight
prefill membership, changes `RUNNING -> PREEMPTED`, resets computed tokens,
clears speculative tokens, increments `num_preemptions`, prepends the request
to waiting, and records its reset ID.

The base `preempted` event closes epoch `e`. The positive profile
`recovery_epoch` and later base `requeued/resumed` events belong to `e+1`, so
validation requires `base.preemption_epoch + 1 == recovery_epoch`.

Instrumentation consequence: logical block and request/epoch identity must be
snapshotted before the free at line 1210, but the profile event may be emitted
only after the state mutation and queue insertion have succeeded. This avoids
recording a victim proposal as a real preemption.

### 2.2 `restore_start`

Runtime source:

- construct the request/job/block mapping:
  `OffloadingConnectorScheduler.update_state_after_alloc()` in
  `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:713-806`;
- job structure (`req_id`, source spec, destination spec):
  `.../offloading/common.py:53-71`;
- real worker submission:
  `.../offloading/worker.py:317-328`.

The scheduler mapping is preparation, not copy start. The proposed profile
boundary is immediately after the rank-0 worker's `submit_load()` returns
success for the exact job.

### 2.3 `restore_done`

Runtime source:

- worker completion and optional transfer size/time:
  `.../offloading/worker.py:340-375`;
- `TransferResult` fields:
  `vllm/v1/kv_offload/base.py:509-514`;
- worker-to-scheduler aggregation:
  `vllm/distributed/kv_transfer/kv_connector/utils.py:50-116,156-168`;
- scheduler reception:
  `Scheduler._update_from_kv_xfer_finished()` at
  `vllm/v1/core/sched/scheduler.py:2531-2561`.

The profile is restricted to one device rank, so it selects the first matching
successful rank-0 worker completion as `restore_done`. Scheduler reception is
a later handoff and is not silently folded into copy time. A future multi-rank
profile must separately freeze whether completion means each worker, the
slowest worker, or a scheduler-visible all-worker barrier.

### 2.4 `scheduler_wakeup`

Runtime source:

- waiting traversal and blocked check:
  `vllm/v1/core/sched/scheduler.py:692-709`;
- readiness consumption and promotion:
  `Scheduler._try_promote_blocked_waiting_request()` at `2498-2513`;
- restored-block state update:
  `_update_waiting_for_remote_kv()` at `2464-2496`.

The exact boundary is the first later scheduler traversal that consumes the
matching `finished_recving`, updates restored blocks, and promotes the request
from `WAITING_FOR_REMOTE_KVS` to `PREEMPTED` for a recovery epoch.

`get_num_new_matched_tokens()` is earlier than H2D job construction and cannot
be called a post-restore wakeup.

### 2.5 `requeue`

The prepend in `_preempt_request()` is the preemption commit and is not counted
as a post-wakeup requeue. Later defers are spread across branches rather than
one hook:

| Post-wakeup runtime branch | Lines | Proposed reason |
| --- | ---: | --- |
| LoRA capacity | `scheduler.py:713-724` | `lora_capacity` |
| DP prefill throttle | `842-849` | `prefill_throttled` |
| token/encoder budget bundle | `873-906` | split into `token_budget`/`encoder_budget`, otherwise `unclassified` |
| block allocation/capacity | `964-986` | `block_capacity` |

The adapter must retain explicit episode-ready state and emit at the actual
branch. It may not infer a reason because the request is still in a queue on a
later scheduler pass. Running-capacity and pause checks occur before request
selection; connector lookup/inflight and encoder-cache defers occur before a
successful restored request is woken or are unreachable once restored tokens
are installed. They are pre-wakeup wait evidence, not post-wakeup requeues.

### 2.6 `admission`

Runtime source:

- pop from queue: `scheduler.py:1008`;
- append to running: `1031`;
- classify `PREEMPTED` into `scheduled_resumed_reqs`: `1036-1041`;
- commit `request.status = RUNNING`: `1050-1051`.

The proposed boundary is after the status commit, with positive scheduled
tokens and the request present in the resumed scheduler roster. Queue
selection, block allocation, and scheduler wakeup are earlier observations and
cannot be relabeled as completed admission.

### 2.7 `first_prefill_or_decode`

EngineCore dispatches a `SchedulerOutput` at:

- ordinary path: `vllm/v1/engine/core.py:495-510`;
- batch-queue path: `core.py:552-555`.

`execute_model()` at those lines is host dispatch, not actual model compute.
The worker forward candidates are:

- MRV1: connector context followed by `_model_forward()` in
  `vllm/v1/worker/gpu_model_runner.py:4351-4387`;
- Ascend MRV1 (`NPUModelRunner`, used by the Ascend NPU worker):
  `vllm_ascend/worker/model_runner_v1.py` — `observe_kv_recovery_first_compute`
  is called immediately before `_model_forward` inside the same
  `maybe_get_kv_connector_output` context (added 2026-08-09; the override
  previously omitted the observe call, which is why the live Ascend run was
  missing `first_prefill_or_decode`);
- V2 full graph: `vllm/v1/worker/gpu/model_runner.py:1285-1292`;
- V2 piecewise/eager: the same file at `1294-1323`.

The worker must receive the explicit `(runtime_request_id, recovery_epoch)`
sidecar and prove the request is a member of that exact scheduler step before
emission. This is a versioned profile child observation inside the active
engine-core-owned base prefill/decode span; it references but does not redefine
the base `prefill_started`/`decode_started` event. A batch forward timestamp
cannot automatically be attributed to every request in the batch.

## 3. Scheduler roster and episode propagation

The authoritative batch membership is
`SchedulerOutput.num_scheduled_tokens: dict[str, int]` in
`vllm/v1/core/sched/output.py:180-220`.

- first-scheduled requests use `scheduled_new_reqs`;
- cached requests use `CachedRequestData.req_ids`;
- MRV1 retains `CachedRequestData.resumed_req_ids` at `output.py:111-126`;
- the scheduler constructs the roster at `scheduler.py:1097-1166`.

V2 merges resumed requests into `scheduled_new_reqs` at
`scheduler.py:1102-1104` and clears the separate resumed list. Consequently,
the worker roster alone cannot distinguish a new request from a recovered
request in V2. The thin adapter must propagate the recovery epoch explicitly;
it cannot re-infer it at the worker.

## 4. Connector handoff chain

The runtime-core connector path is:

```text
get_num_new_matched_tokens(request)
  -> allocate_slots(request)
  -> update_state_after_alloc(request, blocks)
  -> OffloadingConnectorMetadata.load_jobs[job_id]
  -> SchedulerOutput.kv_connector_metadata
  -> worker bind / pre-forward / start load
  -> submit_load(job_id, src_spec, dst_spec)
  -> TransferResult(job_id, size, time)
  -> KVConnectorOutput.finished_recving + worker metadata
  -> scheduler _update_from_kv_xfer_finished()
  -> next schedule() waiting promotion
```

Supporting boundaries:

- lookup: `scheduler.py:781-807`;
- async allocation and `WAITING_FOR_REMOTE_KVS`: `942-1029`;
- connector metadata on scheduler output: `1168-1175`;
- base connector contract:
  `vllm/distributed/kv_transfer/kv_connector/v1/base.py:285-321,357-392,453-522`;
- MRV1 bind/start/output:
  `vllm/v1/worker/kv_connector_model_runner_mixin.py:74-112`;
- V2 bind/start: `vllm/v1/worker/gpu/kv_connector.py:61-95`.

The proposed `transfer_wait` operation observation has one narrower boundary:
`offloading/worker.py:314-315` calls `OffloadingWorker.wait()` for a nonempty
flush set, and the Ascend implementation synchronizes matching end events at
`vllm/v1/kv_offload/cpu/npu_worker.py:431-435`. Scheduler queue residence and
worker shutdown synchronization are not this observation. The flush set is
process-local and can contain jobs for multiple requests. Therefore the
profile's wait-set record is run/process-scoped and joins each request or
recovery episode only through exact member `transfer_id` records; it cannot
carry or infer one request/lifecycle envelope for the whole set.

The connector's `TransferJob` already binds `req_id`, source, and destination,
and the scheduler job counter is shared by load/store jobs at
`offloading/scheduler.py:380-399`. It does not carry a profiler trace ID,
recovery epoch, globally scoped transfer ID, or stable logical block IDs; G1
must add only that bounded sidecar data at reviewed handoffs.

## 5. Request and block identity facts

Runtime facts:

- scheduler-stable internal identity is `Request.request_id`:
  `vllm/v1/request.py:59-102`;
- the preemption count is at `request.py:174-175`;
- `current_step` is scheduler-global, not request sequence:
  `scheduler.py:303-307`;
- `last_sched_seq` is a deferred-free fence, not recovery identity:
  `request.py:148-150`;
- the frontend randomizes engine request IDs in
  `InputProcessor.assign_request_id()` at
  `vllm/v1/engine/input_processor.py:222-240`;
- the original external ID remains on `EngineCoreRequest.external_req_id` at
  `vllm/v1/engine/__init__.py:88-128` but is not copied by
  `Request.from_engine_core_request()` at `request.py:197-222`;
- `OffloadKey` contains a block hash plus four-byte KV-group index:
  `vllm/v1/kv_offload/base.py:27-47`.

Therefore the first profile carries the complete internal request ID and maps
it explicitly to canonical `trace_id:e:0`. It does not parse a randomized ID
or claim to preserve the external client ID. The profile creates a positive
recovery epoch from the committed preemption count.

Connector job IDs and physical CPU/device block IDs are local and reusable.
Formal association uses profile logical block IDs plus process/rank/tier/
direction/transfer scope. Raw `OffloadKey`, token IDs, and prompt data are not
profile output.

## 6. Pressure recovery is not invalid-load recovery

The runtime has a separate invalid-KV-load path:

- detection in `Scheduler.update_from_output()`:
  `scheduler.py:1591-1599`;
- invalid-block truncation:
  `_update_requests_with_invalid_blocks()` at `2563-2664`;
- policy: `_handle_invalid_blocks()` at `2666-2735`.

Synchronous failure leaves the request running, truncates computed tokens, and
skips the step output. Asynchronous failure can pass through
`WAITING_FOR_REMOTE_KVS`, but the failure itself does not increment
`num_preemptions`. The corresponding correctness tests are:

- `tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py:122-187`;
- `tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py:337-477`.

Neither path may be padded with a synthetic `preempt` to satisfy the profile.
Multiple restore attempts also cannot be losslessly represented by PR #4's one
restore pair, so v1alpha1 rejects them rather than selecting a convenient
attempt.

## 7. Device-plugin specialty facts

The alternative `RecomputeCPUOffloadConnector` family is registered at
`vllm_ascend/distributed/kv_transfer/__init__.py:83-87`. It owns a separate
manager/worker state machine:

- request state and group-local CPU blocks:
  `.../recompute_cpu_offload/manager.py:36-53`;
- D2H preemption hook:
  `vllm_ascend/core/recompute_scheduler.py:300-365`;
- state allocation and store mapping: `manager.py:222-367,473-515`;
- H2D lookup/allocation/completion: `manager.py:179-200,384-553`;
- worker transfer path:
  `.../recompute_cpu_offload/worker.py:134-165,200-319`.

This family currently lacks a recovery epoch, loses KV-group boundaries when
flattening block maps, has independent store/load event counters, may bind one
event to several requests, and does not report complete per-transfer bytes,
elapsed time, or failure data. More importantly, its scheduler and offload
APIs are incompatible with the pinned runtime as documented in
`pinned_pair_scheduler_compatibility.md`. It is not the G1 implementation
family for this candidate.

## 8. Thin instrumentation points implemented after G0

The default-off G1 implementation uses the versioned profile, event mapping,
selected implementation family, resolved configuration, and CPU compatibility
tests. It adds calls only at:

1. scheduler preemption commit, carrying the pre-free logical snapshot;
2. connector job construction, adding bounded trace/epoch/logical-block
   sidecars without changing transfer decisions;
3. rank-0 worker successful submit and completion;
4. scheduler finished-receive consumption and each explicit defer branch;
5. completed `PREEMPTED -> RUNNING` admission;
6. exact worker model-forward entry for the admitted request roster; and
7. worker bootstrap, explicitly calling
   `RuntimeLifecycleHooks.reinitialize_after_fork()`.

The calls must be no-ops when tracing is disabled, fail open for serving, fail
closed for evidence, and reuse the existing bounded exporter machinery. Logs,
metrics, or timestamps without IDs remain diagnostics only.

## 2026-08-08 addendum: H2D evidence path wiring

The previous "tiering-manager background migration" hypothesis was refined by a
CPU reproduction and real NPU smoke against the `OffloadingConnector`
scheduler/worker: the CPU->NPU H2D data copy always goes through the connector
worker's `start_kv_transfers` -> `submit_load` path, and the runtime performs it
in two situations:

1. Preemption recovery: a preempted request resumes with its CPU-resident KV.
   This carries a recovery episode and produces the full seven-stage chain.
2. Block-level tiering migration: a still-running request has blocks evicted
   to CPU under device pressure and reloaded without any preemption. This has
   no episode.

Defects that previously dropped the H2D evidence:

- `vllm/v1/kv_recovery_profile.py` `invalidate_transfers` set the
  process-wide `_evidence_disabled` latch on any scheduler discard handoff.
  A preemption that invalidated in-flight store contexts therefore silently
  disabled all later H2D evidence. The latch now stays off for discard
  handoffs; only genuine state-capacity failures disable the observer.
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
  `build_connector_meta` added every in-flight transfer job of a preempted
  request to the discard handoff, including store (D2H) jobs that are the
  evidence anchor for the later restore. It now adds only non-store (load)
  jobs on preemption; terminal and cache-reset handoffs still discard all
  in-flight jobs as abandoned.
- The profiler only accepted `h2d_restore` contexts with a recovery episode.
  Real NPU runs showed the service reloading running requests' blocks without
  preemption, so the adapter dropped them ("no active preemption episode").
  `KVRecoveryTransferContext` now allows `recovery_epoch=None`, and the
  scheduler adapter + worker sink record unassociated H2D as transfer evidence.
- `block_set_chunk` used 64-row chunks, but a 64-row record exceeds the
  4096-byte profile record cap (`encoded record exceeds 4096 bytes`), so the
  H2D block set was dropped as `serialization_failure`. Chunking now uses a
  byte-safe 32-row budget.

Regression coverage: `tests/v1/test_kv_recovery_profile.py`
(`test_connector_flush_invalidates_pending_context_exactly_once`,
`test_h2d_requires_episode_and_d2h_forbids_episode`),
`tests/v1/kv_connector/unit/offloading_connector/test_kv_recovery_worker.py`
(`test_bounded_observer_wait_precedes_explicit_discard_invalidation`),
`tests/v1/kv_connector/unit/offloading_connector/test_kv_recovery_scheduler.py`
(`test_pressure_preemption_preserves_identity_through_real_connector_path`),
and the cross-repository plugin gates
`tests/test_kv_recovery_runtime_integration.py::test_actual_runtime_connector_flow_captures_full_h2d_recovery`
and
`tests/test_kv_recovery_runtime_integration.py::test_actual_runtime_records_unassociated_h2d_migration_without_episode`.
