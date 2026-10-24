# Request Lifecycle Causal Profiler

Historical data for the migrated batch is in [benchmark custody](EVIDENCE.md),
with pinned hashes and an external-directory downloader. Source/runtime behavior is unchanged.

Maintainer: [Shifeng Liu (`Remygred`)](https://github.com/Remygred). This
repository is a vLLM-HUST MOD exposed through the `vllm.general_plugins`
entry-point interface.

This repository is an incubation observability/artifact line for causal request
lifecycle tracing in LLM serving. It targets a single-NPU first implementation
on NPU6 and follows the optimization-repository workflow used by the
`llm-optimizations` workspace.

### ECPA 0.3 packaging boundary

The wheel publishes `org.vllm-hust.request-lifecycle-profiler` for static ECPA
discovery and registers a shared sink with the current vLLM-HUST
`vllm.request-lifecycle-events` v1 EventBus when explicitly selected through
`vllm.general_plugins`. Installation remains inert. The sink observes the
host-owned finished, preempted, and KV-reclaimed events, records their bounded
typed payloads when trace export is configured, and emits launch-bound ECPA
evidence only after a real callback. Import, registration, or an enabled trace
path alone is not `runtime_effective`, NPU qualification, or performance
evidence. The historical KV-recovery adapter remains optional and separate.

The native EventBus currently exposes scheduler terminal, preemption, and KV
reclamation boundaries. It does not expose the complete frontend-to-response
DAG, so activation of this sink must not be described as complete causal
coverage or as closing the blind-attribution study.

The ECPA manifest is bound to the validated vLLM-HUST `>=0.31.1,<0.32`
host line. The validation target is commit
`47c8aa1caa7638d2e5e231eedced45448106fde5`, whose installed version is
`0.31.1.post1.dev5+g47c8aa1ca.empty` and whose request-lifecycle EventBus
reports API `1.0`. This compatibility statement covers plugin registration,
typed callback delivery (finished, preempted, and immediate/deferred KV
reclaimed events), and sink removal in CPU ABI tests. These tests create no
serving requests or native evidence receipts. It does not assert model-serving
effectiveness or performance.

## Current research focus

The profiler joins request identity across frontend, scheduler, KV, executor,
and response events, validates lifecycle completeness and resource ownership,
and ranks mechanisms only against matched evidence. Its open research question
is whether this request-scoped lifecycle DAG changes diagnosis and optimization
decisions more reliably than aggregate metrics, flat timers, or timed raw-log
inspection, while abstaining when the evidence is insufficient.

The corrected Issue #19 experiment is an evidence-valid mechanism `NO_GO`:
resource pathology appeared in 1/10 repetitions and matched latency pathology
in 0/10, below the preregistered 8/10 gate. Reconciliation therefore remains
disabled and will not be implemented for that fixed scenario. This is a
decision-gate result, not a rejection of the Request Lifecycle research topic.
The next research phase is the fresh opaque blind-attribution study described
in [`docs/blind_attribution_contract.md`](docs/blind_attribution_contract.md).

## Research Question

Can identity-preserving request lifecycle DAGs, evaluated with matched
interventions, produce more reliable and actionable mechanism rankings than
the same evidence viewed as aggregate metrics, flat timers, or raw logs, and
abstain when causal evidence is insufficient?

## Repository Map

- Development and performance-evidence policy: see the workspace-level
  `AGENTS.md` (kept local; not part of this repository).
- `src/vllm_request_lifecycle_profiler/`: trace schema, attribution logic, and
  plugin code.
- `contracts/`: historical design notes retained only for technical context;
  they are not approval or activation gates.
- `.benchmarks/`: trace probes and controlled fault-injection entrypoints.
- `third_party/llm-serving-workloads/`: pinned shared workload suite.
- `third_party/traceloom/`: pinned provider of the structured augmented-SQLite
  execution evidence consumed by the blind-attribution adapter.
- `third_party/vllm-hust/`: pinned historical vLLM-HUST hook carrier used only
  as a reviewed patch/evidence reference; new integration starts from current
  runtime main.
- `tests/`: no-NPU trace and repository tests.
- `docs/research_logic.md`: seven-step research framing.
- `docs/experiment_plan.md`: evaluation plan and evidence labels.
- `docs/claim_ledger.md`: current claims and forbidden wording.
- `docs/issue19_public_evidence.md`: sanitized evidence and the scoped Issue
  #19 decision boundary.
- `docs/blind_attribution_contract.md`: next-stage blind custody/reveal,
  baseline, scorer, and counterfactual contract.
- `docs/blind_attribution_candidate_vocabulary.json`: frozen candidate IDs,
  optimization families, adjacency, and error-severity rubric.
- `docs/runtime_fault_attribution_roadmap.md`: path from complete runtime
  hooks to controlled-fault attribution evidence.
- `paper/request_lifecycle_causal_profiler/`: systems-paper scaffold.

## Current Mechanism

For
[benchmark #134](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/134),
`kv_recovery.py` adds an optional pressure-episode schema alongside the base
lifecycle trace. It validates stable request/sequence/block identity and
decomposes copy, restore-to-wakeup, wakeup-to-admission, admission-to-first-
compute, and requeue time. Keeping this optional avoids making historical
non-pressure traces appear incomplete.

The initial profiler represents each request as ordered lifecycle events and
computes complete stage spans. Its deterministic dominant-span rule is a
development localization heuristic, not a causal-root-cause rule. A complete
trace establishes evidence availability, and a long stage narrows an interval;
neither alone establishes a mechanism. Non-abstained causal claims require a
matched comparison, explicit identity/ownership links, confidence and residual
reporting, and a rank-one counterfactual.

TraceLoom owns generic timeline-native execution trees, occurrence-preserving
cost accounting, and lineage to raw profiler rows. This repository consumes
that structured evidence and adds request/lifecycle/epoch joins, state and
resource-ownership edges, matched intervention diffs, rankings, and
abstention. It must not duplicate TraceLoom's execution-tree or profiler-row
representation.

`make offline-intervention-gate` additionally evaluates a CPU-only matched
control/intervention fixture. Its output keeps `dominant_span_localization`
separate from `causal_evidence`: an unpaired long span remains
`localization_only`, while a declared single-stage intervention must produce
the target delta without changing other spans. This is `simulation/model`
readiness and remains `NOT_M0_PROVEN`; it is not controlled live attribution.

### KV-recovery configuration

The optional runtime path is disabled unless both sides opt in. Configure the
profiler process with:

```bash
export VLLM_RLP_TRACE_EXPORT_PATH=/path/to/trace
export VLLM_RLP_COMMUNICATION_MODE=issue2:kv-recovery-v1alpha1
export VLLM_RLP_KV_RECOVERY_RUN_ID=0123456789abcdef0123456789abcdef
export VLLM_RLP_PROFILER_PARENT_COMMIT=<40-lowercase-hex-commit>
export VLLM_RLP_RUNTIME_CORE_COMMIT=<40-lowercase-hex-commit>
export VLLM_RLP_DEVICE_PLUGIN_COMMIT=<40-lowercase-hex-commit>
```

The vLLM runtime configuration must also contain:

```json
{
  "kv_recovery_profile_enabled": true,
  "recompute_scheduler_enable": false
}
```

The connector must resolve to `OffloadingConnector` with
`TieringOffloadingSpec`. Omitting the profiler mode, run ID, runtime switch, or
supported spec keeps the observer disabled. Initialization and observation
failures disable profiling without failing serving.

## NPU and Environment

- Reserved device: NPU6.
- Project environment: `vllm-request-lifecycle-profiler-exp`.
- The shared `vllm-hust-dev` environment may be cloned as a baseline, but
  project overlays must not be installed into it.

## Shared Workloads

Use the pinned `third_party/llm-serving-workloads` submodule as the default
workload source. Repo-local workloads are allowed only for controlled
fault-injection cases. For workspace convenience, `WORKLOAD_REPO` or
`LLM_SERVING_WORKLOADS_SRC` may point at a sibling checkout, but paper evidence
must record the pinned submodule commit.

## Evidence Discipline

Every experiment must carry one evidence label: `real-online`,
`existing-server-probe`, `replay`, `simulation/model`, `projected-profile`, or
`derived-artifact`. Profiler-only results are diagnosis evidence, not runtime
speedups.

## Test

```bash
PYTHONPATH=src pytest -q
make shared-workloads-smoke PYTHON=python3
```

## Next step

Keep the Issue #19 negative evidence and fixed gate unchanged, and do not
implement the rejected reconciliation treatment. Freeze the blind-attribution
contract before any opaque case is revealed, then evaluate two fresh positive
cases and one valid insufficient-evidence/no-single-root-cause negative case.
All comparison arms must receive the same normalized evidence and time budget;
every non-abstained top-1 requires a rank-one counterfactual.

Historical public artifacts and synthetic fixtures remain development inputs
only. They do not count toward blind accuracy. A later performance
implementation claim must separately follow the cross-repository benchmark
performance policy in the workspace-level `AGENTS.md`.

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
