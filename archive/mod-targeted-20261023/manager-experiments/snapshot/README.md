# vLLM-HUST Extension Manager

Project accountability: `ShuhaoZhangTony` (张书豪) is the primary owner,
project lead, accountable owner, and current executor. ECPA is PI-led and
self-driven; student participation is optional and non-blocking. `Apei-520`
is credited only for documented historical contributions and is not an owner
or critical-path dependency.

## Research charter

The research line **Evidence-Carrying Plugin Architecture for LLM Inference
Systems / 大模型推理插件体系结构** studies whether plugin activation can remain
discoverable, composable, verifiable, and reversible under version skew,
multi-process loading, heterogeneous providers, composition conflicts, lifecycle
drift, and runtime effects that cannot be proved statically.

- [Research charter](docs/research-charter.md)
- [Archived process-supervision originals](docs/benchmark-originals.md)
- [Falsifiable evaluation plan](docs/evaluation-plan.md)
- [Executable ECPA architecture](docs/design/ecpa-architecture.md)
- [Protocol state machine](docs/design/protocol-state-machine.md)
- [Threat model](docs/design/threat-model.md)
- [Failure semantics](docs/design/failure-semantics.md)
- [Frozen core decisions](docs/adr/0001-ecpa-frozen-core-decisions.md)
- [Durable coordinator MVP](docs/design/durable-coordinator.md)
- [Cross-implementation open questions](docs/design/open-questions.md)
- [ECPA 0.1 candidate specification](spec/0.1/README.md)
- [ECPA Attestation Profile 0.1 candidate](docs/design/attestation-profile-0.1.md)
- [Host plugin lifecycle evidence 0.1](docs/design/host-plugin-evidence-0.1.md)
- [Formal-real lifecycle source profile](docs/design/formal-real-lifecycle-sources.md)
- [Reference ExposureGate M1](docs/design/reference-exposure-gate.md)
- [Existing plugin-path inventory and next experiments](docs/research/ecpa-existing-path-inventory.md)
- [Page-bound 24-MOD audit](docs/corpus/workshop-mods.json)

`vllm-hust-ext` is a provider-neutral control point for discovering, validating,
configuring, enabling, planning, rendering, and checking vLLM-HUST extensions.
It is not a vLLM distribution, deployment system, or control plane.

The core owns static manifests, compatibility, persistent intent, lifecycle
state, conflict checks, and delegation. Host Providers retain runtime
authority:

- the vLLM Provider renders launch configuration for in-process extensions;
- the Mooncake Provider reuses official vLLM connectors and checks externally
  operated Mooncake services;
- the Production Stack Provider renders Helm/Kubernetes inputs and dry-run
  plans without applying them.

The ECPA 0.1 candidate attestation profile now has a Python producer/verifier,
a separate Go implementation passing the shared positive/negative corpus, and a
signed-evidence adapter for the durable coordinator. It uses RFC 8785 JCS,
detached compact JWS, and Ed25519/EdDSA. This milestone does not supply a real
vLLM-HUST issuer, production trust root/key management, or formal overhead
result, and a valid signature alone never implies `runtime_effective`.

A plugin, KV connector, external KV system, and control-plane policy remain
different kinds. Installing or enabling an adapter never gives this manager
authority to start shared services, change drivers, delete KV data, or mutate a
production cluster.

Production Stack health evidence is not accepted as a bare boolean:
`cluster_reachable=true` requires `cluster_evidence`, and
`rollout_healthy=true` requires both a reachable cluster and
`rollout_evidence` plus structured `component_evidence` for controller
reconciliation, Router traffic, and an autoscaler decision. It also requires a
configured Kubernetes context, a vLLM OpenAI backend endpoint, and structured
`router_data_plane_evidence`: mock traffic cannot claim health, and the
evidence must show a real model plus backend 5xx and recovered 2xx results. A reported
`ownership_conflicts` entry projects `incompatible + degraded`; in particular,
an HPA and a `VLLMRouter` controller may not both own
`Deployment.spec.replicas`. Rendered operator plans keep install, upgrade,
rollback, and uninstall as explicit operator-owned `null` actions.

The official Production Stack controller at commit `1b87c11a` has now
reconciled a `VLLMRouter` into owned RBAC, Service, and Deployment resources in
an isolated Kubernetes 1.34.11 cluster. The official Router forwarded an
OpenAI-compatible completion request to an external test backend, and a real
metrics-server CPU signal scaled a separately owned Router Deployment from one
to three replicas. The separation is intentional: a negative test proved that
placing an HPA on the operator-owned Deployment creates a two-writer conflict.
On the arm64 host `180-ascend-bench`, a Router built from the same upstream
commit returned HTTP 500 for an absent backend, then HTTP 200 and `ROUTER_OK`
from the existing `zai-org/GLM-4-32B-0414` service after only the isolated
Router was reconfigured. The product does not require amd64. The
`vLLM-HUST/production-stack-hust` thin fork now passes a GitHub-hosted arm64
image build and Router entrypoint smoke test on `main`; publication and clean
host reproduction remain gated. No self-hosted Actions runner is required.

Mooncake runtime detection covers the mutually exclusive official CUDA,
CUDA 13, non-CUDA, NPU, MUSA, and EFA wheel variants. Installing more than one
variant is reported as an incompatible/degraded environment instead of picking
one arbitrarily. The experimental Mooncake profile currently declares
`>=0.3.11.post1,<0.4`. Its Store REST and vLLM KV configuration surfaces are
marked explicitly unversioned because upstream does not publish independent
protocol semantic versions; the Manager does not invent `1.0` contracts for
them.

The official 0.3.12.post1 non-CUDA wheel has completed both a two-process 1 MiB
TransferEngine TCP write and an isolated Store put/exist/get/remove round trip
on `a100-dev`. A separate Ascend NPU 4 run completed a real
MooncakeStoreConnector save/load hit: nine keys and 133,191,072 bytes each way,
with local prefix caching disabled. Master outage produced partial save
failures while inference remained available, and recovery restored save/load
without restarting vLLM. That result remains a pinned experimental integration
point rather than a general support claim.

> **0.3 release boundary:** Manifest `0.3`, saved-config schema 2, explicit
> enable intent, composition conflict rejection, and live process-owned runtime
> evidence are the stable ECPA 0.3 surface. Legacy `0.2-experimental` and
> `0.3-experimental` identifiers remain migration inputs. Experimental Host
> Providers, external operator integrations, and individual MOD performance
> claims are not promoted to universal stable support by this release.

The source tree is ECPA `0.3.0`. The machine-readable decision in
[`docs/release-readiness.json`](docs/release-readiness.json) is authoritative
for release automation and authorizes this release only while all required
gates remain backed by checked-in evidence. CI rejects inconsistent versions,
missing evidence, invalid inventory arithmetic, or a `go` decision while any
required gate is blocked.

The pinned pass/fail combinations and lifecycle rollback owners are summarized
in [`docs/support-matrix.md`](docs/support-matrix.md). A passing point does not
implicitly validate the rest of an experimental version range.
Release qualification and MOD performance recommendation are separated by
[`docs/qualification-policy.md`](docs/qualification-policy.md); a Manager
release never implies that every integrated MOD is beneficial.
Configuration migration and rollback rules are documented in
[`docs/versioning-and-migration.md`](docs/versioning-and-migration.md).
Capability-registry discovery, composition resource claims, and explicit
Bundle activation dependencies are documented in
[`docs/manifest-0.3.md`](docs/manifest-0.3.md). Existing
`0.3-experimental` wheels remain readable as a compatibility alias.
The pinned KV-materialization clean-wheel procedure and its evidence boundary
are documented in
[`docs/kv-materialization-runbook.md`](docs/kv-materialization-runbook.md).

Release artifacts are published on the repository's tagged GitHub release.
For reproducible evaluation, pin the tag or reviewed commit explicitly:

```bash
git clone https://github.com/vLLM-HUST/extension-manager.git
cd extension-manager
git checkout v0.3.0
pip install .
pip install bidkv

vllm-hust-ext --version
vllm-hust-ext extension list
vllm-hust-ext extension status org.vllm-hust.bidkv
vllm-hust-ext extension check org.vllm-hust.bidkv
```

`extension list` validates registrations independently. A malformed installed
Bundle is reported as `invalid` (or as a structured `discovery_error` with
`--json`) without hiding unrelated valid Bundles. Targeted operations,
enablement, and `run` remain strict and never skip an invalid selected Bundle.

BidKV 0.2 targets vLLM-HUST `0.28.1rc1.dev319` through the typed
`vllm.preemption-policy` API v1. The main BidKV distribution does not register
the removed private `vllm.victim_selector` entry point and does not monkey
patch the scheduler. Manager resolves the static bundle component into the
host-native `--preemption-policy` option and refuses incompatible official-vLLM
hosts. Source compatibility is currently unverified on the Sage Mate TP4 graph
target; the older vLLM-HUST 0.23 serving record is retained only as historical
evidence and is not inherited by this baseline.

Direct Worker MODs can declare one `vllm.worker-class.v1` component. The vLLM
Provider renders its `module:object` as native `--worker-cls`, preserves an
identical user value, and rejects a different value before launch. The Bundle
must also claim its exclusive process-carrier resource. This projection does
not install the Worker package or establish host, device, runtime-effect, or
performance qualification.

```bash
vllm-hust-ext extension enable org.vllm-hust.bidkv
vllm-hust-ext run -- vllm serve MODEL
```

`run` owns the process tree it launches. On POSIX it places the serving command
in a separate session, forwards `SIGTERM`, `SIGINT`, and `SIGHUP` to the whole
launch group, waits up to ten seconds, then escalates to `SIGKILL`. It also
terminates descendants that survive their direct parent, so a stopped Manager
does not intentionally leave API or worker processes behind. The grace period
can be set before `--`, for example
`vllm-hust-ext run --shutdown-grace-seconds 30 -- vllm serve MODEL`.

StateAxis candidates separate experimental activation from performance
qualification. An active carrier may run unqualified only when its extension
configuration sets `experiment_mode: true`; the rendered plan remains degraded
and labels the launch experimental. Descriptor-only carriers still fail closed,
and production activation still requires the bound qualification record.
Only one trusted in-process StateAxis carrier may be enabled for a process tree;
the Manager rejects a second owner at enable time and rechecks the invariant at
launch time.
This is a process-lifecycle guarantee, not proof that a particular accelerator
driver has released device memory; deployments must verify that separately.

Only one enabled extension may claim vLLM's `--kv-transfer-config` in a single
process. The Manager rejects conflicting connector plans instead of silently
choosing one. Package removal remains separate from runtime intent: `forget`
only removes Manager-owned configuration and enabled intent and never stops a
shared service, clears KV data, or deletes Kubernetes resources.

Installing an extension distribution only makes it discoverable. Enabling is
explicit and stored in the user configuration. Discovery reads installed
distribution metadata and the static bundle manifest without importing its
implementation modules.

For enabled in-process vLLM extensions, activation entries in the official
`vllm.general_plugins` or `vllm.platform_plugins` groups are merged into
`VLLM_PLUGINS` at launch. Existing selections are preserved, the built-in
`ascend` plugin remains selected, and duplicate names are removed in a stable
order. Two enabled extensions may not claim the same plugin name; entry points
in unrelated groups are not projected into vLLM's plugin allowlist.

Lifecycle states are independent: `installed`, `configured`, and `enabled`
describe artifacts and saved launch intent. `runtime_effective` requires a
process-owned observer to prove that the selected implementation handled real
runtime work. For Manager-supervised launches, the Manager creates a fresh
plan/launch binding and accepts the host's strict evidence stream. Status adds
`runtime_effective` only while the reporting process identity is still live;
it never infers the state from discovery, enablement, an environment variable,
or successful plugin import.

Descriptors whose only Python implementation is marked `import_only` or
`legacy_unregistered` remain inspectable but cannot be enabled. Inspection
reports `activation_ready=false` and the blocker instead of pretending that an
inert research package is runnable. Likewise, saved enabled intent survives an
external-service outage, but `run` refuses a new launch until every non-optional
`requires_services` dependency reports healthy.

Third-party host providers register factories only under the project-owned
entry-point group `vllm_hust_ext.providers`. Extension distributions register
static manifests under `vllm_hust.extension_bundles`. Neither namespace claims
an unofficial `vllm.*` entry-point group.

## Organization catalog feed

The Manager validates the organization-wide, immutable extension catalog used
by Workstation Mod Center. The feed is metadata only: listing an entry never
installs or enables it. `qualified` requires passed functional and recovery
evidence plus an immutable install target. An extension with correct behavior
but a negative matched performance result remains available and records
`not-recommended-for-tested-cell`; unverified function or recovery remains a
non-enableable `preview`.

```bash
vllm-hust-ext catalog validate extension-catalog-v1.json
vllm-hust-ext catalog list extension-catalog-v1.json --include-preview
vllm-hust-ext catalog inspect extension-catalog-v1.json org.vllm-hust.bidkv
```

Catalog availability is not runtime state. Workstation and API consumers must
continue to distinguish `installed`, `configured`, `enabled`, and
`runtime_effective`; only a process-owned observer may prove the last state.

Manifest 0.3 is broad enough to compose many MOD shapes, but it is not a claim
that ECPA can activate arbitrary repository contents. A source library or
benchmark needs a real host carrier; an external operator retains its own
lifecycle; an unversioned patch remains inspect-only until a host contract
exists. The current organization counts and per-MOD gates are recorded in the
[support matrix](docs/support-matrix.md).

### Composing vLLM bundles

vLLM plans compare shared environment and `additional_config` entries by key,
CLI values by option, and JSON CLI values as complete objects per option. Thus
an adapter that declares environment variables can compose with a method bundle
that declares none. Bundle-local user configuration and native manifests do not
compete for a shared setting. Different values for the same shared key, exclusive
resource ownership, and ambiguous plugin ownership still reject activation.

Provider authors can project shared writes with `ProviderPlan.config_claims`
(`ConfigClaim.path` is a tuple of key segments). Providers that omit this optional
projection retain conservative whole-field conflict checking, including when
composed with a projected plan. This is a planning contract, not an activation
receipt: real host invocation evidence is still required.
