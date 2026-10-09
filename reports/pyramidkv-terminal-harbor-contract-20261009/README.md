# Candidate Terminal-Bench 2.1 Harbor contract for PyramidKV

This advances PyramidKV #8 and benchmark #254 from verified input/image inventories to a concrete,
reproducible **candidate execution plan**. It does not run a model, create a sandbox, execute a
task, validate the verifier semantically, or report a resolution score. Canonical review and
external execution resources are still required; `ready_for_scored_execution` remains false.

## Frozen preparation

The preparer re-verifies the entire
[source/image audit](../pyramidkv-terminal-source-audit-20261009) and stages all 89 tasks. It
replaces only each task's mutable image tag with its previously verified AMD64 manifest digest. TOML
parsing checks that all other task semantics are preserved, including CPU/memory/storage requests,
network policy and per-task timeouts. Task file permissions are restored from the verified Git tree:
1,035 staged files, of which 12 are executable. The source snapshot and reference files are
unchanged; task instructions and solutions are not published in this report.

The scaffold is
[Harbor v0.24.0 at b53b8134](https://github.com/harbor-framework/harbor/tree/b53b8134e1241686dca7759af188f987ecc48e8b),
using its unmodified Terminus-2 agent (reported agent version 2.0.0). Key installed source files
were checked against that release. B0/B1 jobs differ only in job/output names. Compression
activation must be changed and independently verified in the model server; these job files do
**not** toggle it.

The proposed policy is one trial per task, one concurrent trial, 100 agent turns, temperature zero,
seed 17, up to 4,096 output tokens per model call, and a 32,768-token model context. It disables
thinking, context summarization and additional instructions/skills/MCP tools. Context overflow is a
failure, not silently truncated input. Task-specific timeouts remain unchanged. Fresh task
containers are deleted after log collection; no host directories or Docker sockets are mounted into
the task environment by this plan. CPU and memory limits are requested through Harbor's `limit`
policy, but their actual enforcement must be checked on the future execution backend.

**One trial is not one model request, and task retries are not model retries.** Job retries and
LiteLLM SDK retries are set to zero, while Harbor's LLM wrapper and Terminus query wrapper each keep
three-attempt decorators for eligible errors. Their retry exclusions differ. The upstream fallback
for rejected telemetry parameters also remains. These behaviors are part of the frozen scaffold, not
hidden extra successful trials. All raw traces, errors and attempted-task denominators must be
retained when real execution is authorized and provisioned.

The 32K/4K bounds are the configured serving/response limits, not an exact promise about the
agent-side token estimator. Server-reported usage is authoritative. The zero cost entries only
prevent custom-model pricing lookup; this report makes no monetary-cost claim. The localhost API
address is a candidate placeholder for the current model service; a remote execution host still
needs a verified route and separately configured authentication. No network service was exposed.

## Validation performed

Two complete preparations are byte-identical for contracts, job files, source inventories and schema
receipts. Both job schemas and agent option schemas validate with installed Harbor 0.24.0. A real
Terminus-2 constructor confirms the 32K context and 4K output limits without calling the model or
setting up an environment. Fifteen targeted tests cover matched-arm equality, task membership,
endpoint/credential handling and preservation of task semantics while pinning images.

`schema-environment.json` records the exact dependency artifacts used for this **ARM64 local
validation**. It must not be presented as an AMD64 task execution image or its lockfile. The
original installation report and full local staged tasks remain on the development host. Published
files contain configurations, hashes and receipts, not task contents. `source-receipt.json` binds
the upstream config, agent, LLM and environment sources inspected to establish these semantics.

## Reproduce preparation

Python 3.11 or later is enough for source preparation. For schema validation, use an isolated Python
3.12+ environment with Harbor 0.24.0; do not install its dependencies into the model's runtime venv.
First retrieve and verify the source/image assets with the scripts in the previous report. Then run:

```bash
python scripts/prepare_pyramidkv_terminal.py \
  --assets /path/to/terminal-assets --output /path/to/new-plan \
  --api-base http://127.0.0.1:8000/v1 --validate-harbor
```

The generated job files use paths relative to the plan directory. After all execution gates are
closed and the endpoint/authentication are bound, execute the frozen jobs from that directory with
Harbor's `run -c job-b0.json` / `run -c job-b1.json`, restarting and verifying the corresponding
model arm separately. The preparation command never invokes Harbor `run` and cannot produce task
scores. Do not rewrite the endpoint, budgets or staged task files after observing scored outputs.

## Still required

The execution operator must provide an AMD64 Docker/Harbor backend, resource enforcement and model
connectivity, then freeze that environment's image/dependency identity. Benchmark #254 maintainers
coordinate task/third-party rights and canonical contract review; @Irisuko coordinates MOD
integration. Reference-solution and failure behavior must be validated in fresh sandboxes before
scoring, including the exact verifier reward schema. All 89 tasks and failures must be accounted
for; a nonbinary or missing reward cannot silently become a passed task or disappear from the
denominator.

No real B0/B1 trial, transport validation or verifier validation was performed by this preparation.
These external and execution-dependent gates remain explicit. The report does not promote the
applicability matrix, official v0.18 FP16 leaderboard, website readiness, or any compression
benefit.
