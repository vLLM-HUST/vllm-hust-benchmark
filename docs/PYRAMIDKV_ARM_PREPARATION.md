# PyramidKV shared ARM contract preparation

The user approved preparation of a new SWE V1 / Terminal ARM64 contract on 2026-10-28. The older
V2/AMD64 audits remain historical evidence. Judge calibration requirements are unchanged. This
preparation does not start model, Agent, Docker, oracle or scoring processes, and does not grant
access to another thread's shared workspace. Coordinate actual execution with benchmark #403.

`scripts/prepare_pyramidkv_arm.py` adapts the existing SWE runner settings and Harbor/Terminus job
schema. It does not replace either evaluator. Python 3.11+ and PyArrow are needed for SWE input
preparation. Use a separate CPU environment, with `PYTHONPATH=src` from this repository.

## SWE V1

Obtain the frozen V1 parquet using the input identity in
`src/vllm_hust_benchmark/data/swe_bench_pro_v1_lock.json`. The preparer verifies the exact file hash
before loading rows; V2 is rejected. Supply explicit task IDs from that file, one repository per
plan, and a settings template based on `docs/examples/swe-bench-pro/single-agent.json`:

```bash
PYTHONPATH=src python scripts/prepare_pyramidkv_arm.py swe \
  --parquet /absolute/private/v1.parquet \
  --task-id "$EXPLICIT_TASK_ID" \
  --settings-template /absolute/private/settings-template.json \
  --output /absolute/private/new-swe-plan
```

The chosen tasks must have source-lock coverage for the original evaluator scripts and Dockerfiles.
The two output settings differ only in run identity. Both use the same task IDs, container
resources, Agent budgets, API route, model name, seed and generation settings; the adapter binds the
served name `qwen35-pyramidkv`, seed 17, temperature 0 and thinking disabled. Extra Controller
hooks, task instructions and embedded API keys are rejected. Authentication uses the shared runner's
`api_key_env` field.

The settings template is not proof of installed paths, model reachability or image availability. The
contract retains those gates. After an owner supplies the actual workspace, bind unique run IDs,
runtime paths, immutable images and the model route; freeze a new final contract, then use the
shared runner's `check`, `prepare` and `verify-inputs` flow. Do not run the candidate settings as
though that handoff had already occurred. Keep official grading and unresolved/failure outcomes
separate from runtime success. Record the selection rationale separately: explicit IDs alone do not
establish representativeness or prove that selection was independent of existing results.

## Terminal ARM64 derived image

Reuse the original audited Terminal-Bench 2.1 assets. Supply one explicit task and a historical ARM
control-plan reference carrying its immutable Docker image configuration digest and task-file
hashes. For example, `reports/terminal-openssl-arm64-controls-20261026/controls-plan.json`
identifies the OpenSSL derivation. Its existing successful controls are not current backend
qualification or a PyramidKV model score.

```bash
PYTHONPATH=src python scripts/prepare_pyramidkv_arm.py terminal \
  --assets /absolute/private/terminal-assets \
  --task-id openssl-selfsigned-cert \
  --arm-reference reports/terminal-openssl-arm64-controls-20261026/controls-plan.json \
  --api-base http://127.0.0.1:8000/v1 \
  --output /absolute/private/new-terminal-plan
```

The complete original source/image audit is rerun locally. Instructions, solution and verifier files
must match the supplied ARM reference. Staging restores executable modes and changes only the
`task.toml` image reference to the recorded image ID. Task resource/time limits remain unchanged;
the staged original Dockerfile is historical source, not an ARM rebuild recipe. `force_build=false`
is retained. Obtain the actual ARM image transport/build provenance from its owner and verify the
loaded architecture/configuration before use; this tool does not fetch, build or inspect an image.

The two Harbor jobs retain the existing candidate's context/output/scaffold contract and differ only
in artifact destinations. SDK and trial retries are disabled, but nested upstream Harbor/Terminus
retry decorators remain. Do not claim zero model-request retries.

## Boundaries shared by both candidates

Output directories must be new. Contracts record source and settings hashes and always report
`ready_for_scored_execution=false`, `backend_verified=false` and `server_activation_verified=false`.
These are preparation results, not a substitute for observed runtime admission.

The B0/B1 filename does not toggle compression. Matched server lifecycles, cache policy, actual
compression OFF/ON activation, input/context limits and request-matched scheduler/TP transactions
must be frozen and verified independently. The qualified v0.23 PyramidKV stack is an experimental
comparison, not the shared program's official v0.18 baseline. Neither task quality nor compression
benefit can be inferred from successful preparation, another MOD's outcomes or fixed trace replay.

SWE task rows and staged Terminal tasks stay private. Publish source identities, contracts, hashes
and permissible per-task outcomes in benchmark first, while preserving all failures and original
versus recovered scoring distinctions.
