# PyramidKV SWE V1 / Terminal ARM candidate preparation

The user selected the shared ARM route on 2026-10-28. The decision and preserved boundaries are in
[PyramidKV PR #25](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/pull/25). This report
records CPU-only preparation, not a task trial, backend deployment, execution approval, task score
or performance result. Original V2/AMD64 audits and judge-calibration requirements remain.

## Prepared candidates

The new [preparer](../../scripts/prepare_pyramidkv_arm.py) reuses the shared SWE runner's validation
and the existing Harbor/Terminus configuration builder. It starts no model, Docker, Agent, oracle or
grader process. Source/settings hashes bind the checked code and candidate outputs.

- SWE: the actual V1 parquet matches the shared immutable SHA-256; it contains 731 rows. One
  explicit NodeBB task has the required evaluator/Dockerfile source-lock entries. It is the shared
  bootstrap example, selected for configuration integration, not as a representative or successful
  quality sample. The settings retain the example's resource/time/tool budgets and bind the
  PyramidKV served model, seed and generation settings. Runtime paths are unresolved template
  values.
- Terminal: the complete original Terminal source/image audit was rerun locally. The OpenSSL task
  was selected because a published ARM derivation reference is available for preparation. Task,
  solution and verifier bytes match that reference; staging restores Git executable modes and
  changes only the task image reference. This is not a claim that the ARM image is locally present
  or that old controls qualify a new backend. Harbor 0.24.0 validated both generated job and Agent
  option schemas without environment setup or model calls.

These two examples exercise the preparation interfaces only. They do not replace the five-dataset
objective, freeze a final scoring subset, establish representativeness, or authorize rerunning
another thread's tasks. The future task selection and rationale must be fixed before execution.

## Verification and handoff

`swe-contract.json` and `terminal-contract.json` bind the source, settings, selection and remaining
gates. Their `ready_for_scored_execution`, `backend_verified` and `server_activation_verified`
fields are all false. Both have zero model calls and zero executed tasks. The B0/B1 files differ
only in run/artifact destinations; they do not toggle server compression.

The preparation commands are documented in the [guide](../../docs/PYRAMIDKV_ARM_PREPARATION.md).
Outputs must use new directories. Each published settings SHA and the preparer's source SHA were
checked against the final files. `SHA256SUMS` covers these public preparation records. Task text,
answers, patches and credentials are not included.

Before execution, the resource owners must supply the workspace access method, effective
CPU/memory/storage/network limits and model route; exact ARM images and their transport/build
provenance; and a coordinated model window. The concrete input/scaffold/tool/budget and
fresh-sandbox grading plan still needs qualification. The historical image's config digest is an
identity reference, not proof of current image availability or a reproducible rebuild. Shared #403's
current review state and the distinction between trace performance and live task quality remain in
effect.

Related: [PyramidKV #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8),
[benchmark #403](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/403).
