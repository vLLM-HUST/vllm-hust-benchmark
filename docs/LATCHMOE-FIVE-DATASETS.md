# LatchMoE five-dataset integration boundary

This is an **offline evidence-contract candidate**, not a five-dataset server runner, live worker
collector, model qualification, reviewed scorer contract or V5.4 acceptance. See LatchMoE #94 and
its children #96–#100, and benchmark #254.

## Reuse without transferring another MOD's claims

| Dataset            | Reusable code                                                                                              | Still required before execution/claims                                                                                                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MMLU-Pro           | `mmlu_pro_pyramidkv.extract_answer`, `format_example`, `encode_prompt`; `latchmoe_dataset_pair.score_mmlu` | Freeze exact source, 5-shot prompt/tokenizer, full or named subset IDs, generation parameters and scorer; collect real outputs. Do not reuse PyramidKV activation or compression-eligible selection.    |
| HLE-Verified       | `scripts/audit_pyramidkv_hle.py` source/modality audit                                                     | Choose verified full vs Gold; rights, image protocol, evaluator/judge revision and calibrated scoring. Removing 93 Gold image rows is not complete Gold.                                                |
| SWE-bench-Pro      | `scripts/audit_pyramidkv_swe.py`; frozen Pro V2 task harness                                               | Task rights, AMD64-capable isolated backend, original test-patch bytes, pinned agent/budget, fresh sandbox for each run and independent regrade.                                                        |
| FrontierScience    | `frontierscience_inputs` and candidate `frontierscience_pyramidkv` solver/judge primitives                 | Separate Olympiad/Research contracts and every source row; real judge identity/budget/calibration; replace method-specific protocol/activation before reuse. Fixture verdicts do not calibrate a judge. |
| Terminal-Bench 2.1 | `scripts/audit_pyramidkv_terminal.py`, candidate Harbor configuration builder                              | Task rights, pinned Harbor/Terminus/tools, AMD64 sandbox/image layers, actual verifier reward. Replace PyramidKV model/job names; a B0 filename does not switch the service off.                        |

Existing PyramidKV model outputs, 4096-token compression thresholds and transport-only diagnostics
do not establish LatchMoE quality or path execution. Keep the canonical program `asset-frozen` until
its review/execution gates are met. Do not edit leaderboard scores or website snapshots from these
diagnostics.

## Candidate paired receipt format

`python -m vllm_hust_benchmark.latchmoe_dataset_pair verify-pair --contract CONTRACT.json --b0 B0/receipt.json --b1 B1/receipt.json`
checks **offline consistency only**. CLI errors exit 2. The contract uses schema
`latchmoe-dataset-pair-v1`, an exact primary dataset, and for FrontierScience exactly one of
`olympiad` or `research`. Dataset `provider`, `source`, `revision` must match `SOURCE_IDENTITIES`;
its `track` must match the top-level track. FrontierScience additionally binds `source_sha256` to
the immutable per-track raw source hash. These are source bindings, not proof of local asset
verification. See `COMMON_FIELDS` for required
model/runtime/hardware/dataset/server/evaluator/scaffold/budget fields. Revision fields must be
immutable reviewed identities in real evidence, not labels such as `main` or `latest`.

- `common` is identical in the contract and both effective server receipts. Include all relevant
  package/source/image identities, server settings (APC/MTP/compile/async/parallel/KV), scaffold,
  sampling, retries, tools and budgets as additional frozen fields. The gate's mandatory subset is
  not a complete dataset execution contract.
- `requests` is the ordered immutable list of `{task_id, prompt_sha256, parameters_sha256}`; it is
  identical for both arms. Hash actual prompts/parameters before execution. Research rows with
  repeated group IDs remain distinct tasks. Hashes alone do not prove the underlying values.
- `mod_config` freezes `num_slots`, `indexed_mode` and positive `offload_gb` and any additional
  method settings. `effective_mod` in each receipt equals this configuration plus boolean `enabled`;
  B0 has `enabled=false, offload_gb=0`, B1 has `enabled=true`. Collect from actual server/worker
  state, not launch intentions. Do not hide other MOD activation in either arm.
- `artifacts` maps evidence-root-relative file paths to SHA-256 hashes. `outcomes_path` points to
  raw normalized task outcomes (`task_id`, `prompt_sha256`, `parameters_sha256`, boolean `success`,
  retained `error` on failures). Outcome request hashes must match the manifest; hash verification
  and JSON parsing consume the same captured bytes. Every task remains in both arms, including
  failures. These transport outcomes are **not** SWE/Terminal task resolution or HLE/Frontier
  judgement; preserve the original outputs and dataset-specific grading records as additional hashed
  artifacts.
- `telemetry_path` points to a raw normalized receipt containing worker/process identity,
  `phase=measured_requests_only`, `request_manifest_sha256`, and `before`/`after` nonnegative
  monotonic counters `calls` and `waves`. The collection boundary excludes load/warmup/capture and
  brackets this exact request cohort. Bind all worker/rank/layer records and their raw files in the
  live collector; the gate does not collect or authenticate them. A shared counter reset, process
  replacement or ambiguous workload boundary invalidates the receipt.
- Native B0 counters are zero, including the before snapshot. B1 requires positive measured call and
  wave deltas for `matched-evidence`; enabled with zero deltas is `not-exercised`. The current
  counter convention is for the fixed-slot wave path; another dataplane needs a separately reviewed
  telemetry schema, not made-up zero/native counters.

The output always has `quality_scores=null` and `formal_v5_4_accepted=false`. A hash-valid
self-reported receipt is not proof of successful inference or truthful instrumentation. Missing arms
fail; this gate cannot certify a signed release, OCI image, judge, task rights or sandbox. Current
MOD TP2/indexed guards must be qualified before any live collector/paired runner starts.

## MMLU scorer-only connector

`python -m vllm_hust_benchmark.latchmoe_dataset_pair score-mmlu --cases cases.json --results results.json`
reuses canonical exact answer extraction, cardinality/request matching and failure-inclusive
accuracy. It strips the PyramidKV compression-eligibility subdivisions; LatchMoE's expert caching is
not a long-context-only optimization. Scorer fixtures are not real model results. This command does
not bypass the pair, source, rights or human-review gates.

## Real execution order

1. Resolve blockers, independently verify assets, select reviewed contracts and qualify the actual
   Qwen3.5 BF16 TP2 MOD cell. V5.4 additionally requires signed release and OCI identity.
1. Freeze all task IDs/prompt assets/scorer/scaffold/parameters/budgets before outputs.
1. Run B0/B1 in fresh task-owned processes/sandboxes, capture effective state and request-window
   worker telemetry, retain every raw success/failure and grade in the appropriate track.
1. Check paired receipts plus dataset-specific grading; preserve positive, zero and negative
   effects. Publish immutable canonical evidence before website linkage. Never publish eager
   diagnostics as official graph-mode leaderboard data.
