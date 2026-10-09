# FrontierScience candidate execution and grading protocol

This advances [PyramidKV #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8) and
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254). The implementation
now prepares solver inputs, collects raw responses, invokes a separately configured judge, parses
grades, and aggregates the tracks independently. **No real solver or judge calls were made for this
report; no task score or compression benefit is reported.**

The 100 Olympiad and 60 Research rows retain the earlier frozen source IDs and prompt hashes. All
problem-only prompts remain below the 4,096-token prefill compression threshold. The prepared
protocol is a compatibility evaluation, not an attempt to manufacture compression applicability. The
implementation is an independently authored candidate informed by the
[FrontierScience paper](https://cdn.openai.com/pdf/2fcd284c-b468-4c21-8ee0-7a783933efcc/frontierscience-paper.pdf),
not OpenAI's official evaluator configuration. Olympiad checks reference-answer equivalence;
Research uses its reference rubric, 0–10 points, and success at 7 or more.

## Frozen local behavior

- One solver attempt per source row, temperature 0, seed 17, 4,096 output tokens, no tools, thinking
  disabled, 600-second request timeout and no retry. References are absent from solver prompts and
  enter only the separate judge request.
- Length-limited solver responses remain available for grading and are flagged. Request failures
  remain in the task denominator as incorrect attempts.
- Each track has its own judge instruction. A verdict must occur exactly once, on the final line.
  Ambiguous, malformed, nonfinite or out-of-range grades fail. Judge failures and missing grades
  leave the track score null rather than dropping tasks or fabricating zero quality.
- Research is reported separately as success rate and mean points. Source row IDs prevent the
  duplicate Research group ID from dropping one of its 60 tasks.
- The collector verifies the plan hash, implementation hash, prompt inventory and model identity; it
  records deployment provenance and raw responses. Runtime provenance still needs a fresh capture
  for each actual B0/B1 launch. This does not by itself validate serving optimizations.
- The grader requires an immutable judge model identifier, runtime identity, matching instruction
  hashes, and a checksum-bound passing semantic calibration receipt for both tracks. It verifies the
  model identifier returned by the endpoint and retains all raw grading requests/responses. The
  fixed local judge call budget is 4,096 tokens, temperature 0, 600 seconds, no retries.

Parser and HTTP fixture tests are software checks, **not semantic judge calibration or measured
benchmark tasks**. A calibration receipt must come from an actual independent calibration run, with
its examples, expected assessments, raw responses and runner provenance retained externally; its
receipt has `judge_model_revision`, `instructions_sha256`, `passed`, and `tracks` fields. The grader
checks the receipt's binding and declared pass status; it does not independently certify its
scientific validity. This artifact must not be substituted with the synthetic test fixtures.

## Reproduction and execution

Install the repository and the qualified tokenizer dependencies, or set `PYTHONPATH=src` while
running from its root. Prepare from the fixed local source files and prior published inventory:

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.frontierscience_pyramidkv prepare \
  --assets /path/to/frontierscience \
  --audit reports/pyramidkv-frontierscience-inputs-20261009 \
  --model /path/to/Qwen3.5-35B-A3B \
  --output /path/to/new-plan
```

The command prints the SHA-256 of its generated `contract.json`. `solver-cases.json` contains source
problem token IDs and stays local. The published contract and preparation receipt record its
checksum. Before collection, freeze a judge specification and passing calibration receipt as
described below, then a deployment `runtime.json` containing `model_revision`, `sources`, `command`,
`hardware`, `method_config_sha256`, and the named `arm`.

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.frontierscience_pyramidkv collect \
  --plan /path/to/new-plan --contract-sha256 EXACT_PLAN_CONTRACT_SHA256 \
  --runtime /path/to/runtime.json \
  --judge-contract /path/to/judge.json --judge-contract-sha256 EXACT_JUDGE_CONTRACT_SHA256 \
  --calibration /path/to/calibration.json --output /path/to/new-solver-run
```

A real judge specification must supply `model_revision`, `runtime_identity`, `instructions_sha256`
copied from the frozen contract, and `calibration_sha256`. Freeze that JSON and the actual semantic
calibration receipt before solver generation. Supply credentials through `FRONTIER_JUDGE_API_KEY` if
the selected endpoint requires them; credentials are not written to artifacts. The endpoint must
support the frozen request parameters and return the pinned identity.

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.frontierscience_pyramidkv grade \
  --solver-run /path/to/solver-run --contract-sha256 EXACT_PLAN_CONTRACT_SHA256 \
  --assets /path/to/frontierscience \
  --judge-contract /path/to/judge.json --judge-contract-sha256 EXACT_JUDGE_CONTRACT_SHA256 \
  --calibration /path/to/calibration.json --judge-url https://JUDGE_ENDPOINT/v1 \
  --output /path/to/new-grades
```

All output directories must be new. Keep raw grading requests local unless source redistribution has
been reviewed: they contain reference answers/rubrics. Report the two tracks separately and include
raw response hashes, failure counts, deployment and judge provenance. Do not relabel this candidate
protocol as an official FrontierScience configuration or a v0.18 FP16 leaderboard result.

## What still needs external resources

The current process environment has no configured common external judge credentials. A usable judge
endpoint, pinned model/runtime and actual semantic calibration are still needed before real scores.
Installing a parser or passing the unit suite does not remove that requirement.

For SWE-bench-Pro and Terminal-Bench, this pod has no Docker/Podman/Harbor or Docker socket. A user
namespace probe succeeds, but the combined user/network/mount namespace probe fails with permission
denied, and `/dev/fuse` is absent. A supported isolated execution backend or changed host
permissions is still required; merely installing the CLI would not demonstrate readiness. Task/image
manifests and adapters can continue independently.

The fixed SWE-bench-Pro dataset card **does have a license section**: its harness/tooling use MIT
and task contents retain the licenses of the eleven upstream repositories. A blanket claim of "no
license" is inaccurate. The remaining work is per-repository/task rights and provenance
verification. The missing separately mounted asset bundle is not a hard blocker for publicly
retrievable source files. See `resource-probe.json` for the checks made on this pod.

The collector writes a hash-bound `collection-contract.json` before the first solver request.
Grading rejects a different judge or calibration receipt, preventing post-hoc grader selection.
Preparation with unresolved judge fields is allowed; real collection is blocked until they are
fixed.
