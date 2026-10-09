# PyramidKV: five-dataset adaptation and MMLU-Pro compatibility evidence

This is the canonical experimental report for
[PyramidKV #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8), linked to
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254). The matrix and named
evaluation contract are submitted for maintainer review. They do not promote the organization-wide
five-dataset program to completed or executable, and do not populate the official v0.18 FP16
leaderboard. This run uses a separate BF16 v0.23 graph-mode profile.

## Applicability

[applicability.json](applicability.json) records all five exact dataset revisions, blockers, owners
and unblock conditions. The canonical asset audit at
[e3a7183](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/e3a7183206bd399447475b2d117fef897c0880b6/reports/pujiang-specified-datasets-20261009)
remains authoritative for the collected source bundle. This pod does not mount that bundle; we
re-downloaded only the exact MMLU-Pro test/validation files and verified their upstream SHA-256
values. We do not claim to have repeated the other host's 1,135-file verification here.

| Dataset            | Status          | Concrete remaining work                                                                                                                                                               | Owner / coordination                        |
| ------------------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| MMLU-Pro           | `not-exercised` | All 12,032 frozen five-shot prompts are 898–2,910 tokens, below the >4,096-token prefill threshold. Review the bounded task-accuracy contract; full-test accuracy remains unmeasured. | @Irisuko; benchmark #254 contract review    |
| HLE-Verified       | `blocked`       | Canonical rights gate; choose full vs Gold; freeze task IDs, judge model/prompts, scorer, image handling and runtime.                                                                 | @Irisuko; #254 rights/scorer maintainers    |
| SWE-bench-Pro      | `blocked`       | Canonical rights gate; task/repository/image mapping; provision fresh isolated sandboxes; freeze scaffold, tools, budget and regrader.                                                | @Irisuko; #254 runtime/scaffold maintainers |
| FrontierScience    | `blocked`       | Freeze separate Olympiad and Research manifests, answer extraction and grading contracts/environments.                                                                                | @Irisuko; #254 scorer maintainers           |
| Terminal-Bench 2.1 | `blocked`       | Provision Harbor-compatible sandbox backend; pin task images, scaffold, tools, budgets, scorer and task rights.                                                                       | @Irisuko; #254 runtime/scaffold maintainers |

Owner names identify coordination responsibility from issue #8, not a claim that maintainers have
approved this matrix. [execution-capabilities.json](execution-capabilities.json) records that this
pod has no Docker executable/socket or Harbor executable. Local absence is not a claim that the
canonical source assets are missing organization-wide. No score is manufactured for blocked
datasets.

## Named MMLU-Pro contract

The adapter is [`mmlu_pro_pyramidkv.py`](../../src/vllm_hust_benchmark/mmlu_pro_pyramidkv.py).
`prepare` freezes all 12,032 test IDs, source-row hashes, actual token counts and prompt hashes in
`inventory.jsonl.gz`; `tasks.json` freezes the selected cases without publishing the source
questions. The selection is the first five numeric test IDs per each of 14 categories, plus the
first naturally compression-eligible ID in each category if not already selected. No such eligible
ID exists, so the subset contains 70 tasks. Selection uses only input metadata, before any model
outputs. It is a coverage-oriented compatibility subset, not a random/representative sample or full
MMLU-Pro score.

Five category-specific validation demonstrations (70 total) are included in source order. The
upstream prompt template and three-stage A–J extraction are pinned to TIGER-AI-Lab/MMLU-Pro revision
`f418b116db00b065c2aea046518d8fcf74d39872`. The source's Apache-2.0 license is preserved in
[MMLU-Pro-reference-LICENSE.txt](MMLU-Pro-reference-LICENSE.txt); the pinned dataset card declares
MIT.

The metric is **strict extracted-answer accuracy on this named subset**. Unlike the reference
runner, missing/invalid/out-of-option answers are incorrect instead of randomly guessed. Transport
failures remain in the denominator, with raw failure records preserved. All cases are reported,
including length-limited answers; an extracted valid answer can still count as correct when
length-limited. The upstream extractor's final fallback can select a standalone choice letter in
reasoning; this limitation is retained and documented rather than silently changing the scorer after
seeing outputs.

The Qwen chat template wraps the five-shot CoT prompt with thinking disabled. The target task's gold
answer/rationale is never put in its prompt. Generation is temperature 0, seed 17, max output 2,048,
concurrency 1, no tools and no retries. Unlike the upstream local runner, this profile uses a 32K
context, keeps all five demonstrations and does not add a `Question:` stop string. Over-context
prompts are rejected; no padding or silent truncation is used. These differences prohibit treating
this result as the full upstream leaderboard configuration.

## Runtime and evidence boundary

Both arms use Qwen3.5-35B-A3B revision `59d61f3ce65a6d9863b86d2e96597125219dc754`, BF16 TP2, actual
910B2 physical devices 6/7 (logical 0/1), FULL_AND_PIECEWISE graph capture, MTP2, APC, 32K context
and 8 GiB KV allocation per worker. The identity/command/config/package/image hashes are in the
frozen contract. `VLLM_VERSION=0.25.1` selects an integration ABI; the installed core is
0.23.0+empty and Ascend is 0.23.0.post1. Do not relabel the installed runtime as v0.25.1.

B0 is a fresh server with compression disabled; B1 is a fresh server with compression enabled. The
same loaded provider/plugins and configuration are used in both; only activation differs. The
explicit aligned profile is max_capacity_prompt=1368, beta=2, window=8, maximum aligned physical KV
length 2,048, prefill threshold 4,096. This is not the default 512/beta20 profile. The patched
shared adapter is pinned at `19f322130e1b3841da953b1b421bcf3259e9b8dc` for this host.

Both arms use the same startup-only policy: when the pod's 64 GiB cgroup memory usage exceeds 48
GiB, advise eviction of local model-file page cache with POSIX_FADV_DONTNEED. Events are recorded,
and no cache advice runs during measured requests. Previous workers must release HBM before the next
server starts. B0 precedes B1; this is one paired run, not repeated randomized performance
estimation.

The running container image digest is recorded from this pod's Kubernetes status. Its registry is
private, so independent replay requires registry access or an exported image plus the pinned source
overlays. A digest does not imply public image availability. The runner verifies actual process
argv, activation and selected environment variables, per-request prompt identity and task coverage.
The shared analyzer checks scheduler transactions and TP acknowledgements; for this workload it must
observe zero compression transactions. Telemetry is supporting evidence, never a replacement for
task accuracy or proof of optimization.

The prior LongBench evidence remains supplementary and is linked from
[PyramidKV PR #10](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/pull/10). Its KV-page
reduction must not be attributed to these five datasets. This report makes **no optimization
claim**.

## Publication order

Canonical artifacts and checksums are submitted here first. Website synchronization and global
program readiness changes remain pending canonical contract/matrix review. The four blocked datasets
require their stated rights/scorer/runtime contracts before any task-quality score is admitted.
