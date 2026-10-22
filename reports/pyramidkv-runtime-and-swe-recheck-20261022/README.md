# PyramidKV runtime recovery and SWE runner compatibility audit

This report follows [PyramidKV #1](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1)
and [#8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8). The report date is
Asia/Shanghai; machine timestamps are UTC.

## SWE execution entry: reusable implementation, incompatible frozen inputs

Benchmark [#314](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/314) delivered an independent
Agent entry, pinned input verification, per-step records, and fresh-container official grading.
[#318](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/318) subsequently recorded a submitted
single-task experiment with official result 0/1. These are useful shared assets, not PyramidKV
results.

The audit inspected benchmark commit `08e20fd6e5ad25badb8eb1c44340e6e48297e48e`.
`swe-compatibility.json` records the inspected source hashes and input identities.

| Contract                    | Frozen PyramidKV input                                             | Shared runner                                                      |
| --------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Dataset configuration       | V2, 642 tasks                                                      | V1                                                                 |
| Dataset repository revision | `2d52cb3df914a3fcf80c7f66738b3a88ae37fc50`                         | Same repository revision                                           |
| Parquet SHA-256             | `1067f72c1f43bbc578276888625db3b836cebb535546c9eaf824c5d5dd104a7f` | `c8cd7115496ad4e9a8b21d088cef576a65bf821bb542b24336f13f714cef13f8` |
| Task-image platform         | `linux/amd64`                                                      | `linux/arm64`                                                      |
| Evaluator                   | V2-specific qualification still required                           | Locked V1 source coverage                                          |

The shared repository revision does not make the two configurations interchangeable. The actual
retained V2 parquet was rehashed and passed to the unmodified shared runner's `prepare` function,
using the repository's example settings with only the dataset path replaced. It rejected the file at
`swe_bench_pro_inputs.py:98`, before source preparation, Docker, or model calls. This expected
negative probe is recorded in `v2-rejection-probe.json`; it is not an Agent execution or a prepared
task. The runner also explicitly checks ARM image architecture in `verify_inputs`.

No dataset substitution, removal of input checks, ARM port, or expansion of the V1 result to V2 was
performed. The reusable design is the independent Agent entry and evidence flow. Applying it to the
frozen PyramidKV experiment still requires V2-specific task/evaluator/image qualification, reviewed
agent/tool/budget and rights contracts, and an accessible isolated AMD64 execution backend with a
network route to the model. Local Docker/Podman and the default Docker socket were absent; the
bounded environment-name probe is recorded in `backend-observation.json` and does not prove that no
remote backend exists elsewhere.

## Scoring and release boundaries

The user retained the existing judge-calibration requirement on October 19. Same-model development
grading in benchmark #278/#280 does not authorize relaxing that requirement for PyramidKV. No new
HLE, FrontierScience, SWE, Terminal, or MMLU quality score is reported here.

Shared [PR #23](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/pull/23) is merged, but the
public package check still returned the original 0.10.0 wheel with SHA-256
`456a2bb800dc2cd67a8604495c9235748a9260b0da36b3c84ef1e5f5c7046710`. The installed candidate shared
wheel is a different artifact despite its equal version string. `runtime-artifact-recheck.json`
rehashes all three retained wheels and compares their installed Python/JSON files: shared 23,
PyramidKV 7, Manager 31. This checks the existing candidate installation; it does not qualify the
defective public wheel or widen PyramidKV main's dependency range.

## Candidate service recovered after device reassignment

The previous endpoint was unreachable and its model processes were absent. Current `npu-smi`
reported two idle 910B2 devices, physical 3 and 6, before this launch. The retained candidate
environment was restarted using runtime ordinals 0 and 1. Its actual TP0/TP1 worker PIDs 35910/35911
were subsequently observed on physical 3/6. The cgroup has an 8-CPU quota and a 64-GiB memory limit.

The model is Qwen3.5-35B-A3B in BF16/TP2, using the previously qualified core/Ascend source pair,
32K context, APC, async scheduling, MTP2 and FULL_AND_PIECEWISE graphs. Each worker has an 8-GiB KV
pool. PyramidKV keeps budget 1368, beta 2, window 8, kernel 7 and the 4096-token prompt threshold.
`host-source-recheck.json` records current source heads and working-tree status; the existing
untracked Ascend native-build directory remains present. Model weight files were not fully rehashed.

The original strict smoke runner completed successfully:

- Three actual chat requests returned HTTP 200 and their exact expected short/Chinese outputs.
- Both long requests had 5312 semantic tokens and committed 2048 physical tokens, matched by request
  ID to both TP workers and the scheduler.
- The repeated request reported 2048 cached prompt tokens.
- MTP counters increased by 14 draft tokens and 10 accepted tokens.

`smoke-summary.json` and the raw requests/responses, before/after metrics and matching logs preserve
these observations. They establish functional recovery only. Request timings are incidental smoke
telemetry, not a throughput comparison. The candidate service was left healthy and running; the old
environment and prior evidence remain preserved. No new rollback exercise or public-package
acceptance was performed. The startup-only cache-advice monitor stopped at readiness.

## Evidence verification

`raw.tar.gz` holds the captured launch configuration, startup log, smoke script, requests,
responses, metrics and NPU observations. `raw-members.json` records each exact member's size and
SHA-256; archive members were read back and verified. `SHA256SUMS` covers the report files. Existing
SWE runner and V2 audit tests passed (31 tests); the real V2 rejection probe is recorded separately.

The report contains only this audit and the separately recorded runtime functional check. It adds no
performance comparison, capacity claim, official score, or production-release acceptance. Historical
failed and negative results remain unchanged.
