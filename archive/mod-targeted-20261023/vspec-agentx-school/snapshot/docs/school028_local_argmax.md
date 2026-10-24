# School 0.28 MTP2 Local Argmax

This default-off experimental adapter isolates the existing
`_sample_mtp_local_argmax` mechanism. It does not activate the full vSpec patchset,
adaptive gamma, cohort refill, state remapping, or a different drafter.

## Configuration

- Qwen3.5-35B-A3B BF16, TP2, native MTP2, greedy sampling.
- Current source-pinned BetterScale E24/R36 + KV recompute + FA Microflow B1.
- Original MMLU-Pro 200-request burst, infinite arrival rate, out256.
- APC, async, FULL graphs and the existing state/host budgets remain unchanged.

The additional bundle is `org.vllm-hust.vspec-school028-argmax`. Extension Manager
must also admit `org.vllm-hust.prefill-microflow-school028-fa`; its full source
preflight is reused. The school adapter additionally checks the exact proposer,
logits processor, model interfaces, Qwen3.5 MTP and FIA publisher source hashes. The proposer pin
is the prepared BetterScale donor, not the unmodified installed base package.
No installed host file is edited.

Select `HUST_VSPEC_SCHOOL_MODE=off` or `on` and a writable
`HUST_VSPEC_SCHOOL_EVENT_DIR` in an isolated process. The Manager activation adds
`vspec_school028_argmax` and `HUST_VSPEC_SCHOOL028_ARGMAX=1`; direct activation of
the full `vspec` entry point is rejected. Restarting without the bundle restores
native behavior.

## Qualification

Both arms check real BF16 head weights on both TP ranks, comparing full-logit
sampling with local argmax for 1, 24, 36 and 48 rows. Zero-valued hidden states
exercise tied logits; deterministic nonzero hidden states exercise ordinary
selection. Any token mismatch aborts model loading. These checks do not replace
whole-task accuracy evaluation.

The observer records selector execution by shape and actual replay of the
native merged graph inside the pinned FIA publisher's `inner` callable, rather
than mistaking the draft model or outer publisher for the ACL graph.
Performance admission additionally requires matching
source/import identity, all 200 original requests, Microflow and KV control
receipts, no foreign NPU load, paired cache initialization, clean drain and device
release. Startup failures are diagnostic records, never performance points.

## Status

Source commit `d8cbfb50ee1df8e1664bd080a59e49ed2e47c104` completed three independent
OFF/ON pairs on the original load: 1,200 successful requests, zero failures.
Both ranks passed the real-head equivalence checks and recorded selector capture
and actual merged-draft graph replay during each measurement. Microflow and KV
control, source/import hashes, independent cache initialization, drain and device
release also passed. The full vSpec bundle was not activated.

The ON median output throughput is 817.670406 tokens/s, versus 816.579073 OFF.
The geometric mean of paired throughput ratios changes by -0.009906%, while ON
is 0.875773% below the existing Microflow B1 median of 824.894609 tokens/s.
No metric replaces the current B1. This near-zero paired change is not evidence
of statistical significance or a general claim about other vSpec components.
Task accuracy was not rescored; bounded head checks do not establish whole-task
quality equivalence.

The [benchmark evidence](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/9ab6413d1ea47587953f037d42fea568f4a5a201/reports/vllm-hust-midterm-20261019/vspec-school028)
contains all six raw runs, the predeclared plan, verification tools, and separate
failed startup/qualification archives. `summary.json` in that directory is the
machine-readable source for these figures. Early source-pin, event-logging and
graph-observer errors are preserved, excluded from the paired results, and fixed
in the measured commit.

Targeted installed-package tests: 18 passed, 1 skipped. Python 3.11/3.12 package
CI passed. The unrelated global-runtime full-suite collection fails on the
pre-existing Qwen2 Eagle `WeightsMapper` ABI; it is not reported as a passing
full runtime suite.
