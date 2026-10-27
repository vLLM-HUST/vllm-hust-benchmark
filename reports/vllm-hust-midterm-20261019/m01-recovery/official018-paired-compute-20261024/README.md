# Official 0.18 / vLLM-HUST Own-Compute Pair

[Canonical summary](summary.json) contains independently reviewed same-run F, synchronized T and
per-chip C for the two-chip MMLU-Pro 200-request pair. B0 is official vLLM 0.18.0 / vLLM-Ascend
0.18.0 with disclosed text-only compatibility patches. B1 uses the identified development runtime
and combined vLLM-HUST optimizations, not an official 0.18 runtime.

Reproduce the arithmetic projection with `python scripts/build_m01_paired_compute.py`. Source
results and reviews are byte-preserved JSON copies. Absolute storage paths identify original custody
locations, not offline builder dependencies. No private prompt or output text is included here.

The
[public arithmetic package](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-r7/public-own-metrics-r2.tar.gz)
contains anonymous integer classes covering all 824131 B0 groups and an offline `python3 verify.py`
entry point. It verifies arithmetic and package identity, not the private dependency reconstruction
or model quality. The [remote readback receipt](public-own-metrics-r2-release-readback.json) records
179522 bytes and SHA256 `aa54a9e70da0fa4032a1ae7bbf938e2f919a8883c707fd875417bf51d5cc80a0`.

B0 r2 repairs Native Mamba aligned-block state-copy ancestry and supersedes the r1 F candidate. The
raw 200 requests and timing were not rerun or modified. Five final export hashes/sizes, 26
startup/final source identities and integer group totals were independently checked.

Compute gain is not a task-quality score, hardware peak utilization or independently calibrated
reference attainment. New whole-chain reference time, reference compute, eta and acceptance remain
null pending method and applicability review. Baseline normalization cannot fill those fields.

Original B1 operator-DAG results remain under
[their own method and evidence](../storage-reference-closed-20261024/README.md).
