# Benchmark methodology

## Matched comparison

Baseline and candidate use the same model, request order, hardware, runtime,
dtype, Graph configuration, and source revisions. Model loading, host/device
transfer, compilation, and warmup are excluded from steady-state operator
timing. Device timing synchronizes the NPU around every measured sample.

End-to-end experiments warm both arms with the same workload before recording
results. Reported tables preserve individual run IDs and include means and
sample variance/standard deviation. Correctness must pass before performance is
interpreted.

## Required gates

1. Supported dtype and aligned/non-aligned shape correctness.
2. Finite output, repeat determinism, and comparison with the native result.
3. At least three matched end-to-end rounds for a production-facing claim.
4. Operator profiling to show that the targeted work actually changed.
5. Fail-closed behavior for unsupported shapes and unknown source revisions.

## Current platform

The checked-in results were collected on Ascend 910B2. Each optimization page
records its exact CANN, Torch, torch_npu, Triton Ascend, vLLM, and
vLLM-Ascend identity. Results are not generalized beyond those boundaries.
