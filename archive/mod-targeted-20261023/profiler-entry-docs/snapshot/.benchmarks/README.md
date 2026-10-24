# Benchmarks

Benchmark and probe entrypoints for request lifecycle causal profiling.

## Expected Flow

1. Validate trace schema and attribution rules without hardware.
2. Run NPU6 existing-server probes that collect lifecycle timelines.
3. Pair hook-disabled and hook-enabled runtime runs under the same workload.
4. Run controlled live fault-injection experiments to measure attribution
   accuracy.

All run directories must include `run_metadata.json` and an evidence label.
Profiler-only evidence should be framed as diagnosis or artifact evidence, not
runtime optimization.

The current runtime-hook smoke gate is ready: `make npu6-runtime-hook-pair-plan`
aggregates checked-in hook-disabled/enabled NPU6 runs and reports complete
chains with all required stages. Treat that as readiness for controlled fault
attribution, not as final diagnosis accuracy.

Use `preflight_npu6_trace_probe.py` before an existing-server probe. It checks
the endpoint, authentication, model path, Ascend runtime root, NPU6 ownership,
trace export schema, and pinned workload submodule metadata without launching or
killing any service.

The repo also provides a non-secret vLLM-HUST dev-hub profile at
`.benchmarks/profiles/npu6_vllm_hust_trace.env`. The profile launches NPU6 from
the pinned parent-repo runtime path and does not enable runtime hooks unless
`VLLM_RLP_TRACE_EXPORT_PATH` is set by the caller. Use:

```bash
make managed-start
make managed-health
make npu6-trace-preflight
make top-tier-readiness
```

The preflight is readiness evidence only until real workload requests and trace
exports are collected.
