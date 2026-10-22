# Experiments

The scripts in this directory are reference entry points for reproducing the
checked-in evidence on a separately prepared Ascend environment. They do not
download models or mutate system packages.

## Persistent MatMul allocation

- `persistent_matmul_empty/correctness.py`: shape/dtype/tail correctness gate.
- `persistent_matmul_empty/run_e2e.sh`: matched model benchmark entry point.
- `persistent_matmul_empty/run_profile.sh`: operator profiling entry point.
- `persistent_matmul_empty/prepare_datasets.py`: deterministic dataset subset.

## Linear + SwiGLU fusion

The fused-kernel implementation and graph matcher live under
`src/vllm_hust_operator_optimizations/optimizations/`. Compact synchronized
operator samples and the matched model-level result are under
`results/linear_swiglu_graph/`. The environment-specific service orchestration
is intentionally not embedded because it contained server-local paths; follow
[`docs/benchmark-methodology.md`](../docs/benchmark-methodology.md) and provide
all paths explicitly in your runner.

Every new optimization should add an independent experiment directory, a
portable result summary, and an evidence-boundary document.
