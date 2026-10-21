# OPset for vLLM-HUST

[![PyPI](https://img.shields.io/pypi/v/vllm-hust-opset)](https://pypi.org/project/vllm-hust-opset/)
[![CI](https://github.com/vLLM-HUST/vllm-hust-opset/actions/workflows/ci.yml/badge.svg)](https://github.com/vLLM-HUST/vllm-hust-opset/actions/workflows/ci.yml)

**OPset is a pluggable operator-optimization collection for the vLLM-HUST
Ascend backend, delivering kernel replacements, vertical fusion,
epilogue/prologue fusion, and the performance evidence for each change.**

OPset is distributed as an updateable vLLM-HUST Extension Bundle with
independently gated optimizations. Version `0.3.2` contains:

- `persistent-matmul-empty`: replaces the redundant `torch.zeros` allocation
  in Ascend batch-invariant Triton `linear_persistent` with `torch.empty`.
- `linear-swiglu-graph`: fuses adjacent unquantized gate/up GEMM and SwiGLU in
  the Graph path. The validated decode/small-batch range (`M <= 128`) uses the
  fused Triton kernel; larger dynamic prefill shapes fail closed to the native
  GEMM plus `npu_swiglu` path.

Both optimizations are limited to the batch-invariant Graph/Triton path and do
not change ordinary ACLNN MatMul or unrelated linear implementations. Existing
NPU evidence covers supported dtypes, divisible and non-divisible shapes, and
matched operator/end-to-end measurements. Unknown source structures are
rejected rather than patched.

## Architecture

The distribution exposes two entry points:

- `vllm_hust.extension_bundles` lets `vllm-hust-ext` discover, check and enable
  the `0.3-experimental` bundle.
- `vllm.general_plugins` installs a lazy import hook in each vLLM process. The
  hook changes only enabled target modules and has no device or network side
  effects at import time.

Optimizations are selected with `VLLM_HUST_OPERATOR_OPTIMIZATIONS`. The bundle
manager enables both shipped optimizations; either ID can also be selected
independently.

## Install and use

This distribution succeeds `vllm-hust-operator-optimizations`. Existing
Python imports, extension IDs, environment variables, and the
`vllm-hust-operator-check` command remain available. Upgrade an existing
installation with:

```bash
pip uninstall -y vllm-hust-operator-optimizations
pip install vllm-hust-opset==0.3.2
```

The package records the fixed experiment line in
`compatibility/verified-line.json`. This keeps current results reproducible
without repeatedly rebuilding the environment; the line will be refreshed and
revalidated periodically as the underlying stack advances.

```bash
export VLLM_SRC=/path/to/vllm
export VLLM_ASCEND_SRC=/path/to/vllm-ascend-hust
pip install vllm-hust-opset==0.3.2
vllm-hust-opset \
  --vllm-src "$VLLM_SRC" \
  --vllm-ascend-src "$VLLM_ASCEND_SRC"
vllm-hust-ext extension check org.vllm-hust.operator-optimizations
vllm-hust-ext extension enable org.vllm-hust.operator-optimizations
VLLM_BATCH_INVARIANT=1 vllm-hust-ext run -- vllm serve /path/to/model \
  --block-size 128
```

`TRITON_ASCEND_SRC` may additionally point to the locked Triton checkout. If
it is absent, the checker verifies the installed `triton-ascend==3.2.2`
wheel; this is the mode used by the validated server environment.

Repository maintainers can build with `uv build --no-sources --out-dir dist`
and use `scripts/install_verified_line.sh` to gate and install a local wheel.

If `VLLM_PLUGINS` is manually set, it must include both the Ascend platform
plugin name and `operator_optimizations`; otherwise vLLM intentionally filters
out this runtime entry point.

## Add another optimization

1. Add an `Optimization` implementation under `optimizations/` with a stable
   ID, target module and fail-closed `apply()` method.
2. Register it in `registry.register_builtins()`.
3. Add its ID to the manifest environment value if it should be enabled by
   this bundle version, or publish a separately selectable bundle profile.
4. Add unit correctness tests and NPU shape/tail, operator-share and matched
   end-to-end evidence.
5. Increment `_version.py` and the manifest `extension_version` together.

The framework deliberately keeps every optimization independent so a future
release can add, disable or reject one optimization without changing the
others.

## Validation

```bash
uv run --with pytest pytest
uv run --with ruff ruff check src tests
uv build --no-sources --out-dir dist
```

The linear-SwiGLU Graph path was validated with Qwen2.5-7B,
ShareGPT, three matched baseline/candidate rounds and operator correctness
coverage. Steady-state TPOT decreased by 3.82% and output-token throughput
increased by 5.24%. Release checks remain `extension list`, `inspect`, `check`,
and `run --dry-run`; full NPU validation is refreshed periodically or when the
optimization changes.

## Evidence and documentation

The installable package contains only runtime code and compatibility metadata.
Research methodology, compact raw results, manifests, and evidence boundaries
are maintained in this repository:

- [Evidence index](docs/evidence.md)
- [Benchmark methodology](docs/benchmark-methodology.md)
- [Persistent MatMul allocation optimization](docs/optimizations/persistent-matmul-empty.md)
- [Linear + SwiGLU Graph fusion](docs/optimizations/linear-swiglu-graph.md)
- [Experiment entry points](experiments/README.md)

Large profiler databases and service logs are intentionally excluded from Git.
Compact JSON results are repository-relative and contain the run identifiers,
means, variance, correctness status, and source/runtime identity needed for
review. Large evidence archives should be attached to a matching GitHub
Release with a SHA-256 checksum.

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
