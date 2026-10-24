# Issue 19 M0 serving preflight

The Issue #19 M0 runner must fail closed before it starts a service. Run:

```bash
uv venv --system-site-packages \
  --python "${ISSUE19_BASE_PYTHON:?set ISSUE19_BASE_PYTHON}" .venv-issue19
uv pip install --python .venv-issue19/bin/python \
  --no-deps -e .
uv pip install --python .venv-issue19/bin/python setuptools-rust
uv pip install --python .venv-issue19/bin/python \
  --no-deps 'torch-npu==2.10.0.post4'
VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0 \
  uv pip install --python .venv-issue19/bin/python \
  --no-deps --no-build-isolation -e third_party/vllm-hust
make issue19-m0-preflight ISSUE19_MODEL_PATH="$MODEL_DIR"
```

The dedicated ignored environment inherits the host CANN/Torch-NPU stack,
installs this profiler, and installs the pinned runtime as the repository's
empty carrier. Install Ascend's exact `torch-npu==2.10.0.post4` requirement in
this isolated environment; do not modify a shared environment. The preflight
prepends the two pinned submodules when it
inspects module origins; it does not accept an ambient vLLM checkout.

The command consumes the pinned OASST1 object and emits three local artifacts
under the ignored `.benchmarks/results/m0_issue19_preflight/` directory:

- `projection_manifest.json` contains the deterministic 10 × 64 request set;
- `execution_manifest.json` fixes seeds `2026082000..2026082009`, all six
  scenario targets, the 2.0 second deadline, and worker-exit-last ordering;
- `environment_manifest.json` records the pinned runtime and Ascend commits,
  upstream/release versions, exact Torch requirements, serving interpreter
  origins, model completeness, the real 640-request tokenizer context gate,
  and required observation seams.

The corrected specialty target is BF16 Qwen2.5-14B on one 910B2 with
`max_model_len=10880`, `kv_cache_memory_bytes=2147483648`, eager execution,
and `OffloadingConnector + TieringOffloadingSpec` with 8 GiB of CPU capacity.
The 2-GiB cache exposes exactly 10,880 block-aligned tokens for this model. The
earlier 32K/2-GiB combination was not executable: a 32K admission requires
6 GiB before any request can be served. This correction was made after a
startup-only smoke and before treatment or M0 observations.

The static readiness gate also requires the profiler, runtime, and Ascend
worktrees to be clean. This prevents a manifest from naming a commit that does
not yet contain the exact source being tested. During local implementation the
expected result is therefore `BLOCKED` until the reviewed changes are committed.

The static gate now verifies the model-derived KV capacity and runs a CPU-only
connector contract probe with Ascend's actual tuple-shaped `(k_cache, v_cache)`
representation. A module import is not enough. The runtime connector now
canonicalizes the two independently allocated tensors, and this probe remains
fail-closed against regressions that reject the tuple or omit either component.

Static admission and M0 admission are deliberately separate. With no static
blockers, preflight writes `SMOKE_READY.txt`, which permits a loopback service
smoke only. It writes `READY.txt` only after a separately supplied
`--service-smoke-result` proves that the exact connector configuration reached
HTTP health and produced a complete request-terminal and recovery trace with
zero data loss/writer failures. Otherwise `M0_BLOCKED.txt` remains present.

The execution plan is observer-only: `lifecycle_reconcile=false`, no treatment
is implemented, and worker exit may target only a verified child of the
loopback service launched for that repetition. A statically blocked preflight
writes `BLOCKED.txt`, removes stale readiness markers, returns status 2, and
must not start vLLM or inspect/kill an NPU process.

The selected repository-supported pair is the shared `0.23.0` carrier line in
both HUST repositories. Ascend's own installation documentation names vLLM
`0.23.0` and owns the serving Torch stack (`torch==2.10.0` and
`torch-npu==2.10.0.post4`). The runtime's `torch==2.11.0` pyproject entry is a
build-isolation input, not the active serving dependency in this empty-carrier
configuration. The preflight therefore requires the installed Torch and
Torch-NPU distributions to match Ascend, the empty vLLM carrier to report
`0.23.0`, both imports to originate from the pinned submodules, and a joint
source-compatibility probe to pass with Ascend's documented
`VLLM_VERSION=0.23.0` gate.

The runtime now includes a default-off lifecycle seam for request entry, exact
terminal causes, duplicate abort attempts, engine/worker generation failure,
and persisted bounded resource transitions. When observation is enabled, the
runtime records request KV-capacity leases and offload-transfer ownership;
fatal EngineCore cleanup invalidates still-open EngineCore leases before the
observer is closed. The disabled path does not allocate the auxiliary lease
map. Ascend retains the first-compute seam. None of these callbacks enables
reconciliation or changes scheduling/transfer behavior. A successful
preflight only authorizes the next service smoke; these artifacts remain
`smoke-only`, not real-online M0 or performance evidence.
