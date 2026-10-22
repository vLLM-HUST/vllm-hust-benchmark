# Cascade graph signature-drift fix (D1–D4) — report

> **快照口径**：本文件是 2026-09-09 的修复报告。修复后的**未 shim** 重跑证据见
> `section2-correctness.md`（证据②）与 `section3-performance.md`（证据③）；现行启用状态以
> `section4-active-enablement.md` 为准。本文的代码锚点与数字不改。

- Plugin repo: `/vllm-workspace/vllm-ascend-split-batch-hust` (editable, only tree edited)
- Host (read-only, unmodified): vllm-hust v1 `0.28.1.post1.dev143+gf18cf803c`,
  vllm-ascend-hust main `0.25.1rc2.dev125+hust` — the same baseline the
  `EVIDENCE.md` audit used
- Env: conda `hust` (torch 2.13.0+cpu / torch_npu 2.13.0rc1, CANN 9.1.0,
  triton-ascend 3.2.2, `ascend-kernel==2026.3.9`)
- Card: **6 only** (`ASCEND_RT_VISIBLE_DEVICES=6`, HBM 5% at start of every leg)
- All raw artifacts: `logs/fix-sigdrift/`

## 0. Result

| Item | Result |
| --- | --- |
| D1 `_capture_cudagraphs` / `profiler` | fixed |
| D2 `_update_full_graph_params_if_needed` / `positions` | fixed |
| D3 `update_graph_params` / `num_dcp_pcp_tokens` + 7 positional | fixed |
| D4 `_model_forward` / required `num_tokens_padded` | fixed (see §5 caveat) |
| New regression tests | 13 (`tests/test_host_signature_drift.py`) |
| `python -m pytest -q` | **102 passed** (89 pre-existing + 13 new) — `logs/fix-sigdrift/pytest-ruff.log` |
| `ruff check .` | clean — same file |
| Mutation check (reintroduce each defect) | 4/4 caught — `logs/fix-sigdrift/mutation-checks.log` |
| Unshimmed graph-mode engine start | **`Application startup complete`**, `gate=1, graph_gate=1, kernel_wheel=ok`, 96 capture bodies, 63 cascade replays, 0 TypeError — `logs/fix-sigdrift/serve-on-card6.log` |
| Default-off zero-diff | `gate=0, graph_gate=0`, `cascade-active` 0, capture/replay 0, no errors — `logs/fix-sigdrift/serve-off-card6.log` |
| ON-vs-OFF sanity | token level **0/8** divergence (graph lane); serve text level 1/8; both ≤ historical criterion (§4.5) |

## 1. Defects and exact changes

All four were "import succeeds, signature does not match" failures against
private host seams (the failure mode `docs/pitfalls.md` §2.1 warns about). The
plugin wrappers are now signature-agnostic (`*args, **kwargs`), forward the host
call verbatim, and never let an internal failure escape a patched host method.

### D1 — `_capture_cudagraphs` lost `profiler=`

Host `vllm/v1/worker/gpu_model_runner.py:7059`:
`(self, batch_descriptors, cudagraph_runtime_mode, profiler=None)`; call site
`:6971` passes `profiler=profiler`. The old plugin wrapper declared only
`(self, batch_descriptors, cudagraph_runtime_mode)`.

- before `src/vllm_ascend_split_batch/cascade_runner_patch.py:177`
  ```python
  def _capture_cudagraphs(self, batch_descriptors, cudagraph_runtime_mode):
      orig(self, batch_descriptors, cudagraph_runtime_mode)
  ```
- after `src/vllm_ascend_split_batch/cascade_runner_patch.py:303-327`
  (`_make_capture_cudagraphs_wrapper`; wrapper def at `:313`)
  ```python
  def _capture_cudagraphs(self, *args, **kwargs):
      result = orig(self, *args, **kwargs)
      try:
          _capture_cascade_twins(self, args, kwargs)
      except Exception as exc:
          _warn_drift_once("cascade twin capture", exc)
      return result
  ```
  `_capture_cascade_twins` (`:210`) resolves `batch_descriptors` /
  `cudagraph_runtime_mode` by name with a positional fallback and raises into
  the wrapper's own handler when the host shape moved (fail-open, one warning).

### D2 — pre-model update passed the dropped `positions`

Host `vllm_ascend/worker/model_runner_v1.py:2935`:
`(self, forward_context, num_tokens_padded)`; `positions` was dropped upstream.

- before `cascade_runner_patch.py:288`
  ```python
  self._update_full_graph_params_if_needed(
      forward_context, num_tokens_padded, positions
  )
  ```
- after `cascade_runner_patch.py:381-384` (inside `_make_model_forward_wrapper`)
  ```python
  self._update_full_graph_params_if_needed(
      forward_context=forward_context,
      num_tokens_padded=num_tokens_padded,
  )
  ```
  Keyword call only: a future rename fails open (one warning, `orig` still runs)
  instead of forwarding a stale positional.

### D3 — `update_graph_params` had an extra parameter and forwarded 7 positionals

Host `vllm_ascend/attention/attention_v1.py:592`:
`(update_stream, forward_context, num_tokens, vllm_config, speculative_config=None, draft_attn_metadatas=None)`
(no `num_dcp_pcp_tokens`); the host's own caller `compilation/acl_graph.py:295`
passes 5 positional + `draft_attn_metadatas=`.

- before `cascade_graph_plugin.py:881-936`: wrapper declared
  `num_dcp_pcp_tokens=None` and every `orig_update(...)` call forwarded 7
  positionals.
- after `cascade_graph_plugin.py:782-846` (`_make_update_graph_params_wrapper`;
  wrapper def `:794`)
  ```python
  def update_graph_params(*args, **kwargs):
      ...
      try:
          if _step_is_cascade() and not rp._step_update_done:
              update_stream = _pick_arg(args, kwargs, 0, "update_stream")
              forward_context = _pick_arg(args, kwargs, 1, "forward_context")
              num_tokens = _pick_arg(args, kwargs, 2, "num_tokens")
              ...
              if graph_params is not None and graph_params.attn_params.get(cascade_key):
                  if not gate.decision_for(...):
                      return orig(*args, **kwargs)      # gate-off: verbatim
                  _update_cascade_graph_params(...)
                  return
      except Exception as exc:
          _warn_once("cascade replay update failed on this host build (...); "
                     "delegating to the original update pass", ...)
      return orig(*args, **kwargs)
  ```
  Both non-cascade and gate-off paths now forward the host call byte-for-byte.

### D4 — `_model_forward` required `num_tokens_padded`

Upstream call convention `vllm/v1/worker/gpu_model_runner.py:4550` is
keyword-only (`input_ids=, positions=, intermediate_tensors=, inputs_embeds=,
**model_kwargs`) — no `num_tokens_padded`. The old wrapper had
`num_tokens_padded` as a required leading positional.

- before `cascade_runner_patch.py:274-282`
  ```python
  def _model_forward(self, num_tokens_padded, input_ids=None, positions=None,
                     intermediate_tensors=None, inputs_embeds=None, **model_kwargs):
  ```
- after `cascade_runner_patch.py:337-393` (`_make_model_forward_wrapper`;
  wrapper def `:361`)
  ```python
  def _model_forward(self, *args, **kwargs):
      global _step_update_done
      try:
          if _step_is_cascade():
              forward_context = get_forward_context()
              num_tokens_padded = _pick_arg(args, kwargs, 0, "num_tokens_padded")
              if not isinstance(num_tokens_padded, int):
                  descriptor = getattr(forward_context, "batch_descriptor", None)
                  num_tokens_padded = getattr(descriptor, "num_tokens", None)
              if not isinstance(num_tokens_padded, int):
                  raise RuntimeError("num_tokens_padded is unavailable (...)")
              self._update_full_graph_params_if_needed(
                  forward_context=forward_context,
                  num_tokens_padded=num_tokens_padded,
              )
              _step_update_done = True
      except Exception as exc:
          _warn_drift_once("pre-model graph param update", exc)
      return orig(self, *args, **kwargs)
  ```
  When the padded token count is absent from the call it is recovered from
  `forward_context.batch_descriptor.num_tokens` (the same value the graph
  dispatch recorded, `model_runner_v1.py:3106`); if that also fails the
  pre-model update is skipped with one warning and the stock post-model update
  still runs.

### Additional positional-pinning removed (same class of latent drift)

- `cascade_plugin.py:258-265` `_make_build_wrapper` now calls the host
  `build(common_prefix_len=, common_attn_metadata=, fast_build=)` by keyword.
- `cascade_plugin.py:410-438` and `cascade_graph_plugin.py:889-950`
  `forward_fused_infer_attention` now accept `**kwargs` and forward the host
  method by keyword.
- `cascade_graph_plugin.py:151-163` `build_for_cudagraph_capture` fallback
  forwards by keyword.

### Dispatch wrapper (no defect, hardened)

`cascade_runner_patch.py:139-207` (`_make_determine_batch_wrapper`) is now
`(self, *args, **kwargs)`: it finds `use_cascade_attn` by name (positional
fallback index 4), re-dispatches with it forced `False` while preserving the
original call shape, and on any resolution failure warns once, leaves the
cascade step flag `False` and calls `orig` untouched. The host calls this
keyword-only (`gpu_model_runner.py:4375`, `:6029`); both conventions work.

## 2. Fail-open contract (new, enforced by tests)

- `_warn_drift_once` (`cascade_runner_patch.py:86`) / `_warn_once`
  (`cascade_graph_plugin.py:69`) log **one** warning per cascade sub-path per
  process; per-step spam is impossible.
- Every cascade-specific step runs inside `try/except Exception` and the
  original implementation is invoked on the way out — no exception can escape a
  patched host method.
- Default-off path unchanged: with the graph gate unset the runner patches are
  not installed at all; with the gate set but the decode gate unset, every
  wrapper forwards verbatim (`orig(self, *args, **kwargs)`).

## 3. New tests — `tests/test_host_signature_drift.py` (13)

Signature/convention sources are the **live host** (`inspect.signature` of the
host callables) and the **host source** (keyword sets of the actual call sites
parsed with `ast`), so a host bump fails the guard loudly.

| Test | Drift it catches |
| --- | --- |
| `test_capture_wrapper_accepts_every_keyword_the_host_call_site_passes` | D1 — calls the wrapper with exactly the host call-site keywords (`batch_descriptors`, `cudagraph_runtime_mode`, `profiler`); old code raised `TypeError: unexpected keyword argument 'profiler'` |
| `test_capture_wrapper_failure_falls_back_to_orig` | fail-open for twin capture |
| `test_model_forward_pre_update_passes_only_accepted_params` | D2 — recorder bound to the host signature; asserts no positional args and no `positions` kwarg, and that `num_tokens_padded` is right |
| `test_model_forward_wrapper_supports_host_keyword_only_convention` | D4 — calls with the upstream keyword-only convention and asserts `num_tokens_padded` is recovered from `batch_descriptor.num_tokens` |
| `test_model_forward_failure_falls_back_to_orig` | fail-open for the pre-model update (flag not set, model still runs) |
| `test_update_graph_params_forwards_host_convention_verbatim` | D3 — host call site keywords (`draft_attn_metadatas`) + 5 positional; old code forwarded 7 positional to a 6-parameter host method |
| `test_update_graph_params_gate_off_forwards_verbatim` | D3 — gate-off branch must delegate, not rebuild the arg list |
| `test_update_graph_params_failure_falls_back_to_orig` | fail-open for the replay re-parameterization (one warning) |
| `test_determine_batch_wrapper_accepts_host_keyword_only_convention` | dispatch wrapper accepts both host call sites' keyword-only sets |
| `test_determine_batch_wrapper_redispatch_keeps_host_convention` | cascade re-dispatch forces `use_cascade_attn=False` and keeps the host shape |
| `test_determine_batch_wrapper_fails_open_on_unknown_signature` | future host bump: unknown signature -> stock dispatch + one warning, no raise |
| `test_wrappers_forward_to_orig_via_star_args` | source/AST guard: every cascade wrapper must accept `*args`/`**kwargs` and call `orig(*args, **kwargs)` |
| `test_host_method_forwards_use_keywords_not_positions` | source/AST guard: host private methods are never pinned to argument positions |

**Mutation check** (`logs/fix-sigdrift/mutation-checks.log`): each defect was
reintroduced in the fixed source and the matching test failed with the
historical message:

| Mutation | Test | Failure |
| --- | --- | --- |
| D1 | `test_capture_wrapper_accepts_...` | `TypeError: ... got an unexpected keyword argument 'profiler'` |
| D2 | `test_model_forward_pre_update_...` | `assert ['too many positional arguments'] == []` |
| D3 | `test_update_graph_params_forwards_...` | `TypeError: too many positional arguments` |
| D4 | `test_model_forward_wrapper_supports_...` | `TypeError: ... missing 1 required positional argument: 'num_tokens_padded'` |

## 4. Verification (commands + raw results)

### 4.1 CPU gates

```bash
cd /vllm-workspace/vllm-ascend-split-batch-hust
ruff check .        # All checks passed!
python -m pytest -q # 102 passed, 14 warnings in 35.34s
```
Artifact: `logs/fix-sigdrift/pytest-ruff.log`. Baseline before the change was
89 passed (same command), so the new tests add 13 and nothing regressed.

### 4.2 Signature audit after the fix

`python logs/sig-audit/sig_audit2.py` -> `logs/fix-sigdrift/sig-audit-after.txt`:

```
cascade_runner_patch:149  _determine_batch_execution_and_padding(self, *args, **kwargs)
cascade_runner_patch:313  _capture_cudagraphs(self, *args, **kwargs)
cascade_runner_patch:361  _model_forward(self, *args, **kwargs)
cascade_graph_plugin:794  update_graph_params(*args, **kwargs)
```

### 4.3 Unshimmed graph-mode engine start (the blocker)

```bash
cd /tmp   # never /vllm-workspace
ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
VLLM_ASCEND_ENABLE_CASCADE_DECODE=1 VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1 \
VLLM_ASCEND_CASCADE_MIN_PREFIX=2048 VLLM_ASCEND_CASCADE_MIN_REQS=2 \
VLLM_ASCEND_CASCADE_TRACE=1 VLLM_DISABLE_COMPILE_CACHE=1 \
VLLM_CACHE_ROOT=/tmp/sigdrift_cache_on \
vllm serve /data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --cudagraph-capture-sizes 8 16 --port 8361
# traffic: python /tmp/sigdrift_client.py 8361 8 64 240 /tmp/sigdrift_on.json
```

No shim, no harness-side signature adapter; the plugin on disk is the commit in
§6. Artifact `logs/fix-sigdrift/serve-on-card6.log`, summary
`logs/fix-sigdrift/key-lines.txt`:

```
9:   cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)
42:  (EngineCore) cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)
977: Application startup complete.
capture body SUCCESS: num_tokens=16  x48
capture body SUCCESS: num_tokens=8   x48
replay update: cascade key hit num_tokens=8   x63
replay_swap=True                              x113
stage1 re-bind skipped                        x62
TypeError/Traceback/ERROR inside the serving window (lines 1..1432, before the
first [shutdown] marker): 0 / 0 / 0
```

All 9 client requests returned 200 (warmup + 8 concurrent, 64 tokens each,
`ignore_eos`, prompt_tokens=2442..2447 -> shared prefix >= MIN_PREFIX=2048).
The single `Traceback` in the whole log is the post-`SIGTERM` teardown
`EngineDeadError` (`async_llm.py:829`) at 18:06:17, after the `[shutdown]`
lines; the same teardown artifact appears in the OFF log and is not cascade
related.

> Note: the first ON attempt aborted in `torch/_functorch/.../autograd_cache.py`
> (`AssertionError: expected OutputCode, got GraphModuleImpl`) during engine
> init, **before** any cascade capture. This is the documented
> shared-torch.compile-AOT-cache hazard of this baseline
> (`EVIDENCE.md` §6 item 3), not a cascade defect; the same
> `VLLM_DISABLE_COMPILE_CACHE=1` + fresh `VLLM_CACHE_ROOT` was then applied
> symmetrically to both legs (the failed log was overwritten, so this
> observation cites the existing artifact).

### 4.4 Default-off zero-diff

Same model/CLI/capture sizes on card 6, cascade env **unset** (port 8362),
same client. Artifact `logs/fix-sigdrift/serve-off-card6.log`:

```
9:   cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=ok)
37:  (EngineCore) cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=ok)
130: Application startup complete.
gate0_marker=2  cascade-active=0  capture body=0  replay=0
wheel unavailable=0  TypeError/Traceback/ERROR in serving window=0/0/0
```
All 8 requests 200. With `graph_gate=0` the runner patches are not installed at
all, so this leg runs the stock host methods.

### 4.5 ON-vs-OFF sanity (short, not a full evidence re-run)

Two measurements, both cheap:

1. **Graph lane, token level** (`logs/fix-sigdrift/onoff-on-trace.log`,
   `onoff-runner.log`, token dumps `ev2_off_sigdrift_off.json` /
   `ev2_on_sigdrift_on2.json`): offline `LLM` lane, B=8, shared prefix ~2.44k,
   `max_model_len=4096`, `gen=64`, `ignore_eos`, capture sizes [8,16],
   `MIN_PREFIX=2048`, `MIN_REQS=2`.
   - ON leg reached `gate=1, graph_gate=1`, 96 capture bodies, **63**
     `replay update: cascade key hit`, 1 twin-miss fail-open (the non-uniform
     prefill step), 0 TypeError.
   - **ON vs OFF token divergence: 0/8**; ON vs ON re-run: 0/8 (determinism).
2. **Serve lane, text level** (`sigdrift_on.json` / `sigdrift_off.json`):
   identical prompts, `temperature=0`, 64 tokens -> **1/8** texts differ
   (request 4 flips at token ~2, `"The sum of 4 and 4 is 8."` vs
   `"The answer to 4 + 4 is 8."`). Text-level divergence is a lower bound for
   token divergence.

Criterion cited: the historical graph-mode main acceptance for 14B B=64 was
**6/64 ON-vs-OFF ≤ 8/64 C-口径** (B=32: 1/32; 0.5B: 0/8) —
`cascade-c3-results/RESULTS.md` §1/§2. Both measurements here (0/8 token,
1/8 text) are within that bound. Cascade output is documented as bf16-level
different, not bit-exact (`cascade_plugin.py:248`).

## 5. Residual risk / not verified

1. **D4 is latent on this baseline.** The live callers of `_model_forward` are
   the vllm-ascend overrides `NPUModelRunner.execute_model`
   (`model_runner_v1.py:2451`) and `_dummy_run` (`:3854`), and both pass
   `num_tokens_padded` positionally; the upstream keyword-only convention
   (`gpu_model_runner.py:4550`) is not on this backend's live path. So the
   reported `TypeError` was not reachable through the asc call sites — the fix
   and the new test make both conventions work and guard the keyword-only one
   for the next host bump. The requester's repro (`logs/sig-audit/d4_repro.py`)
   is a stand-in class, not the live call path.
2. **Full evidence #2/#3 re-run not performed** (explicitly out of scope). The
   numbers in §4.5 are a sanity check at B=8/GEN=64; the historical
   B=64/GEN=128 matrix and the TPOT comparison must be re-run unshimmed before
   any `active` flip.
3. **`_capture_cudagraphs` still relies on the plugin's own post-capture
   ordering** (gate bench -> twin capture). If a future host moves the capture
   loop or changes `CUDAGraphMode`, the twin capture fails open (one warning,
   standard graphs only) rather than raising; the feature would silently
   degrade — the new tests only guard the call convention, not the capture
   semantics.
4. **The AOT compile-cache crash is a host/baseline issue** and remains: both
   legs needed `VLLM_DISABLE_COMPILE_CACHE=1`. Any future perf re-run must keep
   it symmetric (as `EVIDENCE.md` §6 item 3 already requires).
5. **One twin-miss fail-open per run** (non-uniform prefill step carrying the
   cascade flag) is expected by design; it logs one warning and replays the
   standard graph.
6. `HOST_CONTRACT.md` was already edited in the working tree by the requester
   to document the parameter-level anchor table and the D1–D4 design rule; it
   is included in the commit (content verified against the fixed code).

## 6. Commit

Single atomic commit in `/vllm-workspace/vllm-ascend-split-batch-hust`:

```
fix(cascade): make graph-mode host wrappers signature-agnostic and fail-open
```

Files: `src/vllm_ascend_split_batch/cascade_runner_patch.py`,
`src/vllm_ascend_split_batch/cascade_graph_plugin.py`,
`src/vllm_ascend_split_batch/cascade_plugin.py`,
`tests/test_host_signature_drift.py`, `HOST_CONTRACT.md`.
Host trees (`vllm-hust`, `vllm-ascend-hust`) and
`EVIDENCE.md` / `section*.md` / plan files untouched.
