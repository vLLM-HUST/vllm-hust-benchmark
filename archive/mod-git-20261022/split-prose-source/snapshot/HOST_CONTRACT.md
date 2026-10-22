# Split-batch host contract proposal

The extracted planner is pure and default-off. Runtime activation requires:

1. `vllm.graph.runtime-key.v1`: validated, bounded opaque graph metadata;
2. `vllm.forward.split-context.v1`: per-forward split mode and stream identity;
3. `vllm.ascend.graph-pool.v1`: separately owned main/parallel graph pools;
4. `vllm.worker.split-executor.v1`: execute a committed plan and restore all
   buffers/context on success, error, or cancellation.

The provider must reject speculative decoding, LoRA, MLA, M-RoPE, non-uniform
decode, unsupported graph modes, and unknown capture sizes. It must never lazily
capture an unbounded key set or modify global forward context outside a scoped
context manager.

## GDN QKV norm tier-2 proposal (default-off)

No `gdn-qkv-norm` plugin or bundle entry is exposed. The accepted kernel
requires the separately reviewed reversible host proposal; the package does
not install it. Registration is withdrawn pending a reachable narrow dispatch
boundary. Source hashes are proposal provenance, not a public registration API.

Prefer IR/CustomOp registration or graph replacement after proving callsite
reachability. The CPU export probe in `tools/probe_gdn_registration.py` checks
the actual registered core boundary, not production graph visibility. An opaque
core cannot be matched internally by an outer FX rule. Layer-level replacement
must not disguise a copied whole-forward implementation as operator registration.
Registration and release gates are defined in `docs/release.md`.

## Cascade attention components (implemented, default-off)

The cascade plugin (`cascade_plugin.py`, `cascade_graph_plugin.py`,
`cascade_runner_patch.py`) rides ONLY the official public surface:

- `vllm.general_plugins` entry point (`load()`).
- `vllm.v1.cudagraph_dispatcher.add_cudagraph_key` / `dispatch` for graph keys.
- Runner methods `_capture_cudagraphs`, `_warmup_and_capture`,
  `_determine_batch_execution_and_padding`, `_model_forward`,
  `_update_full_graph_params_if_needed` (monkeypatched, env-gated).
- vllm-ascend `update_full_graph_params` / `get_graph_params` /
  `GraphParams` for capture/replay parameter plumbing.
- `ACLGraphWrapper` variant-entry table: the cascade twin graph is stored in a
  per-instance side table keyed by the SAME standard `BatchDescriptor`
  (the official frozen descriptor has no cascade field; the table is swapped
  in for the duration of a cascade capture/replay call).

Default-off semantics: with `VLLM_ASCEND_ENABLE_CASCADE_DECODE` unset the
patched callables delegate to the originals unchanged. `BatchDescriptor` and
all other host dataclasses are never modified.

### Anchor map on the current baseline (parameter-level, corrected 2026-09-09)

Host = vllm-hust v1 (`0.28.1.post1.dev143+gf18cf803c`) + vllm-ascend-hust
main (`0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`), CANN 9.1.0.

> **Correction (2026-09-09, post-review).** The first version of this table
> checked symbol *existence* only and asserted "wrappers forward via `*args`".
> That assertion was **false**, and it hid four blocking signature drifts
> (D1–D4 below) — exactly the failure mode `docs/pitfalls.md` §2.1 warns about.
> Existence is not compatibility: a monkeypatched method must match the host's
> *parameter list and call convention*. Use the AST audit
> (`knowledge/evidence/cascade/logs/sig-audit/sig_audit2.py` semantics) before
> any `host.version_range` change.

| Anchor | Host signature (current) | Plugin wrapper | Status |
| --- | --- | --- | --- |
| `_capture_cudagraphs` | `(self, batch_descriptors, cudagraph_runtime_mode, profiler=None)` | was `(self, batch_descriptors, cudagraph_runtime_mode)` | **D1 drift → fixed** |
| `_update_full_graph_params_if_needed` | `(self, forward_context, num_tokens_padded)` | was called with 3 positional (`+positions`) | **D2 drift → fixed** |
| `impl_cls.update_graph_params` | `(update_stream, forward_context, num_tokens, vllm_config, speculative_config=None, draft_attn_metadatas=None)` | had extra `num_dcp_pcp_tokens` and forwarded 7 positional | **D3 drift → fixed** |
| `_model_forward` | `(self, input_ids=None, positions=None, intermediate_tensors=None, inputs_embeds=None, **model_kwargs)`; host calls **keyword-only** | had required leading `num_tokens_padded` | **D4 drift → fixed** |
| `_determine_batch_execution_and_padding` | 12 params (see host `gpu_model_runner.py:4040`) | identical | OK |
| `_warmup_and_capture` | `(self, desc, cudagraph_runtime_mode, profile_seq_lens=None, allow_microbatching=False, num_warmups=None, profiler=None)` | not wrapped (called via `orig`) | OK |
| `AscendTopKTopPSampler` / `AscendSampler` | `vllm_ascend/sample/sampler.py` | subclass replacement | OK |
| `update_full_graph_params` / `GraphParams` / `get_graph_params` / `ACLGraphWrapper` | `vllm_ascend/compilation/acl_graph.py` (:279/:306/:334/:60) | read-only use | OK |
| `ACLGraphWrapper._updatable_graph_replay` (**只在有新机制的被测宿主上存在**;基线 `74f0c0a27` 无此方法) | `vllm_ascend/compilation/acl_graph.py:304` 起;触发条件 `runtime_mode == FULL and use_updatable_graph(self.attn_backend)`(`:297`)。配套:`vllm_ascend/compilation/updatable_graph.py` 的 `UpdatableGraph` 与 `vllm_ascend/utils.py:1773` 的 `use_updatable_graph` —— **旧基线三者在树里都不存在** | 包装该方法,在**宿主重放之前**做 cascade 重参数化(见 `cascade_graph_plugin._wrap_updatable_graph_replay`);特征探测,旧宿主自动 no-op | **新增接缝(2026-09-27)**,机制与顺序约束见 `docs/pitfalls.md` §2.4 |
| `BatchDescriptor` | `vllm/forward_context.py` | unchanged | OK |
| `spec_decode` / `spec_decode.ngram_proposer` / `eplb.core.policy.policy_factory` | present (needs `scipy` + `decorator`) | shims | OK |

**Design rule this establishes**: a wrapper that raises out of a patched host
method kills the engine, which violates the documented fail-open contract. All
cascade wrappers must (a) match the host call convention and (b) delegate to the
original on any internal failure with a single warning.

**Carrier status (2026-09-11)**: the two cascade carriers
(`cascade_plugin:load` / `cascade_graph_plugin:install`) are `active`; the
planner keeps `import_only` (no acceptance evidence, review F7). The D1–D4
signature drifts were fixed and the three acceptance evidences were re-run on
this pinned baseline: default-off smoke PASS
(`EVIDENCE.md` §1), real-model correctness 6/64 like-for-like
(`section2-correctness.md`), performance/gate margin closed
(`section3-performance.md`). release.md §3 enablement verification then
passed on the real model (`knowledge/evidence/cascade/section4-active-enablement.md`):
`extension check` compatible+configured, `run` gate proof
`gate=1, graph_gate=1, kernel_wheel=ok`, cascade twin capture 144/144 with 0
TypeError, `/health` + functional smoke 200. `host.version_range` stays at the
exact verified build (`==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`);
packaging rejects local-version labels in ordered comparators, hence the
point-`==` form.

## fi_sampling component (implemented, default-off)

The fi_sampling plugin (`fi_sampling_plugin.py`, `fi_sampling_route.py`) rides
ONLY:

- `vllm.general_plugins` entry point `fi-sampling` (`load()`).
- vllm-ascend module attribute `vllm_ascend.sample.sampler.AscendTopKTopPSampler`
  replaced by a subclass (`FiSamplingTopKTopPSampler`) whose `forward_native`
  overrides the base; `AscendSampler.__init__` resolves that module attribute at
  construction time, so no other host object is touched. Same variant-entry
  pattern as the fork's own `vllm_ascend/_310p/sample/sampler.py`.
- Read-only probes of `vllm.envs.VLLM_BATCH_INVARIANT` and
  `vllm_ascend.ascend_config.get_ascend_config()` (`enable_reduce_sample`,
  `enable_async_exponential`) for the fallback matrix.

Default-off semantics: with `VLLM_HUST_FI_SAMPLING` unset `load()` returns
before importing `vllm_ascend` or the vendored kernel package, so nothing is
patched and `triton` is never imported. The host source trees are untouched and
no `vllm.platform_plugins` entry is registered. Host-upgrade checklist item
(see `docs/release.md` §4): `AscendTopKTopPSampler` name/`forward_native`
signature and the `AscendSampler` construction site.

## rope_fix component (implemented, default-off, `import_only`)

The RoPE variant defect carrier (`rope_fix_plugin.py`) fixes three *fork* defects
plugin-side (design note `docs/design/rope-variant-defects.md`). It rides ONLY:

- `vllm.model_executor.custom_op.op_registry_oot` — the out-of-tree registry read
  by `CustomOp.__new__` (:109-128) / `PluggableLayer.__new__` (:47-66) at
  **instantiation** time. The carrier **replaces the three rope entries**; it never
  calls `CustomOp.register_oot` itself, because that method asserts
  `reg_name not in op_registry_oot` (duplicate-name guard).
- `vllm_ascend.utils.register_ascend_customop` — **wrapped** (original first, then
  the overrides), and the wrapper is rebound in every already-imported module
  holding a direct reference, `vllm_ascend.worker.worker` included
  (`worker.py:95` binds the function, `:150` calls it).
- `vllm_ascend.ops.rotary_embedding`: the module-level `triton_mrope` binding (bound
  under `HAS_TRITON`; the mirrored `forward_triton` calls the *same* object the host
  call site resolves, and defect ② is simply not installed when it is absent),
  `AscendRotaryEmbedding.forward_oot` (the module-level delegation fallback of the
  llama3 override), `AscendMRotaryEmbedding` (subclassed; the mirrored
  `forward_triton` reads `mrope_interleaved`, `is_neox_style`, `head_size`,
  `cos_sin_cache`, `_ASCEND_TRITON_GRID_LIMIT`), `AscendYaRNRotaryEmbedding`
  (subclassed; only the keyword-only `truncate` default changes) and the private
  `_record_cos_sin_cache` helper.
- `vllm.model_executor.layers.rotary_embedding`: `Llama3RotaryEmbedding`
  (10-argument construction signature mirrored verbatim) and `triton_mrope`
  (9 positional parameters, the last one `is_neox_style`).

### Anchor map (baseline `main@74f0c0a27`)

| Anchor | Host reality (verified) | Carrier | Status |
| --- | --- | --- | --- |
| `REGISTERED_ASCEND_OPS` (utils.py:735-765) | `RotaryEmbedding` / `MRotaryEmbedding` / `YaRNScalingRotaryEmbedding` / `DeepseekScalingRotaryEmbedding`, **no `Llama3RotaryEmbedding`** | creates `op_registry_oot["Llama3RotaryEmbedding"]` | defect ① (coverage gap) |
| `op_registry_oot["RotaryEmbedding"]` (set by the same pass; 310P swaps in `AscendRotaryEmbedding310`) | the implementation the build actually uses for the base rope | resolved at construction and delegated to by the llama3 override | compatibility-build contract |
| `AscendMRotaryEmbedding.forward_triton` (:513-552) | calls `triton_mrope` with **8** positional args | mirrored subclass + `self.is_neox_style` (9th arg) | defect ② |
| `rotary_embedding.triton_mrope` (module global, `HAS_TRITON`) | the kernel object the call site resolves | bound from this module; absent -> defect ② not installed | defect ② precondition |
| `AscendYaRNRotaryEmbedding.__init__` (:281-298) | `truncate: bool = False` (vLLM/HF default is `True`) | subclass changing only that default to `True` | defect ③ |
| `register_ascend_customop` (:688) | early-returns on `_ASCEND_CUSTOMOP_IS_REIGISTERED`; registers every key at :813-814 | wrapped; overrides applied **after** the original returns; rolled back if that eager pass fails | ordering contract |
| `worker.py:95/150` | `from vllm_ascend.utils import register_ascend_customop` + a call inside `NPUWorker.__init__` | the wrapper is rebound in this module too (an attribute patch alone would miss the live call site) | call-site rebinding |
| `CustomOp.register_oot` (:339-351) | `assert reg_name not in op_registry_oot` | never called by the carrier (why the entries are replaced instead) | lever justification |

Host-upgrade checklist item: `tests/test_rope_fix_drift.py` (see
`docs/release.md` §4). Its reverse assertions fail *by design* when upstream fixes
one of the three defects (`upstream fixed defect ①/②/③ → drop the corresponding
override`) and use a second wording when a seam moved
(`anchor drifted → re-audit HOST_CONTRACT`); the seam list above is what that
second wording asks the reader to re-audit. A missing host tree is a **failure**,
not a skip: an unevaluated guard must not look like a passing one.

Default-off semantics: with `VLLM_HUST_ROPE_FIX` unset `load()` returns before
importing `vllm_ascend` / `torch` / `torch_npu` / `triton`, touches no registry
and logs nothing. Fail-open: a missing/renamed seam, a foreign registry entry
(e.g. the 310P variants) or any internal error warns once and leaves the host
implementation in place; a failure *after* the wrapper was installed rolls it back
so `stats()` matches the log.

## GDN in_proj merge carrier (implemented, default-off, fail-closed)

`gdn_merge_inproj_plugin.py` is the plugin-library home of the accepted **F2**
mechanism (load-time `in_proj_ba` -> `in_proj_qkvz` weight merge, one
`[6176, 2048]` `MatMulV2` per GDN layer instead of two). It rides ONLY:

- `vllm.general_plugins` entry point `gdn-merge-inproj` (`load()`), plus the
  descriptor-only bundle `org.vllm-hust.gdn-merge-inproj`
  (`src/vllm_ascend_split_batch/gdn_merge_inproj/`, `implementation[].status =
  import_only`, injection key `VLLM_HUST_GDN_MERGE_INPROJ=1`). The status is
  deliberately **not** `active`: the extension manager must refuse to enable a
  carrier that is fail-closed on this host (same ladder as `rope-fix`).
- the active model runner's `load_model` method, wrapped exactly like
  `gate_fp32_plugin` (same probe list and order, `vllm_ascend.ops` warm-up,
  fail-open on a missing seam). Both carriers wrap the same method; each checks
  its own marker, calls through to whatever it wrapped (so the other hook still
  runs) and is undone by its own `uninstall()`.
- read-only resolution of
  `vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.QwenGatedDeltaNetAttention`
  — the class the model's GDN layers are instances of, because `patch_qwen3_5`
  rebinds the Ascend methods onto it — plus its `in_proj_qkvz` / `in_proj_ba`
  `MergedColumnParallelLinear` shards and the `key_dim` / `value_dim` / `tp_size`
  / `num_v_heads` attributes used to derive the F2 row counts.

Host cohort: vllm-ascend `0.25.1rc2.dev605+hust.20260903.4.gfbe4911bb` (active
stack tree, `/root/vllm-workspace/.stack/ascend-fbe4911`), vllm
`0.28.1.post1.dev143+gf18cf803c`, 2x910B2 TP=2, Qwen3.5-35B-A3B, caliber
`mmlu-pro-eval-traffic`, cohort `028-qwen35-35b-a3b-tp2-fullgraph`.

### Current status: fail-closed on the missing forward-consumer seam

The weight conversion is implemented and CPU-tested, but on this host the carrier
rewrites **nothing** and logs one warning, because the merged weight has no safe
consumer.

`AscendGatedDeltaNetAttention.forward` computes `self.in_proj_qkvz(...)` and
`self.in_proj_ba(...)` inline and then runs `torch.ops.vllm.qwen_gdn_attention_core`
+ `RMSNormGated` + `out_proj` inside the same method body: there is no projection
or custom-op seam. A plugin-side single-matmul consumer would need either

1. per-call Python state inside the traced `forward` (a module attribute or
   thread-local carrying the fused output to the second projection call) — the
   dynamo-unsafe shape the F2 v1 carrier was rejected for, or
2. a copy of the host `forward` body in this repository — host private source,
   which would then drift silently.

Merging without a consumer is not a fallback: it removes no kernel (the stock
second `in_proj_ba` matmul still runs) and feeds a `[*, 6176]` tensor into the
stock `mixed_qkvz.split([qkv_size, z_size])`. `FORWARD_CONSUMER_BLOCKER` in the
carrier is the machine-readable form of this paragraph;
`forward_consumer_available()` is the single gate, and the merge path behind it is
exercised by `tests/test_gdn_merge_inproj_plugin.py` (the fail-closed path and
the merge path are both pinned).

Unblocking condition (any one): the host forward grows an env-gated
merged-consumption branch for the flat `gqa_interleaved_layout=False` layout, or
the projection step becomes its own bindable method / custom op.

Default-off / fail-open semantics: with `VLLM_HUST_GDN_MERGE_INPROJ` unset (or
anything other than the exact token `1`) `load()` returns before importing
`torch` / `vllm` / `vllm_ascend`, touches no class and logs nothing. Enabled: a
missing/renamed host class, a missing runner seam or any internal error warns and
changes nothing; the post-load hook never raises into `load_model`, and a
`load_model` that raises is never post-processed.

Coexistence with the runtime-tree F2 patch (read before reading any serve log):
the accepted F2 measurement came from the runtime-tree patch, which consumes the
**same** env key `VLLM_HUST_GDN_MERGE_INPROJ=1` (the validation ladder exports it
by default, `tools/bench-ladder.sh`). On a host where the patch is applied, a
`=1` run therefore shows the patch's merge *and* this carrier's fail-closed
warning in the same log; with the plugin carrier alone the key has the visible
effect of that one warning and nothing else. The two cannot double-merge even if
the seam appears later: the patch leaves a `[6176, 2048]` weight, which
`plan_merge` refuses on the row-count guard (`layer dims imply (6144, 32)`).

Host-upgrade checklist item — re-audit when the host moves: (a) the
`QwenGatedDeltaNetAttention` name and its `in_proj_qkvz` / `in_proj_ba`
attributes; (b) that GDN layers are still instances of the core class rather than
of a fork subclass; (c) `gqa_interleaved_layout` and the four dim attributes;
(d) the runner probe list and the `load_model` signature; (e) **whether a
merged-consumption seam now exists** — if it does, delete this blocker and run the
L3/L4 acceptance.

Statistics scope: the accepted evidence (L1 card `f2_accepted_verdict_1016`) —
decode `MatMulV2` -30.1/step, device kernel time -15.4% per pattern, TPOT -1.10%
(no regression), L3 8/8 identical, zero new `ConcatD`/`CopyD` — belongs to caliber
`mmlu-pro-eval-traffic` / cohort `028-qwen35-35b-a3b-tp2-fullgraph` **and to the
runtime-tree patch**, not to this carrier: any performance number for this carrier
must be re-accepted with a fresh L4 window on the same caliber. Nothing here may
be extrapolated to another caliber, cohort, `capture_size` or the eager path.

## W1R GMM `tuning_config` / counts dump (tier-2 upstream proposal, default-off)

状态：**提案补丁，未提交上游**（承载目录 `host-proposals/W1R-gmm-tuning-config/`，
`proposal_meta.json` 记录 `upstream_status` 与后续动作）。不是本仓插件 entry，
因此**不出现在** manifest / `vllm.general_plugins` 列表里。

- 宿主文件：`vllm_ascend/ops/fused_moe/routed_experts.py`
  （sha256 `b17e5a10…`，`host.version_range == 0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909`，点钉）。
  基线可复现：`git -C <host repo> cat-file -p 74f0c0a27:vllm_ascend/ops/fused_moe/routed_experts.py | sha256sum`
  → `b17e5a105d59fc280f244bdc4036b4cb41efd1d5a2ff1accca4b56620cc91ab7`。
- 补丁点（**仅两处**，均为宿主私有方法体内部的 `npu_grouped_matmul` 调用）：
  `AscendUnquantizedFusedMoEMethod.apply_gmm1` / `apply_gmm2`。
- 为什么必需 tier-2：宿主没有 `tuning_config` 的公开配置面；插件侧若复制方法体
  （host private source）违反本仓契约；全局 monkeypatch
  `torch_npu.npu_grouped_matmul` 不是已列公开面，且会波及所有 GMM 调用方。
- 开关（均 default-off；两者未设时**零 import / 零 IO / 零 shape 读取 / 无设备差异**，
  但每次 GMM 仍有**两次早退 Python 调用**（tuning kwargs 查询 + dump 早退）；
  宿主文件文本因注入块而不同，**不得声称字节等价**）：
  - `VLLM_HUST_GMM_TUNING_CONFIG`：**单值**，∈ {1,2,8,32}（列表被显式拒绝，避免一臂多变）。
    `tuning_config` 是 CANN 的 tiling hint：首值 = 「各专家预期处理的 token 数」；
    torch_npu 只承诺首值语义。非法值在**首次 GMM 调用（起服 profile/dummy run，非 capture 内）**
    即 fail-fast 抛 `ValueError`，错误缓存后每次继续抛出（不静默降级）。
    `Unknown vLLM environment variable` 警告**不是**「开关未生效」判据。
  - `VLLM_HUST_GMM_COUNTS_DUMP=<path>`：记录 group_list counts 与
    形状/provenance 采样；**不改变计算输出**（hook 恒返回 None，op 结果原样返回）。
    该文件是 **dump 产物，永远不是计时产物**（dump 窗带 env 门控 + IO，属第二变量）。
    capture 策略：仅在 dump 开启时 lazy import 做官方检测 `torch.npu.is_current_stream_capturing()`；
    为 capture 或检测失败 → **读取任何 device tensor 之前返回**，`sample_source` 记
    `skipped_inside_capture` / `skipped_unknown_capture_state`；capture dummy 行绝不写成真实 counts，
    FULL graph replay 不执行 Python。TP=2 时路径支持 `{rank}`，无占位符则自动追加 `.rank<RANK>`。
- 回滚：**该层已从 `apply-runtime-patches.sh` 的 MANIFEST 移除**，脚本里没有可用的
  `--rollback w1r-gmm-tuning`。若运行树仍是部署态，二选一：①直接对**已核过的**
  `host-proposals/W1R-gmm-tuning-config/patch.diff` 执行 `patch -R`（先 `patch -R --dry-run`）；
  ②按 `OWNERSHIP.md §5` 先把层重新加回 MANIFEST，再用 `apply-runtime-patches.sh --rollback`。
  当前该层已**从 MANIFEST 移除**（blocker `fixture-unreachable-fullgraph-decode`），
  重新启用条件与步骤见 `host-proposals/W1R-gmm-tuning-config/OWNERSHIP.md` §5。
  `gen_patch.py --check` 只对**未打补丁的宿主/镜像 blob** 有意义（部署态见上一条回滚说明）。
- CPU 门：`tests/test_w1r_gmm_tuning_patch.py`（补丁正/反向、生成器 byte-identical、
  fail-closed、零模块级 import/IO、白名单解析、dump schema 与 invariant）。
