# Ascend Split-Batch and Full-Graph Parallel

Owner-led research carrier for split-batch planning, dual-stream replay, and full-graph parallel execution. Core owns graph lifecycle seams; this repository owns policy, runners, validation, and evidence.

**Status: default-off.** Two independently gated capabilities are installed:

1. **Split-batch dual-pad planner** — pure planning + prechecks (`import_only`,
   no vLLM behavior change without the host contract).
2. **Cascade two-stage decode** (`vllm.general_plugins` entry
   `cascade-attention`) — eager + graph-mode two-stage cascade attention for
   vllm-ascend, patched entirely from the plugin; official vllm /
   vllm-ascend sources are NOT modified.

Technical ownership belongs to @Raing5Days, @ilnnfover. Source extraction must
preserve exact authorship, license, tests, constraints, and evidence before
activation is considered.

See [MAINTAINERS.md](MAINTAINERS.md) and [PROVENANCE.md](PROVENANCE.md).

## Install

```bash
pip install vllm-ascend-split-batch
```

Published on PyPI (`0.1.3`; wheel + sdist, sha256 and PEP 740 attestations on the
project page). Release receipt: [docs/release.md](docs/release.md) §11.9. This version widens
`host.version_range` from a point pin to the bounded window
`>=0.25.1rc2.dev125,<0.25.2` (rationale §0.3) and carries the UpdatableGraph replay fix;
it was verified on the published bytes, not only on the work tree.

**What you get:** 8 `vllm.general_plugins` carriers and 4 extension bundles,
all **default-off** — installing changes nothing until you enable a capability
via its env var (see the table below). Cascade attention additionally needs the
`ascend_kernel` CCE op wheel, which is **not on any package index** (the
`kernels` extra pins a version that only exists in the kernel repository, so the
extra resolves only with `--find-links`):

```bash
# cascade-only soft dependency; skip it if you do not need the cascade path
pip install "vllm-ascend-split-batch[kernels]" \
  --find-links https://github.com/Raing5Days/vllm-hust-cascade-kernel/releases/expanded_assets/v2026.9.26
```

⚠️ **Host pin.** The manifest pins one verified `vllm-ascend` build
(`host.version_range`; see [docs/release.md](docs/release.md) §0.1). On any other
build the manager reports the extension as `incompatible` and
`vllm-hust-ext run` refuses to launch it — `extension enable` still works, and
the env-var route below is unaffected. Widening the pin requires re-running the
host-upgrade checklist on the target build (§4).

## Documentation

Normative knowledge (architecture & contracts, coding rules for humans and
agents, release process, pitfall history, progress log) lives in
[docs/](docs/README.md) — start there. Repo-level agent rules:
[AGENTS.md](AGENTS.md).

## GDN QKV norm (tier-2, default-off)

The accepted kernel and reversible host proposal live in
`host-proposals/GDN-qkv-norm`. There is no `gdn-qkv-norm` general plugin or
bundle entry point: registration is withdrawn while the target unpack/norm
combination remains inside an opaque core without a proven narrow dispatch.
Do not treat a packaged kernel as plug-and-play callsite replacement.

Scope: Qwen3.5-35B-A3B bf16/ND, TP2, pure non-spec decode T16, FULL graph.
Existing tier-2 activation requires the reviewed host patch plus
`VLLM_HUST_GDN_QKV_NORM=1` and `USE_DEFAULT_FLA_NORM=0` before startup;
unsupported inputs retain native fallback. No automatic patching is provided.
Rollback requires disabling the env and restarting/recapturing; reverse the
host patch only through its separately reviewed procedure.
Acceptance evidence: [host-proposals/GDN-qkv-norm/validation/](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/host-proposals/GDN-qkv-norm/validation/).
Registration decision: `host-proposals/GDN-qkv-norm/registration-closure.json`.
Delivery gates: [docs/release.md](docs/release.md#算子-bundle-最短交付路径).

## Cascade attention plugin (default-off)

Entry point: `vllm.general_plugins` ->
`cascade-attention = vllm_ascend_split_batch.cascade_plugin:load`.

Environment gates (all default off):

| Variable | Meaning |
|---|---|
| `VLLM_ASCEND_ENABLE_CASCADE_DECODE=1` | enable the two-stage cascade decode |
| `VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1` | additionally enable the graph-mode (aclgraph) cascade twin |
| `VLLM_ASCEND_CASCADE_MIN_PREFIX` (8192) | min shared prefix tokens to trigger |
| `VLLM_ASCEND_CASCADE_MIN_REQS` (32) | min batch size to trigger |
| `VLLM_ASCEND_CASCADE_PRECISION` | `bf16` (Tier-0, default) or `fp32` (Tier-1, eager-only) |
| `VLLM_ASCEND_CASCADE_STRICT=1` | force the standard full-KV path (bit-exact reference) |
| `VLLM_ASCEND_CASCADE_TRACE=1` | per-step plugin trace to stdout |
| `VLLM_ASCEND_CASCADE_UPDATE_SKIP_STABLE` (1) | skip the stage-1 graph re-bind when its inputs are step-invariant (0 restores always-rebind) |

With every gate unset the patched methods are no-ops and the serving path is
bit-identical to stock vllm-ascend. On load the plugin emits one INFO line,
`cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=not-probed)`, so a
default-off serve log proves `load()` actually ran (vLLM otherwise swallows
general-plugin load errors). **Disabled discovery touches nothing else**: with
`VLLM_ASCEND_ENABLE_CASCADE_DECODE` unset the plugin injects no env entries,
imports no host module, installs no fail-open shim, imports no kernel wheel and
replaces no attention/graph entry (contract: `docs/release.md` §6).

## Speculative decoding boundary

Cascade admits **one query row per request**. The two-stage path flattens the
shared prefix once and re-derives the per-request suffix from the query-row
count, so MTP verification and chunked prefill (which feed `k + 1` rows) are
rejected by the dispatch gate and skip the cascade twin capture; those steps
keep native attention and the standard FULL graph. `speculative_config` being
set at all is treated as fail-closed, even when a step happens to carry single
rows, because the `k > 1` query layout is neither supported nor measured yet.
This is a guard, **not** MTP support: Split-Batch/dual-pad fails closed the
same way (`planner.precheck_reason` -> `speculative_decode_conflict`). Enabling
Cascade with speculative decoding therefore yields no cascade speedup — it is
a compatibility boundary, not an enabled-Cascade result.

### Cohort scope (what our evidence does and does not cover)

> Authority: [`docs/cohort-and-ownership.md`](docs/cohort-and-ownership.md) — the cohort boundary plus the four
> ownership items returned to this repository; binding rule: [`AGENTS.md`](AGENTS.md) hard constraint 6.

Stated here so the numbers below are not over-extended. **2026-10-13 owner ruling: the baseline moved to the
0.28 image + Qwen3.5-35B-A3B** ("we no longer work on Qwen2.5"). The upstream audit quotes in
[`docs/cohort-and-ownership.md`](docs/cohort-and-ownership.md) §1 are kept verbatim as historical receipts;
the table below is the current claim surface. The Qwen2.5 / 0.25 evidence is retained as an **independent,
historical** set and must not be crossed with the numbers below.

| This repository's evidence covers (0.28 baseline) | This repository's evidence does **not** cover |
|---|---|
| 0.28 image stack: vllm `0.28.1.post1.dev143+gf18cf803c` + vllm-ascend `0.25.1rc2.dev125+hust…`, CANN 9.1.0, 2×910B2 TP=2 | other host builds outside the measured 0.28 image |
| Qwen3.5-35B-A3B (40L: 30 GDN + 10 MoE) hybrid **TP2** | native **MTP2** verification rows / any speculative decoding |
| chunked prefill **on**, APC **off**, `cudagraph_mode=NONE` | `async` scheduling combinations we did not run |
| bf16 | the retired Qwen2.5 / 0.25 line (historical set only) |

⇒ **Do not cross-extrapolate**: the historical Qwen2.5/0.25 numbers do not carry over to the 0.28/Qwen3.5
baseline, and vice versa. Speculative decoding (incl. MTP2) stays **fail-closed** on both domains — a
"cascade ON" run under it measures the native path, not this feature. Entering the unified cohort would
still first require a dual-pad/graph bucket contract for `k + 1` verification rows plus real replay
correctness on that configuration — that work is **not** planned here (see `docs/support-matrix.md` §5 and
the workspace ledger, OPEN-05).


Microbatching: the plugin mirrors the official vllm core gate and keeps the
two-stage path off under ANY microbatching (`use_ubatching`, i.e. DBO or
`ubatch_size > 1`).  On the current vllm-ascend the platform layer resets both
`enable_dbo` and `ubatch_size`, so this gate is dormant there by construction.

Verified evidence — **historical (Qwen2.5 / 0.25 domain)**, Qwen2.5-Coder-14B-Instruct, 910B2, CANN 9.0.1:

- eager off/on/on_fp32 token-identical; graph off/on/on_fp32 token-identical;
- C3-shaped run (B=64, shared ~4.1k tokens, gen128): 0/64 divergent arms;
- graph cascade at that shape: `VLLM_ASCEND_CASCADE_UPDATE_SKIP_STABLE` on/off
  token-identical (64/64), wall 8.51s vs 9.38s (off baseline 6.49s);
- no fail-open events during the verified runs;
- `pytest -q` + `ruff check .` green.

The full evidence set is archived in benchmark; retrieval and custody rules are
in [EVIDENCE.md](EVIDENCE.md). See
[docs/evidence/cascade/README.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/cascade/README.md) for the
per-section reports (`section2-correctness.md`, `section3-performance.md`,
`section4-active-enablement.md`), the reproduction scripts, the raw result
JSONs and the sha256 manifest (including the list of large sweep logs left in
the development workspace). Read its §1 before quoting any number: the Coder
model is a **performance stand-in** and must not be compared with the real
`Qwen2.5-14B-Instruct` on behavior. This whole set is **historical (Qwen2.5 /
0.25)** and must not be crossed with the current 0.28/Qwen3.5 baseline.

## fi_sampling plugin (default-off, evidence-only)

Entry point: `vllm.general_plugins` ->
`fi-sampling = vllm_ascend_split_batch.fi_sampling_plugin:load`.

Routes the sampling step to the vendored `fi_sampling` triton-ascend kernels
(flashinfer-semantics port, W2 package) by replacing the fork's
`AscendTopKTopPSampler` with a subclass at plugin-load time. With
`VLLM_HUST_FI_SAMPLING` unset nothing is imported or patched.

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_HUST_FI_SAMPLING` | `0` | master switch |
| `VLLM_HUST_FI_SAMPLING_SEED` | torch initial seed | fixed base seed (per-call advancing) |
| `VLLM_HUST_FI_SAMPLING_JOINT_MIN_BATCH` | `256` | joint-cell batch floor (W2 bench knee) |
| `VLLM_HUST_FI_SAMPLING_JOINT_MIN_K` | `32` | joint-cell top-k floor |
| `VLLM_HUST_FI_SAMPLING_K1_ARGMAX` | `1` | `0` keeps k=1 on the fork chain (tie-exact) |
| `VLLM_HUST_FI_SAMPLING_TRACE` | `0` | per-step route trace + histogram |

Routing: no top-k/top-p -> FI api1 kernel; k=1 on every row -> argmax;
`B >= 256` with `k >= 32` and top-p -> FI joint kernel; everything else (and
all five mandatory fallbacks: per-request generators, processed
logits/logprobs modes, batch-invariant, reduce-sample, async-exponential)
stays on the fork chain with no device sync.

Evidence ([docs/evidence/w2b-fi-sampling/REPORT.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/w2b-fi-sampling/REPORT.md), Qwen2.5-Coder-14B, 910B2 —
**historical Qwen2.5/0.25 domain**, see the cohort scope above):

- default-off is zero-diff (no host/kernel import, no patch) and NPU smoke is
  6/6 PASS;
- untruncated e2e (4k ctx, B=64, 2 repeats): median TPOT off 112.67 ms vs on
  112.52 ms (**-0.14%**, within noise) with the FI api1 route confirmed on
  1300/1300 steps -- sampling is only ~1% of TPOT at that batch size;
- joint route is unreachable in the 4k-context single-card shape (B_max ~= 122
  < the 256 floor); it only fires with a ~512-token context, so the joint
  branch is a safe, currently dormant branch in normal deployments;
- `pytest -q` 73 passed + `ruff check .` green.

## Extension framework

Extension ID: `org.vllm-hust.split-batch-full-graph`

The Manifest 0.2 descriptor marks the two cascade carriers `active` (the
planner stays `import_only`): `vllm-hust-ext extension enable` injects the
`activation.environment` flags, and installation alone changes no vLLM
behavior because the cascade capability is env-gated at runtime. Declared
host range: **`vllm-ascend >=0.25.1rc2.dev125,<0.25.2`** — a bounded
*compatibility window*, not a point pin: both endpoints were verified on
910B2 hardware (`dev125` historically; `dev605` = `fbe4911bb` on 2026-09-27),
and a failure inside the window is a host-side finding by design (rationale
and the core-pairing caveat: `docs/release.md` §0.1/§0.3).

```bash
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.split-batch-full-graph
pytest -q && ruff check .
```

⚠️ **`vllm-hust-ext` is not on PyPI** (see the org's own note on
[vllm-hust-website](https://github.com/vLLM-HUST/vllm-hust-website): *"No public
vllm-hust-ext PyPI alpha exists yet; install the current source"*). It is the
org's extension-manager package, maintained in
[`vLLM-HUST/extension-manager`](https://github.com/vLLM-HUST/extension-manager) —
this repository only consumes it.

- **Installing this repository's extras never needs it.** Since 2026-09-26 the
  `test` extra no longer pins it, so `pip install "vllm-ascend-split-batch[test]"`
  resolves on its own. (Released 0.1.0/0.1.1 metadata still carries the pin —
  `pip install ...==0.1.1[test]` fails with `Could not find a version that
  satisfies the requirement vllm-hust-ext==0.2.0.dev0`; use the git step below
  for those versions. PyPI files are immutable, hence the next release.)
- **Running the manifest tests does need it** (`tests/test_manifest.py` imports
  `vllm_hust_ext.manifest`). Install the manager from git first, which is exactly
  what CI and `publish.yml` do:

```bash
python -m pip install "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"
```

The `kernels` extra is the one end users need, and it is
resolvable (`--find-links` to the kernel release, see "Install" above).

## Kernel wheel dependency (soft, fail-open)

Both cascade tiers consume the `ascend_kernel` CCE op wheel
(`fa_fp32_stage1` / `lse_merge`) as a SOFT dependency: it is declared only in
the `kernels` extra — base dependencies stay empty. When the wheel is absent
or its ops fail to register, the whole cascade feature is disabled
(fail-open to the standard full-KV path) with a single warning; the
default-off semantics are unchanged. Validated pairing:

| plugin | kernel wheel | torch_npu | CANN |
|---|---|---|---|
| `0.1.3` | `ascend-kernel==2026.9.26` | `2.13.0rc1` | `9.1.0` |
| `0.1.2` | `ascend-kernel==2026.9.26` | `2.13.0rc1` | `9.1.0` |
| `0.1.1` | `ascend-kernel==2026.9.26` | `2.13.0rc1` | `9.1.0` |

The kernel wheel is not on any package index (it is platform-tagged and must be
rebuilt whenever the CANN / torch_npu / soc tuple changes), so it ships as a
**GitHub Release asset**:

```bash
pip install "vllm-ascend-split-batch[kernels]" \
  --find-links https://github.com/Raing5Days/vllm-hust-cascade-kernel/releases/expanded_assets/v2026.9.26
python -c "import ascend_kernel, torch; assert hasattr(torch.ops.npu, 'fa_fp32_stage1')"
```

Verify the installed kernel build by its library fingerprint (the kernel repo's
rule is "read the md5, not the file name"): `libascend_kernel.so` must hash to
`ca8de2d70fe0504d` for `2026.9.26`. Building it from source is documented in the
kernel repo README (`./build.sh`, ~2 min).

Compat tuple & red lines for the kernel side live in the kernel repo README
§4-§5. After bumping the kernel wheel: reinstall, then rerun the plugin
smoke (see [docs/release.md](docs/release.md) §2).

## Operator validation workflow

The repository owns the executable ladder at `tools/bench-ladder.sh`; `opt bench`
dispatches to this checked-in source rather than mutating archived evidence scripts.
The enforced cadence is:

1. static checks and CPU tests;
2. `opt bench serve`, then `opt bench screen <unique-tag>` on a **plain** server
   (`L1×3 + L2×3 + L3×1`), producing `screen.json`;
3. after recording the candidate as `ready`/`validating`, run a bounded light L4
   with `--trace-tier light --screen-report <screen.json>`;
4. use `--trace-tier deep` only with `--collection-reason mechanism-gap`;
5. parse offline and slim immediately after delivery.

`light` defaults to the host's `max_iterations=32` bounded window, disables stack/memory/shape collection,
stamps the capture scope into the opstat, and removes profiler databases after the
CSV/all-kernel result is delivered. `deep` defaults to a full window and retains
the shape database. Plain and profiler-enabled servers cannot be reused across
modes; stop the active server before switching.

### Carrier status is recorded, not gated

`screen.json` carries a `carriers` block (`opt-ops-carriers.v2`), and
`opt bench serve` prints the same summary right after `/health` is ready
(`tools/bench-ladder.sh carriers [serve_log]` prints it as one JSON line). It records the
three things that actually decide whether the plugin carrier is live:

1. `find_spec("vllm_ascend_split_batch")` under the **serving interpreter**
   (`/usr/local/python3.12.13/bin/python` on this host; `/usr/bin/python3` cannot see
   the package and will report a false absence);
2. the husk entry points enumerable from the `vllm.general_plugins` group
   (name + value; an install that failed to register is equivalent to no carrier);
3. functional markers counted from that run's serve log.

Plugin switches are read from the **serve side only**: `cmd_serve` writes the exported
`VLLM_HUST_*` values to `<serve log>.env.json` (`opt-ops-serve-env.v1`, atomic write, never
overwritten) before the server starts, and the probe reads them through `env_source` /
`env_scope` (`serve-process`). The shell that later runs `opt bench screen` does **not**
necessarily inherit those exports, so it is never used as the source. If the snapshot is
missing or does not match the serve log, every value is `null` with
`env_scope`/`env_source` = `unknown` and a `probe_errors` entry — an unset switch is never
reported as `""` or as `0`.

Attribution is **split per carrier** under `attribution` and is never merged into one boolean:

| field | states | meaning |
|---|---|---|
| `husk.cascade` | `effective` / `present-not-observed` / `absent` / `unknown` | `cascade plugin loaded` seen per process |
| `husk.gate_fp32` | same | `Gate fp32 cache populated` seen per worker |
| `husk.gdn_merge_inproj` | `present-but-fail-closed` / `present-not-observed` / `absent` / `unknown` | the in-repo carrier refuses to merge (no safe public seam), model stays stock |
| `runtime_tree.w3f2` | `applied` / `not-applied` / `unknown` | the deployment-tree patch merged `in_proj` (`W3F2: GDN in_proj merge applied`) |
| `runtime_tree.w3f3` | `enabled` / `not-enabled` / `failed` / `unknown` | shared scalar-gate pack patch (`W3F3: …`) |

`husk.gdn_merge_inproj = present-but-fail-closed` and `runtime_tree.w3f2 = applied` routinely
coexist: the fail-closed path is the in-repo plugin declining a merge it cannot consume safely
(see [HOST_CONTRACT.md](HOST_CONTRACT.md), GDN in_proj merge carrier), while W3F2 is a separate
runtime-tree patch. **Never** attribute the W3F2 merge to this plugin, and never treat the
fail-closed warning as a failure — it asserts the opposite (no double merge).

`Unknown vLLM environment variable detected: VLLM_HUST_*` is **not** an absence signal —
it shows up whether or not the carrier is installed, so it is recorded separately under
`unreliable_absence_signals` with `is_absence_signal: false` (see
[docs/pitfalls.md](docs/pitfalls.md) §1.5). Carrier state is a **recording axis**: the probe is
fail-closed (absence, an entry-point enumeration error, missing log, or a missing env snapshot
records `unknown`/`null`/`false`) and never changes `screen`/`serve` exit codes; only
cross-window absolute comparisons require two windows to share a carrier state.

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
