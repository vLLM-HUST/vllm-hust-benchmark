# §2 证据②：正确性对齐（vllm-hust v1 基线）

本文件合并三块同一新基线（vllm-hust v1 / CANN 9.1.0 / 卡 6）的正确性证据：

1. **修复后构建（`ddc0120`，未 shim）** —— 验收口径，本文件主体（§3–§11）。
2. **修复前的签名漂移诊断与 harness shim（`9396b21`）** —— 仅诊断，压缩在 §3.5 / §3.6。
3. **真模型 like-for-like 重跑** —— 关闭"缺口 A"的口径问题（§12）。

- 被测插件（验收）：`vllm-ascend-split-batch-hust` @ `ddc01207fcfb6eb046a7e3282577b594de1751f0`
  （`fix(cascade): make graph-mode host wrappers signature-agnostic and fail-open`）。
- Baseline under test (all versions read from `logs/v1-ev2-rerun/env_and_versions.txt`):
  vllm-hust v1 `0.28.1.post1.dev143+gf18cf803c.empty` (editable `/vllm-workspace/vllm-hust`)；
  vllm-ascend-hust `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27` (editable, main)；
  torch 2.13.0+cpu / torch_npu 2.13.0rc1 / CANN 9.1.0 (`/usr/local/ascend91`, ATB sourced by activate.d)；
  triton-ascend 3.2.2, `ascend-kernel==2026.3.9`, transformers 5.14.1。
- Harness：`ev2_run.py`（C3 验收 harness 的忠实移植）、`ev2_compare.py`（byte-identical 口径）、
  `ev2_natural_prefix.py` —— 全轮**逐字未改**；唯一新增是各 `logs/v1-*/` 下的驱动/日志。
- Raw artifacts：`logs/v1-ev2/`（修复前构建 + shim 诊断）、`logs/v1-ev2-rerun/`（修复后验收）、
  `logs/v1-real-model/`（真模型）。

## 0. Verdict summary

| # | Item | Verdict |
|---|---|---|
| 0 | Kernel-level: CANN-9.1 wheel ops registered + plugin wheel probe | **PASS** (§1) |
| 1a | Graph lane **starts, captures and replays unshimmed** (the prior blocker) | **PASS** |
| 1b | Graph lane ON vs OFF arm divergence, task-specified **ignore_eos** shape, vs historical ≤8/64 | **FAIL — 22/64** |
| 1c | Same lane in the **literal historical config** (natural EOS), vs ≤8/64 | **PASS — 1/64** |
| 1d | Graph lane determinism on_run2 vs on | **PASS — 0/64** (and off vs off2 0/64) |
| 2 | Eager lane re-run | **FAIL on the letter — 22/64 bf16 / 14/64 fp32** (prior 21/64, 14/64); **not a cascade defect** — stand-in + forced post-answer tail (§4) |
| 3 | B=32 lane | **FAIL on the letter under ignore_eos — 7/32** (both lanes); **PASS in literal config — 1/32** |
| 4 | Natural-answer (pre-EOS) agreement | **PASS — 63/64** (B=64), **31/32** (B=32) |
| 5 | Cascade engaged in every ON leg + no spurious fail-open | **PASS** |
| 6 | Evidence #2 as a whole on the stand-in model | **PARTIAL**: graph-lane blocker closed and verified; arm-divergence criterion still above the historical bound under forced continuation (§8) |
| 7 | Real-model like-for-like re-run (closes the criterion gap, 缺口 A) | **PASS — 6/64** (§12) |

**Bottom line.** The `ddc0120` fix is real and numerically transparent: the unshimmed graph lane now
runs, and its 22/64 ON-vs-OFF arm set and per-arm `first_diff_step` values are **identical to the
prior harness-shimmed diagnostic** (§3.5/§3.6) — i.e. fixing the signatures changed nothing
numerically. On the Coder stand-in the historical arm-divergence criterion (≤8/64 at B=64, 1/32 at
B=32) is **not** met under the `ignore_eos` shape (22/64, 7/32) and **is** met in the literal
historical shape (1/64, 1/32); neither is a like-for-like reproduction because the real
`Qwen2.5-14B-Instruct` is not on this box (§9.1). The real model closes that gap: §12 reproduces the
historical graph main acceptance **6/64** exactly.

## 1. Environment, exact commands, symmetric baseline quirks

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate hust
cd /tmp                       # NEVER /vllm-workspace
export ASCEND_RT_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 VLLM_DISABLE_COMPILE_CACHE=1
# per-leg: VLLM_CACHE_ROOT=/tmp/ev2rerun_cache_<tag>   (fresh dir, rm -rf before each leg)
E=/vllm-workspace/knowledge/evidence/cascade
$E/logs/v1-ev2-rerun/rerun_matrix.sh          # batch 1: 17 legs, graph+eager, ignore_eos + literal
$E/logs/v1-ev2-rerun/rerun_matrix_literal.sh  # batch 2: 6 legs, graph literal + trace-run2
$E/logs/v1-ev2-rerun/rerun_compare.sh         # -> comparisons_raw.txt / natural_prefix_analysis.txt
$E/logs/v1-ev2-rerun/engagement_proof.sh      # -> engagement_proof.txt
```

Harness invocation per leg (verbatim `ev2_run.py`, env-knob driven):
`EV2_BATCH={64,32} EV2_SENT=420 EV2_GEN=128 EV2_MAXLEN=8192 EV2_EAGER={0,1} EV2_GRAPH={0,1}
EV2_IGNORE_EOS={0,1} [EV2_TORCHMERGE=1] python -u ev2_run.py <mode> <tag>`

- **`VLLM_DISABLE_COMPILE_CACHE=1` and a fresh `VLLM_CACHE_ROOT` are applied to EVERY leg,
  ON and OFF alike** (documented baseline quirk: the shared AOT cache hard-crashes this
  baseline; see `EVIDENCE.md` §6 item 3 and `fix-sigdrift-report.md` §4.3 note).
  The prior evidence-#2 eager legs did not set it; the graph legs could not run at all.
  Numerical effect: none observed — the graph ON dump is arm-for-arm identical to the prior
  shimmed ON dump (§3.1 vs §3.6).
- Card 6 HBM Usage Rate 5% before and after (`card6_before.txt`, `card6_after.txt`;
  per-leg samples `card6_perleg.txt`). Card 7 was concurrently running another agent's
  evidence-#3 matrix (neighbour power/thermal load); no process of ours ran there.
- No stray python/vllm processes on card 6 after the matrix (checked 19:41 UTC).
- Graph vs eager mode is real in the artifacts: graph legs `enforce_eager=False`,
  `cudagraph_capture_sizes=[32,64]`, `Capturing CUDA graphs (decode, FULL)` ×2
  (`gB64_ie_off.log`); eager legs `enforce_eager=True`, 0 capture events (`eB64_ie_off.log`).

Measured shape (`shape_and_env.txt`): prompt = **4224 tokens** (base 4207); the plugin splits
at block-aligned `shared_len=4096 shared_blocks=32` (48 layers). The prior report/docstring
label this shape "~4356 tok"; the measured value here is 4224 (split at 4096).

**Kernel-level re-verification (item 0, PASS)** — artifact `logs/v1-ev2/kernel_wheel_probe_and_env.txt`:

```
ascend-kernel            2026.3.9
torch.ops.npu.fa_fp32_stage1 registered: True
torch.ops.npu.lse_merge        registered: True
cascade_plugin._probe_kernel_wheel() -> (True, '')
kernel_wheel_available() -> True
```

`ascend_kernel` imports and both custom ops are registered on the CANN 9.1 wheel; the plugin's own
probe agrees. Card 6 free at start (`HBM Usage Rate 5`, `logs/v1-ev2/card6_before_matrix.txt`).

## 2. Criterion and provenance (unchanged)

- **Graph main acceptance**: 14B, B=64, shared≈4356 tok, buckets [32,64], MIN_PREFIX=4096,
  gen128, det → ON vs OFF **6/64 ≤ C 口径 8/64**; on_run2 vs on **0/64**.
  `cascade-c3-results/RESULTS.md:11-24` (§1); criterion text
  `现阶段情况核实/cascade-fp32-总流程记录-20260903.md:53,67`
  ("发散面：bf16 带 1~17 臂/格（C 口径 ≤8/64@B64）").
- **B=32**: ON vs OFF **1/32** — `RESULTS.md:30` (§2).
- **Eager**: Tier-0 bf16 **8/64**, Tier-1 fp32 **5/64** (original gate ≤2/64 not met) —
  `RESULTS.md:117-130` (§6.4).
- **Reference-frame control**: eager-off vs graph-off **7/64** — `RESULTS.md:124`.
- Determinism requirement: on_run2 vs on **0/64** — `RESULTS.md:17`.

The historical numbers were produced on the **real `Qwen2.5-14B-Instruct`** with natural EOS
(`out_tokens` 6469 off / 6634 on for 64 arms ≈ 101–104 tok/arm). This box only has the
`Qwen2.5-Coder-14B-Instruct` stand-in, whose answer is 3–5 tokens; see §8 and §12.

## 3. Item 1 — graph lane, UNSHIMMED (the historical main-acceptance lane)

C3 shape: B=64, sent=420 (4224-tok prompt), gen=128, temperature=0, `enable_prefix_caching`,
`max_model_len=8192`, `enforce_eager=False`, `cudagraph_capture_sizes=[32,64]`, graph gate
`VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1`, `MIN_PREFIX=4096`, `MIN_REQS=2`.

### 3.1 Arm divergence — task-specified `ignore_eos` full-decode shape

| comparison | result | log |
|---|---|---|
| **on(bf16) vs off** | **22/64** | `comparisons_raw.txt` "G-ie B64 graph on(bf16) vs off"; dumps `raw_dumps/ev2_on_gB64_ie_on.json` vs `ev2_off_gB64_ie_off.json` |
| on_run2 vs on | **0/64** | same file, "[DETERMINISM]" |
| on_run2(trace) vs on(trace) | **0/64** | same file, "[DETERMINISM, both traced]" |
| on_run2(trace) vs on_run2(no-trace) | **0/64** | same file, "[TRACE INVARIANCE]" |
| off vs off2 | **0/64** | same file, "[NOISE FLOOR]" |
| on_fp32 vs on(bf16) | **0/64** | same file (graph Tier-1 fp32 in-graph is disabled → same source, as documented) |
| on_fp32 vs off | 22/64 (same arms) | same file |

Diverged arms and first-diff steps (raw, `comparisons_raw.txt`):
```
[1, 7, 8, 9, 18, 22, 26, 27, 28, 29, 30, 36, 38, 42, 47, 53, 56, 57, 58, 59, 62, 63]
arm 1:53  7:13  8:13  9:13  18:13  22:6  26:6  27:83  28:6  29:54  30:56  36:7
arm 38:55  42:55  47:13  53:15  56:15  57:15  58:15  59:15  62:8  63:62
```
This arm set **and every first-diff step** equal the prior harness-shimmed graph diagnostic
(§3.6: "on vs off 22/64", same 22 arms, same steps) and the prior trace leg. So the fix is
numerically transparent.

**Where the divergences live** (`natural_prefix_analysis.txt`): **1 natural-region** (arm 1,
no EOS, first_diff 53), **21 post-EOS** (OFF first_EOS 3–4, first_diff 6–83);
**natural-prefix exact match 63/64**. Decoded both sides is coherent — no garbage.

### 3.2 Arm divergence — literal historical config (natural EOS)

| comparison | result | log |
|---|---|---|
| **on(bf16) vs off** (no ignore_eos) | **1/64** (arm 1, first_diff 53) | `comparisons_raw.txt` "G-lit B64 graph on(bf16) vs off"; dumps `ev2_on_gB64_lit_on.json` / `ev2_off_gB64_lit_off.json` |
| on_run2 vs on | **0/64** | same file, "G-lit B64 ... [DETERMINISM]" |
| natural-prefix match | **63/64** (1/1 natural) | `natural_prefix_analysis.txt` "LANE G literal B=64" |

`out_tokens` = 515 for 64 arms (`gB64_lit_{off,on}.log`) — the stand-in emits EOS after 3–5
tokens, so this config exercises only ~5–8 decode steps/arm (see §8).

### 3.3 Determinism (explicit)

- ON lane is deterministic: on_run2 vs on **0/64**, and it holds **with trace on/off**
  (both 0/64) and across the trace-invariance pair (0/64). Log: `comparisons_raw.txt` §"LANE G".
- OFF lane is deterministic: off vs off2 **0/64** (same file).

### 3.4 Activation proof for the graph lane (item 5)

`gB64_ie_on.log` (`engagement_proof.txt`):
```
INFO [cascade_plugin.py:147] cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)
capture body SUCCESS: num_tokens=32  x48
capture body SUCCESS: num_tokens=64  x48
replay update: cascade key hit num_tokens=32 x1 ; num_tokens=64 x126
wrapper: replay_swap=True x227 / False x700
stage1 re-bind skipped: stable sig shared=4096 tokens=64 x125
cascade twin missing for descriptor BatchDescriptor(num_tokens=4559, ..., uniform=False) x1 (WARNING)
```
OFF leg `gB64_ie_off.log`: `gate=0, graph_gate=0`, capture 0, replay 0, twin-miss 0.
Identical proof for B=32 (`gB32_ie_on.log`: 96 captures, 127 key hits @num_tokens=32) and for
the literal legs (`gB64_lit_on.log`: 96 captures, 125 key hits) — see `engagement_proof.txt`.

### 3.5 修复前（`9396b21`）：graph lane 无法启动 + 签名漂移隔离（DIAGNOSTIC ONLY）

`logs/v1-ev2/graphB64_ie_on.log` (B=64, `cudagraph_capture_sizes=[32,64]`,
`VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1`):

```
INFO  cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)
...
TypeError: _patch_capture_scheduling.<locals>._capture_cudagraphs()
           got an unexpected keyword argument 'profiler'
RuntimeError: NPUModelRunner init failed ... Engine core initialization failed
```

The same failure occurs at every scale (B=4 smoke `logs/v1-ev2/smoke_eager_on.log` hits the sibling
drift even earlier). **Exact errors and root causes** — canonical D-labels
(cf. `EVIDENCE.md` §4 and `fix-sigdrift-report.md`):

| drift | plugin side (`9396b21`) | v1 host side | symptom |
|---|---|---|---|
| **D1（漂移口径）** `_capture_cudagraphs` 增 `profiler=` | `cascade_runner_patch.py:177` 只声明 `(self, batch_descriptors, cudagraph_runtime_mode)` | `vllm/v1/worker/gpu_model_runner.py:7059` 增 `profiler=None`，调用点 `:6971` 传 `profiler=` | `TypeError: ... unexpected keyword argument 'profiler'` → **引擎初始化失败**（graph lane，`logs/v1-ev2/graphB64_ie_on.log:133`） |
| **D2（漂移口径）** `_update_full_graph_params_if_needed` 删 `positions` | `cascade_runner_patch.py:288` 以 3 位置参（含 `positions`）调用 | `vllm_ascend/worker/model_runner_v1.py:2935` 已删 `positions` | `TypeError: ... takes 3 positional arguments but 4 were given`（eager + GRAPH=1，`logs/v1-ev2/smoke_eager_on.log:119`） |
| **D3（漂移口径）** `update_graph_params` 删 `num_dcp_pcp_tokens` | `cascade_graph_plugin.py:928-936` 以 7 个位置参转调 `orig_update` | `vllm_ascend/attention/attention_v1.py:592` 取 6 个（无 `num_dcp_pcp_tokens`） | `TypeError: ... takes from 4 to 6 positional arguments but 7 were given`（`logs/v1-ev2/g2b_gate1_skip_runner_patch.log:184`） |

`HOST_CONTRACT.md` (commit `9396b21`) claimed the anchor map was re-verified on this baseline. It
was not complete: `_update_full_graph_params_if_needed` is listed under "runner five methods ...
wrappers forward via `*args`", but the plugin called it with a fixed 3-arg list, and the two
`_capture_cudagraphs` / `update_graph_params` signatures were not covered.

**Isolation proof** (`ev2_graph_isolate.sh`, all legs fresh `VLLM_CACHE_ROOT`):

| leg | graph_gate | graph_plugin.install | runner_patch.install | outcome |
|---|---|---|---|---|
| g3b | 1 | neutralized | neutralized | engine starts, graph capture OK |
| g1b | 1 | neutralized | **active** | **D1（漂移口径）TypeError** at capture |
| g2b | 1 | **active** | neutralized | engine starts, then **D3（漂移口径）TypeError** at first replay |
| g4 | 1 | active | active | **D1（漂移口径）TypeError** at capture |

Logs: `logs/v1-ev2/g3b_*.log`, `g1b_*.log`, `g2b_*.log`, `g4_gate1_full.log`.

**共性（修复前）**：D1–D3（漂移口径）都是"import 成功、签名不匹配"的**静默失配**（`docs/pitfalls.md` §2.1 /
HANDOFF「最危险：静默失配」），且**违反 fail-open 契约**：文档承诺失败即回落 fork/标准路径，实际是硬崩引擎。

### 3.6 修复前的 harness shim 诊断（DIAGNOSTIC ONLY — not an acceptance result）

To answer "would the twin work if the three signatures were fixed?", `ev2_shim.py` wraps the plugin's
two `install()` functions and adapts D1/D2/D3（漂移口径）**in the harness only** (no host or plugin source was
modified; enabled with `EV2_SHIM_SIG=1`). With the shim:

- `logs/v1-ev2/traceB64_ie_on.log` (with `VLLM_ASCEND_CASCADE_TRACE=1`):
  `capture body SUCCESS` ×96 (48 layers × buckets 32 and 64), `replay update:
  cascade key hit` ×127, `wrapper: replay_swap=True` ×227, and the only
  "twin miss" entries are prefill steps (num_tokens 4559 / 3704), which is the
  documented fail-open. So the twin **does capture and replay** on this baseline
  once the signatures match.
- Correctness under the shim: **on vs off 22/64** (`logs/v1-ev2/comparisons_raw.txt`,
  `shimB64_ie_on` vs `shimB64_ie_off`); determinism **0/64** (off vs off2);
  natural-prefix agreement **63/64**; natural-region divergence **1/64**, the
  other 21 post-EOS — the same shape as the eager lane.
- Token-level: 2077/8192 differing positions (25.35%).

This is a **diagnostic**, never an acceptance verdict: the shipped plugin (`9396b21`) could not
exercise this lane at all. After `ddc0120` the unshimmed lane reproduces exactly this 22/64 arm set
and every first-diff step (§3.1) — i.e. the fix is numerically transparent.

## 4. Item 2 — eager lane re-run

C3 shape with `enforce_eager=True`, graph gate 0.

| comparison | this run | prior run (`ev2` eager, `9396b21`) | log |
|---|---|---|---|
| on(bf16) vs off, ignore_eos | **22/64** | 21/64 | `comparisons_raw.txt` "E-ie B64 eager on(bf16) vs off" |
| on_fp32 vs off, ignore_eos | **14/64** | 14/64 | same file |
| on_fp32 vs on(bf16) | **17/64** | 18/64 | same file |
| on_run2 vs on | **0/64** | 0/64 | same file |
| **off vs off2** | **1/64 (arm 42)** | 0/64 | same file, "[NOISE FLOOR]" |
| torchmerge-on vs kernelmerge-on (ablation) | **0/64** | 0/64 | same file |
| torchmerge-on vs off | 22/64 (same arms) | 21/64 | same file |
| literal on vs off | **1/64** (arm 1 @103) | 1/64 | same file, "E-lit" |
| B=32 on vs off, ignore_eos | **7/32** | 7/32 | same file, "E-ie B32" |

**Stand-in or real defect? Stand-in model + `ignore_eos` forced continuation — not a cascade
defect.** Evidence, all from artifacts:

1. **Locus**: 21 of 22 divergences first differ *after the OFF arm already emitted EOS*
   (OFF `first_EOS` 3–4; `first_diff` 6–83) — `natural_prefix_analysis.txt` "LANE E B=64".
   Natural-prefix exact match **63/64**; the single natural-region divergence is arm 1, a
   long free-running repetition with no EOS.
2. **Literal (natural-EOS) config is 1/64** — same file, "LANE E literal".
3. **The `lse_merge` kernel is exonerated**: forcing the torch (non-kernel) merge changes
   nothing (torchmerge vs kernelmerge **0/64**; torchmerge vs off **22/64**, the same arm set).
   So the difference is the cascade two-stage bf16 path vs the full-KV FIA path, which the
   plugin itself discloses as bf16-level (one `bf16-level difference ... not bit-exact` notice
   per ON process — present in every ON leg).
4. **The eager reference frame itself is not bit-stable on this run**: off vs off2 = **1/64**
   (arm 42). Arm 42 is also the one arm by which this run's 22/64 exceeds the prior run's
   21/64, and it appears in the eager on_fp32-vs-off set too. So the eager arm-level metric
   carries **±1 arm of run-to-run noise** here (the prior run measured 0/64 on the same pair).
   The graph reference frame was stable (0/64).

Numerically the divergence is therefore confined to the forced post-answer tail; the eager
lane is **above** the historical arm-divergence bar (8/64 Tier-0, 5/64 Tier-1) and **within**
it on the natural-answer reading (1/64).

## 5. Item 3 — B=32

| lane | ignore_eos | literal |
|---|---|---|
| graph (unshimmed) | **7/32** (arms 1,21,22,26,27,29,30) | **1/32** (arm 1 @103) |
| eager | **7/32** (same arms/steps) | (not run; eager literal B=32 not in scope) |
| historical (graph) | 1/32 (`RESULTS.md:30`) | 1/32 |

Logs: `comparisons_raw.txt` §"LANE G B=32" / §"LANE G literal ..." / §"LANE E B=32";
dumps in `raw_dumps/`. Natural-prefix match **31/32** in both ignore_eos lanes
(`natural_prefix_analysis.txt`). The literal B=32 number (1/32) meets the historical bound;
the arm differs from the historical arm 22@step36 (ours: arm 1@step103, a no-EOS
free-running arm).

## 6. Item 4 — natural-answer (pre-EOS) agreement

| pair | natural-region divergences | natural-prefix exact match | log |
|---|---|---|---|
| graph B=64 on vs off | 1 (arm 1) | **63/64** | `natural_prefix_analysis.txt` |
| eager B=64 on vs off | 1 (arm 1) | **63/64** | same |
| graph B=32 on vs off | 1 (arm 1) | **31/32** | same |
| eager B=32 on vs off | 1 (arm 1) | **31/32** | same |
| graph literal B=64 | 1 (arm 1) | **63/64** | same |
| eager literal B=64 | 1 (arm 1) | **63/64** | same |

i.e. in every lane the two arms' **natural answers are identical in 63/64 (B=64) / 31/32
(B=32)** cases; only the single no-EOS free-running arm 1 diverges inside the natural region
(at step 53/103).

## 7. Item 5 — activation + fail-open audit (`engagement_proof.txt`)

| check | result |
|---|---|
| Every graph ON leg | `gate=1, graph_gate=1, kernel_wheel=ok`; 96 `capture body SUCCESS` (48 layers × buckets 32/64); 125–127 `replay update: cascade key hit`; 123–126 `stage1 re-bind skipped`; `replay_swap=True` 176–227 |
| Every eager ON leg | `gate=1, graph_gate=0`; `[cascade-active] shared_len=4096 shared_blocks=32 num_tokens={64,32} heads=40 head_size=128` **×48** |
| Every OFF leg | `gate=0, graph_gate=0`; cascade-active 0, capture 0, replay 0, twin-miss 0, wheel warnings 0 |
| **Fail-open warnings** | exactly **one** `cascade twin missing for descriptor BatchDescriptor(num_tokens=4559/4035, ..., uniform=False); step replays the standard full-KV graph` per graph ON process — the documented **prefill** fail-open (non-uniform step). Not spurious. |
| Other fail-open paths | **0** in all 23 legs: no "delegating to the original implementation" (drift), no "capture fell back", no "GraphParams unavailable", no "param lists misaligned", no "replay without cascade_shared_len", no "replay with short real request", no "cascade replay update failed", no "wheel unavailable", no "lse_merge kernel failed" |
| bf16 disclosure notice | exactly 1 per ON leg (documented, `VLLM_ASCEND_CASCADE_STRICT` reference) |
| Crashes | **0 `TypeError`, 0 `Traceback`** in all 24 legs (23 matrix + smoke; `final_verification.txt`) — the D1–D4（漂移口径）drifts are gone |
| Non-trace determinism leg | `gB64_ie_on_run2` (trace off) shows the twin-miss WARNING + bf16 notice, and its dump equals both the traced ON dump and the traced run2 dump (0/64) — so engagement is established transitively where trace lines are absent |

**修复前对照（`9396b21`）**：同一 `VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1`（manifest
`activation.environment` 逐字注入该 flag）下，引擎**硬崩而非回落**到标准 full-KV 路径——
fail-open 契约被违反（`logs/v1-ev2/graphB64_ie_on.log`，§3.5）。修复后（`ddc0120`）为 0 fail-open（上表）。

Observation (not a defect): the **prefill batching descriptor varies run-to-run**
(`4559+3704` in the traced ON/run2/literal legs vs a single `4428` in the untraced run2 leg),
with **no token effect** (0/64 in every determinism pair). Recorded for the next host bump.

## 8. HTTP serving-level check (OFF vs ON, `9396b21` harness, offline matrix is the acceptance artifact)

Script `ev2_serve_compare.py`; logs `logs/v1-ev2/serve_{off,on}_card6.log`（offline 矩阵未重跑 serve；
"启动并可服务"在 `ddc0120` 后由 `fix-sigdrift-report.md` §4.3 的 serve 日志覆盖：`gate=1, graph_gate=1`,
96 captures, 63 replays, 0 TypeError）。
`vllm serve` on card 6, `--enforce-eager --max-model-len 8192
--gpu-memory-utilization 0.85`, cascade ON = decode gate + MIN_PREFIX=4096 +
MIN_REQS=2 (graph gate OFF, because it could not start — §3.5). 8 prompts with a
shared ~4356-token prefix fired **concurrently** (a serial loop gives B=1 and
never opens the MIN_REQS=2 gate — first attempt, `serve_on_card6` in
`logs/v1-ev2/ev2_serve_compare_runner.log`, had `cascade-active` count 0; the
concurrent rerun has 48).

| leg | startup marker | `cascade-active` |
|---|---|---|
| OFF | `gate=0, graph_gate=0` | 0 |
| ON | `gate=1, graph_gate=0` | 48 (`shared_len=4224 shared_blocks=33 num_tokens=8`) |

Result (`logs/v1-ev2/raw_dumps/ev2_serve_compare.json`): **8 prompts, 4 exact
match, 4 diverged**; first-difference character positions 583 / 515 / 144 / 166 —
i.e. all four diverge in the post-answer "Created Question/Answer" continuation,
after the arithmetic answer is already correct in every prompt. Sample:
`logs/v1-ev2/serve_compare_text_sample.txt`.

## 9. Deviations from the historical measurement setup (and their effect)

1. **Model stand-in (largest deviation, stand-in run).** The real `Qwen2.5-14B-Instruct` is absent
   (`/data/shared_models` holds only `Qwen--Qwen2.5-Coder-14B-Instruct` + `strict-*`). The
   historical 6/64 / 5/64 / 8/64 / 1/32 were measured on the real model with **natural EOS**
   (≈101–104 tok/arm). The Coder stand-in answers the C3 arithmetic prompt in 3–5 tokens and
   emits EOS, so:
   - literal config: only ~5–8 decode steps/arm (out_tokens 515 for 64 arms) → the
     arm-divergence metric is dominated by the first few steps (1/64);
   - `ignore_eos`: 128 forced continuation steps/arm, of which ~124 are post-answer
     low-entropy repetition → near-ties flip (21/22 divergences post-EOS, 22/64).
   **The stand-in cannot be compared like-for-like**; §12 removes this deviation by re-running on
   the real model.
2. **Prompt/shape**: measured 4224 prompt tokens split at 4096, vs historical shared≈4356
   →4608 aligned. Same MIN_PREFIX=4096, buckets [32,64], B, gen, temp=0, prefix caching,
   `max_model_len=8192`.
3. **`VLLM_DISABLE_COMPILE_CACHE=1` + fresh per-leg `VLLM_CACHE_ROOT`** (baseline quirk),
   applied symmetrically to ON and OFF, graph and eager. The prior evidence-#2 eager legs did
   not set it. Effect on results: none observed — the unshimmed graph ON dump reproduces the
   prior shimmed ON dump arm-for-arm and step-for-step.
4. **Harness**: `ev2_run.py` offline `LLM` API (a faithful port of the C3 acceptance harness),
   not the original `c3_run_14b_graph.py`. Unchanged from the prior run; `enable_chunked_prefill=False`
   and `async_scheduling=False` are harness constants (vLLM warns about the former).
5. **Hardware/neighbour**: card 6 alone, but card 7 was running another agent's heavy matrix
   concurrently (possible power/thermal neighbour effect). Historical runs were on NPU0 of the
   source machine.
6. **Eager reference-frame instability** (this run): eager off vs off2 = 1/64 (arm 42), so the
   eager arm-level numbers carry ±1 arm of noise; graph off vs off2 = 0/64.

## 10. Gaps / not reproduced

1. **The historical model is not on this box** — the historical criterion cannot be measured
   in its own configuration on the stand-in (§9.1). Closed by the real-model re-run (§12).
2. **HTTP serve-level check not re-run on `ddc0120`** — "starts and serves unshimmed" is covered by
   the fixer's serve log (`logs/fix-sigdrift/serve-on-card6.log`) and §8 of this file. Not duplicated
   to keep card time bounded.
3. **Graph-mode performance** is out of scope (evidence #3).
4. **`0.5B` regression shape** / **8k / 16k prefix shapes** from the W5 sweep not re-run; this section
   covers the C3 main shape (shared≈4.4k, B=64/B=32) plus the §8 serve check.
5. **No tokenizer-level human review of every diverged arm** — locus analysis plus decoded samples
   (`logs/v1-ev2/natural_prefix_analysis.txt`, `serve_compare_text_sample.txt`). The decoded text on
   both sides is coherent (normal near-tie flips, no garbage), matching the historical "正常翻转" finding.
6. **Prefill-batching variance** (descriptor 4559+3704 vs 4428) is observed but not explained;
   no eager-side prefill trace exists to check whether it also varies in the eager lane.

## 11. Real defect vs harness limitation

- **No remaining blocking defect on the graph path**: the three active drifts (D1/D2/D3（漂移口径）) and
  the latent D4（漂移口径）are fixed; the unshimmed graph lane starts, captures, replays, and is
  deterministic; all 23 legs are crash-free with zero drift warnings. Detail:
  `EVIDENCE.md` §4/§5, `fix-sigdrift-report.md`.
- **The arm-divergence numbers are a shape/stand-in limitation, not a new defect**: they are
  numerically identical to the pre-fix shimmed diagnostic and to the pre-fix eager lane, and
  the `lse_merge` ablation (0/64) plus the post-EOS locus show the difference is the
  documented bf16-level cascade-vs-full-FIA difference in a forced-continuation regime the
  stand-in model creates.
- **Measurement caveat to carry forward**: the eager reference frame is not bit-stable on the
  stand-in run (off vs off2 = 1/64), so single-arm deltas in the eager lane are inside the noise band.

## 12. 真模型 like-for-like 重跑（缺口 A 关闭，2026-09-10）

- 卡 7（HBM 5% 前后一致）；插件 `783794a`（clean tree）。
- 模型：`/data/shared_models/Qwen--Qwen2.5-14B-Instruct`（真模型，非替身；8 safetensors/28G）。
- 口径：**字面历史口径**（自然 EOS，`EV2_IGNORE_EOS` 从未设置；B=64/32，sent=420→实测 4224 tok，
  split@4096，gen128，temp=0，prefix caching，max_model_len 8192）。
- 唯一 harness 偏差：`logs/v1-real-model/ev2_run_real.py` 第 68 行 MODEL 路径（diff 已核；
  原 `ev2_run.py`/`ev2_compare.py`/`ev2_natural_prefix.py` 未动，mtime 2026-09-09 不变）。
- 原始产物：`logs/v1-real-model/`（8 腿日志、raw_dumps、全部对比/locus/engagement 文件）。

**判定：PASS — 与历史逐项对上**

| 项 | 本次 | 历史（`RESULTS.md`） | 判 |
|---|---|---|---|
| graph B64 on vs off（**主验收**） | **6/64**（臂 22,32,42,47,58,60） | 6/64（臂集不同） | ✅ ≤8/64 且计数精确复现 |
| on_run2 vs on（确定性） | 0/64 | 0/64 | ✅ |
| graph B32 on vs off | **1/32，臂 22@步36** | 1/32，臂 22@步36 | ✅ **臂与步长 bit 级一致** |
| eager bf16 on vs off（参考帧） | 8/64 | 8/64 | ✅ |
| eager off vs off2（噪声底） | 0/64 | 0/64 | ✅ |
| eager-off vs graph-off（参考系对照） | 7/64 | 7/64 | ✅ |
| natural-prefix 一致（graph B64） | 58/64（6 处发散全部自然区，0 post-EOS） | 58/64 | ✅ |
| out_tokens/臂 | 101.4（16 臂自然 EOS） | ~101（6469/64） | ✅ 形态同源 |

替身轮的 22/64 由此得到解释：真模型自然 EOS 无 forced-continuation 尾部，全部发散在自然区
（bf16 级 cascade-vs-full-FIA 近平局翻转），与历史性质完全相同。

**必须随行的 caveat（跨基线漂移，已量化）**

- `NEW off vs HIST off` = **8/64**、`NEW on vs HIST on` = 9/64：宿主版本差
  （vllm 0.28.1/CANN 9.1.0 vs 历史 0.23.0/9.0.1）单独就造成 ~8/64 臂漂移。
- 因此**臂身份不可跨基线比**（也无需比）；验收口径是 on-vs-off 发散计数，其复现是干净的。
- 本轮跑在 vllm-hust v1 / CANN 9.1.0 / 插件 783794a 上——即**未来 active 态将驻留的基线**。

**对 release 的含义**：证据②的三块卡点全部解除——graph lane unshimmed 运行 ✅（§3）、
自然区逐 token 口径 ✅（58/64 = 历史值）、历史 6/64 重算 ✅（6/64 精确复现）。

## 13. Artifact index

### 13.1 `logs/v1-ev2-rerun/`（修复后验收）

- `env_and_versions.txt`, `shape_and_env.txt`, `card6_before.txt`, `card6_perleg.txt`, `card6_after.txt`
- `rerun_matrix.sh`, `rerun_matrix_literal.sh`, `rerun_compare.sh`, `engagement_proof.sh`
  (drivers only; the harness scripts `ev2_run.py` / `ev2_compare.py` / `ev2_natural_prefix.py`
  live in the parent dir and are unmodified)
- `rerun_matrix_runner.log`, `rerun_matrix_literal_runner.log`, `rerun_compare_runner.log`
- `comparisons_raw.txt` (every comparison, raw comparator output), `natural_prefix_analysis.txt`,
  `engagement_proof.txt` (+ `engagement_proof_batch1.txt`), `final_verification.txt`
  (per-leg TypeError/Traceback/drift/twin-miss sweep; card-6 release check)
- Per-leg logs: `gB64_ie_{off,off2,on,on_run2,on_run2_trace,on_fp32}.log`,
  `gB32_ie_{off,on}.log`, `gB64_lit_{off,on,on_run2}.log`, `gB32_lit_{off,on}.log`,
  `eB64_ie_{off,off2,on,on_run2,on_fp32,on_torchmerge}.log`, `eB64_lit_{off,on}.log`,
  `eB32_ie_{off,on}.log`, `smoke_g_on.log`
- `raw_dumps/ev2_<mode>_<tag>.json` — token dumps for all 23 legs

### 13.2 `logs/v1-ev2/`（修复前构建 + shim 诊断）

- `kernel_wheel_probe_and_env.txt` — item 0 + environment + card check
- `comparisons_raw.txt` — every divergence comparison, raw comparator output
- `natural_prefix_analysis.txt` — per-arm natural-region vs post-EOS divergence
- `raw_dumps/*.json` — token dumps for every leg (incl. `ev2_serve_compare.json`)
- `eagerB64_ie_{off,on,on_fp32,on_run2,off2,on_torchmerge}.log` — Lane A
- `eagerB32_ie_{off,on}.log` — B=32
- `graphB64_ie_{off,on}.log`, `graph_on.log`, `graph_shim_on.log` — Lane B (as shipped)
- `g3b_*`, `g1b_*`, `g2b_*`, `g4_gate1_full.log` — drift isolation
- `traceB64_ie_on.log`, `shimB64_ie_{off,off2,on}.log`, `shim_smoke_on.log` — shim diagnostics
- `serve_{off,on}_card6.log`, `serve_compare_text_sample.txt` — HTTP serving level
- `smoke_eager_on.log`, `smoke_eager_on_gate0.log`, `smoke_card6_before.txt` — first-contact smokes

Scripts (all under `knowledge/evidence/cascade/`):
`ev2_run.py`, `ev2_compare.py`, `ev2_natural_prefix.py`, `ev2_shim.py`,
`ev2_smoke.sh`, `ev2_matrix.sh`, `ev2_eager_matrix.sh`, `ev2_graph_diag.sh`,
`ev2_graph_isolate.sh`, `ev2_graph_fresh.sh`, `ev2_graph_shim.sh`,
`ev2_graph_trace.sh`, `ev2_controls.sh`, `ev2_summarize.sh`,
`ev2_serve_compare.py`.

## 14. Reproduction

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate hust
cd /tmp   # NEVER /vllm-workspace
E=/vllm-workspace/knowledge/evidence/cascade
# 修复后验收（§3–§7、§9）
$E/logs/v1-ev2-rerun/rerun_matrix.sh
$E/logs/v1-ev2-rerun/rerun_matrix_literal.sh
$E/logs/v1-ev2-rerun/rerun_compare.sh
$E/logs/v1-ev2-rerun/engagement_proof.sh
# 修复前诊断与隔离（§3.5/§3.6）
$E/ev2_smoke.sh          # first contact: does cascade activate?
$E/ev2_eager_matrix.sh   # Lane A: eager off/on/on_fp32/on_run2, B=64 + B=32
$E/ev2_matrix.sh         # Lane B: graph as shipped (fails) + literal C3 config
$E/ev2_graph_isolate.sh  # drift isolation (g3b/g1b/g2b/g4)
$E/ev2_graph_shim.sh     # DIAGNOSTIC: graph twin with ev2_shim.py
$E/ev2_graph_trace.sh    # DIAGNOSTIC: twin capture/replay proof
$E/ev2_controls.sh       # determinism + lse_merge ablation
$E/ev2_summarize.sh      # regenerate comparisons_raw.txt
python $E/ev2_serve_compare.py   # HTTP serve-level OFF vs ON (§8)
# 真模型 like-for-like（§12）
ASCEND_RT_VISIBLE_DEVICES=7 python $E/logs/v1-real-model/ev2_run_real.py ...
```
