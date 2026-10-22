# Cascade 翻 active 三项证据（总入口）

> **文件定位**：cascade 线的**唯一证据总入口**。§0–§7 = 新基线（vllm-hust v1，2026-09-09 测量）；
> §8 = 旧基线（0.23.0rc1 / CANN 9.0.1）留档，**不继承**（2026-09-13 由原独立文件 `EVIDENCE.md` 归并入本节）。
>
> **⚠️ 现行状态（2026-09-12 核）**：两个 cascade carrier（`cascade_plugin:load`、`cascade_graph_plugin:install`）为 **`active`**（插件 `c968c73`，启用验证见 `section4-active-enablement.md`）；`planner:plan_dual_pad` 为 `import_only`（无验收证据）。§5 列出的两个缺口（A/B）均已关闭（见 §5 各条与 `section4-active-enablement.md`）。§0–§7 保留作当时证据链（各节数字、gate 9 格、缺陷 D1–D4（漂移口径）一律不改），**引用状态一律以 `section4-active-enablement.md` 为准**。

- 插件仓库：`vllm-ascend-split-batch-hust`，分支 `feat/cascade-attention-plug`，被测 commit `9396b21`
- **新基线**（本次验证对象，版本号均读自归档产物）：
  - vllm-hust **v1** `0.28.1.post1.dev143+gf18cf803c`（`/vllm-workspace/vllm-hust`）
  - vllm-ascend-hust main `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`
  - torch 2.13.0+cpu / torch_npu 2.13.0rc1 / **CANN 9.1.0**（`/usr/local/ascend91`）/ triton-ascend 3.2.2
  - kernel wheel `ascend-kernel==2026.3.9`（**CANN 9.1 重编**，`ops/kernels/ascend-kernel@73bf96a`）
- 旧基线（0.23.0rc1 / CANN 9.0.1，已退役）留档见本文件 §8，**不继承**。
- 详细分节：`section2-correctness.md`（证据②，含真模型 like-for-like）、
  `section3-performance.md`（证据③，含 G6 gate 修复，见其 §13）、`fix-sigdrift-report.md`（D1–D4（漂移口径）修复细节）

## 0. 结论速览（审核后）

| release.md §0 三项证据 | 未修复构建 | **修复后构建（`ddc0120`）** | 依据 |
| --- | --- | --- | --- |
| ① default-off 零回归冒烟 | ✅ 齐备 | ✅ 齐备 | §1 |
| ② 与参考/基线的正确性对齐 | ❌ 不满足（图模式无法启动） | ⚠️ **部分**（阻塞已解除；但强制续写形态发散 21–22/64 > 历史 ≤8/64 口径） | §2 |
| ③ TPOT/latency 性能对比 | ⚠️ 部分（ON 腿需 shim） | ⚠️ **部分**（未 shim 已可测；存在 gate 未覆盖的亏损格） | §3 |

**快照结论（2026-09-09）：cascade 维持 `import_only`**，依据三条：
1. 证据②在**强制续写（ignore_eos 128 步）**形态下发散 21–22/64，高于历史 ≤8/64 口径；字面形态 1/64 通过。该差异与修复无关（修复前后发散臂集合一致），但口径本身尚未在新基线达成一致（stand-in 模型 vs 历史真模型）。
2. 证据③存在**自适应 gate 未覆盖的亏损格**：`(64,4096)` 实测 +6.7% 亏损，而 gate 微基准判该格 `on`、不回退。
3. 阻塞性缺陷 D1–D4（漂移口径）已修复（`ddc0120`），图模式车道首次在新基线跑通，属实质进展。

> **两个缺口的关闭（现状，见 `section4-active-enablement.md`）**：
> 缺口 B 由 gate 分档 bench margin 关闭（`c4121e2`：P≤4096 桶需 ≥25%，使 gate 同时保护 `(32,4096)` 与
> `(64,4096)`；4 腿重验：亏格收回 +6.7%→+2.1%、8k/16k 赢格保留、`(128,4096)` 大 margin 格不受影响，
> 见 `section3-performance.md` §13）；缺口 A（正确性口径待真模型或团队决策）由真模型到位关闭
> （见 `section2-correctness.md` §12）。

**复核方法与范围**：缺陷代码双侧比对、AST 参数级签名审计（`logs/sig-audit/`）、关键数字对归档产物抽查、pytest/ruff/未 shim 启动/变异测试独立复跑、发散臂集合前后一致性比对。证据②③分别为两轮（未修复 / 修复后）构建、分卡 6/7 采集。范围含 §4 的 HOST_CONTRACT 参数级签名（见 :141）。

## 1. 证据一：default-off 零回归冒烟（PASS）

执行：`run_evidence1_default_off.sh`（卡 6，`--enforce-eager --max-model-len 4096 --gpu-memory-utilization 0.85`）
产物：`logs/serve-default-off-v1.log`、`/tmp/ev1.log`

| 项 | 结果 |
| --- | --- |
| 服务就绪 | `READY after 270s` |
| `/health` | **200** |
| chat completions ×3 | **200 / 200 / 200** |
| 启动标记 | **2 条**（API server + EngineCore）`cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=ok)` |
| `cascade-active` 计数 | **0** |
| `wheel unavailable` 计数 | **0** |
| 测后清理 | 端口 8341 释放（curl 000） |

**结论**：装而不开与原生行为零差异；CANN 9.1 重编 wheel 的探针判定为可用（`kernel_wheel=ok`，与证据② §1 的独立探针一致）。

## 2. 证据二：正确性对齐（部分）

### 2.1 修复后构建（`ddc0120`，未 shim）

详见 `section2-correctness.md` §3–§7；数据 `logs/v1-ev2-rerun/comparisons_raw.txt`。

| 项 | 结果 | 溯源 |
| --- | --- | --- |
| 图模式车道启动/捕获/回放（原阻塞点） | **PASS**（首次在新基线未 shim 跑通） | `logs/v1-ev2-rerun/engagement_proof.txt` |
| 图模式 ON vs OFF（`ignore_eos` 满 128 步，任务口径） | **22/64**（> 历史 ≤8/64 → **FAIL**） | `comparisons_raw.txt:5` |
| 图模式 ON vs OFF（历史字面配置，自然 EOS） | **1/64**（**PASS**） | `comparisons_raw.txt:72` |
| 图模式确定性 on_run2 vs on / off vs off2 | **0/64** / **0/64**（PASS） | `comparisons_raw.txt:29,31` |
| eager 复测 | 22/64（bf16）、14/64（fp32）（与修复前 21/64、14/64 同量级） | `comparisons_raw.txt:88` |
| B=32 | 7/32（`ignore_eos`）/ 1/32（字面） | 同上 |
| 自然答案（EOS 前）一致 | **63/64**（B=64）、31/32（B=32）（PASS） | `natural_prefix_analysis.txt:27` |
| 激活与 fail-open | PASS：`gate=1, graph_gate=1`、capture ×96、replay key hit ×125–127；**0 TypeError / 0 drift warning / 0 Traceback**（24 腿） | `final_verification.txt` |

**关键佐证——修复未改变数值**：修复前 eager 发散臂 = `{1,7,8,9,18,22,26,27,28,29,30,36,37,47,53,56,57,58,59,62,63}`（21 臂）；修复后 eager = 该集合 ∪ `{42}`（22 臂），修复后图模式 = 该集合 −`{37}` ∪ `{38,42}`（22 臂）。臂集合高度重合，且 `{42}` 同时出现在 OFF 参考帧的噪声底（off vs off2 1/64）——**说明 21–22/64 的发散是该形态的固有属性，不是修复引入的回归**。`lse_merge` 消融（torch-merge vs kernel-merge）**0/64**，排除 merge 算子责任。

### 2.2 修复前构建（未修复，仅诊断）

| 项 | 结果 | 溯源 |
| --- | --- | --- |
| **图模式车道** | **FAIL：引擎无法启动** | `logs/v1-ev2/graphB64_ie_on.log:133` |
| 图模式（harness shim 后） | 22/64（诊断） | `logs/v1-ev2/traceB64_ie_on.log` |
| eager ON vs OFF（字面 / ignore_eos） | 1/64 / 21/64 | `logs/v1-ev2/comparisons_raw.txt:3,18` |

**历史口径与出处**（未改动）：图模式主验收 14B B=64 on vs off **6/64 ≤ C 口径 8/64**、on_run2 0/64（`knowledge/evidence/c3-legacy/RESULTS.md:11-24`）；eager Tier-1 fp32 5/64、Tier-0 bf16 8/64（`RESULTS.md:117-130`）；B=32 1/32（`RESULTS.md:30`）。

**判定：部分满足。** 阻塞点已解除、确定性与自然答案一致均 PASS；但**强制续写形态下发散 21–22/64 高于历史 ≤8/64**。该差异与修复无关，属**口径可比性问题**：历史 6/64 产自真模型 `Qwen2.5-14B-Instruct`（自然 EOS ≈101 tok/臂），本机只有 `Qwen2.5-Coder-14B-Instruct`（3–5 tok 即 EOS），字面配置仅 ~8 tok/臂（1/64 通过）、强制续写 128 步（21–22/64 不通过）。**要判定证据②通过，需二者之一**：(a) 取到历史真模型重跑；(b) 与团队就"强制续写形态"的判定口径达成一致。

## 3. 证据三：性能对比

### 3.1 修复后构建（`ddc0120`，**未 shim**，验收口径）

详见 `section3-performance.md` §5；数据 `logs/v1-ev3-rerun/analysis.json` → `delta_table`。

| cell | OFF 中位 (s) | ON 中位 (s) | delta | 判定 | n(off/on) | 区间是否分离 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| p420_b32 | 7.71 | 8.66 | **+12.3%** | **LOSS** | 3/3 | 分离 |
| p420_b64 | 10.50 | 11.20 | **+6.7%** | **LOSS** | 5/5 | 分离 |
| p800_b32 | 9.08 | 8.76 | −3.5% | WIN | 3/3 | 分离 |
| p800_b64 | 13.28 | 11.49 | **−13.5%** | WIN | 5/5 | 分离 |
| p1600_b32 | 12.74 | 10.17 | **−20.2%** | WIN | 3/3 | 分离 |
| p1600_b64 | 20.11 | 13.75 | **−31.6%** | WIN | 5/5 | 分离 |

- 所有 6 格 ON/OFF 区间**不重叠**（无"不可判"格）；b64 各 5 次重复，取 3 次子集结论不变。
- **ON 腿确已触发**（14 条 ON 腿全部）：`gate=1, graph_gate=1, kernel_wheel=ok` ×2、`capture body SUCCESS` ×144、`replay update: cascade key hit` ×254–381、**0 TypeError / 0 Traceback / 0 fail-open warning**；OFF 腿 capture/replay/swap 全 0。
- **与 shim 表对比**：5 格差异 ≤2.7pp；唯一方向变化是 `p420_b64` 由"不可判"转为 **决定性 +6.7% 亏损**。说明 shim 对性能中性，旧 shim 数字方向可信。
- **与旧基线拓扑对比**（`archive/cascade-e2e/sweep_*_w5d.log`：+6.9/−3.5/−6.6/−19.6/−21.2/−33.7%）：拓扑复现（4k 亏、8k 小赢、16k 大赢），但 8k/16k 赢幅收窄 2–6pp，`p420_b64` 由小赢翻为亏损、`p420_b32` 亏损加深。

### 3.2 ⚠️ 产品缺口：自适应 gate 未覆盖 `(64,4096)` 亏损格

gate 微基准实测（`logs/v1-ev3-rerun/*.log`，4 次重复一致）：

```
[cas-gate] N=32 P=4096 cascade=648–684us full=429–432us -> OFF   ← 正确回退
[cas-gate] N=64 P=4096 cascade=699–737us full=822–826us -> on    ← 判 on，但 e2e 实测 +6.7% 亏损
```

即 **gate 的微基准与 e2e 结论在 `(64,4096)` 上矛盾**：微基准说 cascade 更快（699 vs 822 µs），e2e 说慢 6.7%。gate 因此不会回退该格，亏损未被覆盖。`p420_b32` 因 gate 判 OFF 而回退，亏损 +12.3% → +0.9%（真实回退生效）。

另需注意：gate 开启的 b64 腿读数为 10.77s，但 `GATE_OVERRIDE=on`（跳过 bench）读数 11.20s = gate-unset 值——说明该差异来自 **bench 子进程预热 NPU**，而非 gate 逻辑本身。这提示 gate 的微基准自身会扰动被测对象。

### 3.3 修复前（未修复构建，shimmed，仅诊断）

见 `section3-performance.md` 附录 B。ON 腿全部带 `(shimmed)`——因 D1–D3（漂移口径）未修复，未打补丁的插件无法启动；shim 仅存在于 harness 进程内。delta：p420_b32 +11.9%、p420_b64 +2.2%（噪声带）、p800_b32 −3.0%、p800_b64 −16.0%、p1600_b32 −19.2%、p1600_b64 −30.8%。**不得作为验收证据**。

### 3.4 单算子锚点（修复后 3 次新测）

- `fa_fp32_stage1 @4356 = 124.6 / 130.3 / 134.7 µs`（±4% 波动，落在旧带 133.9–135.6 内/边缘）；
- `lse_merge @B64H40 = 35.9 / 36.4 / 36.8 µs`——4 次独立测量**均高于旧带上界 33.6**（内部散度 ~3%），判定为 **CANN 9.1 重编后的真实差异（约 +8–10%，2–3 µs）**，不改变"固定开销主导"的结论。

## 4. 阻塞性缺陷 D1–D4（漂移口径）（审核确认）

| # | 位置 | 宿主（当前基线） | 症状 | 性质 |
| --- | --- | --- | --- | --- |
| D1（漂移口径）| `cascade_runner_patch.py:177` | `gpu_model_runner.py:7059` 增 `profiler=None`，调用点 :6971 传 `profiler=` | `TypeError: got an unexpected keyword argument 'profiler'` → **引擎初始化失败** | **活跃阻塞** |
| D2（漂移口径）| `cascade_runner_patch.py:288` | `model_runner_v1.py:2935` 已删 `positions` | `TypeError: takes 3 positional arguments but 4 were given` | **活跃阻塞** |
| D3（漂移口径）| `cascade_graph_plugin.py:881` | `attention_v1.py:592` 已删 `num_dcp_pcp_tokens`（宿主内部 `acl_graph.py:295` 传 6 个，自洽） | `TypeError: takes from 4 to 6 positional arguments but 7 were given` | **活跃阻塞** |
| D4（漂移口径）| `cascade_runner_patch.py:274` | 上游 `gpu_model_runner.py:4550` 全关键字调用 | `TypeError: missing 1 required positional argument: 'num_tokens_padded'` | **潜伏**（见下） |

**D4（漂移口径）的性质：潜伏风险（非活跃阻塞）**。当前后端 `NPUModelRunner` 自带 `execute_model`（`model_runner_v1.py:2093`）
与 `_model_forward`（:2957），调用点 `:2451`/`:3854` 均为**位置传参且首参正是 `num_tokens_padded`**，与插件原签名匹配；
上游那个全关键字调用点（`gpu_model_runner.py:4550`，位于被覆盖的 `GPUModelRunner.execute_model`）不在本后端活跃路径上，
故本次基线不触发，但一旦上游路径被启用即崩。修复一并覆盖两种调用约定。
**判据纪律（教训）**：定性必须落到"实际活跃路径"，不能只照"上游源码长什么样"下结论。

**D4（漂移口径）由主 agent 在审核中补充发现**（两个证据 subagent 均未报出）。实证脚本：`logs/sig-audit/d4_repro.py`。

**共性**：D1–D3（漂移口径）都是"import 成功、签名不匹配"的**静默失配**——正是 `docs/pitfalls.md` §2.1 与 HANDOFF「最危险：静默失配」所指。且它们**违反 fail-open 契约**：文档承诺失败即回落 fork/标准路径，实际是硬崩引擎。

**根因（流程）**：锚点表为**参数级**（符号存在性不足以定位；见 `HOST_CONTRACT.md`）。

## 5. 处置与后续

1. **cascade 维持 `import_only`**（manifest 未改；`extension check` 保持 `compatible+degraded`），**不翻 active**——理由见 §0；现状见本文件 `:3` 与 `section4-active-enablement.md`。
2. **D1–D4（漂移口径）已修复**：插件 commit `ddc0120`（`fix(cascade): make graph-mode host wrappers signature-agnostic and fail-open`）。要点：
   - 所有包装器改 `*args, **kwargs` 并原样转发 `orig(*args, **kwargs)`；按名/按位解析宿主参数（`_pick_arg`）；
   - cascade 专属逻辑全部收进 try/except → 单条 warning + 委托原实现（兑现 fail-open 契约）；
   - 新增 `tests/test_host_signature_drift.py`（13 例：活体 `inspect.signature` 比对 + 宿主调用点关键字集合 AST 断言 + 4 例 fail-open + 2 例源码守卫）。
   - **审核独立复核**：`ruff check .` 干净、`pytest -q` **102 passed**（自跑）；变异测试 4 个缺陷各自复原时对应测试**各失败一次**（`logs/fix-sigdrift/mutation-checks.log`）；未 shim 图模式启动 `logs/fix-sigdrift/serve-on-card6.log`（`gate=1, graph_gate=1` ×2、startup complete、capture ×96、cascade replay ×63、服务窗口 0 TypeError）；default-off `serve-off-card6.log`（`gate=0, graph_gate=0` ×2、`cascade-active` 0）。
3. **本快照时点的两个待关闭缺口**（现状见 `:3`）：
   - **缺口 A（正确性口径）**：强制续写形态 21–22/64 > 历史 ≤8/64。需取历史真模型重跑，或与团队就 stand-in 形态的判定口径达成一致（字面形态 1/64 已通过）。
   - **缺口 B（gate 覆盖）**：`(64,4096)` 实测 +6.7% 亏损而 gate 判 `on`、不回退（§3.2）；关闭方式 = 插件 `c4121e2` 分档
     bench margin（P≤4096 桶需 ≥25%）——`logs/v1-ev3-gatefix/` 4 腿重验亏格收回、赢格保留，详见 `section3-performance.md` §13。
4. 修复后 agent 的快速 sanity（offline 图模式 B=8/G=64 → ON-vs-OFF 0/8 发散）**不替代**上述证据。
5. 其他待办：`lse_merge` 在 CANN 9.1 上复现性 +8–10%（§3.4），需确认是否值得追；`fa_fp32_stage1` 波动 ±4% 需更多重复才能收窄。

## 6. 换环境暴露的缺口（本基线特有）

| # | 现象 | 根因 | 修复 |
| --- | --- | --- | --- |
| 1 | `OSError: libatb.so: cannot open shared object file` → 引擎初始化失败 | 迁移 CANN 9.1 时只 source `ascend-toolkit/set_env.sh`，漏了 ATB | `activate.d/zz-cann910.sh` 补 `source .../nnal/atb/set_env.sh --cxx_abi=1`（已修，记入工作区 AGENTS.md） |
| 2 | `Free memory ... 9.03/60.96 GiB` 启动失败 | 卡 0–5 被他人占用（HBM 85%）；`~/.bashrc` 默认绑卡 0 | 改用空闲卡 6/7 + `--gpu-memory-utilization 0.85`（已记入 AGENTS.md） |
| 3 | 证据③各腿统一加 `VLLM_DISABLE_COMPILE_CACHE=1` | 共享 torch.compile AOT 缓存命中会硬崩新基线（与 cascade 无关） | 对称地用于 OFF/ON 两腿，并在报告中声明 |

## 7. 复现命令

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate hust
# 证据①
bash /vllm-workspace/knowledge/evidence/cascade/run_evidence1_default_off.sh
# 证据②（详见 section2-correctness.md 的命令清单）
cd /tmp && ASCEND_RT_VISIBLE_DEVICES=6 python /vllm-workspace/knowledge/evidence/cascade/ev2_run.py ...
# 证据③
cd /tmp && ASCEND_RT_VISIBLE_DEVICES=7 python /vllm-workspace/knowledge/evidence/cascade/logs/v1-ev3/cascade_sweep_plug.py.frozen ...
# 参数级签名审计（本次审核用）
python3 /tmp/sig_audit2.py
```

## 8. 旧基线留档（0.23.0rc1 / CANN 9.0.1，2026-09-10 已退役；**不继承**）

> 本文件主体（§0–§7）是 v1 基线（新基线）的证据。本节仅归档旧基线（0.23.0rc1）**不可再生的标识事实**与
> 当时的记录数字，原始产物在 `archive/`、`cascade-c3-results/`（均为当时目录布局）。
> **结论一律以 §0–§7 与 `section4-active-enablement.md` 为准。**

### 8.1 环境与提交（不可再生）

- 运行环境：Python 3.12.13 / torch 2.10.0+cpu / torch_npu 2.10.0.post2 / CANN 9.0.1 / 910B2 /
  kernel wheel `ascend-kernel==2026.3.9`（`ascend_kernel-2026.3.9-cp312-cp312-linux_aarch64.whl`）。
- 宿主：`vllm==0.23.0+empty`（`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`）、
  `vllm-ascend==0.23.0rc1`（`f4a08bddd`，tag `v0.23.0rc1`）；旧参考源码树 `vllm/`、`vllm-ascend/`
  已于 2026-09-10 删除（上游 tag 可重 clone）。
- 插件 repo：`vllm-ascend-split-batch-hust` 分支 `feat/cascade-attention-plug`，起点 `224efa2`；
  提交 `02650ea`（fail-open 守卫+单测）、`0af8087`（docs 同步）、`4cec120`（default-off shim 收窄）、
  `08d58c1`（F1/F2 manifest）、`8751839`（F4/F5 翻 active + test_manifest）、`b1e7e61`（kernels extra）。
  wheel 软依赖 + fail-open 守卫（`02650ea`）代码锚点：`cascade_plugin.py:66-94,197-210,457-476,646-652`、
  `cascade_runner_patch.py:66-79,190`、`cascade_gate_self.py:39-50`；单测 `tests/test_cascade_fail_open.py`（11 例）。
- 状态沿革（关键事实）：该基线 2026-09-09 manifest 翻 `active`；`9396b21` 退回 `import_only`；
  2026-09-11 在 v1 基线以点 pin 再翻 `active`（`c968c73`，见 `section4-active-enablement.md`）。
  `dual-pad-planner` 始终 `import_only`（无验收证据）。
- review findings F1–F7 已在该基线全部落实（`docs/release.md` §0/§0.1、`docs/architecture.md` §3）。

### 8.2 三项证据（旧基线口径，数字不继承）

| release.md §0 三项证据 | 结论 | 关键数字 / 溯源 |
|---|---|---|
| ① default-off 零回归冒烟 | 齐备 | serve 级冒烟（卡 6，`logs/serve-default-off-card6.log`）+ 门禁 `pytest -q` 38 passed、`ruff` 绿（`logs/pytest-ruff-after-change.log`）；F7 启动标记 `cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=ok)`（`logs/serve-active-default-off-card6.log`） |
| ② 正确性对齐 | 齐备 | graph 主验收 14B B=64 on vs off **6/64**（≤ C 口径 8/64）、on_run2 0/64；B=32 1/32；eager Tier-1 fp32 5/64、Tier-0 bf16 8/64；算子精度 30/30+30/30、负例 3/3 拒；S1 bit 锚点逐 bit；LSE 摊平 19/19；lse_merge 56/56。溯源 `cascade-c3-results/RESULTS.md:11-24,26-34,104-140` |
| ③ TPOT/latency | 齐备 | W5 e2e 矩阵 8k 段 −6.6%~−28%、16k 段 −21%~−38%（`archive/cascade-e2e/results/sweep_*.json`）；亏格：4.4k×B32 +6.9%、C3 形态 4k shared×B64×suffix260 净亏 ~31%（off 6.49s vs on 8.51s，`ops/kernels/ascend-kernel/README.md:140,189`）；graph 主验收 on 10.95s vs off 11.52s（−5.0%）；单算子 fa_fp32_stage1 @4356 133.9–135.6µs、lse_merge @B64 25.4–33.6µs（`ops/kernels/ascend-kernel/README.md:179-189`） |

> 旧基线 16k 段 ON 腿只有 w5d 单轮有效（w5c 轮 16k 崩于 CUDA graph capture，非插件缺陷）；
> 以上数字属旧基线测量记录，**不作"本机现状"引用**。

### 8.3 旧基线零散 wall / 精度数字（只此一份出处，故留档）

| 数字 | 含义 | 溯源 |
|---|---|---|
| on **10.99 s** | graph 主验收 ON/ON 复现轮 | `cascade-c3-results/RESULTS.md:11-20` |
| **11.51 s** | 源机 M1fix 基线 off（±1% 噪声带） | 旧 `EVIDENCE.md` §2 记录 |
| **8.94 s** | M3 C3 形态 off | 旧 `EVIDENCE.md` §2 记录 |
| **9.38 s** | C3 形态 always-rebind | 旧 `EVIDENCE.md` §3 记录 |
| **10.53 s** | `sweep_on_fill16k_b32.log` workaround 单跑（非标准 sweep） | 旧 `EVIDENCE.md` §3 记录 |
| 抑制因子实测 **0.069** ≈ 理论 **0.059** | per-request 与摊平两形态 merged 输出逐 bit 相等 | `cascade-c3-results/RESULTS.md:132-140`（§6.5） |
| **7.574e-3** | 全链（stage1+FIA+merge vs fp64）精度口径 | `ops/kernels/ascend-kernel/README.md:185` |
