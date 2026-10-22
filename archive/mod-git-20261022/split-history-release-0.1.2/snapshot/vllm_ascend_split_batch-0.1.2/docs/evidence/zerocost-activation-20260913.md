# 证据：zerocost-wiring §3 启用验证（2026-09-13）——**阻塞，未翻 active**

- 仓库：`vllm-ascend-split-batch-hust`，改动前工作树干净 @ **`e308e78`**
  （`fix(zerocost): 补齐交付面——extension_bundles 入口 + 子 manifest（收口缺口）`）。
- Bundle：`org.vllm-hust.zerocost-wiring`（`import_only`，本次**未**改 manifest）。
- 开关：`VLLM_HUST_FI_PREFILL_OUT=1`（①）+ `VLLM_HUST_SKIP_COS_SIN=1`（②）（两者都必须精确 `"1"`）。
- 设备：卡 7（起服前 / OFF 腿后 / ON 腿后 / isolation 腿后 `HBM Usage Rate = 5%`）；
  触卡命令均包在 `flock -x -w 7200 /tmp/npu-card7.lock` 内。
- 环境：conda `hust` / Python 3.12.14 / torch 2.13.0+cpu / torch_npu 2.13.0rc1 / CANN 9.1.0（`/usr/local/ascend91`）/
  910B2；vllm `0.28.1.post1.dev143+gf18cf803c.empty` /
  vllm-ascend `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`（与 manifest 点钉一致）。
- 原始日志：`docs/evidence/zerocost-activation-20260913/raw/`（ext 前后输出、两腿 + isolation 腿 serve 日志、
  chat JSON、`smoke_stage.log`、`evidence_greps.txt`）。

## 0. 结论

| §3 步骤 | 结果 |
| --- | --- |
| baseline `inspect`/`check`/`status`（`import_only`） | `activation_ready=false`、blocker=`import_only`；states 含 `degraded`（**符合纪律**） |
| baseline `enable` | **拒绝**（`rc=2`，`descriptor-only / import_only`） |
| 临时翻转 `import_only → active` 后 `inspect`/`check`/`status` | `activation_ready=true`、blocker=`null`；states 由 `degraded` → `configured` |
| `enable`（翻转后） | **成功**（`rc=0`） |
| `run --dry-run` 注入预览 | 正确注入 `VLLM_HUST_FI_PREFILL_OUT=1` + `VLLM_HUST_SKIP_COS_SIN=1`（外加共存的 cascade/demask 键） |
| default-off 腿（plain `vllm serve`，无 zerocost env） | `/health` 200；2 条 chat 200；**日志零 zerocost 痕迹** |
| **ON 腿 = §3 正式 serve**（`vllm-hust-ext run -- vllm serve`，cascade graph 共存） | `/health` 200；2 条 chat 200；**② ACTIVE，但 ① refused（fail-open）** |
| isolation 腿（plain `vllm serve` + 两 env，cascade 关） | `/health` 200；chat 200；**① ACTIVE + ② ACTIVE** |
| **最终判定** | **未翻 active**：§3 正式 serve 路径下 headline 能力 ① 静默 fail-open（见 §6） |
| manifest | **未改动**（`implementation[0].status` 保持 `import_only`；`host.version_range` / `activation.environment` 未动） |
| 收尾 | 三腿 serve 子树均 `our residual children: <none>`；卡 7 HBM 回 5% |

> 一句话：`check`/`status`/`inspect`/`enable`/`dry-run` 全绿，default-off 零回归，② 在所有腿都 ACTIVE；
> 但 **① 在唯一的 §3 正式 launch 路径（`vllm-hust-ext run`，必然 co-enable cascade graph）下被宿主锚点解析拒绝**，
> 故按 release.md §0 与任务规则**不翻 active**。

## 1. baseline（`import_only`，翻转前）

```
$ vllm-hust-ext extension list
org.vllm-hust.fia-demask 0.1.0.dev0 enabled
org.vllm-hust.rope-fix 0.1.0.dev0 disabled
org.vllm-hust.split-batch-full-graph 0.1.0.dev0 enabled
org.vllm-hust.zerocost-wiring 0.1.0.dev0 disabled
```

- `inspect`（`raw/ext_inspect_before.txt`）：`activation_ready=false`、
  `activation_blocker="extension is descriptor-only and cannot be enabled (implementation status: import_only)"`、
  `activation.environment=[["VLLM_HUST_FI_PREFILL_OUT","1"],["VLLM_HUST_SKIP_COS_SIN","1"]]`、
  `host.version_range == ==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`（满足）。
- `check`（`raw/ext_check_before.txt`）：`states=[installed,discovered,compatible,degraded]`，evidence 两条
  （host 区间满足 + `descriptor-only ... import_only`）。
- `enable`（`raw/ext_enable_before.txt`）：`rc=2`，`cannot enable ... (import_only)`。
- 解读：`disabled`/`degraded` 是**期望值**——未备齐 §3 前 ext-manager 拒绝标可 enable（与 rope-fix P4 同类）。

## 2. 临时翻转后的阶梯（`active`）

为取 §3 证据，先把 `implementation[0].status` 临时改为 `active`（末尾已回滚）：

```
$ vllm-hust-ext extension inspect org.vllm-hust.zerocost-wiring   # raw/ext_inspect_after.txt
  "activation_blocker": null, "activation_ready": true, "status": "active"
$ vllm-hust-ext extension check  ...                              # raw/ext_check_after.txt
  states = [ installed, discovered, compatible, configured ]      # degraded 消失
$ vllm-hust-ext extension enable ...                              # raw/ext_enable_after.txt
  enabled org.vllm-hust.zerocost-wiring   (rc=0)
$ vllm-hust-ext extension status ...                              # raw/ext_status_after.txt
  states = [ installed, discovered, compatible, configured, enabled ]
```

`run --dry-run`（`raw/ext_dryrun_after.txt`，与 ON 腿命令逐字一致）：

```json
{"command":["vllm","serve","/data/shared_models/Qwen--Qwen2.5-14B-Instruct",
  "--gpu-memory-utilization","0.85","--max-model-len","4096","--served-model-name","qwen14b",
  "--port","8353","--compilation-config","{\"cudagraph_capture_sizes\":[32,64,128]}"],
 "environment":{"VLLM_HUST_FIA_DEMASK":"1","VLLM_ASCEND_ENABLE_CASCADE_DECODE":"1",
   "VLLM_ASCEND_ENABLE_CASCADE_GRAPH":"1","VLLM_HUST_FI_PREFILL_OUT":"1",
   "VLLM_HUST_SKIP_COS_SIN":"1",
   "VLLMHUST_EXT_ENABLED_BUNDLES":"org.vllm-hust.fia-demask,org.vllm-hust.split-batch-full-graph,org.vllm-hust.zerocost-wiring"}}
```

⇒ 两个 zerocost 键**正确注入**；`VLLMHUST_EXT_ENABLED_BUNDLES` 为 manager 自有标记。

## 3. default-off 零回归腿（端口 8352，无任何 zerocost env）

- 命令：plain `vllm serve`（**不经** `vllm-hust-ext run`，故 cascade/demask 也都关）+
  `--compilation-config '{"cudagraph_mode":"FULL"}'` + `--gpu-memory-utilization 0.85` + 卡 7。
- `/health` 200（41 次轮询内）；2 条 chat 均 200（`raw/chat_off_{1,2}.json`，内容 `ok1` / `Ok2 …`）。
- **零回归证据**：`grep -nE "zero-cost wiring|zerocost|VLLM_HUST_FI_PREFILL_OUT|VLLM_HUST_SKIP_COS_SIN" raw/serve_off.log`
  → `<none>`（`raw/off_activation.txt`）——载体未被激活、未 import、零日志。
- ⚠️ 日志里**确有一条 Traceback**（`raw/serve_off.log:150-157`），但它是**我们主动 kill 引发的关停竞态**
  （时序 `07:27:55`：两条 chat `200 OK` → `stop` 送 SIGTERM → `EngineCore: trigger received signal=SIGTERM` →
  之后 `AsyncLLM output_handler failed … EngineDeadError`）。与 rope-fix P4 §3 **逐字同型**，**不是** zerocost 缺陷。

## 4. ON 腿 = §3 正式 serve（端口 8353，`vllm-hust-ext run -- vllm serve`）

- 命令：`vllm-hust-ext run -- vllm serve … --compilation-config '{"cudagraph_capture_sizes":[32,64,128]}'`
  （capture-size 封顶 = pitfalls §3.5；本路径必然 co-enable cascade graph）。
- `/health` 200（45 次轮询内）；2 条 chat 均 200（`raw/chat_on_{1,2}.json`）。

引擎日志（`raw/on_activation.txt`）：

```
14: cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)          # API server
59: (EngineCore) cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)
60: (EngineCore) FIA decode de-mask is ACTIVE (VLLM_HUST_FIA_DEMASK=1): ...
61: (EngineCore) zero-cost wiring ① refused (VLLM_HUST_FI_PREFILL_OUT=1): a host anchor moved;
                  the process stays on the stock FIA call + copy path
73: (EngineCore) zero-cost wiring ② is ACTIVE (VLLM_HUST_SKIP_COS_SIN=1): update_cos_sin now
                  records positions only and get_cos_and_sin_slice() materialises the rope buffers on demand.
```

⇒ ② 生效；**① 未生效（refused / fail-open）**。① 的完整异常（`raw/serve_on.log:62-71`，逐字）：

```
RuntimeError: install.<locals>.forward_fused_infer_attention is not defined in
/vllm-workspace/vllm-ascend-hust/vllm_ascend/attention/attention_v1.py
```

（fail-open 契约如实兑现：一条 warning + 整项回滚，进程继续走宿主原生 FIA + 拷贝路径，② 不受影响。）

## 5. isolation 腿（端口 8354，plain `vllm serve` + 两个 zerocost env，cascade 关）

用于把自变量定到"是否 cascade graph 共存"：

```
9:  INFO cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=ok)
40: (EngineCore) zero-cost wiring ① is ACTIVE (VLLM_HUST_FI_PREFILL_OUT=1): ...
41: (EngineCore) zero-cost wiring ② is ACTIVE (VLLM_HUST_SKIP_COS_SIN=1): ...
```

⇒ cascade 关时 **①、② 均 ACTIVE**（复现 Y1 的验证条件）。VLM 组合同 §3/§4 参数（`{"cudagraph_mode":"FULL"}`）。

## 6. 根因（host-anchor 解析穿不过双层插件 wrapper）

`zerocost_wiring._host_func()` 的设计前提（Y1 §1.2）是"沿 `__closure__` 链解析出**宿主文件里的那个函数**"，
但它**只接受 `co_filename == 宿主文件` 的 cell**：

```python
if callable(candidate) and getattr(candidate.__code__, "co_filename", None) == host_file:
    inner = candidate; break
if inner is None: break          # 一旦闭包里放的是"另一个插件 wrapper"，链就断
```

链长：
- `VLLM_ASCEND_ENABLE_CASCADE_DECODE` 关时，`cascade_plugin.load()` **仍无条件**把
  `AscendAttentionBackendImpl.forward_fused_infer_attention` 换成 `_make_forward_wrapper.<locals>.forward_fused_infer_attention`
  （pass-through，闭包直接含宿主原函数）⇒ `_host_func` **能**解析 → ① ACTIVE（isolation 腿）。
- `VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1` 时，`cascade_plugin.load()` 追加调用
  `cascade_graph_plugin.install()`（`cascade_plugin.py:704-706`），后者在
  `cascade_graph_plugin.py:889` 定义 `install.<locals>.forward_fused_infer_attention` 并在 `:953`
  覆盖到类上，其闭包里的 `orig_forward` 是**上一层插件 wrapper**（`co_filename=cascade_plugin.py`），
  **不是**宿主文件里的函数 ⇒ `_host_func` 在第一次迭代就 `break` → `RuntimeError` → ① 整项回滚。

即：① 的解析只支持**一层**插件 wrapper；`graph_gate=1` 的 cascade 会叠成**两层**，把 ① 打回原生路径。
`vllm-hust-ext run`（release.md §3 的正式 launch）在本工作区**必然**注入
`VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1`（split-batch-full-graph 已 enable），所以这是**默认路径下的必现**问题，非偶发。

Y1 报告里"真机已验证：cascade wrapper 在场时 ① 仍 ACTIVE（`raw/on/serve2.log:41`）"的结论**成立但有条件**：
该次 `serve2.log` 实测 `cascade plugin loaded (gate=0, graph_gate=0, …)`——只有 decode gate（无 graph），即只有一层 wrapper。
registry O11 / MAIN.plan.md 的相关表述因此**需要收敛为"仅当 cascade graph 关闭时成立"**。

## 7. 收尾自检

- 三腿 `stop` 后均打印 `serve[<tag>] stopped; our residual children: <none>`（按 **setsid 进程组**收尾，未触他人进程）。
- `HBM Usage Rate`：起 5% → OFF 腿后 5% → ON 腿后 5% → isolation 腿后 5%。
- 翻转期间 `enable` 写入了 manager 的 enabled 集合；收尾已 `disable` 回退
  （`raw/ext_disable_restore.txt` / `ext_list_restored.txt`：`zerocost-wiring … disabled`，与任务前一致；
  fia-demask / split-batch-full-graph 未动）。

## 8. 判定

**release.md §3 启用验证未通过**：`check`/`status`/`inspect`/`dry-run` 与 default-off 腿均绿，但 §3 正式
`vllm-hust-ext run -- vllm serve` 路径下 headline 能力 **① 静默 fail-open**（仅 ② 生效）。按 release.md §0
（"三项证据齐备" 之外，§3 要求"门控生效的证明"）与任务规则（"若任一环节不通过：不要翻 active"），
**manifest 保持 `import_only`**。三项验收证据（Y1：default-off 零差异 / 逐位正确 / prefill TTFT −1.09%）
本身不受影响，但它们在 **cascade graph 关闭**的配置下取得，不能覆盖本路径。

## 9. 复现命令

```bash
E=docs/evidence/zerocost-activation-20260913
# 两腿（OFF + §3 正式 serve/ON）；内部 flock -x -w 7200 /tmp/npu-card7.lock
nohup bash $E/raw/run_smoke.sh > /tmp/ws3-zerocost-smoke.log 2>&1 &
# isolation 腿（cascade 关，① 应 ACTIVE）
nohup flock -x -w 7200 /tmp/npu-card7.lock bash $E/raw/zerocost-iso.sh > /tmp/ws3-zc-iso.log 2>&1 &
```

## 10. 未决 / 建议

| # | 项 | 内容 |
| --- | --- | --- |
| B1 | **① 解析穿双层 wrapper** | `_host_func` 需要在"闭包 cell 指向另一个插件 wrapper"时继续下钻（先找 host-file 命中，否则跟随第一个函数型 cell 再迭代），才能兼容 `graph_gate=1` 的 cascade 共存；改动后需补"双层合成 wrapper"单测 + 真机 ON 腿复验。 |
| B2 | 文档纠偏 | registry O11 / MAIN.plan.md 里"cascade wrapper 在场时 ① 仍 ACTIVE"需限定为"仅 decode gate；graph gate 下会 fail-open"。 |
| B3 | manager 状态 | 翻转期间 `enable` 已把 bundle 写入 enabled 集合；**收尾已 `disable` 回退到任务前状态**（`raw/ext_disable_restore.txt`）。 |
| B4 | 翻转动作 | 一旦 B1 修复且 §3 ON 腿出现 ①+② ACTIVE，翻 active 就是 `implementation[0].status: import_only → active` 一行 + 还原
`test_zerocost_bundle_is_*` 守卫（本包已把该守卫的 docstring 更新为如实记录 blocker）。 |

---

# 追加（同日第二次运行）：**修复后 ON 腿通过，已翻 `active`**

> 本节是**最新结论，覆盖 §8 的判定**。§0–§10 的失败记录**原样保留留档**（它准确定位了
> 根因，也是本次修复的回归基线）。

## 11. 修复：`_host_func` 跨任意层 wrapper 溯源

`zerocost_wiring._host_func(func, host_file)` 的判据从「**只跳一层**：闭包 cell 的
`co_filename == 宿主文件`」改为「**遍历整个 `__closure__` 图，要求恰好一个可达函数其 code 属于宿主文件**」：

- 先经 `_unwrap_callable()` 归一化 bound method（`__func__`）/ `functools.partial`（`func`）；
- 以 `seen`（按 `id`）+ 显式栈遍历闭包图，收集所有 `co_filename == host_file` 的候选；
- 候选按同一性去重后**必须恰好为 1**：0 个（锚点漂移 / 断链）或 ≥2 个（歧义）一律 `raise`
  ⇒ 沿用既有 fail-open 契约（`install()` 内一条 warning + 整项回滚），**不静默跑偏**；
- 仍是 `host_func.__code__ = new_code` **原地替换**（未 `setattr` 到类上，理由见
  `knowledge/operator-advantage-registry.md` 判读规则 12）；② 路径与 cascade / cascade_graph 插件未动。

单测（`tests/test_zerocost_wiring.py`，新增 6 项）：两层 / 三层叠加 wrapper 仍能溯源；
双层叠加下端到端确认 `_zc_fi_out` 到达 adaptor 且恒等拷贝被跳过；同文件双候选（歧义）与断链（0 命中）
都拒。**回归有效性已对照验证**：同一合成用例下旧的一层逻辑在两层 wrapper 处即 `RuntimeError`。

CPU：`pytest -q` **277 passed**（新增 6）、`ruff check .` 干净。

## 12. §3 重跑（真模型 `/data/shared_models/Qwen--Qwen2.5-14B-Instruct`，卡 7）

- Bundle：`org.vllm-hust.zerocost-wiring`，本次**已翻 `active`**（`implementation[0].status`）。
- 原始日志：`docs/evidence/zerocost-activation-20260913/raw-fix/`（阶梯输出、两腿 serve 日志、
  chat JSON、`evidence_greps.txt`、`meta.txt`、`fix-smoke.sh`）。
- 设备：卡 7；触卡命令包在 `flock -x -w 7200 /tmp/npu-card7.lock` 内；
  HBM 起 / OFF 腿后 / ON 腿后 = **5% / 5% / 5%**。

| §3 步骤 | 结果 |
| --- | --- |
| `inspect`（翻转后） | `activation_ready=true`、`activation_blocker=null`、`status=active` |
| `check` | `states=[installed,discovered,compatible,configured]`（`degraded` 消失） |
| `status` | 同上 |
| `enable` | **成功**（`rc=0`）；enabled 集合含 zerocost |
| `run --dry-run` | 正确注入 `VLLM_HUST_FI_PREFILL_OUT=1` + `VLLM_HUST_SKIP_COS_SIN=1`（+ 共存 cascade/demask 键） |
| default-off 腿（plain `vllm serve`，无 zerocost env） | `/health` 200（41 poll）；2 chat 200（`ok1` / `Ok2 …`）；**日志零 zerocost 痕迹**（`evidence_greps.txt` 为空） |
| **ON 腿 = §3 正式 serve**（`vllm-hust-ext run -- vllm serve`，co-enable cascade graph） | `/health` 200（37 poll）；2 chat 200（`ok1` / `Ok2 …`）；**① ACTIVE + ② ACTIVE，0 fail-open(refused)** |
| 收尾 | 两腿子树 `residual children: <none>`；manager `disable` 回退（`raw-fix/ext_list_restored.txt`）；卡 7 HBM 回 5% |

ON 腿引擎日志（`raw-fix/serve_on.log`，逐字）：

```
14: cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)          # API server
47: (EngineCore) cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)
48: (EngineCore) FIA decode de-mask is ACTIVE (VLLM_HUST_FIA_DEMASK=1): ...
49: (EngineCore) zero-cost wiring ① is ACTIVE (VLLM_HUST_FI_PREFILL_OUT=1): the eager prefill
                  FIA leg now writes into the caller's output buffer through the op's .out
                  overload and the two redundant TensorMove copies per layer are guarded by an
                  exact identity predicate.
50: (EngineCore) zero-cost wiring ② is ACTIVE (VLLM_HUST_SKIP_COS_SIN=1): update_cos_sin now
                  records positions only and get_cos_and_sin_slice() materialises the rope
                  buffers on demand.
```

⇒ **上一程的 `zero-cost wiring ① refused … a host anchor moved` 已消失**；`grep "refused"` 为空。

唯一 ERROR / Traceback（`raw-fix/serve_on.log:165-172`）是**主动 kill 引发的关停竞态**——
时序 `08:04:25`：两条 chat `200 OK` → `stop` 送 SIGTERM → `EngineCore: trigger received signal=SIGTERM`
→ 之后 `AsyncLLM output_handler failed … EngineDeadError`。与 default-off 腿历史同型，**非** zerocost 缺陷。

## 13. 判定

**release.md §3 启用验证通过**（`check`/`status`/`inspect`/`enable`/`dry-run` 全绿；default-off 零回归；
§3 正式 serve 下 ① 与 ② **都 ACTIVE**、0 fail-open、`/health` 200、chat 200）。
⇒ 三项验收证据齐备 + §3 门控生效 ⇒ **翻 `implementation[0].status: import_only → active`**，
`test_zerocost_bundle_is_import_only_with_its_own_keys` 守卫同步改为
`test_zerocost_bundle_is_active_with_its_own_keys`（断言 `active` + `activation_blocker is None`）。

## 14. 复现命令

```bash
E=docs/evidence/zerocost-activation-20260913
# 阶梯 + default-off 腿 + §3 正式 ON 腿（内部 flock -x -w 7200 /tmp/npu-card7.lock）
nohup bash -c "exec 200>>/tmp/npu-card7.lock; flock -x -w 7200 200 && \
  bash $E/raw-fix/fix-smoke.sh" > /tmp/ws3b-fix-smoke.log 2>&1 &
```
