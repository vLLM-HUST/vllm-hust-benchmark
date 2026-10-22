# §3 启用验证：cascade 翻 `active`（release.md §3，2026-09-11）

- 仓库/分支：`vllm-ascend-split-batch-hust` @ `feat/cascade-attention-plug`，改动前被测 commit **`783794a`**（工作树干净）
- Bundle：`org.vllm-hust.split-batch-full-graph`（vllm-hust-ext 0.2-experimental）
- 翻转：`implementation[]` 的**两个 cascade carrier** `cascade_plugin:load` 与
  `cascade_graph_plugin:install` 由 `import_only` → `active`；`planner:plan_dual_pad`
  保持 `import_only`（无验收证据，review F7）。其余 manifest 字段（host pin、
  activation.environment、protocols、extension_version）未动。
- 真模型：`/data/shared_models/Qwen--Qwen2.5-14B-Instruct`（8 safetensors / 27.51 GiB）
- 环境：conda `hust` / Python 3.12 / torch 2.13.0+cpu / torch_npu 2.13.0rc1 / CANN 9.1.0 /
  triton-ascend 3.2.2；vllm-hust `0.28.1.post1.dev143+gf18cf803c`（`.empty`）/
  vllm-ascend `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27` / kernel wheel `ascend-kernel==2026.3.9`
- 设备：卡 7（910B2，起服前/收尾后 HBM 5%）；全程 `flock -w 7200 /tmp/w3-npu.lock`
- 原始日志：`logs/v1-active-enablement/serve-active-enablement.log`（1438 行，逐字节复制）

## 0. 结论

| §3 步骤 | 结果 |
| --- | --- |
| `extension check` | **compatible + configured**（无 `incompatible`/`degraded`） |
| `extension status`（enable 后） | compatible + configured + **enabled** |
| `run --dry-run` 注入预览 | 环境变量注入 2 个 cascade 门控 flag，命令未改动 |
| 正式 `run -- vllm serve` | 启动成功；startup snapshot + cascade 门控证明见 §3 |
| `/health` | **200** |
| 功能冒烟（1 条 chat） | **200**，返回 `hello cascade` |
| cascade engagement | **capture 级成立**：`capture body SUCCESS` ×144（3 bucket × 48 层）、0 失败、0 TypeError |
| 收尾 | serve 进程全部退出、端口 8341 释放（curl 000）、卡 7 HBM 回 5%、flock 及时释放 |

## 1. manifest diff（摘要）

```diff
@@ implementation[] (cascade_plugin) @@
-      "status": "import_only"
+      "status": "active"
@@ implementation[] (cascade_graph_plugin) @@
-      "status": "import_only"
+      "status": "active"
```

- `implementation[].status`：`cascade_plugin` / `cascade_graph_plugin` → `active`；`planner` 不变。
- `host.version_range` = `==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`（**未改**；点 `==` 钉住实测宿主，
  packaging 有序比较符拒 local label，理由见 release.md §0）。实测 `vllm-ascend` 版本号与之一致。
- `activation.environment` = `{VLLM_ASCEND_ENABLE_CASCADE_DECODE: "1", VLLM_ASCEND_ENABLE_CASCADE_GRAPH: "1"}`
  （**未改**，复核为可注入 flag；enable 时注入，不 enable 则不注入）。
- 守护测试同步（`tests/test_manifest.py`）：`test_descriptor_is_discoverable_and_activatable`
  （`activation_blocker(manifest) is None`）+ `test_only_cascade_carriers_are_active`
  （两 cascade carrier `active`、planner `import_only`），替换原 “全 import_only” 守卫。

## 2. `extension check` / `status`（原文）

```
$ vllm-hust-ext extension check org.vllm-hust.split-batch-full-graph
{
  "extension_id": "org.vllm-hust.split-batch-full-graph",
  "provider": "vllm",
  "states": [ "installed", "discovered", "compatible", "configured" ],
  "evidence": [
    "host version 0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27 satisfies the declared range",
    "protocol vllm.graph.runtime-key is not independently versioned; compatibility is governed by host ranges and acceptance evidence",
    "protocol vllm.forward.split-context is not independently versioned; compatibility is governed by host ranges and acceptance evidence",
    "protocol vllm.ascend.graph-pool is not independently versioned; compatibility is governed by host ranges and acceptance evidence",
    "protocol vllm.worker.split-executor is not independently versioned; compatibility is governed by host ranges and acceptance evidence",
    "vLLM launch configuration can be rendered"
  ]
}
```

```
$ vllm-hust-ext extension enable org.vllm-hust.split-batch-full-graph
enabled org.vllm-hust.split-batch-full-graph
$ vllm-hust-ext extension status org.vllm-hust.split-batch-full-graph
  "states": [ "installed", "discovered", "compatible", "configured", "enabled" ]
```

（翻转前 `check` 为 compatible+degraded（`activation_blocker` 非空）；`active` carrier 使
`activation_blocker` 归一为 `None`，故 `enable`/`run` 放行。）

## 3. `run --dry-run` 注入预览（与正式启动命令逐字一致的 preview）

```
$ vllm-hust-ext run --dry-run -- vllm serve /data/shared_models/Qwen--Qwen2.5-14B-Instruct \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8341 \
    --compilation-config '{"cudagraph_capture_sizes":[32,64,128]}'
{
  "command": [
    "vllm","serve","/data/shared_models/Qwen--Qwen2.5-14B-Instruct",
    "--max-model-len","4096","--gpu-memory-utilization","0.85","--port","8341",
    "--compilation-config","{\"cudagraph_capture_sizes\":[32,64,128]}"
  ],
  "environment": {
    "VLLM_ASCEND_ENABLE_CASCADE_DECODE": "1",
    "VLLM_ASCEND_ENABLE_CASCADE_GRAPH": "1",
    "VLLMHUST_EXT_ENABLED_BUNDLES": "org.vllm-hust.split-batch-full-graph"
  },
  "native_extension_manifests": {}
}
```

注入正确：2 个 cascade 门控 flag 进入进程环境（`VLLMHUST_EXT_ENABLED_BUNDLES` 为 manager 自有标记），
命令未追加 `--additional-config`（`activation.additional_config` 为空），无 native manifest。

## 4. 正式 launch 与启动日志关键行（原文）

启动命令（`cd /tmp`；命令本身省略 `--enforce-eager` 以走图模式车道）：

```bash
cd /tmp && HF_HUB_OFFLINE=1 VLLM_DISABLE_COMPILE_CACHE=1 ASCEND_RT_VISIBLE_DEVICES=7 \
  VLLM_ASCEND_CASCADE_MIN_PREFIX=4096 VLLM_ASCEND_CASCADE_MIN_REQS=2 VLLM_ASCEND_CASCADE_TRACE=1 \
  nohup flock -w 7200 /tmp/w3-npu.lock \
  vllm-hust-ext run -- vllm serve /data/shared_models/Qwen--Qwen2.5-14B-Instruct \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8341 \
    --compilation-config '{"cudagraph_capture_sizes":[32,64,128]}' \
    > /tmp/cascade-active-serve2.log 2>&1 &
```

**extension startup snapshot / 门控生效证明**（API server 与 EngineCore 各一次，逐字）：

```
INFO 09-11 04:57:01 [cascade_plugin.py:147] cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)
(EngineCore pid=501372) INFO 09-11 04:57:32 [cascade_plugin.py:147] cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)
```

即 `VLLM_ASCEND_ENABLE_CASCADE_DECODE`/`..._GRAPH` 由 manager 注入后，插件内部门控读到
`gate=1`、`graph_gate=1`，且 kernel wheel 探针 `ok`。

**cascade engagement（capture 级，逐字）**：

```
(EngineCore pid=501372) [cas-trace] wrapper: cascade_flag=False capture_window=True replay_swap=False
(EngineCore pid=501372) [cas-trace] capture body SUCCESS: num_tokens=32  layers-so-far=48
(EngineCore pid=501372) [cas-trace] capture body SUCCESS: num_tokens=64  layers-so-far=48
(EngineCore pid=501372) [cas-trace] capture body SUCCESS: num_tokens=128 layers-so-far=48
(APIServer pid=501064) INFO:     Application startup complete.
```

- `capture body SUCCESS` = **144**（`num_tokens` ∈ {32,64,128}，每 bucket 48 层；与 gatefix 腿 288=144×2 同口径）
- `cascade graph capture failed` = **0**；`TypeError` / `Traceback` = **0**
- 三 bucket 图全部按 cascade twin 捕获成功（capture window `active=True`），标准图请求经 `orig` 完成。
- 未出现 step 级 replay：本冒烟为 1 条小 prompt（共享前缀远小于 `MIN_PREFIX=4096`），
  且 4096 上下文下 `(32,4096)`/`(64,4096)` 两格被 W2 gate 判 OFF（与 section3-gatefix 预注册一致）——
  即“无 replay”是 gate 正确回落，不是失效。step 级 replay 数字仍以
  `section3-performance.md` §7.1（cascade key hit 254–381）为准。

## 5. `/health` 与功能冒烟（原文）

```
$ curl -s -m 5 http://127.0.0.1:8341/health   ->  200
$ curl ... /v1/chat/completions -d '{"model":"...Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"Reply with exactly: hello cascade"}],"max_tokens":16,"temperature":0}'
{"choices":[{"message":{"role":"assistant","content":"hello cascade",...},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":35,"completion_tokens":3,"total_tokens":38}}   ->  200
$ echo "cascade-active (eager) count: $(grep -c cascade-active log)"   ->  0   （图模式预期 0）
```

## 6. 校验门槛（本改动）

- `pytest -q`：**190 passed**（含 4 例 manifest 守护：discoverable/activatable、only-cascade-active、
  host 点 pin、activation.environment flag、extension_version 一致）
- `ruff check .`：All checks passed

## 7. 偏差与发现（必须与结论同读）

1. **capture-size 上限是必需的**。首次尝试（工具默认 `cudagraph_capture_sizes`，从 1 递增的完整网格）
   cascade twin 捕获在 `_full_graph_fia_cascade` 的
   `torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace` 处 **NPU OOM**
   （`Tried to allocate 418.00 MiB ... 224.63 MiB free`），插件按契约 fail-open
   （`cascade graph capture failed; capturing the standard graph instead`，日志 143+ 条），
   但 warmup 反复重试导致引擎长时间不就绪。改用与 `section3-performance.md` §2 一致的
   `cudagraph_capture_sizes=[32,64,128]` 后 0 失败。**结论**：cascade 图车道在 14B/`--gpu-memory-utilization 0.85`
   下需约束 capture-size 网格（与已验证 harness 一致）；默认全网格会 OOM → capture 级 fail-open
   （引擎仍可起，但 cascade twin 缺失、无加速）。此为配置面发现，非正确性缺陷。
2. **冒烟命令较任务建议增加 3 个 env + 1 个 flag**：`CASCADE_MIN_PREFIX=4096`、`CASCADE_MIN_REQS=2`、
   `CASCADE_TRACE=1`（均取自 `section3-performance.md` §2 的 ON 腿口径）与 capture-size 上限。
   目的：在 4096 上下文中允许 gate bench / 探针可达并取证。`--max-model-len 4096`、
   `--gpu-memory-utilization 0.85`、`HF_HUB_OFFLINE=1`、`VLLM_DISABLE_COMPILE_CACHE=1` 均按任务约束。
3. **step 级 replay 未在本冒烟触发**：4096 上下文下无法同时满足“共享前缀 ≥ `MIN_PREFIX=4096`”与
   “stage-2 后缀 > 0”；且小 batch 落 `(32,4096)`/`(64,4096)`，gate 判 OFF（设计如此）。若需要在 §3 冒烟中
   看到 replay 行，需 `--max-model-len` > 4096 且凑到 gate 判 on 的 padded bucket（如 ≥8k 前缀桶）。
   本条目如实记录，未追加第二次 served 运行（锁定窗口需让给排队的 profiling 任务）。
4. **锁定协作**：启动瞬间卡 7 的 `flock` 由另一 profiling 任务（`profiles/qwen14b-instruct-hotspot-20260910`
   的 `stage_a.sh`，17 格矩阵）持有，本 serve 在 `flock -w 7200` 队列中等待其 Stage A 于
   04:56:26 结束后立即获得锁并起服；收尾后 8 秒内锁被该任务下一段（`npu_rest.sh`）接走。
   本 serve 实际占用窗口 ≈ 05:00 收尾前的数分钟（起服→health→冒烟即收）。

## 8. 判定

release.md §3 启用验证**通过**：`check`/`status` 无误、`run --dry-run` 注入正确、正式
`vllm-hust-ext run -- vllm serve` 启动成功、门控生效（`gate=1/graph_gate=1`）、cascade twin
capture 144/144 成功且 0 TypeError、`/health` 与功能冒烟均 200、收尾干净。结合三项证据
（`EVIDENCE.md` §1 + `section2-correctness.md` §12 真模型 6/64 + `section3-performance.md` §13）与团队确认，
cascade 翻 `active` 成立。
