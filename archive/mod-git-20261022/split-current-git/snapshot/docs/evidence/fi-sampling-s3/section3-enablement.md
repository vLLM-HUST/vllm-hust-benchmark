# §3 启用验证：fi-sampling（release.md §3，2026-09-11）

- 仓库/分支：`vllm-ascend-split-batch-hust` @ `feat/cascade-attention-plug`，改动前被测 commit
  **`37a014a`**（工作树干净）。
- Bundle：`org.vllm-hust.split-batch-full-graph`（vllm-hust-ext `0.2-experimental`；cascade 两 carrier
  已 `active`，planner `import_only`）。
- 真模型：`/data/shared_models/Qwen--Qwen2.5-14B-Instruct`（8 safetensors）。
- 环境：conda `hust` / Python 3.12.14 / torch 2.13.0+cpu / torch_npu 2.13.0rc1 / CANN 9.1.0 /
  triton-ascend 3.2.2；vllm-hust `0.28.1.post1.dev143+gf18cf803c`（`.empty`）/
  vllm-ascend `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`。
- 设备：卡 7（910B2，起服前/收尾后 HBM **5%**）；全程 `flock -w 7200 /tmp/w3-npu.lock`。
- 原始日志：`raw/extension-ladder.txt`、`raw/run-dryrun.txt`、`raw/serve.log`（219 行，逐字节复制）、
  `raw/health-smoke.txt`、`raw/fi-s3-serve.sh`（启动脚本）。

## 0. 结论

| §3 步骤 | 结果 |
| --- | --- |
| `extension check` | **compatible + configured + enabled**（`activation_ready=true`，无 incompatible） |
| `run --dry-run` 注入预览 | 注入 cascade 两 flag + `VLLM_HUST_FIA_DEMASK`；**不含 `VLLM_HUST_FI_SAMPLING`**（见 §2 偏差） |
| 正式 `run -- vllm serve` | 启动成功；startup snapshot + 门控证明见 §3 |
| fi-sampling 门控生效 | **成立**：`fi_sampling sampling path is ACTIVE` ×1，路由 **`fi_api1` 20 条 trace + 直方图 75/75**、`$fork=0` |
| 共存 | cascade `cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)` ×2；fia-demask `de-mask applied` ×1 |
| `/health` | **200** |
| 功能冒烟（2 条 chat） | **200 / 200**（无截断腿正确返回，`finish_reason` 正常） |
| 健康性 | 0 `TypeError`、0 `Traceback`、0 OOM；收尾 `Application shutdown complete`、端口拒连、卡 7 HBM 回 5% |
| CPU 门 | `pytest -q` **226 passed** / `ruff check .` 全绿 |

## 1. 前置门（CPU，本仓）

```
$ python -m pytest -q          -> 226 passed, 14 warnings in 41.18s
$ ruff check .                 -> All checks passed!
```

## 2. `extension check` / `status` 与 `run --dry-run`（原文见 raw/）

- `extension list`：`org.vllm-hust.fia-demask ... enabled`、`org.vllm-hust.split-batch-full-graph ... enabled`。
- `extension check org.vllm-hust.split-batch-full-graph`：
  `states = [installed, discovered, compatible, configured, enabled]`，
  `activation_ready = true`，`activation_blocker = null`。
- `run --dry-run -- vllm serve <model> --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8441
  --compilation-config '{"cudagraph_mode":"FULL"}'` 注入：

```json
"environment": {
  "VLLM_HUST_FIA_DEMASK": "1",
  "VLLM_ASCEND_ENABLE_CASCADE_DECODE": "1",
  "VLLM_ASCEND_ENABLE_CASCADE_GRAPH": "1",
  "VLLMHUST_EXT_ENABLED_BUNDLES": "org.vllm-hust.fia-demask,org.vllm-hust.split-batch-full-graph"
}
```

### 2.1 偏差（必须与结论同读）：dry-run **不注入** `VLLM_HUST_FI_SAMPLING`

- **事实**：dry-run 实际输出的 enable 环境里没有 fi-sampling 的开关。原因明确——`fi_sampling_plugin`
  **不是**本 bundle（或任何 bundle）的 `implementation[]`/`component` 条目，既不在
  `activation.environment` 里，也没有独立 bundle。目录侧对此有显式记录：catalog 条目
  `org.vllm-hust.fi-sampling` 的 `enablement.blocker` = 「not registered in the extension manifest」，
  `knowledge/surveys/catalog/README.md` 亦写明「**当前刻意不注册**」。
- **后果**：release.md §3 里「由 manager 注入 enable env」这一步，对 fi-sampling **当前不可达**。
- **本次处置（如实记录，未改 manifest）**：启用开关由**环境面**提供（`VLLM_HUST_FI_SAMPLING=1`
  随 `vllm-hust-ext run` 的父进程环境透传——manager 的 `run` 以 `os.environ.copy()` 起子进程），
  与历史 e2e harness（`bench_fisampling_refresh_e2e.sh`）同口径。本次 §3 因此验证的是「**fi-sampling
  在真实 manager 启动的 serve 中门控生效、与 cascade/fia-demask 共存**」，而**不是**「manager 注入
  fi-sampling 开关」——后者是 qualified 翻转的前置缺口（见 review 包 §6）。

## 3. 正式 launch 与启动日志关键行（原文）

启动（`cd /tmp`；`raw/fi-s3-serve.sh`）：

```bash
cd /tmp && HF_HUB_OFFLINE=1 VLLM_DISABLE_COMPILE_CACHE=1 ASCEND_RT_VISIBLE_DEVICES=7 \
  VLLM_HUST_FI_SAMPLING=1 VLLM_HUST_FI_SAMPLING_TRACE=1 \
  nohup flock -w 7200 /tmp/w3-npu.lock \
  vllm-hust-ext run -- vllm serve /data/shared_models/Qwen--Qwen2.5-14B-Instruct \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8441 \
    --generation-config vllm \
    --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[32,64,128]}' \
    > /tmp/fi-s3-raw/serve.log 2>&1 &
```

**extension startup snapshot / 门控生效证明**（逐字）：

```
INFO 09-11 15:41:03 [cascade_plugin.py:147] cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)
(EngineCore pid=1199290) INFO 09-11 15:41:32 [cascade_plugin.py:147] cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)
(EngineCore pid=1199290) fi_sampling sampling path is ACTIVE (VLLM_HUST_FI_SAMPLING=1): untruncated sampling uses the flashinfer-semantics triton-ascend kernel, ...
(EngineCore pid=1199290) WARNING - FIA decode de-mask applied: dropped torch.Size([2048, 2048]) int8 atten_mask and set sparse_mode 3->0 on a pure-decode FIA call ...
(APIServer pid=1198388) INFO:     Application startup complete.
```

即：同一 serve 内 **cascade 门控（manager 注入）× fia-demask 门控（manager 注入）× fi-sampling 门控
（环境面）三者同时生效**，互不冲突、0 `TypeError`。

**fi-sampling 路由生效证据**（逐字）：

```
[fi-sampling trace] route=fi_api1 B=1 k=None..None top_k=False top_p=False generators=0 logprobs_mode=raw_logprobs
[fi-sampling histogram calls=75 max_B=1] {'fi_api1': 75}
```

无截断（无 top-k/top-p）请求 100% 落 `fi_api1`；`route=fork` 出现 **0** 次。`enable_async_exponential`
缺失的漂移告警按设计出现 **1** 次（可归因，不触发回退；与
`docs/evidence/w2b-fi-sampling/EVIDENCE-refresh-2026-09-10.md` §2 一致）。

## 4. `/health` 与功能冒烟（原文见 raw/health-smoke.txt）

```
$ curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8441/health   ->  200
# 冒烟 1：'{... "content":"Reply with exactly: hello fi-sampling"}, max_tokens 32, temperature 0.7'
{"choices":[{"message":{"role":"assistant","content":"hello fi-sampling",...},"finish_reason":"stop"}], ...}  ->  200
# 冒烟 2：80-token 无截断（驱动直方图）
{"choices":[{"message":{"role":"assistant","content":"1, 2, 3, ... 22",...},"finish_reason":"length"}],
 "usage":{"completion_tokens":80,...}}  ->  200
```

## 5. 健康性与收尾

- `TypeError` = 0、`Traceback` = 0、OOM/OutOfMemory = 0；`cascade graph capture failed` = 0。
- `kill -TERM <serve pid>` → `Application shutdown complete`；端口 8441 拒连（curl 000）；
  残留 serve/EngineCore 进程 **0**；卡 7 HBM 回 **5%**。
- 卡 7 `flock` 队列：起服前 NZ stageA-e2e 任务占锁约 50 min，本 serve 排队获得；收尾后释放。

## 6. default-off 零差异（引用近期覆盖，未重复跑）

本基线（2026-09-10）已覆盖：`docs/evidence/w2b-fi-sampling/EVIDENCE-refresh-2026-09-10.md` §3.4 的
**3 条 OFF 腿**日志中 `fi_sampling` 出现 **0** 次（`fi_sampling sampling path is ACTIVE` = 0、
trace = 0、直方图 = 0），对应 `bench/results/logs_fisampling_refresh/serve_api1_off_{a,b,c}.log`；
启用侧 ON 腿 `fi_api1 1300/1300`。该证据与本记录同基线（`879e0ff` 干净树），故不再重复 default-off 冒烟。

## 7. 判定

release.md §3 启用验证**部分通过**：

1. **通过**：`check`/`status` 无误、正式 `vllm-hust-ext run -- vllm serve` 启动成功、fi-sampling
   门控生效（`ACTIVE` + `fi_api1` 100%）、与 cascade/fia-demask 共存、`/health` 200、冒烟 200、收尾干净。
2. **不可达（前置缺口）**：`run --dry-run` 不注入 `VLLM_HUST_FI_SAMPLING`——fi-sampling 未注册 manifest，
   §3 的「manager 注入 enable env」这一面无法演示。翻 catalog `qualified`（其硬门槛含
   `enablement.allowed=true`、`installation` 非空）前，须先补注册与不可变安装目标。
   该决策留给团队 review（见 `docs/evidence/fi-sampling-review.md` §6）。
