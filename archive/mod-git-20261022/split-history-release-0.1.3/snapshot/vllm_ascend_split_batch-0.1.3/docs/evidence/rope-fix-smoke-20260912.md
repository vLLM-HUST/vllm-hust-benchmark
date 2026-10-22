# 证据：rope carrier 入口点刷新 + default-off/ON 冒烟（2026-09-12）

- 仓库：`vllm-ascend-split-batch-hust`，改动前工作树干净 @ **`4239dfc`**
  （`docs(evidence): fix the rope-fix test count (35 new, not 34)`）。
- Bundle：`org.vllm-hust.rope-fix`（`import_only`，本次**未**改 manifest）。
- 开关：`VLLM_HUST_ROPE_FIX=1`（默认 0 = 不 import `vllm_ascend`/`torch`/`torch_npu`、不 patch、日志零输出）。
- 设备：卡 7（起服前/收尾后 `HBM Usage Rate = 5%`）；两腿 serve 均在
  `flock -w 7200 /tmp/w3-npu.lock` 内（`raw/run_smoke.sh`）。
- 原始日志：`docs/evidence/rope-fix-smoke-20260912/raw/`（pip 日志、ext 输出、两腿 serve 日志、
  聊天响应、证据 grep、阶段时间线 `smoke_stage.log`）。

## 0. 结论

| 步骤 | 结果 |
| --- | --- |
| 1. 入口点元数据刷新（`pip install --no-deps -e .`） | **成功**（rc=0，重装 `vllm-ascend-split-batch 0.1.0.dev0` editable） |
| 2. `extension list` | **`org.vllm-hust.rope-fix 0.1.0.dev0 disabled` 已被发现**（刷新前不在列表里） |
| 3. `extension inspect` | `activation.environment = [["VLLM_HUST_ROPE_FIX","1"]]`；`activation_ready=false`，blocker = `import_only`（**符合 manifest 纪律，非缺陷**） |
| 4. `extension check` | states = `[installed, discovered, compatible, degraded]`，evidence 明示 `descriptor-only / implementation status: import_only` |
| 5. default-off 冒烟（8453，mml 4096） | `/health` 200；2 条 chat 均 200；**日志零 rope carrier 痕迹** |
| 6. ON 冒烟（8454，`VLLM_HUST_ROPE_FIX=1`） | `/health` 200；2 条 chat 均 200；**三行 carrier 证据齐全**（wrapped / ACTIVE / overrides in place，三个键全部施加） |
| 7. 收尾 | 我方 serve 子树无残留（`our residual children: <none>`）；HBM 回 5% |
| 8. manifest | **未改动**（`host.version_range`、其它 bundle 均未触碰） |

## 1. 入口点刷新（步骤 1）

```bash
cd /vllm-workspace/vllm-ascend-split-batch-hust
pip install --no-deps -e .        # 工作区既定先例：本任务唯一许可的 pip 用法
```

- 原始输出：`raw/pip-install-editable.log`（`Successfully built` + `Successfully installed`，rc=0）。
- 动机（实测前后对照）：刷新**前**已安装 dist 的 entry points 只有 2 个 bundle
  （`org.vllm-hust.fia-demask`、`org.vllm-hust.split-batch-full-graph`），
  `pyproject.toml:20-30` 中的 `org.vllm-hust.rope-fix`（bundle 与 `vllm.general_plugins`
  各一条）**未进元数据**；刷新后两者均出现。

## 2. bundle 发现 / 校验（步骤 2-4）

```
$ vllm-hust-ext extension list
org.vllm-hust.fia-demask 0.1.0.dev0 enabled
org.vllm-hust.rope-fix 0.1.0.dev0 disabled          # <- 新增（刷新前不存在此行）
org.vllm-hust.split-batch-full-graph 0.1.0.dev0 enabled
```

- `extension inspect`（`raw/ext_inspect_rope_fix.txt`）：`bundle_id=org.vllm-hust.rope-fix`、
  `host.version_range == ==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`（满足）、
  `activation.environment=[["VLLM_HUST_ROPE_FIX","1"]]`、`activation_blocker=import_only`。
- `extension check`（`raw/ext_check_rope_fix.txt`）：`states=[installed, discovered, compatible, degraded]`，
  evidence 两条：host 版本满足区间；`extension is descriptor-only and cannot be enabled
  (implementation status: import_only)`。
- 解读：`disabled`/`degraded` 是**期望值**——carrier 尚未备齐三项证据（`docs/release.md` §3），
  ext-manager 拒绝把它标成可 enable；开关注入仍由 `VLLM_HUST_ROPE_FIX` 走
  `vllm.general_plugins` 直接生效（`rope_fix_plugin:load` 内部再判 `is_enabled()`）。

## 3. default-off 冒烟（步骤 5，端口 8453）

- 命令：无 `VLLM_HUST_ROPE_FIX`（显式 unset）、`--max-model-len 4096`、
  `--compilation-config '{"cudagraph_mode":"FULL"}'`、`--gpu-memory-utilization 0.85`、卡 7。
- `/health` 200（36 次轮询内）；2 条 chat completion 均 200（`raw/chat_off_{1,2}.json`）。
- **零回归证据**：`grep -nE "RoPE|rope-fix|rope_fix" raw/serve_off.log` → `<none>`
  （载体未被激活、未 import、日志零输出）。
- ⚠️ **日志里确有一条 Traceback**（`raw/serve_off.log:150-157`），但它是**我们主动 kill 引发的关停竞态**：
  时序为 `04:30:07 [shutdown] API server: shutdown triggered` → EngineCore SIGTERM/teardown
  → 之后 `AsyncLLM output_handler failed … EngineDeadError`；两条 chat 在此之前已经 200。
  **不是** rope 或任何功能缺陷（判定依据：Traceback 只出现在 shutdown 段之后，
  且同一日志里两个请求都是 `200 OK`）。

## 4. ON 冒烟（步骤 6，端口 8454）

`VLLM_HUST_ROPE_FIX=1`，其余同 default-off。`/health` 200；2 条 chat 均 200（`raw/chat_on_{1,2}.json`）。

引擎日志（`raw/serve_on.log`，EngineCore 进程）三行证据齐全：

```
39: RoPE fix: vllm_ascend.utils.register_ascend_customop is wrapped (call sites rebound: vllm_ascend.utils); the rope overrides are applied right after the host's own registration pass.
40: RoPE variant defect carrier is ACTIVE (VLLM_HUST_ROPE_FIX=1): op_registry_oot[Llama3RotaryEmbedding, MRotaryEmbedding, YaRNScalingRotaryEmbedding] will be pointed at the plugin subclasses right after vllm-ascend registers its custom ops -- ① llama3-scaling onto the Ascend rope kernel, ② the missing triton_mrope is_neox_style, ③ the YaRN truncate=True default of vLLM/HF.
43: RoPE fix: rope overrides in place: Llama3RotaryEmbedding -> created (defect ①: the fork never wired it); MRotaryEmbedding -> replaced AscendMRotaryEmbedding; YaRNScalingRotaryEmbedding -> replaced AscendYaRNRotaryEmbedding.
```

- 三键**全部施加**：①新建并接线、②③替换宿主类；`grep -nE "Traceback" raw/serve_on.log` → `<none>`。
- 覆盖时机符合设计（§2 时序）：包装体先装、host 注册返回后施加。

## 5. 收尾自检（步骤 7）

- 两腿 `stop` 后均打印 `serve[<tag>] stopped; our residual children: <none>`
  （reaper 只处理**我方 serve pid 的子进程**，不做 host 级 `VLLM::*` 匹配——两次运行之间的
  第一次尝试用了过宽的匹配，已在本文件 §7 记录并修正）。
- `HBM Usage Rate`：起 5% → OFF 腿后 5% → ON 腿后 5%。

## 6. 本阶段边界（未做/不含）

- **不含性能证据**：缺 llama3-scaling / Qwen2.5-yarn / Qwen2.5-VL checkpoint，三缺陷在真机上
  无法按变体分别压测；本证据只覆盖"载体可装载 + 开关生效 + default-off 零痕迹"。
- **不含 310P**（无真机）。
- **未改 manifest**：`org.vllm-hust.rope-fix` 保持 `import_only`，`host.version_range` 未动，
  其它 bundle 未动。

## 7. 复现命令与一次失败留档

```bash
E=docs/evidence/rope-fix-smoke-20260912
bash $E/raw/run_smoke.sh          # 内部 flock -w 7200 /tmp/w3-npu.lock + 锁超时重试
```

- **run-1（04:22–04:25）失败留档**：`set -u` 下 `local tag="$1" pidfile="...${tag}..."` 同一语句
  在 bash 5.2.15 报 `tag: unbound variable`（已单独验证），脚本在 OFF 腿 chat 后中断，
  遗留 serve pid 1736869 + `VLLM::EngineCor` 1738085；两者已按 pid 杀净并复核 HBM=5%，随后 run-2 重跑。
  时间线与处置原文见 `raw/smoke_stage.log` 的 `### run-1 aborted` 段。
- run-1 的 OFF 腿日志已被 run-2 覆盖（同一路径 `serve_off.log`）；run-2 是本文所有引用的来源。
