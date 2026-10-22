# 证据③：TPOT/latency 性能对比（vllm-hust v1 基线，2026-09-09）

> 本文件是 `EVIDENCE.md` §3 的正文，合并了「未 shim 修复后构建」重跑版（**验收口径**，产物
> `logs/v1-ev3-rerun/`）与它取代的「`ON (shimmed)`」旧版（产物 `logs/v1-ev3/`，插件 `9396b21`）。
> 验收数字一律取未 shim 版；旧 shimmed 版中未被重跑覆盖的记录（eager 车道、shim 定性、独立 gate micro-bench、
> 产物清单）压缩保留在 §5.1、§13 与附录 B。**本文每个数字都可回溯到 `logs/v1-ev3-rerun/`（验收）或
> `logs/v1-ev3/`（旧 shimmed）下的具体文件。**
>
> - 被测插件（验收）：`vllm-ascend-split-batch-hust` HEAD **`ddc0120`**
>   （`fix(cascade): make graph-mode host wrappers signature-agnostic and fail-open`），工作树干净。
> - ON 腿**全程无 harness shim**：`PYTHONPATH` 仅含 CANN 的 acl/tbe 两项
>   （`run_matrix_rerun.progress` 首行已记录），旧 `logs/v1-ev3/shim/sitecustomize.py` 未被加入、
>   `/tmp` 下无 `sitecustomize.py`/`*.pth`。

## 0. 结论速览

| 问题 | 结论 |
| --- | --- |
| 未 shim 的 graph ON 腿能否跑通 | ✅ **能**：**14 条 ON 腿全部** `gate=1, graph_gate=1`、`capture body` ×144、`cascade key hit` ×254–381、**0 TypeError / 0 fail-open warning** |
| 未 shim 的 delta 表 | ✅ **齐备**：6 格 × ≥3 轮（b32 3 轮 / b64 5 轮），**每格 ON 与 OFF 的取值区间不相交**，判定不含"噪声带内不可判" |
| 赢/亏拓扑 | ✅ 复现：4k 两格亏、8k/16k 四格赢；`p420_b64` 从旧基线的 −3.5% 小赢变为 +6.7% 决定性亏，已由 §13 的 G6 修复收回至 +2.1% |
| 与旧 shimmed 表差异 | 极小：最大差 4.5pp（`p420_b64`，该格由"不可判"变为"决定性亏"）；其余 5 格 ≤2.7pp |
| W2 自适应 gate | `p420_b32` 实测回落、亏格收回（+12.3%→+0.9%）；`p420_b64` 的 gate 覆盖缺口已由 §13（插件 `c4121e2` 分档 margin）关闭 |
| 单算子锚点 | `fa_fp32_stage1 @4356 = 124.6–134.7µs`（旧带 133.9–135.6，抖动大）；`lse_merge @B64H40 = 35.7–36.8µs`（旧带 25.4–33.6，**稳定高出上沿 ~8–10%**，判定为 CANN 9.1 重编后的真实差异） |
| 证据三判定 | **满足（unshimmed）**：6 格无一净亏（4k 两格中 `p420_b64` 由 §13 收入噪声带）、赢格保留 |

## 1. 环境（本次测量对象）

| 项 | 值 | 溯源 |
| --- | --- | --- |
| 宿主 vllm | `0.28.1.post1.dev143+gf18cf803c`（vllm-hust **v1**，git `f18cf803c5`） | `importlib.metadata`；`git -C vllm-hust rev-parse --short HEAD` |
| 宿主 vllm-ascend | `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`（main，git `74f0c0a27`） | 同上 |
| torch / torch_npu | `2.13.0+cpu` / `2.13.0rc1` | 同上 |
| CANN | 9.1.0，`V100R001C11SPC001B243`（`/usr/local/ascend91`） | `ascend_toolkit_install.info` |
| 芯片 | 910B2 ×2 物理卡；**本次固定卡 7** | `npu-smi info` |
| kernel wheel | `ascend-kernel==2026.3.9`（CANN 9.1 重编） | `pip show` |
| 插件 | `vllm-ascend-split-batch 0.1.0.dev0` @ **`ddc0120`**（editable，工作树干净）；`vllm-hust-ext 0.2.0.dev0` | `git rev-parse`；`pip list` |
| 模型 | `/data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct`（48 层 / hidden 5120 / bf16） | 冻结 harness |

卡 7 测前空闲：`npu-smi info -t usages -i 7 | grep -i 'HBM Usage Rate'` → **5%**（卡 0–5 为邻居噪声源，
其中卡 2/3 全程 ~56 GB；卡 6 属另一 agent，未使用）。

## 2. 复现命令（逐条）

冻结 harness 与旧 run **逐字节相同**（sha256
`3dbcd45630ac3dabece70f36f47cfeefa74fd2eca72696bb55d3912c1a852557`，同时等于
`/vllm-workspace/archive/cascade-e2e/cascade_sweep_plug.py`）：

```bash
sha256sum logs/v1-ev3-rerun/cascade_sweep_plug.py.frozen \
          /vllm-workspace/archive/cascade-e2e/cascade_sweep_plug.py   # 两个 hash 相同
```

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate hust
cd /tmp                                   # 严禁 /vllm-workspace 作 CWD
bash logs/v1-ev3-rerun/run_matrix_rerun.sh          # 6 格 × OFF/ON 交替，b64×5 + b32×3
bash logs/v1-ev3-rerun/run_gate_and_anchors_rerun.sh # W2 gate(B=32) ×2 + 单算子锚点 ×1
bash logs/v1-ev3-rerun/run_gate64_rerun.sh          # W2 gate(B=64) ×2（补测亏格）
bash logs/v1-ev3-rerun/run_b64_control.sh           # b64 时间对照：OFF/ON 再各 2 轮
bash logs/v1-ev3-rerun/run_gate_override64.sh       # GATE_OVERRIDE=on 诊断（跳过 bench）
python3 logs/v1-ev3-rerun/analyze_rerun.py          # 只读日志 → analysis.json/report
```

**与旧 run 的两处口径变化（明确披露）**：

1. **ON 腿去掉 shim**（`PYTHONPATH` 不含 `logs/v1-ev3/shim`）。这正是本次重跑的目的。
2. **OFF/ON 轮次交替**（`off_a, on_a, off_b, on_b, …`）而非先 OFF 后 ON。理由：机器存在随时间
   的缓慢漂移（§7.3 的 OFF 后段样本上升可见），交替使漂移同时作用于两条腿。每格中位数不受顺序影响。

统一约束（与旧 run 一致）：`cd /tmp`、`ASCEND_RT_VISIBLE_DEVICES=7`、
`--gpu-memory-utilization 0.85`、`enable_prefix_caching=True`、`cudagraph_capture_sizes=[32,64,128]`、
`max_num_batched_tokens=32768`、`enable_chunked_prefill=False`、`async_scheduling=False`、
`max_tokens=128`+`temperature=0`+`ignore_eos=True`。
ON 腿 env：`VLLM_ASCEND_ENABLE_CASCADE_DECODE=1`、`VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1`、
`VLLM_ASCEND_CASCADE_MIN_PREFIX=4096`、`VLLM_ASCEND_CASCADE_MIN_REQS=2`（+ `..._TRACE=1` 取证）。

**一处口径偏差（与旧 run 相同，必须披露）**：所有腿统一 `VLLM_DISABLE_COMPILE_CACHE=1`。原因 =
共享 `/root/.cache/vllm` 的 torch.compile AOT 缓存命中会硬崩本基线（`logs/v1-ev3/sweep_off_b64_r2.log:124`
`TypeError: 'NoneType' object is not callable`，与 cascade 无关）。OFF/ON 对称施加，且编译在计时区外。

## 3. 形状矩阵

| cell | 共享前缀 token（标称） | B | gen | 备注 |
| --- | --- | --- | --- | --- |
| `p420_b64` | 4201 | 64 | 128 | 主图形态 |
| `p800_b64` | 8001 | 64 | 128 | 8k 段 |
| `p1600_b64` | 16001 | 64 | 128 | 16k 段 |
| `p420_b32` | 4201 | 32 | 128 | **文档记录的亏格（4.4k×B32）** |
| `p800_b32` | 8001 | 32 | 128 | 边界格 |
| `p1600_b32` | 16001 | 32 | 128 | 16k 段 |

每格输出恒为 `128×B`（§6 输出 token 校验：4096 / 8192，全部腿一致）。

## 4. 原始数据（每轮墙时，均可回溯）

每行 = 一次独立进程的一次 `llm.generate` 墙时；完整 PER-RUN 段见 `analysis_report.txt`。

| cell | OFF 各轮 (s) | ON(unshimmed) 各轮 (s) |
| --- | --- | --- |
| `p420_b32` | 7.91 / 7.68 / 7.71 | 8.64 / 8.66 / 8.69 |
| `p800_b32` | 9.01 / 9.12 / 9.08 | 8.74 / 8.97 / 8.76 |
| `p1600_b32` | 12.74 / 12.76 / 12.62 | 10.14 / 10.48 / 10.17 |
| `p420_b64` | 10.42 / 10.50 / 10.86 / 10.42 / 10.83 | 11.28 / 11.20 / 11.20 / 11.14 / 11.26 |
| `p800_b64` | 13.01 / 13.27 / 13.28 / 13.42 / 13.74 | 11.48 / 11.51 / 11.53 / 11.49 / 11.39 |
| `p1600_b64` | 19.64 / 19.89 / 20.11 / 20.37 / 20.44 | 13.75 / 14.09 / 13.53 / 14.04 / 13.64 |

来源：`sweep_off_b32_{a,b,c}.log`、`sweep_on_graph_b32_{a,b,c}.log`、
`sweep_off_b64_{a,b,c,d,e}.log`、`sweep_on_graph_b64_{a,b,c,d,e}.log`（每行 `SWEEP mode=... wall=...s`）。
b64 的 d/e 两轮来自 `run_b64_control.sh`（§7.3）。

## 5. Delta 表（赢 **和** 亏，中位数；负 = cascade 更快）

新基线（vllm-hust v1 / CANN 9.1 / 910B2 卡 7），graph 模式，**ON 腿未 shim**：

| cell | n (off/on) | OFF 中位 (s) | ON 中位 (s) | delta | 判定 | spread off/on % | ON 区间 vs OFF 区间 | 结果文件 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `p420_b32` | 3/3 | 7.71 | 8.66 | **+12.3%** | **LOSS** | 3.0 / 0.6 | `[8.64,8.69]` > `[7.68,7.91]` 分离 | `sweep_off_b32_*.log` / `sweep_on_graph_b32_*.log` |
| `p420_b64` | 5/5 | 10.50 | 11.20 | **+6.7%** | **LOSS** | 4.2 / 1.3 | `[11.14,11.28]` > `[10.42,10.86]` 分离 | `sweep_off_b64_*.log` / `sweep_on_graph_b64_*.log` |
| `p800_b32` | 3/3 | 9.08 | 8.76 | **−3.5%** | **WIN** | 1.2 / 2.6 | `[8.74,8.97]` < `[9.01,9.12]` 分离 | 同上 |
| `p800_b64` | 5/5 | 13.28 | 11.49 | **−13.5%** | **WIN** | 5.6 / 1.2 | `[11.39,11.53]` < `[13.01,13.74]` 分离 | 同上 |
| `p1600_b32` | 3/3 | 12.74 | 10.17 | **−20.2%** | **WIN** | 1.1 / 3.4 | `[10.14,10.48]` < `[12.62,12.76]` 分离 | 同上 |
| `p1600_b64` | 5/5 | 20.11 | 13.75 | **−31.6%** | **WIN** | 4.1 / 4.1 | `[13.53,14.09]` < `[19.64,20.44]` 分离 | 同上 |

数据源：`logs/v1-ev3-rerun/analysis.json` → `delta_table`（由 `analyze_rerun.py` 只读日志生成）。

**噪声评估**：全部 6 格、每腿 ≥3 轮的 run-to-run spread 最大 **5.6%**（`p800_b64` OFF：
13.01→13.74），其余均 ≤4.2%，低于任务给的 10% 阈值。**6/6 格的 ON 区间与 OFF 区间完全不相交**，
因此本表没有"噪声带内不可判"的格子——这比旧 shimmed 表（`p420_b64` 不可判）更强。

**b64 只用前 3 轮（a/b/c）的子集**（与旧 shimmed 表同为 3 轮，便于逐格对照）：
`p420_b64` 10.50→11.20（**+6.7%**）、`p800_b64` 13.27→11.51（**−13.3%**）、
`p1600_b64` 19.89→13.75（**−30.9%**）。**判定与 5 轮一致**。

**亏格如实并列（任务硬要求）**：
1. `p420_b32` **+12.3%**（7.71→8.66s）——文档记录的 4.4k×B32 亏格；比旧基线（+6.9%）**更深**。
2. `p420_b64` **+6.7%**（10.50→11.20s）——旧基线此格是 **−3.5% 小赢**，新基线**翻成决定性亏**
   （5/5 轮区间分离，非噪声）。这是本次重跑最值得注意的行为变化；§13 的 G6 修复把它收回至 +2.1%。

### 5.1 eager 车道（附带测量，非量产路径）

`enforce_eager=True`，cascade 只走 eager 两段式（`..._GRAPH` 未设）：

| cell | OFF (s) | ON (s) | delta |
| --- | --- | --- | --- |
| `p420_b64` | 17.89 | 18.80 | +5.1% |
| `p800_b64` | 17.06 | 18.74 | +9.8% |
| `p1600_b64` | 20.63 | 20.76 | +0.6% |

**全为亏**，与旧基线记录的 eager 车道开销方向一致（`cascade-c3-results/RESULTS.md:38`
eager off 13.62 / on(bf16) 15.53）。**eager 车道不构成量产主张**，此处仅作
"cascade 确实被激活"的独立证明（§7.2）。各 1 轮，未做重复。日志：
`cascade_sweep_ev3.py` + `sweep_off_eager_b64_a.log` / `sweep_on_eager_b64_a.log`（`logs/v1-ev3/`）。

## 6. 与两个参照的对照及含义

### 6.1 对照 (a)：旧 shimmed 表（`logs/v1-ev3/analysis.json`，插件 `9396b21`）

| cell | shimmed delta | **unshimmed delta** | 差 | 判定变化 |
| --- | ---: | ---: | ---: | --- |
| `p420_b32` | +11.9% | **+12.3%** | +0.4pp | LOSS → LOSS |
| `p420_b64` | +2.2%（噪声带内不可判） | **+6.7%** | +4.5pp | 不可判 → **决定性 LOSS** |
| `p800_b32` | −3.0% | **−3.5%** | −0.5pp | WIN → WIN |
| `p800_b64` | −16.0% | **−13.3%**（3 轮子集） | +2.7pp | WIN → WIN（赢幅略收窄） |
| `p1600_b32` | −19.2% | **−20.2%** | −1.0pp | WIN → WIN |
| `p1600_b64` | −30.8% | **−30.9%**（3 轮子集） | −0.1pp | WIN → WIN |

**含义**：去掉 shim 后，**未 shim 的 ON 腿性能与 shim 下的诊断值在噪声级一致**（≤2.7pp，
除 `p420_b64` 外）。也就是说：(i) 旧 shimmed 表在**性能方向上是可信的**，可以继续当诊断基线；
(ii) shim 本身没有给 cascade 带来额外开销或加速；(iii) 唯一的方向性变化是 `p420_b64`
——旧 shimmed 该格 OFF spread 4.5% > 中位差 2.2%，被判"不可判"；本次 5/5 轮区间分离，坐实为亏。
因此 `p420_b64` 从"边界格、方向朝亏"升级为**明确的亏格**。

旧 shimmed 的 per-leg 原始墙时与激活计数见附录 B。

### 6.2 对照 (b)：旧基线拓扑（`archive/cascade-e2e/sweep_*_w5d.log`，单轮）

| cell | 旧基线 delta（w5d） | **unshimmed delta** | 变化 |
| --- | ---: | ---: | --- |
| `p420_b32` | +6.9%（7.52→8.04） | **+12.3%** | 亏格加深 5.4pp |
| `p420_b64` | −3.5%（10.46→10.09） | **+6.7%** | **小赢 → 决定性亏**（10.2pp 翻转） |
| `p800_b32` | −6.6%（8.82→8.24） | **−3.5%** | 赢幅收窄 3.1pp |
| `p800_b64` | −19.6%（13.31→10.70） | **−13.5%** | 赢幅收窄 6.1pp |
| `p1600_b32` | −21.2%（12.65→9.97） | **−20.2%** | 基本持平 |
| `p1600_b64` | −33.7%（19.91→13.20） | **−31.6%** | 略收窄 |

**含义**：赢/亏**拓扑完整复现**（4k 亏、8k 小赢、16k 大赢），但 8k/16k 段的赢幅整体收窄
2–6pp，且 `p420_b64` 由小赢翻为亏。定性结论不变：cascade 的收益区间在 **≥8k 共享前缀**，
4k 段（尤其 B≥32）是稳定亏区。

## 7. ON 腿确实生效的证据（逐条，未 shim）

### 7.1 启动标记 + 图孪生捕获 + 逐 step 消费

每条 ON-graph 腿都满足下表（本次共 **14 条** ON 腿：主矩阵 6 + b64 时间对照 2 + gate 4 +
`GATE_OVERRIDE` 诊断 2；下表列 12 条，override 2 条见 §8.3）。以 `sweep_on_graph_b64_a.log` 为例：

| 证据 | 内容 | 计数 |
| --- | --- | --- |
| 启动标记 | `cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)` | 2 条（API+EngineCore） |
| **图孪生捕获** | `capture body: num_tokens={32,64,128} shared_len=4096` | **144**（3 桶 × 48 层） |
| 捕获成功 | `capture body SUCCESS: ... layers-so-far=N` | **144** |
| **逐 step 消费** | `replay update: cascade key hit` | **380**（3 形状 × 128 step = 384，即 380/384 decode step 走 cascade 重参数化） |
| 图变体选择 | `replay_swap=True` | 1730 |
| 负对照 | `twin miss`（非均匀 prefill step 回落标准图） | 1351（fail-open，设计内） |
| **崩溃/漂移** | `TypeError` / `Traceback` / fail-open warning | **0 / 0 / 0** |

全部 12 条 ON 腿的计数（`analysis.json` → `engagement`）：

| 日志 | 类型 | gate 标记 | capture body | cascade key hit | TypeError |
| --- | --- | --- | ---: | ---: | ---: |
| `sweep_on_graph_b32_a.log` | 主矩阵 | `gate=1, graph_gate=1` | 144 | 381 | 0 |
| `sweep_on_graph_b32_b.log` | 主矩阵 | `gate=1, graph_gate=1` | 144 | 381 | 0 |
| `sweep_on_graph_b32_c.log` | 主矩阵 | `gate=1, graph_gate=1` | 144 | 380 | 0 |
| `sweep_on_graph_b64_a.log` | 主矩阵 | `gate=1, graph_gate=1` | 144 | 380 | 0 |
| `sweep_on_graph_b64_b.log` | 主矩阵 | `gate=1, graph_gate=1` | 144 | 380 | 0 |
| `sweep_on_graph_b64_c.log` | 主矩阵 | `gate=1, graph_gate=1` | 144 | 380 | 0 |
| `sweep_on_graph_b64_d.log` | b64 对照 | `gate=1, graph_gate=1` | 144 | 379 | 0 |
| `sweep_on_graph_b64_e.log` | b64 对照 | `gate=1, graph_gate=1` | 144 | 381 | 0 |
| `sweep_on_gate_b32_r1.log` | W2 gate | `gate=1, graph_gate=1` | 144 | 254 | 0 |
| `sweep_on_gate_b32_r2.log` | W2 gate | `gate=1, graph_gate=1` | 144 | 254 | 0 |
| `sweep_on_gate_b64_r1.log` | W2 gate | `gate=1, graph_gate=1` | 144 | 379 | 0 |
| `sweep_on_gate_b64_r2.log` | W2 gate | `gate=1, graph_gate=1` | 144 | 379 | 0 |

> 与旧 shimmed 腿对比：shimmed 的 capture body 亦为 144、cascade hit 379–381——去掉 shim 后
> **激活行为不变**。gate 腿 hit 数较低（254/379）是因为 `p420_b32` 被 gate 回落到标准图（§8），
> 对应命中数减少。

### 7.2 OFF 对照（必须全零）

全部 6 条 OFF 腿：`cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=ok)`，
`capture body=0`、`replay update: cascade key hit=0`、`replay_swap=True=0`、`TypeError=0`
（`sweep_off_b{32,64}_{a,b,c,d,e}.log`）。**OFF 腿零 cascade 活动**，是合法负对照。

### 7.3 b64 时间对照（排除机器漂移）

`p420_b64` 的 +6.7% 是本表唯一"方向翻转"的格子，故额外做了时间对照：
`run_b64_control.sh` 在主矩阵之后（19:27–19:39）再跑 OFF/ON 各 2 轮，
`on64_d/e = 11.14 / 11.26`，与主矩阵的 `11.20/11.20/11.28` 一致；
`off64_d/e = 10.42 / 10.83`，与 `10.42/10.50/10.86` 一致。
**ON 腿没有随时间漂向 OFF**，`p420_b64` 的亏格不是漂移假象。
（对照同时暴露 OFF 腿在 1 小时内有 +2–4% 的缓慢上升，故本表采用 OFF/ON 交替 + 中位数。）

## 8. W2 自适应 gate 在亏格中的行为（实测）

### 8.1 bench 决策表（引擎内，未 shim）

`VLLM_ASCEND_CASCADE_ADAPTIVE_GATE=1`；bench 在隔离子进程中执行，逐格打印 `[cas-gate]`。
两次重跑（`sweep_on_gate_b32_r1/r2.log`）的判定一致：

| (N, P) | r1 cascade/full (µs) | r2 cascade/full (µs) | 判定 |
| --- | --- | --- | --- |
| 32 : 4096 | 672 / 430 | 684 / 429 | **OFF** |
| 32 : 8192 | 688 / 772 | 698 / 776 | on |
| 32 : 16384 | 690 / 1463 | 709 / 1468 | on |
| 64 : 4096 | 734 / 822 | 737 / 826 | **on** |
| 64 : 8192 | 732 / 1498 | 737 / 1496 | on |
| 64 : 16384 | 727 / 2903 | 679 / 2907 | on |
| 128 : 4096 | 831 / 1603 | 817 / 1603 | on |
| 128 : 8192 | 829 / 2979 | 835 / 2979 | on |
| 128 : 16384 | （未出） | （未出） | 默认 on |

与附录 B 的旧 shimmed 独立 bench **定性完全一致**：唯一 OFF 格仍是
`(32, 4096)`；`(128, 16384)` 第 9 格仍因服务进程旁 HBM 不足未出（"8 cells benched"）。

### 8.2 回落确实发生（亏格 `p420_b32` 被收回）

`p420_b32` 的运行时共享前缀是 block 对齐的 **4096**，命中 bench 的 OFF 格。两次 gate 腿均出现：

```
[cas-trace] replay update: gate off num_tokens=32 shared=4096
[cas-trace] wrapper: gate off (shared=4096 N=32) -> standard graph
```

| cell (B=32) | OFF 中位 | ON 无 gate | **ON + gate** | gate vs OFF | gate vs 无 gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `p420_b32` | 7.71 | 8.66 (+12.3%) | **7.78**（7.77/7.79） | **+0.9%** | 亏格被收回 |
| `p800_b32` | 9.08 | 8.76 (−3.5%) | 8.98（8.97/8.98） | −1.2% | 略回落但仍在 OFF 之下 |
| `p1600_b32` | 12.74 | 10.17 (−20.2%) | 10.34（10.29/10.38） | −18.9% | 不损伤赢格 |

来源：`sweep_on_gate_b32_r{1,2}.log` 的 `SWEEP ... wall=`。→ **W2 gate 在 B=32 亏区自动回落
FULL，把 +12.3% 亏拉回 +0.9%（≈持平），且不损伤 8k/16k 赢格**。

### 8.3 `p420_b64`：gate **不**回落，亏格未被覆盖（关键缺口，已由 §13 关闭）

B=64 的 gate 腿（`sweep_on_gate_b64_r1/r2.log`）显示：bench 判 `(64, 4096) -> on`
（cascade 699/711µs vs full 822/824µs），因此 **`p420_b64` 不发生回落**——
gate 腿中 `gate off` 的 replay/wrapper 各只有 1 条，且都是 `num_tokens=32`（warmup 哑跑）。

| cell (B=64) | OFF 中位 | ON 无 gate | ON + gate | gate 判定 |
| --- | ---: | ---: | ---: | --- |
| `p420_b64` | 10.50 | 11.20 (+6.7%) | 10.77（10.78/10.75） | **(64,4096)=on → 不回落** |
| `p800_b64` | 13.28 | 11.49 (−13.5%) | 11.21（11.15/11.26） | on |
| `p1600_b64` | 20.11 | 13.75 (−31.6%) | 13.56（13.36/13.77） | on |

**注意（confound，已用独立 A/B 定位）**：gate 腿的 b64 每格比无 gate 腿快 ~0.2–0.4s，
但 gate 并未回落（判定为 on），所以这**不是 gate 逻辑的功劳**。为定位原因，跑了
`GATE_OVERRIDE=on`（`run_gate_override64.sh`，文档语义 = 强制全 on 且**跳过 bench 子进程**）：
`ovr64_r1/r2 = 11.33/11.07`（p420）、`11.58/11.26`（p800）、`13.68/13.43`（p1600），
**与无 gate 腿一致**（p420 中位 11.20），且 `[cas-gate]` 行数为 0（bench 确未跑）。
→ 结论：b64 gate 腿的加速来自**bench 隔离子进程对 NPU 的预热**（~17–22s 满负载后设备进入
不同状态），不是 gate 消费路径。这也解释了为何 b32 gate 腿没有同向差异（落在 spread 内）。

**产品含义**：W2 gate 的微基准把 `(64, 4096)` 判为 cascade 更优，而 e2e 该格实为 **+6.7% 亏**。
即 **gate 的覆盖边界与 e2e 亏区不一致**：它只保护 `(32, 4096)`，不保护 `(64, 4096)`。
这是一条真实的产品行为缺口（**已由 §13 的 G6 修复关闭**）。

## 9. 单算子锚点（重导）

`logs/v1-ev3-rerun/kernel_anchor_w0_custom{,_r2,_r3}.log`
（`cascade-c3-results/probes/w0_b2_timing_probe.py 200 custom`，卡 7，min-of-3，精确 14B 头几何 H=40/KVH=8/D=128）：

| 项 | 本次 3 次实测 (µs) | 旧基线记录（`ops/kernels/ascend-kernel/README.md:179-189`，CANN 9.0.1；旧基线数字不作"本机现状"引用） | 结论 |
| --- | --- | --- | --- |
| `fa_fp32_stage1 @4356` | **134.7 / 130.3 / 124.6** | 133.9–135.6 | 抖动大（±4%），中位 130.3 略低于旧带；**量级一致** |
| `fa_fp32_stage1 @8192` | 134.5 / 133.0 / 125.5 | "4k→8k 平坦" | 一致（平坦） |
| stage-2 FIA @260 B64 | 190.6 / 188.8 / 175.1 | （未单独记录） | — |
| `lse_merge @B64H40` | **36.8 / 36.4 / 35.9**（+ 旧 shimmed 腿 35.7） | 25.4–33.6 | **稳定高出旧带上沿 ~8–10%** |
| cascade 3-op 合计 @C3 | 362.1 / 355.5 / 335.6 | — | — |

**关于 `lse_merge` 是否为新基线真实差异**：4 次独立测量（旧 shimmed 腿 35.7 + 本次 36.8/36.4/35.9）
内部 spread 仅 ~3%，却一致高于 CANN 9.0.1 记录的 33.6 上沿约 2–3µs。同一 probe 的
`fa_fp32_stage1` 同期抖动 ±4%（124.6–134.7），说明机器负载不能解释 `lse_merge` 的稳定抬升。
故判定：**`lse_merge` 在 CANN 9.1 重编 wheel 上存在约 +8–10% 的真实差异（很可能来自 CANN 9.1
的算子实现/编译差异）**，但绝对量 2–3µs 不改变"两段式固定开销主导"的定性。
未重跑 30/30、54/54、S1 bit 锚点（引用 `ops/kernels/ascend-kernel@73bf96a` 的 CANN 9.1 重验留档）。

## 10. 测量口径与局限（必须与数字同读）

1. **邻居噪声**：卡 0–5 有他人负载（卡 2/3 全程 ~56 GB）；本表用 OFF/ON 交替 + 中位数 +
   区间分离来抵抗，但无法完全消除。OFF 腿在 1 小时内观察到 +2–4% 的缓慢上升（§7.3）。
2. **`VLLM_DISABLE_COMPILE_CACHE=1`**：共享 AOT 缓存命中会硬崩本基线；OFF/ON 对称施加，
   编译在计时区外。**该 flag 本身是基线缺陷，不是 cascade 的**。
3. **stand-in 模型**：`Qwen--Qwen2.5-Coder-14B-Instruct`，`ignore_eos=True` + 128 步固定生成，
   与旧 W5 口径一致；prompt suffix 仅 ~20 token，与 C3 的 260 token 不同（C3 亏格未复核，见 §11）。
4. **墙时口径**：只有端到端 `llm.generate` 墙时，无 TPOT 分步曲线；一进程内 3 个前缀顺序执行，
   prefix-cache 增长对 OFF/ON 对称。
5. **b64 为 5 轮、b32 为 3 轮**：b64 多出的 2 轮是为排查 `p420_b64` 方向翻转而加的时间对照；
   3 轮子集判定与 5 轮一致（§5）。
6. **gate 腿的 bench 预热效应**（§8.3）：`ADAPTIVE_GATE=1` 的进程比无 gate 快，原因是 bench
   子进程预热设备而非 gate 逻辑；报告 gate e2e 数字时已扣除该归因。

## 11. 缺口 / 未覆盖

| # | 项 | 状态与原因 |
| --- | --- | --- |
| G1 | 未 shim 的 graph ON 腿 | ✅ **已补齐**（12 条腿，0 TypeError） |
| G2 | B=128 格 | 未测（时间预算）；旧基线 3 格（p420/p800/p1600 × B128）均赢（−6.0%/−27.9%/−38.4%），未复核 |
| G3 | C3 形态（4k shared × B64 × suffix 260）的 −31% 亏格 | 未测；本 sweep suffix 仅 ~20 token，该亏格只能引用旧记录 `ops/kernels/ascend-kernel/README.md:140,189` |
| G4 | W2 gate 引擎内第 9 格（128:16384） | 服务进程旁 HBM 不足，两次 gate 腿均只出 8 格（独立 bench 9/9 全出，见附录 B） |
| G5 | 单算子 30/30、54/54、S1 bit 锚点 | 未重跑，引用 `ops/kernels/ascend-kernel@73bf96a`（CANN 9.1 重验）留档 |
| G6 | `p420_b64` gate 覆盖缺口 | ✅ **已关闭（2026-09-10，插件 `c4121e2`）**，见 §13 |
| G7 | 长稳定期 / TPOT 分步曲线 | 未测（只有端到端 128-step 墙时） |
| G8 | bench 预热效应的机理 | 已证明来自 bench 子进程（`GATE_OVERRIDE` A/B），但未定位到具体设备状态（DVFS/allocator） |

## 12. 判定

- **证据三（未 shim，修复后构建）：满足**。6 格 × ≥3 轮，ON 腿逐格有激活证明，OFF 腿全零，
  赢亏并列，每格区间分离。
- **拓扑**：4k 两格亏（+12.3% / +6.7%）、8k/16k 四格赢（−3.5% … −31.6%）。与旧 shimmed 表在
  噪声级一致（最大差 4.5pp），与旧基线拓扑同形但 8k/16k 赢幅收窄 2–6pp。
- **`p420_b64` 方向翻转（旧 −3.5% → 现 +6.7%）**，5/5 轮区间分离，不是噪声；§13 的 G6 修复把它
  收回至 +2.1%。
- **W2 gate**：B=32 亏格被正确回落收回（+12.3%→+0.9%）；B=64 亏格由 §13（分档 bench margin）覆盖。
- **`lse_merge` 在新基线稳定高出旧带 ~8–10%**，判定为 CANN 9.1 重编后的真实差异（绝对量 2–3µs）。

## 13. G6 修复验证：小前缀分档 bench margin（插件 `c4121e2`，2026-09-10）

> 本节关闭 §8.3 / §11-G6 记录的产品缺口"p420_b64 gate 不回落"。它**不改动** §1–§12 的任何数字，只追加验证。
> - 被测插件：`vllm-ascend-split-batch-hust` `feat/cascade-attention-plug` @ **`c4121e2`**
>   （`fix(cascade): e2e-calibrated small-prefix bench margin`），工作树干净。
> - 原始产物：`logs/v1-ev3-gatefix/`（4 条腿日志 + `EXPECTATIONS.md` 预注册 + `run_gatefix_all.sh` + 冻结 harness）。
> - 对照基线：`logs/v1-ev3-rerun/` 的 OFF 中位（未重跑 OFF 腿，代码路径未变）。
> - 环境与 §1 完全一致，仅插件版本推进 `ddc0120` → `c4121e2`；harness 逐字节同冻结版（sha256 `3dbcd456…` 已复验）。

### 13.1 归因

bench 探测用逐请求不相交 block table（W0 防故障），测的是"无 prefix-cache 复用"世界；
4k 共享前缀（14B 几何 K+V ≈16.8MB）在引擎内可被 FULL 路径经 L2 完全复用 → cascade 流量优势归零，
只剩引擎路径固定开销 → e2e 净亏。

### 13.2 单一 margin 不可行

留档 `(64,4096)` margin +10.7/+10.8% 与 `(32,8192)` +10.9/+10.1% 几乎相同而 e2e 符号相反；
任何统一阈值必误判其一。

### 13.3 修复内容（插件 `c4121e2`）

- `cascade_gate.py`：新增 `SMALL_PREFIX_MAX=4096` / `SMALL_PREFIX_MARGIN=0.25`
  （env：`VLLM_ASCEND_CASCADE_GATE_SMALLP_MARGIN` / `..._SMALLP_MAXPREFIX`，非法值回退默认）；
  `_margin_for(shared)` 按 prefix 桶选 margin；`bench_all` 判定行与 `[cas-gate]` 打印披露所用 margin；
  docstring 记录不相交 block table 盲区机制与 `SUFFIX=256` 保守性依据。
- `tests/test_cascade_gate.py`：+9 例（留档数字回放：`(64,4096)` 双轮 OFF、`(32,8192)` 双轮 on、
  `(128,4096)` on、边界 `<=` 语义、env 覆盖与非法回退、`_margin_for` 默认分档）。全套 **111 passed**、
  `ruff check` 干净（评审自跑复核）。
- 评审补强 2 处：docstring `Env:` 清单收录新 env；autouse fixture 补新 env 清理。
- `P ≤ 4096` 桶要求 cascade 快 ≥25%（`SMALL_PREFIX_MARGIN`），其余桶维持 2%；默认
  `MIN_PREFIX=8192` 永不 bench 出小桶 → **默认行为零变化**。

### 13.4 bench 决策表（4 腿一致）

预注册：`logs/v1-ev3-gatefix/EXPECTATIONS.md`（发跑前落盘，含判定表、区间带与
"任一不满足=修复无效，不调参重跑凑数"纪律）。

| (N,P) | gatefix 实测 cascade/full (µs) | margin | 判定 | 预注册 |
| --- | --- | --- | --- | --- |
| (32,4096) | 683/432 · 692/430 | 25% | OFF | OFF ✓ |
| **(64,4096)** | **733/825 · 733/825 · 712/826** | **25%** | **OFF** | **OFF ✓（修复点）** |
| (128,4096) | 823/1597 · 832/1593 · 823/1601 | 25% | on | on ✓ |
| (32,8192) | 693/780 · 671/778 | 2% | on | on ✓ |
| (64,8192) | 727/1489 · 725/1494 | 2% | on | on ✓ |
| (32,16384) | 702/1465 · 686/1465 | 2% | on | on ✓ |
| (64,16384) | 742/2928 · 723/2914 | 2% | on | on ✓ |
| (128,16384) | 缺席（HBM，同旧） | — | 默认 on | 允许缺席 ✓ |

注：bench 数字逐腿小幅漂移（±3% 内），判定全部稳定；`(64,4096)` bench margin
11-14% 恒低于 25% 阈，`(128,4096)` 48-49% 恒高于 25% 阈——两侧均有 ≥1.8× 余量。

### 13.5 e2e 墙时（每腿 2 轮，对照 §5 的 OFF 中位）

| cell | OFF 中位 | gatefix gate 腿（r1/r2 → 中位） | delta | 判定 |
| --- | ---: | --- | ---: | --- |
| p420_b32 | 7.71 | 7.97 / 7.75 → 7.86 | +1.9% | 回落保持 ✓ |
| **p420_b64** | **10.50** | **10.51 / 10.94 → 10.725** | **+2.1%** | **亏格收回 ✓**（未回落 ON 为 +6.7%） |
| p800_b32 | 9.08 | 9.16 / 9.04 → 9.10 | +0.2% | ≈持平 ✓（噪声带内） |
| p800_b64 | 13.28 | 11.37 / 11.34 → 11.355 | **−14.5%** | 赢格保留 ✓ |
| p1600_b32 | 12.74 | 10.55 / 10.21 → 10.38 | −18.5% | 赢格保留 ✓ |
| p1600_b64 | 20.11 | 13.66 / 13.50 → 13.58 | **−32.5%** | 赢格保留 ✓ |

p420_b64 的 r2（10.94）高于 OFF 带顶 10.86 约 0.7%，但**两腿均低于未回落 ON 带
[11.14,11.28] 的下沿**（§5 的 5/5 轮区间分离证明未回落不可能低于 11.14），
且回落 trace（§13.6）确认标准图生效——判为回落成立 + 机器漂移（§7.3 记录 OFF 腿 1 小时 +2-4% 漂移；
本次 4 腿跨 13 分钟）。

### 13.6 激活与回落证据（逐腿）

| 腿 | loaded | capture body | cascade key hit | replay_swap | gate-off trace | TypeError |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| gate32_r1 | 1 | 288 | 254 | 1606* | 1×(32,4096) | 0 |
| gate32_r2 | 1 | 288 | — | — | 1×(32,4096) | 0 |
| gate64_r1 | 1 | 288 | **254** | 1606 | 1×(32,4096) + **1×(64,4096)** | 0 |
| gate64_r2 | 1 | 288 | — | — | 1×(32,4096) + **1×(64,4096)** | 0 |

\* 仅 r1 腿统计了 key hit/swap（口径对照用）。`capture body` 288 = 144 捕获 × 2 行（发起+SUCCESS）。
trace 按配置首现打印（与 §7 gate 腿同口径）。

- **步级独立佐证**：§7 的 gate64_r1 `cascade key hit`=379，gatefix=254，
  Δ=125 ≈ p420_b64 的 decode step 数 → (64,4096) 的 cascade 消费在 step 级关闭。
- 留档 8 格回放 8/8 分类正确；爆炸半径恰为缺口格 `(64,4096)`，未伤任何真赢格。

### 13.7 局限（与数字同读）

1. 每腿仅 2 轮（与 §7 gate 腿同规模），判定依赖"带判定 + 区间分离"而非单点；
   `p420_b64` 的 r2 单轮越带顶 0.7% 已按漂移归因（§13.5），未追加第 3 轮。
2. 阈值 25% 由留档误差带（bench +10.7% vs e2e −6.7%，17.4pp 翻转）标定，
   覆盖 `(64,4096)`（11-14%）与 `(128,4096)`（48-49%）两侧各 ≥1.8× 余量；
   换几何（kv 头数/头维）或换 CANN 后需重标定，env 可调不须改码。
3. B=128 格 e2e 未复测（沿 §11-G2）；`(128,4096)` 依赖旧基线 −6.0% 赢的旁证。
4. 子预期未命中（如实记录）："p420_b64 显著低于旧 gate 腿 10.75-10.78"未达成，
   10.725 vs 10.765 仅 −0.4%（噪声级）；G6 判据（亏格收回+赢格保留）不受影响。
5. **G6 判定：关闭**。性能侧证据为"6 格无一净亏 + 赢格保留"。

## 附录 A：产物清单（`logs/v1-ev3-rerun/`，验收）

- `cascade_sweep_plug.py.frozen`（冻结 harness，sha256 `3dbcd456…`，与上游逐字节相同）
- 运行脚本：`run_matrix_rerun.sh`、`run_gate_and_anchors_rerun.sh`、`run_gate64_rerun.sh`、
  `run_b64_control.sh`、`run_gate_override64.sh`
- 进度日志：`run_matrix_rerun.progress`（含 PYTHONPATH 取证）、`run_gate_rerun.progress`、
  `run_gate64_rerun.progress`、`run_b64_control.progress`、`run_gate_override64.progress`
- harness 日志：`sweep_off_b{32,64}_*.log`、`sweep_on_graph_b{32,64}_*.log`、
  `sweep_on_gate_b32_r{1,2}.log`、`sweep_on_gate_b64_r{1,2}.log`、`sweep_on_override_b64_r{1,2}.log`
- 单算子锚点：`kernel_anchor_w0_custom.log`、`kernel_anchor_w0_custom_r{2,3}.log`
- 结果：`results/sweep_*.json`（22 个 token-id 产物）、`analysis.json`、`analysis_report.txt`
- 分析脚本：`analyze_rerun.py`（只读日志；相对旧 `analyze.py` 仅改 LEGS 映射 + 加 `UNDECIDABLE` 判据，
  本表无格落入该判据）
- G6 修复（§13）：`logs/v1-ev3-gatefix/`（4 腿日志 + `EXPECTATIONS.md` + `run_gatefix_all.sh` + 冻结 harness）

## 附录 B：修复前 shimmed 构建（插件 `9396b21`）的补充原始记录

> 以下为被 §1–§12 取代的旧 shimmed 版的唯一未被重跑覆盖的记录（定性描述 + 原始墙时 + 独立 bench）。
> shim 的定性：旧版 ON 腿**无法在未打补丁的插件上启动**（3 处宿主签名漂移 D1/D2/D3（漂移口径）硬崩引擎），
> 数字是用 harness 侧 `logs/v1-ev3/shim/sitecustomize.py` 在内存里把 3 处调用改成新宿主签名后测得的
> （插件树、宿主树磁盘内容均未改动）。D1–D3（漂移口径）的逐条症状与修复见 `EVIDENCE.md` §4 与
> `fix-sigdrift-report.md`。

### B.1 shimmed 每轮墙时（3 轮，`logs/v1-ev3/`）

| cell | OFF 各轮 (s) | ON(graph,shimmed) 各轮 (s) |
| --- | --- | --- |
| `p420_b32` | 7.72 / 7.70 / 7.73 | 8.70 / 8.64 / 8.57 |
| `p800_b32` | 9.03 / 8.99 / 9.21 | 8.76 / 8.82 / 8.74 |
| `p1600_b32` | 12.74 / 12.75 / 12.77 | 10.39 / 10.22 / 10.30 |
| `p420_b64` | 10.49 / 10.58 / 10.96 | 10.99 / 10.81 / 10.80 |
| `p800_b64` | 13.06 / 13.35 / 13.36 | 11.37 / 11.22 / 11.22 |
| `p1600_b64` | 19.65 / 19.83 / 19.65 | 13.48 / 13.60 / 13.59 |

来源：`sweep_off_b{32,64}_{a,b,c}.log`、`sweep_on_graph_shim_b32_{a,b,c}.log`、
`sweep_on_graph_shim_b64_{c,d,e}.log`（每行 `SWEEP mode=... wall=...s`）。shimmed 中位与
§6.1 的 delta 表一致。

### B.2 shimmed 激活证明（逐腿，`analysis.json` → `engagement`）

- graph ON 腿（`sweep_on_graph_shim_b64_d.log`）：启动标记 `gate=1, graph_gate=1, kernel_wheel=ok`；
  `capture body: num_tokens={32,64,128} shared_len=4096` **各 48 条**（3 桶 × 48 层 = 144）+ 
  `capture body SUCCESS` 144；`replay update: cascade key hit` 380（`num_tokens=64` 359 + `num_tokens=32` 21）；
  `wrapper: cascade_flag=True capture_window=False replay_swap=True` ×1830、`capture_window=True` ×300。
  各 ON-graph 日志 key hit 计数：`b32_a` 380 / `b32_b` 379 / `b32_c` 380 / `b64_c` 381 / `b64_d` 380 / `b64_e` 381，`capture_body` 均 144。
- OFF 腿（`sweep_off_b64_a.log`）：`gate=0, graph_gate=0`，`capture body` 0、`cascade key hit` 0。
- eager ON 腿（`sweep_on_eager_b64_a.log`，无 shim、无图捕获）：`gate=1, graph_gate=0, kernel_wheel=ok`，
  `[cascade-active] shared_len=4096 shared_blocks=32 num_tokens=64 heads=40 head_size=128` **×48**；
  OFF 腿计数 0。这是"两段式确实被调用"的最直接证明。
- 说明：graph 模式下**没有** `[cascade-active]` 打印（该打印只在 eager 前向里，`cascade_plugin.py:395`）；
  graph 腿的等价逐 step 证据是 `capture body` + `replay update: cascade key hit` + `wrapper: replay_swap=True` 三件套。
- fail-open：graph ON 腿有大量 `twin miss`（`b64_d` 1451 条，`b32_b` 401 条），对应 `uniform=False` 的
  prefill/混合 descriptor，未捕获孪生的 step 回落标准图（`cascade_graph_plugin.py:697-706`）。

### B.3 独立 gate micro-bench（shimmed 期，卡 7，BNSD，suffix=256）

`logs/v1-ev3/gate_bench_v1.json`（`python3 -m vllm_ascend_split_batch.cascade_gate_self` standalone）：

| (N, P) | full (µs) | cascade (µs) | 判定 |
| --- | ---: | ---: | --- |
| 32 : 4096 | 424.4 | 690.5 | **OFF** |
| 32 : 8192 | 769.1 | 703.3 | on |
| 32 : 16384 | 1467.4 | 700.9 | on |
| 64 : 4096 | 832.0 | 721.5 | on |
| 64 : 8192 | 1492.6 | 721.0 | on |
| 64 : 16384 | 2892.8 | 716.6 | on |
| 128 : 4096 | 1590.3 | 821.5 | on |
| 128 : 8192 | 2951.8 | 824.5 | on |
| 128 : 16384 | 5793.0 | 887.6 | on |

→ 9/9 格均有有效数据，**唯一 OFF 格 = (N=32, P=4096)**。第 9 格 `128:16384`（887.6µs）
是引擎内 bench 未产出的那一格（§8.1 记"未出"）的独立测值。与旧基线
`archive/cascade-e2e/results/gate_out5.json` 定性一致（旧：32:4096 full 422.4 / cascade 678.9 → OFF）。

### B.4 旧版产物清单（`logs/v1-ev3/`）

- `cascade_sweep_plug.py.frozen`（冻结 harness，sha256 `3dbcd456…`）
- `cascade_sweep_ev3.py`（eager 车道用，含 `off_eager`/`on_eager`）
- `shim/sitecustomize.py`（harness 侧 3 处签名 shim，内存生效）
- 运行脚本：`run_matrix.sh`、`run_on_graph64.sh`、`run_gate_and_anchors.sh`、`run_repeats.sh`
- 进度日志：`run_matrix.progress`、`run_on_graph64.progress`、`run_gate.progress`、`run_repeats.progress`
- harness 日志：`sweep_off_b{32,64}_{a,b,c}.log`、`sweep_on_graph_shim_b{32,64}_{a,b,c,d,e}.log`、
  `sweep_{off,on}_eager_b64_a.log`、`sweep_on_gate_b32.log`、
  `smoke_on_b64.log`（D1（漂移口径）原始崩溃）、`sweep_on_graph_shim_b64_{a,b}.log`（D2 原始崩溃）
- 结果：`results/sweep_*.json`（17 个 token-id 产物；含 2 个早期中断轮
  `sweep_off_off_r1.json` / `sweep_off_validate.json`，不计入 delta 表）、`gate_bench_v1.json`、
  `kernel_anchor_w0_custom.log`、`analysis.json`、`analysis_report.txt`
- 早期诊断留档（不计入 delta 表）：`run_primary.progress`、`sweep_off_b64_r2.log`
  （compile-cache 硬崩）、`sweep_on_graph_b64_r{1,2}.log`（D1（漂移口径）未打 shim 的崩溃）
