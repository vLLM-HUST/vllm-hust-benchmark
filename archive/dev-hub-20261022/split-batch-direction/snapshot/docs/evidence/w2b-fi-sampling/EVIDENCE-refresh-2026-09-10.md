# W2b fi_sampling — e2e 证据刷新（新基线复测，终版 2026-09-10）

> **本文件是 `REPORT.md`（2026-09-09，旧基线口径）的增量刷新，不修改其正文。**
> `REPORT.md` 的全部数字仍绑定旧基线，且其原文已声明"不跨宿主继承"。凡引用 W2b 结论处，
> 本文给出新基线复测值。

- 日期：2026-09-10
- 落点：插件仓库 `vllm-ascend-split-batch-hust`，分支 `feat/cascade-attention-plug`
- **被测 commit：`879e0ff`**（`fix(fi-sampling): tolerate removed ascend-config knobs in
  _fallback_flags`），**工作树干净**（`dirty_tracked=false`）——本刷新终版数字全部产自该 commit。
- 环境：conda `hust` / Python 3.12.14 / 卡 7（`ASCEND_RT_VISIBLE_DEVICES=7`，跑前查 5% 空闲）/
  CWD=`/tmp` / `HF_HUB_OFFLINE=1` / `VLLM_DISABLE_COMPILE_CACHE=1`；`flock /tmp/w3-npu.lock` 全程持有。
- **未提交任何 commit**；未 `pip install`/`uninstall`。

## 0. 结论速览

| 项 | 新基线结论 |
|---|---|
| **宿主漂移事故（headline）** | as-shipped 插件在新基线上曾 **100% 走 fork 链且静默**：`enable_async_exponential` 被上游删除 → `_fallback_flags()` 抛 `AttributeError` → fail-closed，而日志仍打印 `ACTIVE`。已修复并**入库 `879e0ff`**。事故与证据见 §2 |
| 无截断 e2e A/B（api1，主收益区） | on/off 3 轮交替，Δmedian = **−0.052%**，噪声带 **0.368%** → **噪声带内不可判**（W2b 同口径） |
| FI 路由生效证据 | ON 腿 **`fi_api1` 1300/1300 ×3 腿**（`max_B=64`），OFF 腿**零 `fi_sampling` 痕迹**——与旧 W2b 命中数**完全一致** |
| joint 可达性 | **可达且复现**：512 上下文形态实测 `max_B=504`，直方图（calls=200 快照）`{'fork': 136, 'fi_joint': 64}`；旧基线为 `{'fork': 111, 'fi_joint': 64}` |
| 步时基数 | 新基线 TPOT median ≈ **74.9 ms/步**（FULL 图模式），旧基线 ≈ **112.7 ms/步**；投影基准必须重算（§3.3） |
| 健康性 | 0 `TypeError`；OFF 腿日志 `fi_sampling` 出现 **0** 次；7 腿端口 slot 前后皆空闲；无残留进程；卡 7 回落 5% |
| 质量门 | route+plugin 单测 **46 passed**；插件全量 **190 passed / 0 failed**；`ruff check .` 全绿；NPU 冒烟 **6/6 PASS** |
| **catalog 材料** | **齐备**——切片数字全部产自入库 commit `879e0ff`（干净树），无遗留前置。见 §9 |

## 1. e2e 协议（新基线口径，预注册规则跑前写定、不变）

沿用 W2b `REPORT.md` §3.1 的原协议（`vllm bench serve --backend openai-chat`、random 数据集、
`--ignore-eos`、固定长度、TTFT/TPOT/ITL 50/95/99 分位、每腿独立 server 生命周期、
`--generation-config vllm`），**仅新增两处新基线必需项，且两腿对称施加**：

| 新增项 | 理由 |
|---|---|
| `--compilation-config '{"cudagraph_mode":"FULL"}'` | 本基线默认 `FULL_AND_PIECEWISE` 引擎初始化即崩（宿主缺陷，`docs/pitfalls.md` §1.3） |
| `VLLM_DISABLE_COMPILE_CACHE=1` | 共享 AOT 缓存命中硬崩（同 §1.3） |

`--enforce-eager`（旧报告用）**已移除**：本任务要求图模式；旧报告的 eager 口径不可跨基线继承。

- api1（无截断 / 主收益区）：`--max-model-len 4096 --max-num-seqs 512`；random 1024 in / 256 out，
  `--num-prompts 256 --max-concurrency 64`，**无 top-k/top-p**；腿序 `off_a, on_a, off_b, on_b,
  off_c, on_c`（交替，抗机器漂移）。
- joint（可达性复测，不做 TPOT 归因）：`--max-model-len 512 --max-num-seqs 512`；random 128 in /
  64 out，`--num-prompts 512 --max-concurrency 512 --top-k 50 --top-p 0.95`；1 条 ON 腿（开 trace）。
- ON = `VLLM_HUST_FI_SAMPLING=1`（+ `VLLM_HUST_FI_SAMPLING_TRACE=1`），OFF = 不设任何变量。

**预注册判定规则**（跑前写入 `fisampling_refresh_e2e.json` 的 `meta.preregistration`，本次与上轮
**逐字不变**）：`delta_frac=(median_on−median_off)/median_off`（负 = ON 更快，W2b 符号口径）；
同臂 `spread=(max−min)/median`；`noise_band=max(spread_off, spread_on)`；
`|delta_frac| < noise_band → 噪声带不可判`，否则 WIN/LOSS。**禁止调参重跑凑显著。**

## 2. 宿主漂移事故与修复（本刷新最重要的发现）

### 2.1 事故：as-shipped 插件在"新宿主删掉一个旋钮"时静默失效

在 `879e0ff` **之前**的宿主树上，插件在新基线上**完全不生效且不报错**：

```
# 修复前 ON 腿（本次单腿复现，本目录 bench/results/logs_fisampling_refresh/prefix_drift/）
fi_sampling sampling path is ACTIVE (VLLM_HUST_FI_SAMPLING=1): ...        # 声称已激活
WARNING - fi_sampling could not read the vllm/vllm-ascend config
          (AttributeError("'AscendConfig' object has no attribute 'enable_async_exponential'"));
          routing every request to the fork sampling chain.
[fi-sampling trace] route=fork reason=batch_invariant B=1
[fi-sampling trace] route=fork reason=batch_invariant B=8      ×20（trace 上限）
histogram calls=1300 max_B=64] {'fork': 1300}                  # 1300/1300 全 fork，0 次 fi_api1
```

- **根因**：vllm-ascend 上游 commit `4f0a38a95`（`[Refactor] asnc exponential optimization
  unset`，#12306）删除了 `AscendConfig.enable_async_exponential`。静态复核：
  `AscendConfig.__dataclass_fields__` 共 45 项，含 `enable_reduce_sample`、**不含**
  `enable_async_exponential`；`/vllm-workspace/vllm-ascend-hust` 全树 grep 该名字 **0 命中**。
- **失效机理**：`fi_sampling_plugin._fallback_flags()` 用**普通属性访问**读该旋钮 → 抛
  `AttributeError` → 被最外层 `except` 捕获 → 按设计 **fail-closed**（五类回退全命中）→
  每一次 `forward_native` 都走 fork 链。即 fail-closed 在"宿主删了一个旋钮"这个新场景上
  从"安全"变成了"永久失效"——不崩溃、不报错、日志还显示 ACTIVE。
- 该腿 TPOT median **75.14 ms**，与 OFF 腿（74.92 ms）无差别 → 失效态零开销，但此时 e2e A/B
  **对 FI kernel 无任何信息量**（ON 腿 ≡ OFF 腿）。这正是"旧证据不可跨宿主继承"的一个实证：
  宿主一次普通重构即可让已 review 的移植能力静默归零。

### 2.2 修复（已入库 `879e0ff`）

`_config_flag(config, name)`：`getattr(config, name, _MISSING)`——**宿主未定义该旋钮 =
该特性在上游已不存在 = 关闭**，并按旋钮去重 warn 一次（漂移可见，不静默）；
旋钮**存在但读取出错**仍向上抛，使 `_fallback_flags` 保持原有 fail-closed 语义
（`test_fallback_flags_fail_closed_when_ascend_config_raises` 不动）。
新增 2 例回归测试：
`test_missing_async_exponential_knob_is_host_drift_not_a_fallback`、
`test_present_reduce_sample_knob_still_falls_back`。

修复后 ON 腿日志（`serve_api1_on_a.log`）唯一相关告警变为可归因的漂移声明，不再触发回退：

```
fi_sampling: this vllm-ascend host does not define the ascend-config knob
'enable_async_exponential'; treating the feature it gates as disabled (feature removed upstream).
[fi-sampling trace] route=fi_api1 B=1 k=None..None top_k=False top_p=False ...
```

> **§3–§5 的全部数字产自该修复已入库后的干净树（`879e0ff`）**；修复前的失效态证据单独留档于
> `prefix_drift/`，不参与任何性能汇总。

## 3. e2e 结果（api1，无截断，6 腿，HEAD `879e0ff`）

### 3.1 逐腿表（终版）

- 服务：`vllm serve` Qwen2.5-Coder-14B-Instruct，`--generation-config vllm --gpu-memory-utilization
  0.85 --compilation-config {"cudagraph_mode":"FULL"}`，卡 7，独立端口 slot（8351–8356）。
- 工作负载：4096 上下文，1024 in / 256 out，256 prompts，concurrency 64，num_warmups 8。
- 腿运行墙时 = 起进程→kill server 全程（含 FULL 图模式编译，缓存禁用）。

| 腿 | median TPOT (ms) | mean | p95 | p99 | 输出吞吐 (tok/s) | 成功/失败 | ready(s) | 客户端(s) | 腿墙时(s) | 路由直方图 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| off_a | 74.98 | 72.16 | 75.65 | 77.29 | 808.52 | 256/0 | 175 | 120.2 | 296.9 | — |
| on_a | 75.02 | 72.24 | 76.11 | 77.35 | 807.12 | 256/0 | 180 | 120.0 | 301.6 | `{'fi_api1': 1300}` |
| off_b | 74.94 | 72.10 | 75.75 | 77.21 | 808.70 | 256/0 | 165 | 120.5 | 287.1 | — |
| on_b | 74.92 | 72.05 | 75.90 | 77.27 | 809.10 | 256/0 | 185 | 118.5 | 305.2 | `{'fi_api1': 1300}` |
| off_c | 74.96 | 72.14 | 75.68 | 77.30 | 807.98 | 256/0 | 165 | 118.1 | 284.7 | — |
| on_c | 74.75 | 71.93 | 75.43 | 77.09 | 809.81 | 256/0 | 175 | 118.2 | 294.9 | `{'fi_api1': 1300}` |

矩阵总墙时 16:44:58 → 17:24:55 ≈ **40.0 min**（含 6×10s 腿间隔 + joint 腿）。

### 3.2 判定（机械执行预注册规则）

| 指标 | off | on | 变化 |
|---|---:|---:|---:|
| 3 轮 median TPOT 的中位数 | **74.961** | **74.922** | **−0.052%** |
| 3 轮值 | 74.94 / 74.96 / 74.98 | 74.75 / 74.92 / 75.02 | — |
| 同臂 spread | 0.041 ms (**0.055%**) | 0.275 ms (**0.368%**) | — |

**|Δmedian| = 0.052% < 噪声带 0.368% → 噪声带内不可判（NOISE-BAND / NOT DECIDABLE）**，
与 W2b 旧基线结论一致。逐腿配对（同 index on vs off）亦无方向性：
a **+0.054%** / b **−0.022%** / c **−0.289%**，符号在三条配对上翻转。

### 3.3 步时基数变了：投影必须重算

- 旧基线步时 ≈ **112.7 ms**；新基线（FULL 图模式）≈ **74.9 ms**，即基数 **降了约 33%**。
- W2 单算子的"省 0.9 ms/步 @B=64"（fork 1.27 ms → FI 0.35 ms）是**旧基线 eager 实测值**，
  本刷新**未**在新基线重测单算子，故只能作为**借用上界**：
  `0.9 / 74.9 ≈ 1.2%`。实测 |Δ| = 0.052%，落在噪声带内且远低于该上界，与"采样占步时 ~1%"的
  旧判断在量级上仍然自洽。
- **结论不变**：无截断路径在该形态下无 e2e 可测收益；**要给正收益结论需要采样占步时更大的形态**
  （大 B / 更大词表），本单卡 4k 上下文 KV 容量限制下不可达（§4）。

### 3.4 FI 路由生效证据（与原报告同口径）

| 腿 | `fi_sampling ... ACTIVE` 行 | `fi_sampling` 出现次数 | 漂移 warn | trace 行 | 最终直方图 | max_B |
|---|---:|---:|---:|---:|---|---:|
| off_a / off_b / off_c | **0** | **0** | 0 | 0 | — | — |
| on_a / on_b / on_c | 1 | 2 | 1 | 20 | `{'fi_api1': 1300}` | 64 |

- 首条 trace：`[fi-sampling trace] route=fi_api1 B=1 k=None..None top_k=False top_p=False ...`
  → "无截断走 FI api1"在真实服务下 **100% 生效**，命中数与旧 W2b **逐位相同（1300/1300）**。
- 复现：`grep -c 'fi_sampling sampling path is ACTIVE' serve_api1_on_a.log` = 1，off 腿 = 0；
  `grep -o 'histogram calls=.*max_B=[0-9]*\] {[^}]*}' serve_api1_on_a.log | tail -1`。

## 4. joint 可达性复测

| 形态 | `--max-model-len` | concurrency | 实测 max_B | 直方图快照 | 结论 |
|---|---:|---:|---:|---|---|
| 512 上下文 | 512 | 512 | **504** | `{'fork': 136, 'fi_joint': 64}` @calls=200 | **可达**（joint cell 命中） |

- 旧基线 probe4：`{'fork': 111, 'fi_joint': 64}` @512 ctx → **可达性结论在新基线依旧成立**，
  且 `max_B` 更高（504，贴近 512 ctx 上限）。
- 小 B top-k cell 仍按设计留 fork 且**零同步**：trace 头部
  `route=fork B=8 k=None..None top_k=True top_p=True`——`B < 256` 不触发每步 top-k 主机读取，
  因此 `k=None` 使 joint 判据不成立 → fork（这是 `REPORT.md` §4 排序修复的设计意图，非缺陷）。
- joint 腿 TPOT median **223.13 ms**（512 并发 64 输出 token，KV-bound），**不做 A/B 归因**
  （按旧报告 §3.3 的降级原则：该 cell 在常规 4k 部署下仍是死分支）。
- **如实记录**：本任务给出的"512+ 上下文凑 B≥256"条件在新基线上**满足且富余**；
  4k 上下文形态（常规部署）下 joint cell 依旧不可达的旧结论未被本次测量推翻，也未被重测。

## 5. 与旧证据对照（新基线 vs 旧基线）

| 项 | 旧（W2b REPORT, 2026-09-09） | 新（本刷新终版, 2026-09-10 @ `879e0ff`） |
|---|---|---|
| vllm | 0.23.0 | `0.28.1.post1.dev143+gf18cf803c` |
| vllm-ascend | 0.23.0rc1 | `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27` |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post2 | 2.13.0+cpu / 2.13.0rc1 |
| CANN | 9.0.1 | 9.1.0（`V100R001C11SPC001B243`） |
| triton-ascend | 3.2.1 | **3.2.2** |
| 宿主源码树 | `/vllm-workspace/vllm`、`vllm-ascend`（已删） | vllm-hust v1 / vllm-ascend-hust main（editable） |
| 图模式 | `--enforce-eager` | `--compilation-config {"cudagraph_mode":"FULL"}` + `VLLM_DISABLE_COMPILE_CACHE=1` |
| api1 轮次 | 2/臂 | 3/臂（交替） |
| api1 median TPOT off→on | 112.67 → 112.52（**−0.14%**） | 74.961 → 74.922（**−0.052%**） |
| api1 判定 | 噪声带不可判 | **噪声带不可判**（带 0.368%） |
| FI 路由命中 | `fi_api1` **1300/1300** | `fi_api1` **1300/1300**（×3 腿） |
| joint 可达性（512 ctx） | 可达 `{'fork':111,'fi_joint':64}` | 可达 `{'fork':136,'fi_joint':64}`，max_B=504 |
| as-shipped 是否可用 | 可用 | **曾不可用**（宿主漂移 → 100% fork），已由 `879e0ff` 修复；终版数字产自修复后 |

## 6. 验证门槛结果

| 门槛 | 命令 | 结果 |
|---|---|---|
| 功能前置（route+plugin 单测） | `python -m pytest tests/test_fi_sampling_route.py tests/test_fi_sampling_plugin.py -q` | **46 passed**（原 44 + `879e0ff` 新增 2 例漂移回归）；route 单测为**纯逻辑、无 NPU 用例**（grep `npu/torch_npu/ASCEND/skipif` 零命中） |
| 插件全量单测 | `python -m pytest -q` | **190 passed / 0 failed**（旧 W2b 口径为 77 passed / 7 failed cascade 锚点漂移，本基线已无失败） |
| lint | `ruff check .`（line-length 88） | **All checks passed** |
| NPU 冒烟 | `cd /tmp && ASCEND_RT_VISIBLE_DEVICES=7 python docs/evidence/w2b-fi-sampling/smoke_npu_fi_sampling.py` | **6/6 PASS**（`logs/logs_smoke_npu_refresh-2026-09-10.txt`） |
| e2e A/B | `bash bench/bench_fisampling_refresh_e2e.sh`（flock + 卡 7） | 6 腿 api1 + 1 腿 joint，7/7 ok |
| 健康性 | 全部腿日志 | 0 `TypeError`；OFF 腿 `fi_sampling` = **0** 次；7 端口 slot 跑前跑后皆空闲；无残留 `vllm serve` 进程；卡 7 回落 5% |

## 7. 复现

```bash
# 前置（新基线回归）
cd /vllm-workspace/vllm-ascend-split-batch-hust
python -m pytest tests/test_fi_sampling_route.py tests/test_fi_sampling_plugin.py -q   # 46
ruff check .

# NPU 冒烟
cd /tmp && ASCEND_RT_VISIBLE_DEVICES=7 python \
  /vllm-workspace/vllm-ascend-split-batch-hust/docs/evidence/w2b-fi-sampling/smoke_npu_fi_sampling.py

# e2e 矩阵（6 腿 api1 + 1 腿 joint，约 40 min；卡 7）
nohup flock /tmp/w3-npu.lock -c 'cd /tmp && ASCEND_RT_VISIBLE_DEVICES=7 bash \
  /vllm-workspace/vllm-ascend-split-batch-hust/bench/bench_fisampling_refresh_e2e.sh' \
  > /tmp/fisampling_refresh.log 2>&1 &

# 事故复现（修复前行为）：临时回退 §2.2 修复后跑单腿
cd /vllm-workspace/vllm-ascend-split-batch-hust
git revert --no-commit 879e0ff      # 或 git stash 该文件改动（若尚未入库）
cd /tmp && ASCEND_RT_VISIBLE_DEVICES=7 bash \
  /vllm-workspace/vllm-ascend-split-batch-hust/bench/bench_fisampling_refresh_e2e.sh \
  --leg on on_prefix 8357 api1
cd /vllm-workspace/vllm-ascend-split-batch-hust && git revert --abort 2>/dev/null || true
```

## 8. 产物清单

| 文件 | 内容 |
|---|---|
| `bench/results/fisampling_refresh_e2e.json` | 终版：预注册 meta（基线/版本/git=`879e0ff`/判定规则）+ 7 腿逐腿行 + 机械判定 |
| `bench/results/logs_fisampling_refresh/serve_api1_{off,on}_{a,b,c}.log` | 终版各腿 server 日志（OFF 腿零痕迹 / ON 腿 ACTIVE + 直方图） |
| `bench/results/logs_fisampling_refresh/serve_api1_{off,on}_{a,b,c}.json` | 终版各腿 `vllm bench serve` 原始指标 |
| `bench/results/logs_fisampling_refresh/client_*.log`、`leg_*.log` | 客户端输出 / 腿子进程 stdout |
| `bench/results/logs_fisampling_refresh/serve_joint_on_j.{log,json}` | joint 可达性腿（终版） |
| `bench/results/logs_fisampling_refresh/prefix_drift/` | **事故证据（修复前）**：ON 腿 server/client 日志 + 原始指标（100% fork + AttributeError 警告） |
| `bench/results/logs_fisampling_refresh/wip_3ea2e84_wip_fix/` | 中间轮（HEAD `3ea2e84` + 未提交修复）产物，已归档，不参与终版汇总 |
| `bench/results/logs_fisampling_refresh/orchestrator.log` | 终版矩阵编排日志 |
| `bench/results/logs_fisampling_refresh/pytest_route_plugin.txt` | 单测记录（190 passed） |
| `bench/bench_fisampling_refresh_e2e.py` / `.sh` | 本刷新 harness（W3.1 `bench_w31_e2e.py` 腿编排风格 + W2b serve 协议） |
| `logs/logs_smoke_npu_refresh-2026-09-10.txt` | NPU 冒烟 6/6 |

## 9. 给 catalog 的一句话

**材料齐备，可进入 `qualified` 评审**：切片数字全部产自入库 commit `879e0ff`（干净树）——
无截断 e2e 三腿交替（Δmedian **−0.052%** < 噪声带 0.368% → 噪声带不可判，W2b 同口径）、
FI 路由 **`fi_api1` 1300/1300 ×3 腿** 100% 生效、OFF 腿零痕迹、joint 512-ctx 可达
（`max_B=504`）、0 `TypeError`、单测 **190 passed**/lint 全绿/NPU 冒烟 6/6；
且 §2 的宿主漂移事故已修复入库并留档证据。**唯一须随材料声明的边界**：`qualified` 仅覆盖
新基线（`879e0ff`）已验证的 `host.version_range`，且"无 e2e 正收益（噪声带）"与"joint 在常规
4k 部署下是死分支"两条形态事实须一并写入 catalog 结论，不得只写"已生效"。

## 10. 遗留项

1. **宿主漂移类缺陷面未清**（§2）：本插件其余宿主耦合面（`AscendTopKTopPSampler` 签名、
   `logprobs_mode` 取值、`vllm.envs` 名称）**尚未逐项按新宿主复核**；本次只暴露并修了 1 处。
   建议按 `REPORT.md` 复审同款方式做一次"宿主面清单核对"。
2. **单算子基线未在新 triton-ascend 3.2.2 / 图模式下重测**：§3.3 的 1.2% 上界借自旧基线 eager。
   若要给解析型收益投影，需在 `flashinfer-migration/sampling` bench 上按新基线重跑单算子。
3. **joint cell 在常规 4k 部署下仍是死分支**：本次只复测了 512 上下文可达性，未测 4k 形态。
4. **e2e 收益未显形（继承 `REPORT.md` §6.1）**：形态受限，非路由失效（路由证据 100% 命中）。
5. **`K1_ARGMAX` / RNG 语义差异**：`REPORT.md` §6.3/§6.4 遗留项在本基线未重测（本次两腿均未触发
   k=1 路由与 per-request generator 回退）。
