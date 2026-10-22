# W3.1 — fi_gelu（gelu_and_mul triton 融合）结题报告

- 日期：2026-09-10
- 分支/commit：`feat/cascade-attention-plug` @ `51f5818`（代码）+ 本报告随证据 commit
- 立项依据：`knowledge/handoffs/gelu-fusion-kickoff.md`（W1b 调研：exact-erf 缺口、4.5× 上界、
  "与 W3 批次合并立项"）
- **判定：能力落地、默认关闭、不建议生产启用**——e2e A/B 噪声带内不可判（预注册投影 0.2–0.6%，
  实测 |Δmedian|=0.311% < spread 1.95%），低于工作区 1% 立项门槛。

## 1. 交付物

| 件 | 位置 | 状态 |
| --- | --- | --- |
| triton 融合 kernel（exact-erf） | `src/vllm_ascend_split_batch/fi_gelu/` | 29 单测全绿（含尾块 13823/13825、gate/up 方向钉死、非连续守卫） |
| 宿主接线（default-off） | `fi_gelu_plugin.py` + pyproject entry `fi-gelu` | env `VLLM_HUST_FI_GELU=1` 才生效；缺省零 import（子进程断言）；patch 目标 `GeluAndMul.forward_oot`（NPU OOT 分发实测 + 活体签名守卫）；fail-open；13 测试 |
| eager bench | `bench/results/bench_w31_gelu.json` | 四方案对照 |
| 图模式 bench | `bench/results/bench_w31_gelu_graph.json` | 三方案 × {eager, graph-replay}，replay 全部 matched_ratio=1.0 |
| e2e A/B | `bench/results/bench_w31_e2e.{json,_summary.txt}` + `logs_w31_e2e/` | 6 腿交替，判定 NOISE-BAND |
| 预注册 | `bench/results/EXPECTATIONS-w31-e2e.md` | 跑前写定，判定规则机械执行 |

## 2. 三段证据链（结论的完整依据）

### 2.1 eager：负结果

triton-ascend host launch 恒定 ~110µs（空核=满载，grid 无关；cProfile 定位在 per-launch
Python/JIT 路径）。融合 kernel（out= 预分配）146µs ≈ fp32 cast 链 130µs，离内存下界 31µs 差
4.8×。`libdevice.erf` 不可 lower，`tl.erf` 可用（maxAbs 5.25e-07）。

### 2.2 图模式：方向反转但幅度小

**triton kernel 可被 ACL graph 捕获**（replay bit 精确 matched_ratio=1.0、重复 replay 稳定、
捕获后 eager 复用无串态）——这是一个独立的可复用事实。replay 计时（µs/次）：

| 方案 | B=64 | B=256 |
| --- | ---: | ---: |
| fp32 cast 链 | 53.5 | 83.6 |
| 两步 bf16（npu_gelu+mul） | 39.4 | 74.8 |
| **vllm 现役 bf16 直算链（F.gelu+mul）** | **34.2** | **60.5** |
| **fi_gelu 融合** | **23.7** | **60.5** |
| 纯拷贝下界 | 12.2 | 16.5 |

vs 现役路径：B=64 **−31%**、B=256 **打平**。launch 完全摊薄后融合赢在单算子层。

### 2.3 e2e：噪声带（预注册判定）

6 腿交替（off/on ×3）、64×128 token、FULL 图模式、卡 7：

| | off | on |
| --- | ---: | ---: |
| 中位墙时 | 7.071s | 7.093s |
| spread | 1.95% | 1.85% |

**Δmedian = −0.311%（ON 略慢），|Δ| < spread → 噪声带内不可判**（机械规则，W2b 同口径）。
实现在时间测试区内确证生效：ON 腿 `fi_gelu activation path is ACTIVE` ×1/腿，OFF 腿零痕迹，
token 总量 8192 全对。投影一致性：|0.311%| 落在预注册 0.2–0.6% 带内。

### 2.4 为什么立项预期（2–3%）没有兑现

kickoff 的 4.5×/2–3% 上界 = eager 墙时 140µs vs 内存下界 31µs。两个修正把它压到 0.45%：
1. serving 走图模式，现役路径在图里只占 34µs（eager 的 140µs 大半是 launch/派发，图里不存在）；
2. 融合只能省到 23.7µs → 每步每层 10.5µs × 48 层 = 0.50ms，本 harness 实测步时 ~55ms
   → 投影 ~0.9%（预注册按 112ms 步时写 0.2–0.6%，两口径都 <1% 门槛）。

## 3. 决策与状态

- **不建议生产启用** `VLLM_HUST_FI_GELU=1`（无可判收益）；接线保留为 default-off 能力 +
  triton 入图可行性参照，OFF 语义零差异已证（子进程纯净断言 + OFF 腿日志零痕迹）。
- gelu_and_mul 在 01 总表评级改为：**✅A(silu) / 🟡B(gelu)——融合不立项（e2e 低于门槛），
  exact-erf 语义缺口继续由 `npu_gelu`（原语级，exact-erf）覆盖**。
- W2b 同款教训再次成立：eager 墙时占比 ≠ 图模式可省空间，立项投影必须按图模式 replay 口径。

## 4. 附带发现（与 gelu 无关，已另行记录）

**宿主基线缺陷**：本基线（torch 2.13 + vllm-ascend-hust main）默认
`cudagraph_mode=FULL_AND_PIECEWISE` 启动即崩（`fusion_pass_compile AssertionError:
expected OutputCode, got GraphModuleImpl`），插件 inert 时同样崩 → 已记入
`docs/pitfalls.md` §1；规避：`--compilation-config '{"cudagraph_mode":"FULL"}'`。

## 5. 复现

```bash
cd /tmp && export ASCEND_RT_VISIBLE_DEVICES=7
python3 -m pytest /vllm-workspace/vllm-ascend-split-batch-hust/tests/test_npu_w31_gelu.py -q      # 29
python3 -m pytest /vllm-workspace/vllm-ascend-split-batch-hust/tests/test_npu_w31_gelu_plugin.py -q # 13
python3 /vllm-workspace/vllm-ascend-split-batch-hust/bench/bench_w31_gelu.py
python3 /vllm-workspace/vllm-ascend-split-batch-hust/bench/bench_w31_gelu_graph.py
bash /vllm-workspace/vllm-ascend-split-batch-hust/bench/bench_w31_e2e.sh   # 6 腿，~30min
```
