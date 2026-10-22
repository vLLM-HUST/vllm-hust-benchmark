# gate-fp32（D4 const-hoist）cohort 与三项证据（2026-10-13）

> 载体：`vllm_ascend_split_batch.gate_fp32_plugin:load`（bundle `org.vllm-hust.gate-fp32`）。
> 作用：把 MoE router/gate 权重常量的 fp32 副本外提到加载期（`gate.weight_fp32`），
> 归零每步每个 MoE 层的 `Cast[256,2048]`（40/步）。default-off，`VLLM_HUST_GATE_FP32=1` 才生效。

## 1. cohort 边界（本载体的证据只在此域内成立）

| 维度 | 取值 |
|---|---|
| host | 0.28 镜像默认栈：vllm core `0.28.1.post1.dev143+gf18cf803c`、vllm-ascend `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909`、CANN 9.1.0、Python 3.12.13、2×910B2 TP=2（版本门放行依据见 [host-028-2026-10-11.md](host-028-2026-10-11.md)） |
| 模型 | Qwen3.5-35B-A3B（40 层：30 GDN + 10 MoE） |
| serve | console script + `--no-enable-prefix-caching` + 双 inductor 缓存禁用 + `--compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,32]}'` + TP=2 / max_model_len 8192 |
| 负载 | L1 单请求（~2K prompt / 256 tok 贪心, 10 iters）；L2 `128 题 2048→256 @ 并发32`；L4 4 请求 128 tok 剖析窗 |

**不得外推**：本页不主张 0.25 线 + Qwen2.5 cohort 的结论，也不覆盖 MTP / APC /
`FULL_AND_PIECEWISE`。与 `docs/cohort-and-ownership.md` §2 的 Qwen2.5 cohort 是**并存的两个域**。

## 2. 三项证据

> 📌 **口径标注（2026-10-13）**：下列三项证据是在当时那条**退化 serve 栈**上测得的
> （`cudagraph_mode=NONE` ⇒ `mode=NONE`，inductor 亦被关闭，见 [pitfalls.md](../pitfalls.md) §1.3）。
> 该口径**不影响优化本身的结论方向**（`Cast[256,2048]` 归零、逐字符一致、开销来源判断均独立成立）；
> 引用其中的**数字**时标注此口径即可。后续实验一律走恢复后的默认配置
> （`mode=VLLM_COMPILE + cudagraph_mode=FULL + cudagraph_capture_sizes=[1,2,4,8,16,32]`）。

1. **default-off 零回归冒烟**：不注入 env 起 serve，载体不装载（serve 日志 `Gate fp32 cache
   populated` 计数 = 0），行为与原生一致；另有子进程用例证明 `load()` 关闭时不 import
   torch/vllm/vllm_ascend（`tests/test_gate_fp32_plugin.py::test_load_default_off_imports_nothing`）。
2. **正确性（L3 数值护栏）**：开/关各 8 条固定 prompt 贪心输出**逐字符一致 8/8 PASS**
   （`ladder-l3.py compare`）。理由：缓存副本 = 在线 `weight.to(torch.float32)` 的同一转换。
3. **性能（3× 交替 A/B，全部同向）**：`OFF ON OFF ON OFF ON` 配对中位值 ——
   L2 `out_tok_s` **+2.16%**、L2 mean TPOT **−2.15%**、L1 e2e **−1.00%**、L1 TPOT **−1.00%**；
   device 侧 `Cast[256,2048]` 由 20480 → **0**（4 请求窗）。墙钟收益大于 device 预算，
   说明还省了 eager 下每步 40 次的 host 发射。

## 3. 复现

```bash
# 载体开关
VLLM_HUST_GATE_FP32=1   # default-off；未设 = 原生

# L3 正确性（开/关各一次）
opt bench l3 && opt bench compare <OFF>-l3.json <ON>-l3.json     # 期望 identical=8/8 -> PASS
# 交替性能
bash /root/包体恢复/scripts/d4-ab-interleave.sh                  # OFF ON OFF ON OFF ON
```

容器侧原始产物（未随仓入库）：`/root/包体恢复/records/ladder/ab-1013-141051/`（12 份 l1/l2 + summary）、
`1013-1610-d4-gate-fp32-ab-interleaved.json`、`1013-1216-d4-gate-fp32-active.json`（per-shape）。
