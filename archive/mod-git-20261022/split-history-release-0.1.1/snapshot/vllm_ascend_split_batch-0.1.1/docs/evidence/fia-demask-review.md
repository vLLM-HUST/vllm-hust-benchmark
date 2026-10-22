# Review 材料：fia-demask（decode FIA 去掩码）

> 一页材料，可独立阅读。结论先行：**已翻 active（§3 全绿）**——正确性构造性等价 + 逐 token
> 全等，六格全量 TPOT geomean **+6.43%**（全部 ≥1% 门槛），default-off 零差异。

## 1. 它是什么

不做任何新 kernel：vllm-ascend 的 decode 路径调 CANN 官方 FIA
（`npu_fused_infer_attention_score`）时每次都携带一张 2048×2048 int8 因果模板掩码
（来源 `attention_mask.py get_splitfuse_attn_mask`），而在纯 decode（每请求 Q_S=1、
`pre_tokens=next_tokens=INT_MAX`、`sparse_mode=3`）下该掩码**恒为恒等**——构造性冗余。
插件以 op 面包装方式（公开 `torch_npu` 符号，含 `.out` 与 workspace 查询，宿主自身在
`batch_invariant.py` 亦 patch 同一符号——模式有先例）对匹配谓词的调用去除掩码并将
`sparse_mode` 3→0；prefill/混批/滑窗/非 TND 调用一律原样放行。

## 2. 证据链（全部可溯源）

| 项 | 结果 | 出处 |
| --- | --- | --- |
| CPU 守护 | 41 tests（demask 35 + manifest 守护）+ ruff 全绿 | `tests/test_fia_demask.py`、`tests/test_manifest.py` |
| 门控证据 | OFF 腿 observe 行（mask 存在）/ ON 腿 `de-mask applied` + `ACTIVE` 行；/health 200 | `profiles/.../probe-fia/`、`raw/serve_demask_{off,on}.log` |
| 逐 token 等价 | greedy 8/8 **bit 级全等**（diff 为空） | `probe-fia/e2e/parity_diff.txt` |
| e2e decode 账（双格深钻） | p1600×B32 **+3.10%**、B64 **+4.61%** TPOT（3+3 轮） | `probe-fia/E2E-mask-removal.md` §3 |
| **六格全量 + e2e 账** | TPOT +3.30~+10.11%（p420/800/1600 × B32/64，**geomean +6.43%**）；e2e random 1000/1000 吞吐 **+3.52%**（42 轮） | `E2E-mask-removal.md` §3.1 |
| 单算子口径 | 去掩码使 FIA 单调 −8…−23%（中位 −12%），108 cell sweep | `probe-fia/REPORT.md` §(c) |

**两条必随事实**：(1) 每腿首轮 bench 为预热离群（四组对称出现，OFF 52.1/70.8 vs ON
51.1/68.6 ms），判读看 r2/r3（腿内散布 0.2%）；(2) 专用探针预测 ~3.1%（B32 口径）与
e2e 实测 +3.10% 吻合；B64 +4.61% 超预测，方向一致、已注记。

## 3. 交付物与提交

- 实现：`src/vllm_ascend_split_batch/fia_demask_plugin.py`（commit `026197e`）
- 激活面：独立 bundle `org.vllm-hust.fia-demask`（commit `72a4c13`；决策理由
  design note §8——单能力独立开关，enable 不再与 cascade 耦合）
- 设计：`docs/design/fia-decode-demask.md`（宿主读码 + 等价论证 + activation 决策）
- 验收：`profiles/qwen14b-instruct-hotspot-20260910/probe-fia/E2E-mask-removal.md`

## 4. default-off 语义

`VLLM_HUST_FIA_DEMASK` 未置 `1` 时 `load()` 在 import torch 之前返回——不 patch、
不注入、日志零输出，与原生逐字节同路径（测试覆盖两态）。

## 5. 已知边界

- 谓词保守限定探针证据域（TND + sm3 + INT_MAX + 方形 int8 掩码 + 累计长度==T）；
  其余调用零改写（fail-open 计数可观测）。
- op 面签名漂移时安装拒绝、进程保持原生（fail-closed）。
- `npu_fused_infer_attention_score` 未来若宿主改走 v2/v4 专属入口，安装期即拒绝
  （不会半启用）——升级宿主时按 release.md §4 清单复核。
