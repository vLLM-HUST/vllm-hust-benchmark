# P1-2：专家权重截断误差全量扫描（离线）

目标：把 MMLU-Pro / PPL 上观测到的量化差异，拆到 **权重截断** 这一步，逐层、逐投影、
逐专家定位 W4A8 相对 W8A8 / BF16 的误差来源。不占卡，纯 CPU 离线。

- 范围：40 层 × 256 专家 × 3 投影 = **30,720** 个专家模块（routed experts）。
- 参考真值：BF16 原始权重（`/data/jxd/Qwen3.5-35B-A3B`），融合张量
  `experts.gate_up_proj [256,1024,2048]` 与 `experts.down_proj [256,2048,512]`。
- 被测：`/data/gcy/Qwen3.5-35B-A3B-W4A8`（int4/ssz）与
  `/data/jxd/Qwen3.5-35B-A3B-W8A8`（int8）。
- MTP 层专家在两个量化产物里都是 `FLOAT`，不参与本扫描；因此本扫描的误差
  只覆盖 40 个主层。

## 反量化口径（由数据反推，不是假设）

`probe_layout.py` 在第 0 层第 0–2 号专家上枚举了全部候选组合，用余弦对 BF16 真值判定：

| 待定项 | 结论 | 证据 |
|---|---|---|
| BF16 `gate_up_proj` 的行序 | gate = 行 `[0:512]`，up = 行 `[512:1024]` | cos 0.99994 / 0.99996 |
| int4 打包的 nibble 序 | **低半字节 = 偶数输出行**，高半字节 = 奇数行 | 该序 cos 0.9908–0.9934；反序 cos ≈ 0.03 |
| `weight_offset` 语义 | 全零（两臂抽样 abs max = 0.0），故 `w ≈ q·scale` | 符号歧义实际不存在 |
| W8A8 反量化 | `w ≈ q·scale`，`scale [out,1]` 每输出通道 | cos 0.99994–0.99997 |
| W4A8 反量化 | 同上，`q ∈ [-8,7]` 有符号 int4 | cos 0.9908–0.9934 |

`weight_offset` 全零这一点与"非对称量化"的常见假设相反：本产物实际是
**逐输出通道对称** 量化，`scale_bias` 是运行时激活量化误差的校正项
（`down_proj` 为 `[out,16]`，其余 15 列为 0），属于算子侧，不是权重误差，
留给 P1-3。

### 死通道（模型侧事实，必须先知道）

BF16 参考权重里有一部分输出通道的数值停在 bf16 最小正规数量级
**±7.824e-38**，即数值上等于 0。它们会把逐通道相对误差统计带偏（一条死通道
就是 100%）。本仓库的口径：逐通道相对误差只在"活通道"（行最大绝对值 > 1e-10）
上统计；模块级 `rel_fro` 仍按全部元素计算。

- gate/up 各 59,093 条死通道（占 1.13% 输出行），集中在 layer 0（该层每个专家的
  gate/up 都有，单模块最高 82%、中位 32%）；`down_proj` 没有。
- 两个量化产物都把死通道量化为 0（scale 被钳到 1.19e-7，即 2^-23），
  绝对误差 7.8e-38，**不是量化损伤**。

## 指标

每个模块记录（对 W4A8、W8A8 各一套）：

- `rel_fro = ||ŵ−w||_F / ||w||_F`、`cos`、`snr_db`
- 逐输出通道相对误差的 `max` / `p99` / `mean`（只在"活通道"上统计，
  见下节死通道说明）；另有 `live_rel_fro`（只在活通道上算的 Frobenius 相对误差）
- `abs_err_max`、`abs_err_rel`（除以 `max|w|`）、`row_bias_rel`
- 量化码统计：`q_absmax`、`q_clamp_frac`（贴到 ±7 / ±127 的比例）、`q_zero_frac`
- `theory_rel`：按实际 per-channel scale 计算的理论均匀舍入噪声底线
  `sqrt(Σ_i n_in·s_i²/12)/||w||`，用来判断"误差是否已经贴着该位宽的极限"
- 权重形状统计：`w_std`、`w_absmax`、`w_row_norm_cv`、`scale_cv`

聚合层（`aggregate_layer_error.py`）：

- 每（层, 投影）的**集合相对误差** `sqrt(Σ_e ||e_e||² / Σ_e ||w_e||²)`
- 误差在 256 个专家间的集中度：top-8 / top-32 份额、Gini、max/median
- 逐专家索引视图（跨 40 层平均）与最差/最好比
- 层号趋势（首 5 层 vs 末 5 层）
- 误差与权重形状的 Spearman 相关

## 复现

```bash
PY=/root/miniconda3/envs/vllm-hust-dev/bin/python   # numpy + torch, 无 NPU 依赖

# 1) 口径探针（秒级）
$PY -I probe_layout.py

# 2) 全量扫描（40 层，多进程；单层 shard 落盘，可断点续跑）
$PY -I scan_weight_error.py --workers 24 --resume

# 3) 聚合
$PY -I aggregate_layer_error.py

# 4) 二次诊断（同投影内相关、贴边码、误差方向、能量占比）
$PY -I analyze_extras.py

# 5) 谱抽样（120 个模块；务必关掉 BLAS 多线程，否则在 192 核机器上会严重过订）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -I sample_spectral.py
```

产物：`out/layer_error_experts.csv`（30,720 行）、`out/layer_error_layer.csv`
（120 行）、`out/layer_error_summary.json`、`out/layer_error_extras.json`、
`out/spectral.json`、`out/layer_error.jsonl`（原始逐模块）。

本次耗时：全量扫描 542 s（24 进程）；聚合与二次诊断各数十秒；谱抽样约 45 s。
