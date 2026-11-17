# P1-3 反量化 / 算子数值探针

单卡（卡 7）算子级探针。目的：把 W4A8 与 W8A8 在 MoE 专家层上的误差**拆成四份**，
而不是只留一个 PPL/MMLU-Pro 分数：

| 代号 | 含义 | 怎么造的 |
|---|---|---|
| **B** | BF16 权重 × 浮点激活 | 直接读 BF16 的 `gate_up_proj`/`down_proj`，fp32 einsum |
| **A** | 反量化 int4/int8 权重 × 浮点激活 | 离线反量化 `w = q·scale`，fp32 einsum |
| **A2** | 反量化权重 × 反量化激活 | 激活先走 `npu_dynamic_quant`（int8 逐 token）再反量化 |
| **A2q** | A2 + 中间激活也被编成 int8 | swiglu 输出做理想逐 token 对称量化后再做 down 投影 |
| **C** | 运行时内核的真实输出 | 用**与运行时逐字节相同**的打包/scale/scale_bias 调 `grouped_matmul_swiglu_quant_v2` + `npu_grouped_matmul` |

归因公式（全部是 rel_fro）：

```
权重截断      A − B
激活量化      A2 − A
格式强加下界  A2q − A2      （中间激活必须是 int8，W4A8/W8A8 都有）
算子实现误差  C − A2q       （gmm1 用 C − quantfloor）
合计（vs A2） C − A2
合计（vs BF16）C − B        ← 这一项才是真正落到模型上的偏差
```

## 文件

| 文件 | 作用 |
|---|---|
| `run_probe.sh` | 环境启动器（CANN 9.1 / ATB 9.1 / custom_transformer，`ASCEND_RT_VISIBLE_DEVICES=7`） |
| `smoke_ops.py` | 前置冒烟：卡可见、bf16/int8、`enable_custom_op()`、三个融合算子的 schema |
| `st_io.py` | numpy-only safetensors 读取（与 P1-2 同一份） |
| `probe_gmm.py` | 主探针：9 层 × 4 专家 × 3 种输入分布 × 2 臂，写 `out/probe_raw.jsonl` |
| `probe_scale_bias.py` | 附加探针：scale_bias 乘 α ∈ {0,.25,.5,.75,1,1.25,2}，写 `out/scale_bias_sensitivity.json` |
| `aggregate_probe.py` | 汇总成 `out/dequant_probe.json` + 四个 CSV |

数据文件：

| 文件 | 内容 |
|---|---|
| `out/probe_raw.jsonl` | **权威数据**：修复后的 54 行全量扫描 |
| `out/probe_raw_pre_sbfix.jsonl` | 修复前的同一次扫描，留档；只有 `no_scale_bias.gmm1_C_vs_A2` 一列失效 |
| `logs/probe_gmm_rerun.log` / `logs/probe_gmm_prefix.log` | 对应的两次运行日志 |
| `out/scale_bias_algebra.json` | **负结果**：试图从外部反解 assist 项的代数结构，全部候选形式失败（cos≈0.05），已放弃，仅留档 |

## 跑法

```bash
./run_probe.sh smoke_ops.py
./run_probe.sh probe_gmm.py --probe-scale-bias --validate
./run_probe.sh probe_scale_bias.py
python3 aggregate_probe.py            # 纯 numpy，不需要 NPU
```

主探针全量（54 行）在卡 7 上约 3 分钟；α 探针（70 行）约 2 分钟（大头是读盘与 numpy 参考）。
`run_probe.sh` 的缓存落在本目录 `cache/`，不碰 `/data/jxd/cache` 里其它实验。

## 权重载荷是"运行时同款"，不是近似

`w4a8_payload()` / `w8a8_payload()` 逐字复刻 vllm-ascend 的 MoE 权重构造
（`vllm_ascend/utils.py:maybe_trans_nz`、`quantization/methods/w4a8/w4a8.py`）：

* **W4A8**：`transpose(1,2)` → `npu_format_cast(FRACTAL_NZ)` → `.view(torch.int32)` 打包；
  `w13 [E,2048,128]`、`w2 [E,512,256]`；scale 以 **int64 位模式**传入
  （`w13 [E,1024]`、`w2 [E,1,2048]`）；`scale_bias` 按运行时口径合并
  （w13 取 `[C,1]` 转置求和，w2 沿第 1 维求和 → `[E,2048]`）。
* **W8A8**：`transpose(1,2)` → NZ cast，不打包 int32；`w13_scale` fp32，
  `w2_scale` 必须是 **bf16**（运行时按 `params_dtype` 分配），`output_dtype = w2_scale.dtype`。

正确性校验：如果打包/反量化口径错了，内核输出与离线参考不可能落在
**1.5%** 这个量级（这 1.5% 基本就是强制 int8 输出编码的下限 1.7%）。
`--validate` 每次都会打印 `h1` 与参考激活的 absmax 是否一致（逐位相同）。

## 踩过的坑（复现时不要再踩）

1. `torch.ops._C_ascend` 是**惰性命名空间**：`dir()` 看不到动态注册的算子，
   必须先调 `vllm_ascend.utils.enable_custom_op()`，再用 `hasattr` 判断。
   schema 用 `str(op.default._schema)` 取（`FunctionSchema` 没有 `__dict__`）。
2. `group_list` 必须 **int64**（int32 直接报 `groupList not implemented for DT_INT32`）。
   W4A8 v2 算子收**计数**（`group_list_type=1`），W8A8 的 nz 算子收**累加和**（type 0）。
3. W8A8 的 gmm2：`scale` 若是 fp32 会报 `scale[0] dtype FLOAT and y dtype BF16 not supported`，
   `y` 若是 fp32 会报 `y[0] FLOAT not supported`；两个都得是 bf16。
4. A8W4 场景 `weight_assist_matrix` **不能为 None**（`must not be nullptr`）。
   想测"关掉 scale_bias"只能传全零矩阵。
5. **`npu_grouped_matmul` 会原地改写它的 int8 输入 `x`**。实测把 gmm1 的 `h1`
   （int8 `[128,512]`）传给 gmm2 后，`h1` 有 92% 的码字被改、幅度约 3.5×。
   所以任何 gmm1 指标都必须在调 gmm2 **之前**落地。
   这个坑一开始伪装成"α 曲线在 α=1 处也不为 0"，排查花了一轮：
   `probe_scale_bias.py` 和 `probe_gmm.py` 的 `--probe-scale-bias` 分支都曾中招，
   两处都已修复（见两文件里同名注释）。旧数据留在
   `out/probe_raw_pre_sbfix.jsonl`，其 `no_scale_bias.gmm1_C_vs_A2` 一列作废
   （8.57 → 修复后 1.69），其余列不受影响。

## 边界

* 激活是**合成**的（高斯 / t3 重尾 / 加尖峰三种），不是从模型里抓的真实 hidden state。
  探针指标对整体缩放不敏感（对称量化与 bf16 都是相对量），所以只有**分布形状**是假设。
* 覆盖 9 层（0,5,10,15,20,25,30,35,39）× 4 专家（0,64,128,192）× 32 token/专家。
  这是"代表性抽样"，不是全量；全量权重侧扫描是 P1-2。
* MTP 层专家在两个产物里都是 FLOAT，不在量化影响范围内。
* 这里量的是**单层模块**误差，不含跨层的误差传播；PPL/MMLU-Pro 的端到端影响是另一层实验。
