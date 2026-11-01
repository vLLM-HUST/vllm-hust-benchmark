# B0 参考有效算力计算方法

本方法采用官方 B0 的同工况实测有效算力作为共同参考，分别计算 B0、B1 的有效算力与参考达成率。 参考实现固定为 vLLM 0.18.0 / vLLM-Ascend 0.18.0；B1
保留运行记录中的实际优化组合。 方法版本为 `project-b0-reference-compute/v1`，自 2026-11-01 的本次项目比较采用。

## 1. 计算公式

对实现 `i∈{0,1}`：

```text
F_i = Σ_g F_i,g
C_i = F_i / (k_i × T_i × 10^12)                 [TFLOPS/芯片]

C_ref = C_0
T_ref,i = F_i / (k_i × C_ref × 10^12)           [秒]
η_compute,B0,i = 100 × C_i / C_ref
              = 100 × T_ref,i / T_i             [%]

η_compute,B0,1
  = 100 × F_1 × k_0 × T_0 / (F_0 × k_1 × T_1)

有效算力提升率 = 100 × (C_1/C_0 − 1)
```

`F_i,g` 来自该运行实际输出依赖所需的运算分项；矩阵乘加计 2 次基础运算。 加、减、乘按源码计数；特殊函数、比较、除法的调用量单列，不给它们人为指定 FLOP 等价值。
两卡参与的总算量只计一次，再除以全部参与芯片数。每组使用自己的算量和对应同步计时； 统计开销保留在计时中。

B0、B1 的必要草稿、缓存状态依赖和算量可以不同。即使两组输出 token 数相同， `η_compute,B0` 也只有在 `F_0/k_0=F_1/k_1` 时才等于耗时比；不能以吞吐倍率代入。

## 2. 参考及判定

本次项目参考目标为 `η_compute,B0≥90%`，即有效算力达到官方 B0 的 90%。 判定由计算结果与目标直接比较；低于目标输出“未达标”，不裁剪到 90% 或 100%。

B0 的 100% 是参考归一化身份；B1 超过 100% 表示有效算力超过官方 B0。 本方法使用固定的实测参考，不将其称为硬件峰值或最优执行的物理上限。

V1.5 原独立参考方法使用独立标定与依赖/资源模型推导 `T_ref`。该方法的中期 50%、 最终 90% 及历史结果原样保留。本页的“已达标”专指 **B0 参考目标**，
不是将新的归一化百分比追认为原独立参考方法的验收结果。 归档的 47.5033 秒仍属于原 B1 独立参考图，不用于本方法。

## 3. 本次采用的配对

采用已完成必要算量导出和审计的官方 B0 / B1 配对，进行一次追溯复算，不新增模型成绩。 模型 Qwen3.5-35B-A3B，BF16、TP2、两片 Ascend 910B2；同一
MMLU-Pro 200 请求、 282265 个输入 token，每题上限 256 个输出 token，保留 EOS。两组均输出 51200 token。
每组独立新建服务和缓存；计时使用该组同步边界。

这是一组固定输出预算的算力工况。该预算下正确数为 B0 21/200、B1 19/200。 完整回答的 143/200、153/200 属于另一组 max8192
实验，仍用于完整回答质量、吞吐与成本。 两组记录不互换算量、时间或质量成绩。

## 4. 复算与来源

```bash
python scripts/recompute_b0_reference_compute.py --check
```

计算器从[运算分项](../reports/b0-reference-compute-20261101/accounting-parts.json)求和，
与[已审计配对](../reports/vllm-hust-midterm-20261019/m01-recovery/official018-paired-compute-20261024/summary.json)
逐项校验，再进行除法和目标判断。 [计算报告](../reports/b0-reference-compute-20261101/README.md)列出全部过程，
[生成结果](../reports/b0-reference-compute-20261101/summary.json)与网站使用相同数据。

## English summary

Use measured official B0 effective compute as the common reference: `C_i=F_i/(k_i T_i 10^12)`,
`C_ref=C_0`, `η_compute,B0,i=100 C_i/C_ref`. Work counts and synchronized elapsed time belong to
each arm; all participating chips and host accounting overhead are included. The project target is
at least 90% of this B0 reference. B0's 100% is a normalization identity; values above 100% are
valid. This is a retrospective analysis of the audited 200-request, 256-token pair, not a claim of
90% of physical peak or acceptance under the original independently calibrated V1.5 reference. The
full-answer experiment remains a separate result.
