# B0、B1 有效算力与相对提升

分别计算两组有效算力，再以官方 B0 作性能比较基线。版本 `b0-effective-compute-comparison/v2`，2026-11-02 订正。

## 1. 算力与比较公式

```text
F_i = Σ_g F_i,g
C_i = F_i / (k_i × T_i × 10^12)       [TFLOPS/芯片]
R_C,i = 100 × C_i/C_0                 [%]
G_C = 100 × (C_1/C_0 − 1)             [%]
R_C,1 = 100 × F_1 × k_0 × T_0 / (F_0 × k_1 × T_1)
```

每组使用自己的必要算量、全部参与芯片数与同轮同步耗时。矩阵乘加计2次基础运算；基础标量按源码计数，特殊函数等调用量另列。统计开销保留在原计时内。只有 `F_0/k_0=F_1/k_1`
时，算力比才等于耗时比。

`R_C,0=100%` 是归一化身份，`R_C,1=184.9705%` 表示 B1 算力提升84.9705%。这些比较值不产生考核达标标志。

## 2. 3.4.1 的参考与验收

按 [V1.5 原文](../reports/vllm-hust-midterm-20261019/milestone-v15/project4-v1.5.json) 第302—312段：

```text
η_i = 100 × T_ref,i/T_i
C_ref,i = F_i/(k_i × T_ref,i × 10^12)
η_i = 100 × C_i/C_ref,i
```

参考时间由相应精度和运行条件下独立标定的计算、访存、通信能力，结合必要工作量、阶段依赖、资源重叠与必要调度推导。两组采用相同参考方法；工作量或执行路径不同则分别核算参考时间。方法和适用范围在正式测量或复算前确定，不以B0、B1实测耗时反向拟合。

中期条件为有效且同口径的 F、无插桩同步 T 和 T_ref，以及
η≥50%；最终90%沿用其对应阶段要求。B0实测算力比值不能替代该效率。当前配对保留了统计开销，也没有两组完整的同口径独立参考验收记录，因此本次3.4.1计量验收未通过。

## 3. 订正记录

撤回v1中“达到B0算力90%即可达标”的门槛和两组达标结论。该条件允许算力比B0低10%，不适合作为本优化项目的验收标准。不能把门槛改成100%、提升90%，或依据本次184.9705%倒推一个能通过的阈值。

原始测量和算量保留；[v1报告](../reports/b0-reference-compute-20261101/README.md)仅供追溯，当前结果见[订正报告](../reports/b0-compute-comparison-20261102/README.md)。本次没有新增模型成绩。

## 4. 工况与复算

Qwen3.5-35B-A3B，BF16、TP2、两片910B2；MMLU-Pro200请求，282265输入token，每题输出上限256 token、保留EOS。两组均输出51200
token；正确数B0 21/200、B1 19/200。完整回答143/200、153/200属于另一组max8192实验，分别保留。

```bash
python scripts/recompute_b0_reference_compute.py --check
```

计算器从[运算分项](../reports/b0-reference-compute-20261101/accounting-parts.json)求和，核对[原配对](../reports/vllm-hust-midterm-20261019/m01-recovery/official018-paired-compute-20261024/summary.json)，生成当前报告和网站数据。它不生成目标值、目标差距或达标布尔值。

## English summary

Compute each arm as `C_i=F_i/(k_i T_i 10^12)` and report its descriptive baseline ratio
`R_C,i=100 C_i/C_0`. The measured gain is84.9705%; no acceptance verdict follows from that ratio.
The former90%-of-B0 target is withdrawn because it permits a regression. Outline efficiency remains
`η_i=100 T_ref,i/T_i`, with independently calibrated resources, actual dependencies, matched work
and uninstrumented synchronized timing. Preserve the original50% midterm and corresponding90% final
criteria; do not fit the reference to measured performance or substitute a new gain threshold.
