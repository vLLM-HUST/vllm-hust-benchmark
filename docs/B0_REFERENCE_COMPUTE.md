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

参考采用 **B0 阶段标定与依赖约束模型**。固定官方 B0
的版本、精度和运行条件，标定必要计算、访存、通信及主机处理的阶段时间；记录输入形状、实际工作量、数据分布和全部参与资源。先复用已有标定，补充缺失的形状或阶段。

串行且互不重叠的阶段直接累加；跨卡并行与计算/通信重叠按数据依赖、资源占用及既定调度顺序计算完工时间。对包含必要资源顺序边的无环依赖图：

```text
finish(v) = duration(v) + max_{u in predecessors(v)} finish(u)
T_ref = max_{v in sink_nodes} finish(v)
```

无前驱节点的起始时间为0。每项必要计算、搬运、通信和同步计入一次；阶段时间已包含的访存不重复叠加。必要主机处理和调度按实际路径标定，可消除的排队空洞、冗余工作及采集开销另列。阶段划分、标定统计规则、依赖关系与适用范围在正式测量或复算前固定。

F、T、T_ref对应同一测量区间和工作量。采集记录用于识别路径、核查算量及阶段标定；验收T使用加载和预热后的无插桩同步墙钟时间，保留完整执行区间内的真实开销。B0、B1采用同一参考方法，按各自工作量核算T_ref；精度或路径差异对应的标定和映射分别归档。

**本次里程碑要求 η≥50%，即 T≤2T_ref。后续90%目标对应 T≤T_ref/0.9≈1.111T_ref。**
两个阶段均衡量同工况参考效率。C_ref是固定B0阶段能力及执行约束下的工况参考值；采用更快实现后可超过该参考，η按实值报告。硬件MFU、设备利用率和B1相对B0的算力提升分别报告。

本页确定参考计算方法，新的阶段标定结果另行归档。现有256-token配对含统计开销，缺少两侧完整的同口径参考验收记录，当前3.4.1计量验收未通过。原实测F、T和算力比较值保持原记录。

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
and uninstrumented synchronized timing. For the current milestone require η≥50% (T≤2T_ref); the
later90% target means T≤T_ref/0.9. Calibrate necessary B0 stages, then apply data dependencies,
resource ordering and overlap. Sum only serial nonoverlapping stages; include necessary host,
transfer and synchronization work once. Freeze the stage model and aggregation before formal
calculation. This workload reference is scoped to calibrated B0 capabilities and execution
constraints; hardware MFU and B0-relative gains are separate.
