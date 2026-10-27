# ascend180：GEMM+SwiGLU 独立算子复现

2026-10-26。验收基线为 **普通 ACLNN 分开执行（F.linear + npu_swiglu）**，不是原生 Triton。

**不合格：融合算子相对 ACLNN 时延增加 48.37%–82.31%。**

当前独立算子实测不支持以该融合实现替代普通 ACLNN 路径。没有复现端到端 +5.24%，也没有证明取消了 GM 往返。删除多余清零仍是有用的小补丁，与融合验收分开判断。

## 条件与结果

物理卡2、7分别独立测量；各卡三轮配对、每轮每项50次同步墙钟计时，10次预热排除JIT；BF16，K3584、N18944，seed20261026。使用未修改 OPset
kernel（source `7d3cf948da548220fb2e0a32bf2aff80b5647dc4`），验收基线为普通 ACLNN
`F.linear + npu_swiglu`。另加载宿主 `linear_persistent` 加 `npu_swiglu` 作为 Triton 背景对照，不作为采用判定基线。这是 eager
同步 microbenchmark，包含host launch开销，非Graph replay或HTTP性能。

每格为三轮均值平均，单位ms：

| 物理卡 |   M | ACLNN基线（ms） | 融合（ms） | 时延增加（不合格） |
| ------ | --: | --------------: | ---------: | -----------------: |
| 2      |   1 |          0.2333 |     0.4253 |            +82.31% |
| 2      |  32 |          0.2485 |     0.4432 |            +78.37% |
| 2      | 128 |          0.3246 |     0.4828 |            +48.71% |
| 7      |   1 |          0.2380 |     0.4233 |            +77.87% |
| 7      |  32 |          0.2700 |     0.4618 |            +71.01% |
| 7      | 128 |          0.3298 |     0.4893 |            +48.37% |

时延增加 = (融合耗时 / ACLNN耗时 - 1) ×
100%，按原始未舍入三轮均值计算，表内毫秒值仅用于展示。两卡、三个shape全部慢于ACLNN。卡7部分轮次有波动，原始50样本和SD完整保留，不筛掉慢轮。

背景对照：融合相对指定原生Triton路径降低约6.8%–14.0%时延，但这是较弱性能对照，不能作为相对ACLNN的验收收益；原始Triton数值保留在证据中。

所有基线/候选在三个shape、两卡上通过事前 FP32 reference 的 finite 与 `rtol=.02, atol=.125`
门槛；误差指标在JSON。这不证明任务质量，也没验证跨batch不变性。普通ACLNN不自动满足batch-invariant语义要求，不能以它更快就宣称可无条件替代该语义合同。

## 物理GM证据

实测生成的kernel元数据包含非零workspace（4096/131072/524288 bytes），源码单kernel显然不能当成“没有workspace”。

额外将M32的缓存ttadapter在同机CPU离线lowering，保存 `evidence/fused-m32-offline.hivm.mlir`：

- 195–207行：两份32×128 FP32 tile经 `fixpipe_nz2nd_float_to_float_4d_to_2d` 从 `cc` 写到 `gm` workspace。
- 257/270行：Vector执行 `sync_block_wait`。
- 264/277行：`load_gm_to_ubuf_1d_float` 从GM读入UB，随后执行激活与乘法。

**这份离线lowering明确仍经过GM，不是寄存器直连SwiGLU。**
未做设备时间线分析，因此没有证明有效Cube/Vector流水或收益来源。离线命令基于同一ttadapter和公开compiler选项；其object与实测缓存object二进制比较不相同，故不冒称此IR是实测binary的精确反汇编。

离线调用（无设备计算）：

```sh
bishengir-compile <M32-cache>/_linear_swiglu_kernel.ttadapter \
 --target=Ascend910B2 --enable-auto-multi-buffer=true \
 --enable-auto-bind-sub-block=true --enable-hfusion-compile=true \
 --enable-hivm-compile=true --enable-triton-kernel-compile=true \
 --save-temps=<task>/offline-ir -o <task>/offline-ir/kernel
```

## 资格、资源及失败披露

原始checker失败：Core `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` / Ascend
`1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8` 不匹配作者锁定版本。Torch2.10.0 / torch-npu2.10.0.post4 /
Triton-Ascend3.2.2，CANN目录9.1.0。没有修共享源码或全局包，没有绕过checker声称插件集成通过；独立算子迁移结果不替代端到端验收。

首次错误地把物理编号当runtime index，进程894046实际进入物理5，另一进程因非法device
index失败。主动终止894046，首次所有数据作废并保留日志。Fletcher强调4/5用于压力测试后，后续仅使用2/7。runtime映射为逻辑0/1/2/3→物理2/4/5/7；修正入口逐次查询并核验PID落卡，正式结果的placement文件提供证据。

正式进程均exit0，已释放物理2/7的三类协作锁；结束检查两卡无进程，4/5压力测试进程仍运行。没有停止其他人的任务。远端只保留 `/root/lumi-opset-study-20261026`
的源码、venv、cache和证据，无后台服务。

## 决策意义

- **普通 ACLNN 是本次性能验收基线；融合时延增加48.37%–82.31%，不合格。** 对较弱Triton对照的改善不能替代这一结论。
- 也不应说“融合所以省掉GM”：离线lowering反而清楚展示GM workspace及同步。
- 当前实现不是普通ACLNN路径的性能替代品；若确实要求batch-invariant Triton，局部优化仍可能有用。
- 去多余清零的机制判断不变，本轮没有另外重测其独立贡献。当前native Triton baseline仍含zeros，因此融合对比也不是剥离清零贡献后的纯epilogue消融。

[预注册及映射修正](contract.md) · [探针](probe.py) · [资源与映射门禁](launch.py) · [原始证据](evidence/)
