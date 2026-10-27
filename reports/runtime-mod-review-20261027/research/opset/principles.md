# OPset：局部算子消冗余与融合

2026-10-26 只读研究。源码 `7d3cf948da548220fb2e0a32bf2aff80b5647dc4`，发布说明 0.3.2；未安装、未独立跑 NPU。Fletcher
初步认为值得看，本轮不预写最终采用 comment。

## 当前实际上有两项优化

### persistent-matmul-empty

将 Ascend batch-invariant Triton `linear_persistent` 的输出分配由 `torch.zeros((M,N))` 改为 `torch.empty`。前提是
kernel 写满所有有效输出，原先清零是多余的 kernel launch 和内存写入。没有改变 GEMM 算术，不是压缩、丢 token 或近似缓存。

通过 lazy import hook 定位模块，用 AST 替换函数中目标调用；检查单个 persistent kernel launch
等结构。该结构检查不是对任意未来内核完整写入的形式证明，仍需固定版本兼容检查和 tail correctness。作者报告 36 shape/dtype cases、108 launches，含
FP16/BF16/FP32 和非整除边界。

### linear-swiglu-graph

FX pass 匹配 `npu_swiglu(unquantized_gemm(x, gate_up_weight, None))`，替换为注册的 custom op：无 bias、非量化路径。

```text
原生：gate/up GEMM → 写 [M,2N] 中间张量 → SwiGLU 读回 → 写 [M,N]
融合：同一 Triton kernel 内累加 gate/up → gate*sigmoid(gate)*up → 写 [M,N]
```

K 循环共享输入 tile 的载入，分别计算两组 FP32 accumulator；激活在 kernel 内完成，消掉显式框架级中间张量和独立 SwiGLU 调用；**不能据此断言消除了物理 GM
往返**。910B 的 Cube→Vector 数据交换仍可能由编译器插入 GM workspace 和同步。数学目标相同，但中间舍入/累加与原生不同，不宣称逐 bit 恒等。

只对 `M <= 128` 走融合，M 是展平后的 token 行数，不简单等于请求数。FP16/BF16；BLOCK_N=128、BLOCK_K=256。大 M custom op 回退
`F.linear + npu_swiglu`。历史大 tile 曾 CBUF overflow，大 M 多 tile 曾出错，因此不是任意 prefill 可用的融合。

## 集成和开销位置

环境变量选择独立优化；general plugin 安装 lazy import hook，另有 extension bundle 管理入口。图匹配发生在编译期，不是每个 decode step 在
Python 中重新搜图。实际 fused kernel 仍是 forward 中的计算，但它替代原有计算，非在旁边追加打分/搬运工作；不是以异步掩盖成本。

声明及实测资格范围是 Graph + batch-invariant Triton，不扩展到普通 ACLNN MatMul、量化 linear 或任意 MoE 路径。plugin.activate
本身只检查优化选择变量，不能误称它单独强制校验 batch-invariant 开关；部署和 pattern/目标路径共同决定是否命中。

## 已读历史证据（作者测量）

固定证据在 benchmark `819b1975f346dca514f58880ca62518850100b2d` 的迁移快照；迁移没有新验收。

- 去清零：ZerosLike calls 59,365→1,057；device share 2.4809%→0.0836%；该 profile 总 device operator time
  −2.7218%。不是自动等于线上吞吐提升。端到端三轮数据存在第一轮 baseline 冷启动异常，不能直接拿均值做因果生产结论。
- 融合微基准：Qwen2.5-7B MLP，K=3584、N=18944，BF16；M=1/32/128 的算子时延分别降低 15.93%/17.92%/16.79%。
- 融合端到端：Qwen2.5-7B-Instruct BF16、910B2、CANN 9.0.0、Graph、batch-invariant，ShareGPT 64 请求、并发4、输出128；8请求
  workload warmup 后三轮配对。TPOT 19.3481→18.6088ms（−3.82%），output throughput
  187.5615→197.3891tok/s（+5.24%）。六轮64/64成功，匹配正确性 prompts 输出相同；不是大规模质量保证。
- 无 workload warmup 时 JIT 曾导致负收益；较大 M 出错/溢出的负证据保留。
- 包当前 verified-line 写 CANN 9.1.0-beta.3，融合历史报告写9.0.0；区分历史 run 与当前兼容元数据，不拼成同一配置，也不相加两个优化的收益。

## 初步工程判断

去多余清零有明确减少实际工作的机制；融合有范围限定的端到端正收益报告，但物理访存减少和流水重叠尚未证实，不能以源码级融合直接推出。但首先要确认目标部署真的走该 Triton 路径；不能为了拿此路径内
+5% 就假定切换整套后端优于当前最佳原生配置。去清零尤其适合上游原地修复；融合适合编译 pass/kernel 层吸收，插件可作为验证与独立开关载体。

## 固定来源

- [源码 README](https://github.com/vLLM-HUST/vllm-hust-opset/blob/7d3cf948da548220fb2e0a32bf2aff80b5647dc4/README.md)
- [去清零实现](https://github.com/vLLM-HUST/vllm-hust-opset/blob/7d3cf948da548220fb2e0a32bf2aff80b5647dc4/src/vllm_hust_operator_optimizations/optimizations/persistent_matmul_empty.py)
- [融合 kernel](https://github.com/vLLM-HUST/vllm-hust-opset/blob/7d3cf948da548220fb2e0a32bf2aff80b5647dc4/src/vllm_hust_operator_optimizations/optimizations/linear_swiglu_kernel.py)
- [图匹配](https://github.com/vLLM-HUST/vllm-hust-opset/blob/7d3cf948da548220fb2e0a32bf2aff80b5647dc4/src/vllm_hust_operator_optimizations/optimizations/linear_swiglu_fusion_pass.py)
- [清零历史报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/vllm-hust-opset/snapshot/docs/optimizations/persistent-matmul-empty.md)
- [融合历史报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/vllm-hust-opset/snapshot/docs/optimizations/linear-swiglu-graph.md)

## Cube/Vector 物理路径复核与评审（2026-10-26）

Fletcher 指出：Ascend Cube/Vector 协作需过 GM，除非形成有效流水，否则不能套用 GPU 寄存器 epilogue 的收益解释。此点纠正了本轮最初将源码级融合等同于消除
GM 中间读写的错误表述。

AscendNPU IR 官方 CV 文档说明，910 系列 Cube 结果经 fixpipe 从 L0C 写到 GM，再供 Vector 使用；编译器可插入 GM workspace。因此一个
Triton kernel 内的 SSA accumulator 并不证明物理片上直连。源码虽以 tile 为单位计算两路 dot 后做 sigmoid/mul，但没有本轮已检查的 lowered
IR/设备 trace 来证明 Cube/Vector 流水或具体搬运量。

收益仍可能来自减少 launch/host dispatch、改变 GEMM tiling，以及编译器生成的 tile
流水/局部缓存行为；这些是待验证解释，不是本轮事实。报告的约16–18%微基准收益和5.24%端到端吞吐增幅只能表明指定对照下整体实现较快，不能归因成“省掉 GM”。收益归因需要同 GEMM、同
tiling 对照及 lowered IR/设备时间线，不启动额外设备实验。

我们的当前判断：**GEMM 多余清零删除有用；GEMM+SwiGLU 融合收益机制存疑，保留指定配置下的实测报告，不认可未经证明的免 GM 搬运解释。**

[AscendNPU IR：Cube–Vector optimization](https://ascendnpu-ir.readthedocs.io/en/latest/sources/developer_guide/features/CV/CVOptimization.html)

## 本机后续实测

[ascend180 两卡算子复现](reproduction/README.md)：物理2/7上融合相对原生Triton时延降低6.8–14.0%，但耗时仍是普通ACLNN的1.48–1.82倍。原始插件兼容检查版本不符，非端到端验收；离线M32
lowering显示GM workspace和Cube→Vector同步，不证明取消GM往返。首次映射错误被停止、数据作废，详情完整披露于报告。

## 最终采用判断

Fletcher/Lumi：**融合不采用；去多余清零有用，但属于应上游修复的小补丁，不值得独立做成一个 MOD。**
普通ACLNN分开执行在本轮测试更快，仅击败较慢Triton基线不能支撑我们的部署投入。保留batch-invariant合同差异，不据此断言作者刻意挑弱基线。[共同评审](../comments.md)
