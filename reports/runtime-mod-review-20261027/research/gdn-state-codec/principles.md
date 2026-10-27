# GDN State Codec：窗口化量化递归状态

2026-10-26，只读研究；源码 `be812c6c6acf0cb628c41be8e4a6941fc84107a5`，未安装或启动硬件实验。

## 目标和机制

对象是Qwen混合模型Gated DeltaNet的固定尺寸递归状态矩阵，不是随上下文长度增长的标准attention KV cache。每head的128×128
FP32状态为64KiB；每步更新和读出需要访问此状态。目标是减少状态带宽与表示体积，也有状态快照/分支共享的CPU参考合同，不能将后者当作当前线上能力。

参考设计源于LeapQuant公式的独立重实现（未审阅该论文，本轮只描述本仓库）：p=16窗口、INT8残差、平滑/缩放向量、r=4高精度低秩补偿，以及窗口内高精度rank-one更新记录。

```text
当前状态 ≈ 量化窗口起点 + 低秩补偿 + 窗口内逐token更新（含衰减）
```

窗口内部直接从编码表示和更新记录计算投影/读出，不每token还原、重写整块FP32状态；16步边界合并记录、重新量化，完整路线还做低秩拟合。减少矩阵写回需支付每步读记录和投影、边界合并/拟合/量化成本。属于有损状态表示，不是无损压缩。

最新已测`quant_only + vector + replace + boundary_power_iterations=0`取消了低秩拟合成本，不等于完整rank4方案。参考设备边界19,456bytes/3.37x缩小只算边界表示，不包括更新buffer、scratch、原生state等；实际宿主仍分配ssm_state，不能宣传为总HBM节省3.37x。

## 系统实现

插件替换Qwen3.5 GDN的`_forward_core`，prefill仍调用原生chunk GDN，在decode首次触达slot时编码状态；后续AscendC
kernel在当前stream执行window step，每16步执行boundary refit。

- 默认shadow：原生state正常更新，另跑codec，与原生输出比较；误差标量`.cpu()`且每步日志，是观察模式，不能用于宣称加速。
- replace：以codec输出替代原生recurrent kernel；初始编码和边界成本在forward内支付，不是旁路后台压缩。decode取cache index存在`.item()`。
- eager-only、Qwen3.5 generation、128×128固定geometry；排除graph和speculative decoding。
- 当前HEAD已添加mixed/multi-request回退到原生并清空codec slot，不应重复历史“并发2必终止”为当前已测事实。
- 但源码存在待验证的状态衔接风险：replace分支跳过原生state更新，`fall_back_to_native`只清空codec
  slot，已读路径未见将最新编码状态写回ssm_state。单请求replace跑过后切入多请求/混合回退时可能读取旧dense state。这是静态审阅发现，未设备复现，不能称回退质量已通过。

## 最新历史报告（不是本轮HEAD实测）

2026-10-09的0.2.0报告绑定mod
`c7faafe98298ef620f47fcf03bf1a7b30f36bccb`、Qwen3.5-35B-A3B、TP2、910B2、CANN9.1、eager。四组328/457/648/1033-token
prompt，每cell预热1次、保留3次，输出128 tokens：

- 总耗时变化：−0.60%到+0.81%，方向混合、低于1%，没有稳定端到端收益。
- TTFT全部恶化：+2.09%到+4.43%。
- 四组输出hash全部不同；无task score，不能将hash变化等同于语义质量下降，也不能宣称等价。
- 当时真实并发2触发gate，HTTP200内嵌500，engine终止；当前有源码回退改动但无本轮验证。

当前catalog `performance_qualified=false`、`unqualified-experiment`，默认关闭。历史组件或其他候选收益不自动继承为当前版本资格。

## 初步判断

机制确实比小辅助函数补丁实质：尝试避免每token大矩阵读写，但有损量化、窗口合并和复杂状态生命周期必须共同划算。当前公开已读结果不支撑采用；不替Fletcher预写最终comment。runtime.py为868行集中处理ABI、编码slot、boundary和decode，后续若要修改应先检查这些职责的理解成本，本轮不修改上游。

## 固定来源

- [README](https://github.com/vLLM-HUST/gdn-state-codec/blob/be812c6c6acf0cb628c41be8e4a6941fc84107a5/README.md)
- [状态表示合同](https://github.com/vLLM-HUST/gdn-state-codec/blob/be812c6c6acf0cb628c41be8e4a6941fc84107a5/native/specs/GDN_STATE_CODEC_STATE_CAPSULE_M0.md)
- [窗口递归公式](https://github.com/vLLM-HUST/gdn-state-codec/blob/be812c6c6acf0cb628c41be8e4a6941fc84107a5/native/specs/GDN_STATE_CODEC_WINDOWED_GDN_M2.md)
- [runtime](https://github.com/vLLM-HUST/gdn-state-codec/blob/be812c6c6acf0cb628c41be8e4a6941fc84107a5/src/gdn_state_codec/runtime.py)
- [宿主forward](https://github.com/vLLM-HUST/gdn-state-codec/blob/be812c6c6acf0cb628c41be8e4a6941fc84107a5/src/gdn_state_codec/host_gdn.py)
- [0.2.0负向在线结果](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/819b1975f346dca514f58880ca62518850100b2d/archive/mod-git-20261022/gdn-state-codec/snapshot/evidence/qwen35-agentx-v020-expanded-20261009/ROUND-RESULTS.md)

## Fletcher 补充的既有工作与最终 comment

该方向此前已在 BetterScale 做过详细的 LeapQuant 实验；据 Fletcher 回顾，同学将这项工作集成成当前 MOD。核心架构障碍是 Ascend AIV/AIC 无法共享
SRAM，协作与数据搬运成本使收益不稳定。

**最终 comment：参考 BetterScale 已有 LeapQuant 实验；此前做过相当详细的工作，拿不到稳定收益。** 本轮不重复读论文、不重做实验。上述历史关系与架构归因来自
Fletcher 的工作回顾，尚未在本轮重新查阅对应实验材料，不编造实验路径或数据。[集中 comment](../comments.md)
