# MuxPD 首个 D TP collective：相同输入的形状敏感性

**本次固定边界上，实际native混合批输出A与恢复原payload形状的私有重放C逐字节一致；将相同各rank D输入缩成单行的重放B，稳定改变了每rank 2048个输出元素中的625个。**
最大绝对差为0.00390625，RMS为9.40213873039046e−05。四rank结果相同，同模式两次重放稳定，历史前缀复现与observed/plain桥全部通过。

这证明当前首层、首个seed D的TP规约边界具有相同输入下的payload形状敏感性。B/C使用原TP组，但均为排空后的私有重放；**实际prefix
graph仍未观察**。本结果不能证明07338完整答案退化已经修复，也没有新增吞吐或Goodput结论。原full200仍为native165/200、旧raw166/200、prefix167/200，prefix新增07338错误并有11次length，答案不退化门继续FAIL，运行时PR27保持实验草稿。

此前[首个GDN卷积诊断](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/f5a69f466ec134b0d6f2a7dd89d2e03af1bf7810/reports/muxpd-conv-diagnostic-20261029)得到A=B=C的负结果。本轮把观测边界推进到同一首层GDN后实际调用的原TP
`_all_reduce_out_place`。沿革中的conv v1库身份/RPC失败仍保持FAIL，独立加载smoke与conv v2结果作为绑定的外部依赖保留，不被当成本轮DTP数据。

固定配置为Qwen3.5-35B-A3B、BF16、TP4/EP4、同步、FULL_DECODE_ONLY图[1,2,4,8]、HCCL_DETERMINISTIC=true、batch
invariance关闭。物理卡0,1,2,7对应logical0..3，型号910B2；该物理分配与历史实验不同。原batch_id7的07151
seed产生恰好1个输出后释放7个peer，目标07338到208个输出时supervisor abort，原输出预算仍8192。未加入eager模型回退、case特判或延长预算。

观测组只复制实际D行的规约前输入、原调用返回后的D行及小型映射metadata。首个mixed payload实际形状为[9533,2048]、BF16、stride[2048,1]，seed
sequence与token
offset均为0。原collective只执行一次，返回alias保持不变。排空并恢复四个hooks后，使用原TP组按B0、B1、C0、C1顺序重放；每次从同一不可变D输入创建独立存储。B为[1,2048]；C保留[9533,2048]形状，其他行置零，D输入回填实际offset。每个分支必须收到四rank一致的ready回执后才调用collective。

| 比较                 | 每rank范围         | 四rank结果                                              |
| -------------------- | ------------------ | ------------------------------------------------------- |
| A实际mixed ↔ B单行   | 2048个BF16输出元素 | 625个不同；max abs 0.00390625；RMS 9.40213873039046e−05 |
| A实际mixed ↔ C原形状 | 2048个BF16输出元素 | 逐字节相同；差异0                                       |
| B单行 ↔ C原形状      | 2048个BF16输出元素 | 625个不同；max abs 0.00390625；RMS 9.40213873039046e−05 |
| B0↔B1、C0↔C1         | 完整各分支输出     | 逐字节相同                                              |
| 跨rank A/B/C输出     | 对应完整分支       | 四rank逐字节相同                                        |

C改变了其他行内容和执行时间，因此原协议要求A=C才能将B/C差异联系到实际mixed边界；本次该门通过，`actual_mixed_attribution_allowed=true`与`supports_geometry_sensitivity=true`。这仍不是对具体HCCL算法或缓存函数指针的直接观测，也不证明native与真实prefix每层输入/state相同。

observed于2026-10-29 15:19:11 UTC终态，plain于15:22:45
UTC终态；两份controller均PASS、exit0、errors为空、owned进程为空，四卡idle。四rank观测收据各确认一次native调用、四次shadow调用和恢复四个hooks。两次executor关闭日志均有grace期后SIGTERM，没有SIGKILL，也没有controller额外强制清理；不能表述为全部worker自然退出。

独立CPU审计不导入producer/analyzer，直接核验28对raw/NPY、32对历史完整保留前缀、8对observed/plain前缀、16份maps/RPC/helper回执及19份冻结producer文件。两组各八请求完整保留前缀匹配两份历史native全量运行，observer未改变本次通过208步的轨迹。每rank估计额外tensor峰值39,084,176字节，低于64MiB预算。没有将这些观测成本当成性能结果。

关键身份：冻结清单SHA `4711d530564675c5632a9e95a6d6f3c21b1041047fd4ee523c9db96b425d3a4c`；分析SHA
`7f10818956a5dd69fa06040e31a59ef270ca91e75188a7e7fd63d20f4e00be15`；独立审计摘要SHA
`faa5fff28b97050b4537f5ad4606de19f102b6fdd871edc8cb6aa3a9934b397a`。完整生产者、两次终态运行、原始张量、checkpoints、失败沿革和分析都应随原始包保留。

`external-dependencies-v1.json`明确列出48个历史依赖文件，包括两次native、一次raw和一次prefix完整控制的验证输入，以及此前conv失败/成功沿革。raw虽然不是本轮实验角色，仍是冻结analyzer要求的历史输入。模型、实际源码/库路径和历史绝对路径不全部包含在新包中，不宣称解压即可独立复跑。归档的本地完整性、远端上传及匿名公开读回须分别记录；任何发布状态都不能替代答案质量门。
