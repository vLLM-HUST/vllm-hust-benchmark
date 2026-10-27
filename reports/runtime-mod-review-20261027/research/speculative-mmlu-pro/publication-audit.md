# 发布前证据审计：35B vSpec、DiffSpec 与 OPset

2026-10-27。审计只读取既有运行原件、回收运行目录源码/输入并离线重算，没有追加设备实验。本文修正此前“性能收益混入graph runtime优化”可能造成的过度因果解读；不改变原始结果。

## 核心结论：整体配置对照成立，单因素归因不成立

35B比较的是**宿主原生固定MTP2**与**vSpec adaptive + refill +
动态宽度runtime整套配置**。不是“原生不开graph、插件开graph”：两臂均配置FULL_AND_PIECEWISE、async scheduling、APC、chunked
prefill；TP2+EP、BF16、B16和输出上限相同。

| 已核验差异      | 原生MTP2              | vSpec adaptive                                                  | 可支持的判断                                        |
| --------------- | --------------------- | --------------------------------------------------------------- | --------------------------------------------------- |
| 插件            | `VLLM_PLUGINS=ascend` | `ascend,vspec`                                                  | 基线没有加载vSpec                                   |
| MTP预算         | 固定2                 | 上限6，候选2/4/6                                                | 并非相同固定预算                                    |
| Capture buckets | 7项，3至48            | 48项，1至112                                                    | 两臂按各自宽度生成capture配置，并非同一bucket集合   |
| 调度            | 宿主原生              | adaptive online + refill8，cohort refill启用                    | 未隔离refill与gamma选择的贡献                       |
| Graph runtime   | 宿主固定宽度路径      | adaptive graph-key、dispatch、query-width及多gamma runnable支持 | 存在配置/实现差异；无逐hook消融，不能量化其性能贡献 |
| 私有兼容层      | 不加载vSpec兼容层     | 加载API/动态宽度适配和统计补丁                                  | 这是迁移后制品，不是未经修改的官方制品验收          |

配置见
`q35-engine-configs.json`、`q35-environments.json`；15个有效运行原件的command/config/exit/result均随证据保存。12个35B正式运行的command环境与冻结设置逐项核对；confidence
margin为空，解析为None；未启用synthetic acceptance。实际序列仍有变化，不将严格配置等同于已证明数值/任务无损。

**正式可发布表述：未通过控制变量分离自适应策略、refill、动态宽度graph/runtime支持与兼容层的独立贡献。1.17×是整套配置相对宿主原生MTP2的观测耗时比，不是自适应策略的因果收益。**

**不能发布为事实的表述：额外graph优化已经被证明造成提速；原生MTP没有开启graph；本次35B提速来自放宽接受阈值。**
同样不能因缺少消融否定整套配置测得的速度差。某些runtime改造本就是实现动态gamma所需，未证明是可独立移植到原生MTP的通用优化。

## 源码依据与开关边界

vSpec固定源码：`4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae`。回收运行目录的 `src/` 与该上游tarball逐文件比较，仅
`adaptive/runtime.py` 两处不同，与保存的 `vspec-stats-compat.patch`
一致：缺失逐位置尝试计数时不伪造字段；空调度步丢弃不存在的反馈。运行目录的harness、兼容层和配置与本地保存字节一致。这是发布时回收比对，不冒称拥有每次运行时的完整源码快照。

- [patches.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/patches.py)：adaptive开关驱动
  `apply_adaptive_patches`，MTP分派到 `apply_mtp_patches`。
- [adaptive/runtime.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/adaptive/runtime.py)：`_add_adaptive_decode_graph_keys`、`_initialize_full_graph_runnables`、`apply_adaptive_patches`
  中的dispatcher/capture/query-width适配。配置开启adaptive full graph；没有逐hook执行计数或阶段性能剖析，不能将每个分支都写成实测生效并有收益。
- [adaptive/refill.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/adaptive/refill.py)：`install_online_refill_patch`由cohort
  refill开关安装调度钩子；本轮开关为1。
- [backends/mtp.py](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/backends/mtp.py)：`mtp_strict_graph`主要为FULL
  graph及eager fallback检查，不应单独称作提速优化；`HUST_VSPEC_MTP_DEVICE_COUNTS`默认0且正式command未设置；local argmax
  reduction在两臂配置均为false。draft parallel graph updates和merged full开关也为0，不可将这些列为本轮加速来源。

## 离线重算结果

- 对15个有效运行重新提取答案，逐题验证prediction/correct/invalid、选题ID/顺序、输出长度及退出码；所有结果与保存记录一致。12个35B运行、3个DiffSpec短对照均无抢占计数增量和连续token0退化。
- 35B质量203/280、202/280、193/280；截断92/94/93。配对区间按原冻结方法重算，仍跨0。原生性能重复40/40/37，因果归因仍受后端变化和截断限制。
- 三轮56题耗时中位数比：adaptive相对target-only为1.53167，相对MTP2为1.17243。输出序列及长度不同，不是等输出kernel加速，不应称作vSpec策略“提升50%”。
- DiffSpec有效B16/C04首轮：target-only/Eagle/DiffSpec分别232.12/95.58/91.34 output tokens/s；四rank
  witness显示实际DiffSpec proposer/cache，但默认linear检索计数仍为0。无完整三臂280题结果。
- OPset按两卡原始三轮mean_ms重算，融合比ACLNN时延增加48.37475%–82.31454%，发布取48.37%–82.31%。此前48.36%–82.30%由表中已舍入值计算，现改为原始精度；采用结论不变。该项是eager独立算子诊断，不是正式graph服务成绩。

## 未被本次审计补足的证据

未追加runtime单因素消融、无充分输出预算的质量复测或非确定性根因实验；未补全DiffSpec280题/三轮。目标模型仅有本地路径、config/index及文件大小/mtime身份，不宣称不可变上游权重revision。选题是14学科每科20题的screen，非完整MMLU-Pro。各MOD未独立测量的采用意见仍是评审判断，不是测量事实。不得作为官方榜单或V5.5验收结果。
