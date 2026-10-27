# StateAxis：异步KV恢复、自动前缀缓存、流水权重加载

2026-10-26，只读独立仓库与历史报告；未运行设备。三个项目均依赖StateAxis宿主，不是普通vLLM安装即得完整数据面的独立实现。以下评语由 Fletcher
于2026-10-27给出；属于方向与采用建议，不代表已完成新增适配或验证。

## 源码身份

- async-kv-transfer：`03143f99a5e9718f93310af8228ea349e4f45897`
- automatic-prefix-cache：`92327b78ab2d8aa717544ede3621b0f109df31fd`
- pipelined-weight-loading：`c28890ec48ed69dad9e980be3e4e6d10422465d2`

## 1. Async KV Transfer

**评语：原理正确；需适配 Qwen3.5-35B-A3B，并选择合适的缓存压力 workload 进一步验证。**

意图是host-tier命中后异步H2D恢复KV，并让其他设备工作推进，不是跨节点分布式KV传输。

独立仓库的Python主要是固定配置与Manifest，Rust `TransferCoordinator`负责：

- 按state_id、generation、snapshot SHA身份合并重复恢复；相同state/generation却不同snapshot拒绝；
- unique work和correlation分别上限64；
- begin_next发lease，commit/abort校验身份；StateAxis激活合同max_retries=0；
- 不包含ACL stream、DMA或snapshot内存分配。其commit只是协调器完成记账，不自行证明设备已经完成。

真正device stream、snapshot内存、数据发布/rollback由宿主负责。不能仅凭该Rust队列就称已证明异步DMA，也不能因其没有DMA就否认历史宿主实现。

历史R5 Qwen2.5-14B BF16、单910B2，control/candidate各3 fresh processes：host-tier-hit p50
592.597→333.716ms（−43.686%）；并发pair
makespan607.648→355.473ms；3/3候选观测到设备overlap，HBM持平。是限定组件工作负载的正结果，非通用serving吞吐；当前拆分版未继承资格，新的deterministic-worker生命周期测试不执行NPU。

## 2. Automatic Prefix Cache

**评语：有潜力，主要来自 recurrent state checkpointing；需高效异步实现，不能干扰计算。**

在具备可复用前缀的特定数据集上，recurrent state checkpointing 有望支持前缀匹配与状态复用。这里指待实现和验证的潜力，不是当前索引已提供任意前缀 checkpoint
的结论；关键是高效的异步 checkpointing，避免状态保存与搬运干扰主计算。

以内容找到可复用的已有StateAxis resident prefix state，避免调用者显式携带state身份。不是Prefix Router那样挑实例，也不自带device cache搬运。

Rust实现
`PrefixCacheIndex<V>`：namespace/salt隔离，128-token块哈希后构建root；按已有条目的长度由长到短匹配请求前缀，命中后再核对完整token序列防碰撞；宿主is_resident确认state仍有效，更新LRU。索引返回泛型state值与reused_tokens，实际复用由宿主完成。

这是已登记前缀状态条目的索引，不是自动为任意共同token前缀创建可恢复checkpoint。容量满时只淘汰未leased条目，全被占用就拒绝插入。Rust库可参数化，但当前Python准入合同固定max_entries=2/block_tokens=128，是很窄的实验配置。

历史3+3 closure：token exact、lifecycle clean，吞吐开销0.0256%；8次插入、6次淘汰，**零cache
hits**。只证明该工作负载下插入/淘汰无明显伤害，没有验证命中路径的加速。独立版未获性能资格。

## 3. Pipelined Weight Loading

**评语：很实用的小技术，建议推广到系统默认主线，做成 default-in（默认启用）。**

优化worker启动到Ready，不是运行时逐层offload/换入权重，也不是decode加速。

宿主方案是两个64MiB pinned staging
slot：读文件/增量校验下一chunk时，上一chunk异步H2D；复用slot前等event，全部pack身份/数据验证通过才事务性Ready。固定plan digest、packed
root、geometry和execution artifact一致；无ECPA授权则保留serial_pageable_sync。独立仓库没有loader内核，主要发布配置/授权合同，真正
`ResolveWeightLoadPlan/PinnedWeightLoader` 内置于StateAxis host/worker。

历史Qwen2.5-14B、同一29.54GB 48+1pack，两种page-cache条件各3+3：

| process-to-Ready中位数 |  control |           candidate |
| ---------------------- | -------: | ------------------: |
| cold                   | 344.168s | 53.895s（−84.340%） |
| warm                   | 324.809s | 36.392s（−88.796%） |

**不是纯流水消融**：control是scalar SHA + pageable + sync H2D；candidate同时改成ARM SHA2/OpenSSL SHA256 +
pinned双buffer + async
H2D。不能将84–89%全部归因于DMA重叠。两边都校验实际载入全部字节；cold/warm以私有inode和mincore验证，无全局drop-caches。TPOT门槛通过，说明该测试启动优化未伤害稳态，不代表稳态加速。

另有MiniMax layer0 EP8 1+1 loader点估计−54.748%，不是整模型serving；更新Qwen3.5 TP2 pre-sharded
checkpoint诊断Ready337.70→301.62s（−10.68%），但没有ECPA激活、未控制cache状态，不能混为同一个机制/资格。

## 综合判断

三者分别处理“选中可复用状态”“将host状态恢复到设备”“启动时加载权重”，需求有区别。前两者可在宿主层组合，但此轮未证明独立包已打通组合。

异步KV恢复和权重加载有历史特定场景正向结果；自动前缀缓存历史验收零命中，收益证据缺失。都需区分历史宿主数据面、拆分库/合同与当前端到端资格。没有证据支持把这三个统称为已生产验证的性能MOD。

## 来源

- [async源码](https://github.com/vLLM-HUST/stateaxis-async-kv-transfer/blob/03143f99a5e9718f93310af8228ea349e4f45897/native/src/lib.rs)
- [APC源码](https://github.com/vLLM-HUST/stateaxis-automatic-prefix-cache/blob/92327b78ab2d8aa717544ede3621b0f109df31fd/native/src/lib.rs)
- [weight合同](https://github.com/vLLM-HUST/stateaxis-pipelined-weight-loading/blob/c28890ec48ed69dad9e980be3e4e6d10422465d2/src/stateaxis_pipelined_weight_loading/__init__.py)
- [async历史报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-async-kv-transfer/snapshot/README.md)
- [APC历史报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-automatic-prefix-cache/snapshot/README.md)
- [weight详细历史报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/ddc4f6a3290e447260fc8f6c03ea36d3ca514e23/archive/mod-git-20261021/stateaxis-pipelined-weight-loading/snapshot/evidence/ISSUE15_WEIGHT_LOADER_RESULT.md)
