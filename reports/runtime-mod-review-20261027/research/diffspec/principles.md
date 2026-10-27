# DiffSpec：缩短 Eagle3 草稿上下文的投机解码

2026-10-26，固定源码 `1294c95b6d9b2d9cbed0d0cffe31202fe5a6a254`。本轮阅读 proposer、runtime 关键函数、打分 kernel
与资格报告；未实测，未独立审计所有验证分支。README 的 SC26 标签按作者声明记录，不作为独立核实的发表状态。

## 动机和机制

长上下文增加 target decode 成本，也会增加 draft 的 attention 成本。DiffSpec 尝试让一层 Eagle3 draft 只看精简的历史工作集，降低猜测成本，同时借
target 的 attention 信号选取可能有用的远端上下文，避免只保留最近窗口导致接受率下降。

1. 周期性选取每个请求最后已提交的 target query 及其 softmax LSE。
1. Triton kernel 直接扫描选定 target attention 层的 paged K，以 Q·K 与 LSE 计算 chunk relevance；不物化完整 attention
   矩阵，但仍有额外 K 读取和打分计算。
1. 按分数选历史 chunks，保留最新 chunk，检索间隔内滚动更新尾部。配置默认 chunk=64、budget=2048、初始 interval=4；README 架构段仍写
   32-token，不应覆盖配置事实。
1. 按所选 token 位置，从 **draft 自己的 canonical pre-RoPE K/V** gather 数据，再以局部位置重新施加 draft RoPE，写入 compact
   working cache。不是直接把 target 的 K/V 拷进 Eagle3。
1. draft 生成候选，由 target 验证；另有深度自适应和显式 tree 分支。

主要近似发生在 proposer 上，不像 KVCompress 那样删掉 target 参考历史。设计意图是保留 target 验证语义；不能仅凭这一点声称全部后端和采样模式已证明逐 token
等价。Tree 路径还涉及 target attention mask、验证与接受路径提交适配，不是完全不触碰 target 执行实现。

## 实现边界与开销

- 两份 draft 视图：canonical 原始 KV 与 compact working KV。缩小工作集不等于删除全部 canonical 状态或同比缩小总内存。
- 工作缓存重建实际执行 index_select、RoPE 和写入，不只是改索引。
- 周期检索存在 `.to('cpu').tolist()` 用于比较 chunk 集合，单请求深度反馈存在 `.item()`；有 device→host 反馈同步成本。
- 线性深度反馈代码只在 `valid_counts.numel()==1` 时更新，连续高接受周期后增深、连续低接受周期后回退；不能宣称任意并发下都进行逐请求最优预算控制。
- 活跃 runtime 资格限定 Eagle3 单层、TP4、PP1、graph、BF16，且禁用 APC 和 async scheduling；不是任意模型通用投机框架。
- `runtime.py` 1883 行承担检索、状态、树与 attention 路径，本轮只沿相关函数定位；后续修改需计入耦合与回归成本。

与 vSpec 区别：vSpec 首轮阅读的重点是动态 gamma/吞吐反馈控制；DiffSpec 更显著的实现特征是 target-guided draft context selection 与
compact draft cache。两者都包含自适应内容，不是互斥分类。

## 已读到的结果

历史 Llama-3.1-8B、单层 Eagle3、TP1、batch=1、greedy 的 decode TPS 报告有正向收益，但排除了 TTFT，不能搬到当前 Qwen TP4 lane。

固定 Qwen3.8-27B + VirVen/Qwen3.5-27B-EAGLE3-v2 的 TP4 graph 验收报告记录：

- 接受 103/534 draft tokens，19.29%。
- output throughput P50：target-only 47.72 tok/s，DiffSpec 14.00 tok/s。
- TTFT P50：0.314 s → 0.459 s。
- 两轮确定性 10 请求 suite 中 9/10 与基线文本完全一致，剩余解释题被记录为语义等价；不是完整逐 token 证明。
- 并发、取消、恢复、5425-token context 等功能检查通过；该配置不推荐作为加速。

报告固定的是历史候选 `c78f55c7...`，不是本轮 HEAD 的独立重测。缺少相同配置下对原生 Eagle3 的消融时，也不能从 target-only 对比归因出 context
selection 自身的净贡献。

## 初步判断

这是“让草稿便宜且仍有足够接受率”的长上下文投机研究，不是无损 target KV 压缩。它有清晰的收益机制，也有检索、复制、重定位、同步与额外验证成本；已读当前 Qwen 配置收益为负。我们的最终
comment：**没有用，草稿接受率太低了。** 针对上述已测配置，不采用、不继续投入；不扩展为所有 draft 配对的结论。

## 固定来源

- [proposer](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec/blob/1294c95b6d9b2d9cbed0d0cffe31202fe5a6a254/diffspec/proposer.py)
- [runtime：accept_target_queries / \_rebuild_working_cache](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec/blob/1294c95b6d9b2d9cbed0d0cffe31202fe5a6a254/diffspec/runtime.py)
- [paged chunk kernel](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec/blob/1294c95b6d9b2d9cbed0d0cffe31202fe5a6a254/diffspec/kernels.py)
- [历史 TP4 资格与负向结果](https://github.com/vLLM-HUST/vllm-ascend-hust-diffspec/blob/1294c95b6d9b2d9cbed0d0cffe31202fe5a6a254/docs/evidence/sage-mate-20260904-tp4-graph.md)
