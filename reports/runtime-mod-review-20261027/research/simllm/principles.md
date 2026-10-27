# SimLLM：借用相似请求的状态，减少 prefill

研究日期：2026-10-26。只读源码固定于
`vLLM-HUST/vllm-ascend-simllm-hust@28f2297c58cec2249576dff2ccdd251c662ad8d6`。本轮是机制阅读，不是硬件复现、全量审计或独立质量认证。

## 核心问题与代价

长 prompt 的 prefill 要逐层构造后续生成所需的状态。若许多请求高度相似，但 token 前缀不完全一致，严格 prefix cache 只能复用真正相同的前缀。SimLLM
尝试扩大可复用范围：把一个历史请求的各层缓存状态移植给新请求，少计算一段 prompt，再继续 forward 和 decode。

它缓存的是模型内部状态，不是最终答案，也不是通过检索给 prompt 添加文本。两类复用必须分开：

- **Exact**：token 相同的前缀，且具有匹配的完整状态；语义上意图保持原计算。实际数值等价仍依赖宿主与实现。
- **Semantic / approximate**：不同 prompt 的状态借用；相似度只是启发式，不是 KV 等价证明，可能改变输出。

与 vSpec 对照：vSpec 主要减少 decode 的昂贵串行轮次，严格路径验证草稿；SimLLM 主要减少 prefill，近似路径不重新精确验证全部被跳过内容。

## 实际接入链

`vllm.general_plugins` → `plugin.register` → import hook → 包装 Ascend V1
`NPUModelRunner.execute_model` / `_model_forward`。

1. 在 worker 构建 batch 前查找历史任务。
1. 命中后协调修改 worker 侧 `num_computed_tokens`、本轮请求 token 数与总数。
1. 在 shortened forward 前恢复目标请求的物理缓存块/混合状态。
1. 执行未跳过的 prompt 部分，之后正常 decode。
1. 完整 prompt 处理结束后保存可复用快照，按任务数与字节预算淘汰。

宿主仍拥有请求调度、块分配和真正模型执行。纯 decode 步骤有快捷绕过路径，不意味着逐个生成 token 都进行语义搜索。默认缓存上限为 1024 项及 512 MiB；恢复和匹配自身也有开销。

[宿主合同](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/28f2297c58cec2249576dff2ccdd251c662ad8d6/HOST_CONTRACT.md)；[runner patch](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/28f2297c58cec2249576dff2ccdd251c662ad8d6/src/vllm_ascend_simllm/patch/patch_model_runner.py)。

## “相似”具体是什么？不能只看模块名字

### 通用 embedding / SimHash 路径

源码确有池化 embedding、随机投影 SimHash、邻近 Hamming bucket 和 cosine 筛选。但当前
`SimLLMPreprocessor.extract_embeddings` 实际读取的是模型 **input token embedding layer**，不是完整 transformer
前向后的语义表示；`embedding.py` 的泛用 docstring 不能替代调用点事实。默认 mean pooling 对词序不敏感。

### 当前 Qwen3.5 默认 tail_anchor 路径

默认 hybrid mode 为 `tail_anchor`。这条 guarded hybrid 路径的 embedding/LSH fallback 被明确关闭，匹配主要在 CPU 上完成：

- 用 `Counter(token_ids)` 得到词频向量，以 cosine ≥ 0.95 筛选。
- 解码 prompt，提取中文弯引号 `“…”` 中 2–24 字符的主题；最高频主题须无并列。
- 要求主题 anchor 和聊天模板 signature 匹配。
- `tail` 另要求 secondary context topics 一致，`tail_anchor` 不做这一额外约束。
- 可接受候选中优先最长可复用跨度，再比较相似分数；不是无条件选 cosine 最大项。

这些条件是特定格式的近重复检测，不是通用语义等价判断。例如，词序交换可以不改变 token-count
cosine，但会改变语义；在很长的共同背景中改变一个关键数字或否定词，也可能仍有很高相似度。这是由表示方式推导的风险，不是本轮测得的失败率。

[实际 embedding 调用](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/28f2297c58cec2249576dff2ccdd251c662ad8d6/src/vllm_ascend_simllm/hooks/preprocess.py)；[匹配与缓存管理](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/28f2297c58cec2249576dff2ccdd251c662ad8d6/src/vllm_ascend_simllm/kv_manager.py)。

## Qwen3.5 hybrid 状态：最值得留意的边界

Full-attention 的 KV 可按 token 前缀讨论，但 Gated DeltaNet 等 recurrent 层的终态是整个已读前缀的累积，不能从完整源终态任意截出较短前缀的状态。

因此 guarded exact 路径要求：有效 recurrent 快照、**完整 source 成为 target 的相同前缀，且 target 更长**；当前 scheduled span
也须超过源前缀。不满足就不能按这条路径复用。相同完整 prompt 不一定在这个 hybrid exact 实现中命中。

默认 `tail_anchor` 的近似命中采用不同折中：

- 恢复历史 full-attention KV；
- 将 recurrent 状态清零，而不是假装它与新 prompt 精确一致；
- 至少保留配置指定的目标后缀供重新计算，默认 512 token，实际跨度还受源长度和调度 chunk 限制。

这不是“精确修正过的缓存”：重算尾部不能证明被跳过前缀的事实与依赖已恢复。尾部的 attention 仍依赖借来的 KV，recurrent 层也没有重新读取完整前缀。

实现还允许把足够增长的近似 target 提升为新的近似缓存源。收益动机是避免复用跨度随着缓存轮转缩短；它不把近似来源变成精确来源，需要另行评估连续复用的质量风险。

## 效果证据如何读

- README 报告 default tail_anchor 的两轮指定回放吞吐分别提高 20.39%、18.78%；这些是作者限定负载上的声明。本轮未下载并复算该外部冻结回放。
- README 另有历史 4.061x 数字，属于 5000 条合成近重复请求、每条仅生成一个 token 的单次固定卡位比较；不是当前默认配置或长生成的普遍收益。
- 已读的早期 2026-09-29 报告中，5000 请求同为单 token 输出时吞吐只提高 3.2%，TTFT p95 反而上升 4.0%。不同策略/阶段的结果不能混为一个插件分数。
- 仓库 MMLU-Pro 诊断明确记录零复用事件；同分不证明近似路径无损，时间波动也不能归因为 SimLLM。
- 关闭原生 prefix cache 的对照不能单独证明胜过已合理配置的原生 APC；仍需区分“增加缓存”与“近似扩展缓存”的各自贡献。

[README（固定版本）](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/28f2297c58cec2249576dff2ccdd251c662ad8d6/README.md)；[早期回放](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/28f2297c58cec2249576dff2ccdd251c662ad8d6/benchmarks/results/qwen35_similarity_20260929/README.md)；[数据集适用边界](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/28f2297c58cec2249576dff2ccdd251c662ad8d6/docs/five-primary-dataset-applicability.md)。

## 我们当前的判断

它是“以近似状态复用换取 prefill 成本下降”的探索，有意义的目标场景是长输入、近重复请求和有机会热缓存的服务；不是一般性的无损推理加速器。对默认 Qwen3.5
路径，更准确的名字是带主题/模板门限的近重复状态复用，而非通用语义缓存。

下层最重要的问题是：**真正发生近似复用的请求，任务正确性损失是多少？** 需要改变关键数字、否定词、指令和上下文顺序的成对请求，固定阈值且与原生 APC 对照，并同时记录命中、跳过
token、质量、TTFT 与输出吞吐。这里仅提出辨别问题，未执行实验。

维护观察：活跃 worker patch 为 1837 行，交织匹配、调度记账、状态恢复和快照；本轮以函数级定位控制阅读范围。若后续修改，内部 API
和状态生命周期耦合会增加回归成本，不宜把它当成简单的入口插件。

## 共同评审 comment（2026-10-26）

Fletcher 的判断：**“它的 KV 匹配完全不靠谱，只在新闻之类的场景有点用，支撑不起任何要求精度的场景。”**

我们的采用结论：当前相似匹配不构成 KV
状态等价保证，不把这条近似路径推荐给要求精度保证的任务。新闻等容忍近似任务仅作为可能的探索方向；本轮没有新闻任务实验，不能把这句话当作已验证的适用范围。此判断针对相似复用，不把严格相同前缀复用一并判为不可用。

补充匹配粒度：默认 guarded hybrid 路径是整条 prompt 的 Counter cosine 与主题/模板门限，不是段落 diff。近似命中后跳过长度为
`min(匹配跨度 - 1, 目标长度 - 重算尾长, 本轮调度长度 - 1)`；从源开头复制物理 KV 块，未重新计算的目标前缀不要求逐 token
相同。整条输入高相似不能保护被跳过区域里新出现的关键事实。
