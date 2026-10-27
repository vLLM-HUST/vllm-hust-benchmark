# Prefix Router：缓存感知的多实例请求路由

2026-10-26，只读源码。HEAD `7dcb522c32862671d25065ebc840f4ab7009fa29`；未安装、未启动服务或测试设备。

## 问题与作用

多个模型服务实例各自有APC缓存。普通轮询可能将共享长prompt的新请求发给无缓存的实例，重复prefill；只追最长命中又可能把热点压到单节点。此项目提供独立HTTP
Router，在缓存命中和已在途工作间选择后端。

它不搬KV、不做语义近似、不修改attention kernel，也不是引擎内scheduler。每个TP2实例整体算一个后端，不能把两个TP
rank当成两个路由目标。所有后端都经相同HTTP路径访问。

## 缓存索引

Router使用与worker一致的tokenizer、chat template、hash algorithm和hash-block
size，调用宿主链式block哈希构建prompt指纹。通过KV事件维护各后端/DP
rank/group的缓存block哈希集合，逐块查最长连续前缀；hash包含parent链，不是对任意中间片段相似匹配。

多KV group分别计算命中，再取最小值；受prompt长度减1和block对齐限制。hash-block size与实际cache group block
size必须正确配置。不支持的指纹输入（例如cache_salt、LoRA、chat template override等）走default_backend，不能假装已安全估算。

这是路由估计，不是锁住缓存：选择后到worker执行前缓存仍可能被淘汰。实际复用仍由worker的APC决定，索引过期通常损害收益而不是让router替换错误KV。

## 两种策略

- Prefix：优先最长缓存命中，再以负载/轮询等处理平局。
- Lifecycle：选择最低启发式成本，并原子记录在途请求，防止连续到达时都看到同一个空闲节点。

默认近似成本：

```text
(已在途未完成prefill的tokens + 新请求未命中的tokens) / block_size
+ 活跃prompt去重后的block数量（含新请求新增部分）
```

两项默认权重均1。active_request_weight默认0，track_output_blocks默认false。活跃共享prompt
block通过引用计数去重，不把同一公共前缀按请求数重复收费；但这只是负载代理，不证明共享KV可以让多个query的decode计算免费。

通过HTTP/SSE响应观察请求接纳、首段生成、完成/断开，更新prefill完成状态和释放计数；不读取scheduler快照。输出长度及硬件服务率没有完整建模，权重未经时延校准，不能当作准确TTFT预测器。

## 工程实现和边界

- 每个backend独立aiohttp连接池，默认512连接；有界排队和超时，满载返回503。解决长SSE占连接时全局100连接池互相干扰的问题。
- KV事件有序号、缺口检测、replay和publisher epoch/snapshot恢复；缺口期间撤销缓存可信度。
- README说明当前0.29.1宿主仅提供无版本sequence-frame
  replay，无法证明重启身份或恢复超出环形buffer的事件。默认拒绝，`allow_unversioned_replay=true`只是best-effort部分索引，不等价于完整恢复。
- manifest仍为import_only，Extension Manager不启动/停止/健康检查此服务，enable应拒绝；用户需显式启动外部进程。安装可发现不等于部署验收通过。
- Router增加tokenization/hash、HTTP转发和中心索引/生命周期维护成本；是否划算依赖真实共享前缀、APC、事件可靠性与负载。

## 证据与初判

有实质源码，包括HTTP转发、索引、事件恢复、生命周期计数、连接隔离和测试；不是只有接口。`experiments/sharegpt`提供四worker、Prefix/Lifecycle配对、固定Poisson到达和三重复的实验脚本，但本轮未见当前README给出已完成性能结果。脚本不是实验结论，也不能把单机四worker写成多机验证。

该方向针对真实多实例部署需求，和单算子补丁不是同一层次。当前仍需补齐宿主事件恢复协议与真实端到端验收；不预写Fletcher最终comment，不声称已可生产或有定量收益。

## 固定来源

- [README](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/7dcb522c32862671d25065ebc840f4ab7009fa29/README.md)
- [prefix索引](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/7dcb522c32862671d25065ebc840f4ab7009fa29/src/vllm_hust_prefix_router/core/prefix_index.py)
- [lifecycle计分](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/7dcb522c32862671d25065ebc840f4ab7009fa29/src/vllm_hust_prefix_router/core/lifecycle.py)
- [请求指纹](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/7dcb522c32862671d25065ebc840f4ab7009fa29/src/vllm_hust_prefix_router/adapters/vllm_hust/request_fingerprint.py)
- [恢复协议](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/7dcb522c32862671d25065ebc840f4ab7009fa29/src/vllm_hust_prefix_router/events/recovery.py)
- [实验方法](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/7dcb522c32862671d25065ebc840f4ab7009fa29/experiments/sharegpt/README.md)

## 最终评审方向

Fletcher/Lumi：**缓存感知路由确实有需求，值得做；最好接入分布式KV存储，优先复用已有缓存，而不是重算。**
将缓存可获取性、搬运成本和节点负载共同用于路由，比仅依据各实例本地缓存是否命中更完整。当前实现没有因此自动获得分布式KV存储/传输能力，也不把方向认可等同于生产资格。这里只记录研究方向，不启动新增实现。[集中comment](../comments.md)
