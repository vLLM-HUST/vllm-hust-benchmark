# Core Attention Boundary：CPU batch 元数据边界辅助函数替换

2026-10-26；只读源码，未安装、未开展性能或NPU测试。仓库 `xmdhb/vllm-hust-core-attention-boundary`，HEAD
`b5bc92a513a4ad043e39c7e2e3d6c4f4351114ff`。

## 做什么

不是 attention kernel、不是上下文/注意力稀疏边界，也不切分或调度batch。它替换 Core 中两个辅助函数：`split_decodes_and_prefills` 与
`split_decodes_prefills_and_extends`。

前提：batch已经按 decode→short_extend→long_extend→prefill 重排。输入是每个请求本轮query token数的累积偏移
`query_start_loc_cpu`、序列总长、请求/token总数。输出是各段的请求数和token数，供attention后端选择/切分不同执行路径。

例如query长度[1,1,4,8]、seq长度[101,201,104,8]，threshold=1：decode有2请求/2tokens，extend有1请求/4tokens，prefill有1请求/8tokens。这里extend指有历史上下文的续算，不是必然用户发起的新会话。

## 实际改动

对照其manifest指定宿主Core `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` 的
`vllm/v1/attention/backends/utils.py`（本轮在ascend180只读）：

- 原生：构造CPU布尔mask，`any`、`int().argmax().item()`找边界，少量Python分支及索引计算。
- 插件：仍构造mask，但 `_first_true_index` 用Python逐项循环，每项取 `.item()`，找到第一个True就退出，没有则返回请求数。
- 得到Python索引后，又包成torch scalar，做 `where`、`stack`、`.cpu().tolist()`，转换回Python整数元组。
- 保留纯decode快路径、uniform/padding与short extend的处理，增加空请求分支和切片限制。
- 注入实际传的是`query_start_loc_cpu`和`seq_lens_cpu_upper_bound`。**这里的item主要是CPU tensor取标量，不能宣称减少了NPU同步。**
  若另行直接给helper传device tensor，逐项item可能反而同步，但不是该插件默认注入路径。

## 接入与成本

启用时检查vLLM版本0.23.x和宿主函数源码关键字，然后monkeypatch函数，并遍历已导入模块替换引用。默认关闭，kill
switch优先。源码关键字匹配不是严格等价性证明；函数签名及所有特殊分支语义仍需测试。

`EVIDENCE=1`只记录安装/首次执行标记，并非性能证据。每次调用仍会检查该环境变量；启用证据时进入去重锁。包导入不依赖torch，只是延迟导入，实际边界计算仍用torch。

最坏情况Python查找随batch请求数线性增长；原生argmax也线性，但在底层实现。提前停止或省mask
dtype转换可能有局部收益，不能仅凭写法断言更快；张量重包装也有额外开销。它不减少attention FLOPs、KV读取或token数。

## 证据状态与初判

manifest为0.1.0.dev0、schema0.2-experimental；`admission_status=ported_cpu_validated_npu_smoke_runtime_blocked`，`evidence_ids=[]`。库内有CPU
helper/activation/guard测试，未见端到端吞吐或该helper时延benchmark。此状态是作者元数据声明，本轮未复验。

当前只够确认它是一个很小的CPU元数据处理补丁，不是attention算法优化。是否值得采用应先证明该helper是热点且替换确实更快；现有Python逐项扫描不足以自证优化，尚不替Fletcher作最终comment。

## 固定来源

- [README](https://github.com/xmdhb/vllm-hust-core-attention-boundary/blob/b5bc92a513a4ad043e39c7e2e3d6c4f4351114ff/README.md)
- [边界实现与注入参数](https://github.com/xmdhb/vllm-hust-core-attention-boundary/blob/b5bc92a513a4ad043e39c7e2e3d6c4f4351114ff/src/vllm_hust_core_attention_boundary/boundary.py)
- [插件入口](https://github.com/xmdhb/vllm-hust-core-attention-boundary/blob/b5bc92a513a4ad043e39c7e2e3d6c4f4351114ff/src/vllm_hust_core_attention_boundary/plugin.py)
- [manifest与资格状态](https://github.com/xmdhb/vllm-hust-core-attention-boundary/blob/b5bc92a513a4ad043e39c7e2e3d6c4f4351114ff/src/vllm_hust_core_attention_boundary/manifests/vllm-hust-extension-v0.2.json)

## 最终评审 comment

**目前缺 benchmark，收益不清，没有看到什么实质性的实现。** 指未见有证据支撑的实质性优化，不否认已有CPU辅助函数替换代码。[共同评审](../comments.md)
