# 压缩方法架构

[English](methods.md) | 简体中文

## 运行时分层

自 0.6 起，本插件自包含运行，不再使用已删除的 vLLM-HUST 压缩生命周期：

1. `plugin.py` 是显式启用的 `vllm.general_plugins` 入口。调度侧立即挂接，
   Ascend worker 在对应模块真正加载时才挂接，因此管理器和纯 API 进程不会
   提前导入 NPU runner。
2. `calibration.py` 在服务权重加载前生成缺失的模型绑定产物，也提供相同生成器的
   命令行入口。
3. `stateful.py` 管理 scheduler/KV manager 状态。在同步调度边界提交已完成的
   压缩，并释放旧 block table 尾部。
4. `provider.py` 校验当前宿主对象、在 worker 镜像事务、把语义位置转换为物理
   slot，并把选择和物化委托给具体方法。
5. `methods/<name>/` 负责算法选项、兼容性检查、评分、校准和 KV 物化。

适配层明确依赖主 README 列出的宿主内部符号。这些接口尚未冻结，所以只能在
已测试版本线上提供支持；接口变化时必须拒绝启动并重新验收。

## 配置契约

Extension Manager 保存一个 JSON 对象，必须包含 `schema_version`、`provider`、
`method` 和 `method_config`。完整示例见
[`examples/triattention.json`](../../examples/triattention.json)。未知字段、未知方法、
不满足 block 对齐的值及无效校准文件都会报错，不会被静默忽略。仅在
`auto_calibrate` 启用时生成缺失文件，详见校准产物文档。

`min_output_tokens_for_compression` 是非负的请求输出门槛。当请求的最大生成长度低于
该值时，scheduler 和 worker 都跳过事务。`0` 保持原有的始终可压缩行为；公开
benchmark 推荐值为 `64`，可消除 32-token 检索负载上的无效压缩开销。

标准运行时的 scheduler、全注意力 manager 与内核缓存块均为 128 token。对于
明确支持的 Qwen3.5 混合模型，Ascend 可把全注意力 manager 页提升为
1,152 或 2,048 token，同时保留 128-token 内核块。适配层校验 scheduler
对齐等于所有 cache manager 的最小公倍数，再把每个注意力页展开为 9 或 16 个
内核块。method 的最大物理 token 数必须能被当前注意力页整除，但无需被跨组
对齐粒度整除。

`score_layer_stride=8` 会从 48 个已校准层中均匀抽取 6 层用于选择，但仍会物化
全部 48 层 K/V。在冻结的公开 Qasper 与 LongBench-v2 集合上，该配置保持了基线
质量，并比原值 `4` 获得更好的同机端到端结果。

## V3 位置策略

`position_policy=global` 保留旧的全局 top-k 行为；显式选择 `v3` 时执行论文的
三段式位置策略：

1. 开头 `protected_prefix_window` 个 token 强制保留；
2. 末尾 `protected_recent_window` 个 token 强制保留；
3. 中间上下文切成 `position_segments` 个等长区间，把全局驱逐数按比例精确分配
   到各区间，再在区间内保留得分最高的 token。

实现使用整数累计配额，所以即使区间长度不完全相同，最终也严格保留
`kv_budget` 个位置。对于 NPU 上等宽的中段区间，所有区间合并为一次批量
top-k，并复用依赖形状的小型索引计划；区间不等宽或存在零配额时仍走原有
逐段路径。优化只改变执行开销，不改变配额规则或受保护位置。

## Qwen3.5 混合架构路径

Qwen3.5-35B-A3B 的 40 个文本层由 10 组“3 个 Gated-DeltaNet 层 + 1 个全注意力
层”组成。方法只绑定第 3、7、…、39 层的全注意力缓存，循环状态组保持不变。
其 256 维注意力头只有 64 维使用 RoPE，因此校准产物会为剩余 192 维保存
`q_pass_mean`。插件自有的分页键融合评分算子现可在一次 NPU 核执行中，把这部分
直接内容点积加到旋转维度的三角评分上；非均值聚合仍可回退 Torch 路径。部分
RoPE 融合路径已通过真实昇腾 NPU 的数值检查。随后的 SWE 官方客户端配对
缩小了性能退化幅度，但尚未扭转；详见 [Frontier 测试](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/frontier-benchmarking-0.9.zh.md)。

张量并行时，校准行按本地 KV 头切分，并在选择前对逐层分数执行最大值 all-reduce，
保证各 rank 选择相同位置。TP 大小必须整除 KV 头数。Qwen3.5 的
`mamba_cache_mode=align` 现由插件的事务式缓存处理支持，`none` 也可使用；
其他混合布局以及 PP/DP/DCP/PCP 仍会拒绝启动。

## 方法契约

实现 `methods/base.py` 中的 `KVCompressionMethod`：

- `name`：稳定的注册和配置名称；
- `runtime_spec`：供 scheduler 适配层使用的 block 对齐阈值、重算窗口、最大
  物理长度和目标区要求；
- `compatibility_reasons(worker)`：无副作用的算法兼容检查；
- `bind_model_runner(runner, layer_caches)`：公共缓存校验后分配状态；
- `compress(request)`：同步完成一次物化并返回新物理长度，以及可选的逐层长度。

对于达到方法压缩阈值的请求，自动前缀缓存 admission 会把可复用前缀限制为
`max(0, min(prompt_tokens - 1, prompt_tokens - required_recompute_tokens))`。
这样即使启用 chunked prefill，也会实际执行方法声明的末尾 Query 后缀；当 prompt
短于要求的窗口时，从 token 0 开始完整重算。APC 关闭、低于阈值或输出长度不满足
压缩门槛的请求继续使用宿主原有 admission 路径。
需要观察 Query 的方法必须声明不短于 `query_window_tokens` 的重算窗口。启用 APC
时，适配层会在启动时检查所需的宿主缓存查找接口；接口缺失则明确拒绝启动。

### 逐层物理状态

`CompressionResult.per_layer_physical_num_tokens` 可以为每个已绑定的全注意力层
返回一个 `(layer_name, length)`。层名必须完整且不重复；长度须为正整数，且不超过
`runtime_spec.max_physical_num_tokens`。scheduler 按声明的组上限预留私有块。
worker 在同一次输出确认后的事务中切换逐层锚点；之后每个层的 KV 写入槽位和注意力
长度独立随 decode 前进。若再次压缩，`CompressionRequest.per_layer_physical_num_tokens`
包含当前逐层长度。GDN 继续使用语义长度。

`None` 或全部等于组上限的结果继续使用原有共享元数据路径，包括 TriAttention。
可能返回不等长结果的方法必须在模型加载前声明
`requires_per_layer_physical_state`。eager 路径仍可在事务执行时发现未预先声明的
不等长结果；图捕获不允许这样做，因为首次捕获前每层必须已经持有独立且地址稳定的
slot 缓冲区。

已声明路径支持 eager 和通过校验的 Ascend FIA `FULL_AND_PIECEWISE` 更新接口。
FULL 重放按层名重新绑定每个 attention task，piecewise prefill 仍执行 Python Query
观察钩子。对于已限定的 Qwen3.5 MTP2，适配层还会通过 speculative common metadata
给辅助 draft-attention cache 独立的序列长度和 slot；被拒绝的 draft 位置会在下一步
覆盖同一个逐层物理槽位。其他图模式、注意力后端、投机方法和上下文并行路径继续
失败关闭。CPU 契约测试不代表外部方法已通过服务验证。

方法只能使用已校验的 cache binding，不能修改 scheduler 拥有的 request 或
block table。返回长度必须为正、不超过声明上限；使用逐层长度时必须覆盖全部层。

### 可选 Query 观察接口

`query_window_tokens` 默认为 `0`，现有方法（包括 TriAttention）不会安装观察钩子。
方法将其设为正整数时，必须实现 `capture_query(layer, query, spans)`、
`complete_query_observation(observation)` 和
`discard_query_observation(request_id)`。

`query_layer_indices` 默认为 `None`，即观察所有已绑定的全注意力 cache 层。混合
投机运行时可能额外绑定一个不参与目标模型 prefill 的 draft cache。Query 方法可
返回自己实际消费的目标层编号；适配层只等待这些层完成观察，同时仍要求辅助 cache
包含在逐层物化结果中。

host 仅在 prefill 中对已绑定的全注意力层调用 `capture_query`。`query` 是注意力层
应用位置编码后的 Query 张量；每个 `QueryBatchSpan` 给出一个请求在张量中的左闭右开
行区间。方法在 `bind_model_runner` 中分配并持有地址稳定的缓冲区，可跨分块 prefill
累积末尾窗口。完整模型前向和采样成功之前不会发布观察结果。最终 prefill 步骤有
授权压缩计划时，`complete_query_observation` 收到请求 ID、原始
`CompressionPlan`、语义长度、窗口长度和参与的层编号。同一计划也会传给
`CompressionRequest.plan`。方法使用窗口前须检查各层数据完整。压缩完成以及请求
结束、抢占、重启或执行失败时，host 调用 `discard_query_observation` 清理。

若 prefill 中有方法请求的层未触发观察钩子，包括图路径跳过 Python 钩子的情况，
host 会失败关闭。当前 CPU 契约测试不构成任何图模式或外部方法的服务支持证据；
必须在精确的 host 与设备组合上完成图重放、兼容性和正确性验证后才能启用。

## 注册其他方法

仓库内部可调用：

```python
from vllm_ascend_kvcompress import register_method

register_method("my_method", create_method)
```

外部包可声明：

```toml
[project.entry-points."vllm_ascend_kvcompress.methods"]
my_method = "my_package.method:create_method"
```

名称只允许小写字母、数字、连字符和下划线；重复名称、非法 factory 或名称不匹配
都会失败关闭。

## 验收清单

- 只解析本方法拥有的选项，拒绝未知值；
- scheduler 限制必须确定且满足 block 对齐；
- 服务启动前校验模型、RoPE、校准、dtype 和 cache layout；
- `compress` 返回前完成所有层的物化；
- 补齐注册、配置、事务和数值测试；
- 在精确宿主 revision 上完成压缩关闭/开启的正确性、质量、吞吐、时延和 HBM
  对照验收。
