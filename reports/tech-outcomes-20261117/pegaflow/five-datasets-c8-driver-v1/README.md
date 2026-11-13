# BidKV/PegaFlow 五数据集 C8 执行适配器

公开的是项目自行编写的 HTTP/SSE、统计和 C8 驱动源码；`.py.source` 去掉末尾 `.source` 后可执行。原私有 Engine 源码、题目、答案、对话及完整原始 SSE 不在本目录分发。

本次先在重启后的 81（物理 6/7，逻辑 0/1）验证 BidKV 的官方 0.23 隔离实验载体。C8 沿用原冻结的 4K/16K、8 并发、60 秒预热与 300 秒提交窗口，并完整排空。原 C8 是 max_tokens=256、ignore_eos=true 的固定长度合成压力检查，与全部保留 EOS 的五项正式数据集分开。冻结清单见 [admission-plan.json](admission-plan.json)。

全部实际 `/tokenize` 需匹配离线冻结 token ID 哈希；请求盐唯一，并用引擎 prefix-hit 增量和外部加载计数交叉确认零复用。缓存 usage 为 null 只表示未知，不能写成零。SSE 原始字节以 base64 保留，记录内容首 token 与含 reasoning 的首生成事件，正式有限批次使用前者。请求无自动重试。

独立合成 HTTP 测试覆盖自然 EOS、固定长度边界、空缓存详情、缓存命中、流内错误、缺少 DONE、总超时、重叠批次的墙钟分母、单 token TPOT 排除和有限样本分位数：9 项通过。CPU 通过不代表 NPU 资格或正式测试完成。

首次模型启动因执行环境覆盖了 CANN Python 路径而失败，零生成；失败保留。第二次仅修正继承路径，模型参数未变。原镜像可选依赖检查问题保留，不宣称全局 pip check 通过。

BidKV 实验载体：[代码 PR #28](https://github.com/vLLM-HUST/vllm-hust-bidkv/pull/28)（Draft，非已发布支持路径）。两臂使用相同 prepared scheduler；Native 关闭策略，Candidate 启用策略。没有调度调用的工况不能证明 BidKV 性能收益。PegaFlow 的旧 SWE 缓存复用实验继续单列，不冒充本轮冻结五数据集结果。
