# HLE Gold 独立性能

新队列request-throttling-datasets-20261018-gold-performance-r3，原质量完成后停止r2并drain，再运行完整16阶段。文字200题/图片93题，每臂各1预热和3正式，预热保留排除；实际输出除以vllm测量duration，各指标分别取三次正式中位数。max256、自然EOS、temp0、seed0、rate
inf、无客户端slot限制。

原始hash/输入/命令/逐遍算术与中位数私有归档核验PASS。TTFT/TPOT仅原vllm汇总，未保存逐请求timing，不能独立重算其均值。普通单轮透传，不直接归因节流机制；不代表完整正确解题费用。

公开核验：`python verify.py`；私有原始包不公开，SHA见checkpoint.json。
