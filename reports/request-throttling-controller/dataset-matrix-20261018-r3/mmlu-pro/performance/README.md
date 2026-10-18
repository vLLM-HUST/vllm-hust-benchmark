# MMLU-Pro独立性能

同服务内Native/Controller各1次预热和3次正式、交替执行。固定原200题输入、outputcap256、自然EOS、temperature0、request
seed0、request-rate inf，无客户端slot限制。原vllm duration计算实际输出率，逐指标取正式三遍中位数，预热保留排除。

Native正式290.111247/355.630973/337.200931，Controller290.787346/371.564952/360.525152
tok/s；中位337.200931/360.525152。6正式均200/200、0失败、51200实际输出。meanTTFT中位69.299432/66.989813秒；meanTPOT39.680193/37.906218毫秒。延迟仅原vllm
aggregate，未留逐请求timing，不能声称独立重算延迟均值。

`python verify.py --archive raw-and-plan.tar.gz`独立核验原JSON/hash、冻结共享输入与各遍记录、原命令、summary、actual
output÷duration及中位数PASS。公开不带archive只核验公开算术与校验和。私有包、提示、token-ID与内部地址不公开。

首次采集沿用Frontier输入路径导致在checkpoint创建前失败；只读确认不存在检查点后改为冻结MMLU客户端实际共享输入路径，唯一创建归档。原失败日志保留，无重复模型调用、无修改测量。

本项是普通单轮透传性能，不宣称节流因果收益，也不是完整正确解题费用或统计显著性证明。
