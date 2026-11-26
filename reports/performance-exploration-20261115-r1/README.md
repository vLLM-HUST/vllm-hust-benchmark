# B1 内部优化探索 · 2026-11-15

Tradeoff v3
收录24次不同的模型试验：22次完整MMLU200、2次启动失败；34处来源引用按实测ID/指标/seal去重，真正重复全部保留。范围为双Ascend910B2、Qwen3.5-35B-A3B
BF16、固定200/max8192/EOS/temp0/seed0/burst200、MTP2/FULL。原始映射文件核验两张910B2，不能由目录名推断硬件。

这些是B1-r6内部配置观测，**原ND E09不是正式B0**；不替换固定五数据集B0/B1或四目标结论。容量、State策略、原生多流、decode
padding、布局/router与组合的全部正负结果见[来源绑定指标](../../website-publication/data/optimization-exploration-20261115-v1/tradeoff.json)、[完整试验CSV](../../website-publication/data/optimization-exploration-20261115-v1/all-trials.csv)、[取舍图](../../website-publication/data/optimization-exploration-20261115-v1/tradeoff.svg)、[完整试验表](../../website-publication/data/optimization-exploration-20261115-v1/all-trials.svg)和[provenance](../../website-publication/data/optimization-exploration-20261115-v1/provenance.json)。失败没有零吞吐/零分；不挑最好一轮、不拟合跨配置曲线。菱形仅表示吞吐/平均TPOT/平均TTFT的观测非支配点，不考虑质量。

E19关闭态相对原ND吞吐+5.629%、meanTTFT−8.573%、meanTPOT−0.553%，正确149→143/200，截断18→20；E19Off→E18On吞吐−0.299%、meanTPOT−0.490%，正确143→154/200。只作带范围的历史观测，不将组合约5%收益归给多流；分数/序列变化不能证明模型或引擎单方根因。所有原生比较与配对对错转换保留在指标JSON。E20的E28/必需GDN96已完成：吞吐1102.424
tok/s，相对原ND吞吐+3.272%、meanTTFT−11.558%、meanTPOT+8.720%、P95TPOT+9.045%，正确149→137/200、截断18→19，性能初筛不通过；不能由E26/E32端点插值或隐去本次负结果。

native原clock不含启动/加载/预热/评分/报告；TTFT含burst排队，生成量会变化。每轮冷成本见Engine的对应报告。TorchDynamo/Triton封口范围不包含CANN，未知格式/cache/临时内存保留未知。没有独立质量等价、稳定性、MFU或等预算Autotune贡献证明；MOD默认关闭，不自动部署。

来源为[Engine固定发布提交](https://github.com/vLLM-HUST/vllm-hust-engine/commit/a039ae0ea16296bc3367d31ea8630413b6ddd7c6)的payload-free投影，原始字节/来源摘要保持不变。[MOD0.1.2离线源包](https://github.com/vLLM-HUST/vllm-hust-engine/releases/download/performance-exploration-20261115-r1/tradeoff-tuning-mod-v0.1.2.zip)
SHA256为`2ddfaa0280fd8be0b415b166ac41ff06950ecc32b7b6a6424f464297d489d9d6`，已回读验证；Engine私有原件需仓库授权，不把仅导入投影视为重新执行或私有raw逐项复验。

复现导出：恢复对应Engine实验目录后执行`scripts/export_optimization_exploration.py --engine-experiment-root /absolute/path/to/optimization-20261115 --output /absolute/path/to/new-benchmark-staging`。网站仍按已合并固定benchmark
commit装配，网站仓库只提交展示代码。
