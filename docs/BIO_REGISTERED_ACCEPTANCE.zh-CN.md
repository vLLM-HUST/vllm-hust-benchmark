# 注册式生信Benchmark验收摘要

公共注册、驱动接口及原生任务编排的操作说明见[接入指南](BIO_ONBOARDING.zh-CN.md)。模型桥接及机器可读验收摘要维护在Scientist配套分支的[验收记录](https://github.com/vLLM-HUST/scientist-lightweight-vllm/blob/feat/benchmark-cross-series/docs/benchmark/acceptance_cross_series_20261123.json)。

新两模型RNA-FM、GenomeOcean-100M在单910B2/FP16
eager下完成45用例、1812正常/恢复请求、6219参考行为检查、57拒绝/恢复和两次host异常关闭，95项CPU回归通过。独立verify和报告字节重建通过。以注册版原型为基线，跨系列扩展复用九个未改变的公共驱动、执行、计量、注册及报告源文件。

ESM-2与RNA-FM复用序列表征契约，ProGen2与GenomeOcean复用因果生成契约。旧原型完整39用例按封存原件与受测实现身份核验、重算复用；旧七任务另跑真实原生回归。这不是一次四模型84用例的新测量。

RNA-FM两个长序列contact用例共78请求存在与原作者FP16同位置的NaN，明确记录为科学输出无效，不能将参考行为passed解释为科学有效性通过。原件和权重尚未作为公开附件上传；验收JSON提供摘要与来源身份，实际下载地址不由本地路径代替。本功能属于原生生信任务项目评测，不进入正式榜单或网站成绩发布。
