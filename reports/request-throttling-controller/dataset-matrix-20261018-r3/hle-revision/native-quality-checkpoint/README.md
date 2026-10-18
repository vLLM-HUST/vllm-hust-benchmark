# HLE Revision Native完整质量检查点

1143题（1025文本、118图片）完整求解、判分与report全部SUCCEEDED。303/1143＝26.509186%，39判分未决保留完整分母；文本285/1025，图片18/118。求解0请求失败、判分0传输失败；703solver长度截断、39judge长度截断全部保留，严格JSON评分不救回。求解24876.917583秒、judge4849.104744秒。

本轮独立TP2：Qwen3.5-35B-A3B BF16、Core/Ascend0.23.0、APC关闭/full自动KV/context65536/maxseq16，quality
client16，solver16384/judge8192。使用同Native
Qwen替代judge、单次全量质量，不冒充官方排行榜复现或Controller成对收益。Controller/独立性能尚未完成，不把Native分数填入网站Controller列，也不覆盖历史Native。

私有原始archive逐1143题两阶段固定图文请求、原响应/finish_reason/usage、记录hash/solver-judge绑定、冻结源码与输入hash独立核验，并用冻结客户端重解析grade、逐字节重算summary.json/per_case.csv/per_case.json全部PASS。归档物化plan内容，未改动在跑源码或调用模型。真实smoke队列计划一并归档。

公开`python verify.py`只核验公开一致性；`private-verification.json`为实际私有`--archive`核验回执。私有原始包/提示/答案/图片payload/地址不公开，SHA见checkpoint.json。
