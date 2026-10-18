# HLE Revision Controller完整质量检查点

1143题（1025文本、118图片）完整求解、判分与report全部SUCCEEDED。308/1143＝26.946632%，32判分未决保留完整分母；文本287/1025，图片21/118。求解0请求失败、判分0传输失败；691solver长度截断、32judge长度截断全部保留，严格JSON评分不救回。求解24524.314683秒、judge3854.084044秒；这些阶段墙钟不当独立性能。

匹配Native303/1143，分数保留101.650165%。两臂contract仅solver.base_url不同，judge完全一致，输入、源码、生成配置及client16一致。使用同Native
Qwen替代judge、单次全量质量，不冒充官方排行榜复现。独立性能仍由原队列继续，未完成成对性能核验前不发布网站Controller成绩、不覆盖历史Native。

本轮独立TP2：Qwen3.5-35B-A3B BF16、Core/Ascend0.23.0、APC关闭/full自动KV/context65536/maxseq16，quality
client16，solver16384/judge8192。

私有原始archive逐1143题两阶段固定图文请求、原响应/finish_reason/usage、记录hash/solver-judge绑定、冻结源码与输入hash独立核验，并用冻结客户端重解析grade、逐字节重算summary.json/per_case.csv/per_case.json全部PASS。归档物化plan内容，真实smoke队列计划一并归档。首次下载少1141134字节被SHA校验拒绝，保留不完整传输文件；核对远端原包大小及hash后，仅补传同一原包尾部，完整SHA通过，没有重复归档或模型重试。

公开`python verify.py`只核验公开一致性；`private-verification.json`为实际私有`--archive`核验回执。私有原始包/提示/答案/图片payload/地址不公开，SHA见checkpoint.json。
