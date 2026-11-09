# 完整回答联合测量与连续运行复核

日期：2026-10-30。固定B0：vLLM 0.18.0 / vLLM-Ascend 0.18.0；优化候选： vLLM 0.28.1.post1.dev143+gf18cf803c /
Ascend 0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909。

## 本轮结果

| 指标                 | B0 · 0.18/0.18 | B1 · 0.28.1/0.25.1rc2 |
| -------------------- | -------------- | --------------------- |
| 正确/题数            | 143/200        | 153/200               |
| 长度终止             | 23             | 16                    |
| Output token/s       | 495.8359       | 944.4738              |
| Mean TTFT (ms)       | 610288.1058    | 290879.2940           |
| Mean TPOT (ms/token) | 41.6217        | 21.3205               |
| 元/百万输出token     | 28.7176        | 15.0763               |

固定200题子集正确数未下降。逐题24题改善、14题退步，差值+5.00个百分点；全部响应和原评分器fallback保留。长度终止不隐去，也不事后改判。
吞吐变化+90.4811%，平均TTFT/TPOT分别改善52.3374%/48.7757%，每百万输出token租赁估算降低47.5014%。

| 方案 | 正常结束正确/总数 | 长度终止正确/总数 | 完成输出token | 请求区间秒数 | 双卡AI Core采样均值 |
| ---- | ----------------- | ----------------- | ------------- | ------------ | ------------------- |
| B0   | 137/177           | 6/23              | 706796        | 1425.463447  | 33.5844%            |
| B1   | 147/184           | 6/16              | 661040        | 699.902978   | 62.9772%            |

这是一轮固定子集配对，不代表全集质量、统计非劣或所有负载。新完整回答质量和计时不回填旧out256的M01算量。

## 输入、资源与计量

- 原MMLU-Pro固定200题顺序及物化请求不变，服务端分词哈希全部核对；输出上限8192、temperature=0、seed=0、EOS开启。
- 同Qwen3.5-35B-A3B源权重、BF16、TP2、同物理5/6两张910B2，双方计费资源均为两卡。
- 同时提交200请求，无额外任务预热，服务模型加载和图捕获完成后开始。请求总超时及读超时均6600秒；不事后增加交互式SLO或质量容差。
- 两侧max-model-len=65536、max-num-seqs=24、max-num-batched-tokens=4096、MTP2、设备KV预算20GiB/rank。B0为FULL_AND_PIECEWISE、CANN8.5.1及原text-only兼容补丁；B1为FULL、CANN9.1.0、E24/R36、逐请求重算及Microflow组合。
- B1将同一状态对象内同dtype lane合并为一块锁页分配，不跨对象共享；主机State预算4GiB/rank。8GiB/rank失败记录保留，不将更低预算当成原配置。
- 每臂独立服务、新空运行/编译缓存、逐请求唯一salt；只读模型及OS文件页缓存共享。连续资格结束后先释放服务和卡，再启动联合轮，不导入前轮答案或状态。
- B0完成后，闲置B0运行时已移存并逐文件核验后删除本地副本；这会改变OS页缓存环境，与4GiB预算变化共同披露，不单独归因于预算。不声称全局冷缓存一致。
- 吞吐分母为首次提交至全部响应完成的区间；同一批成功响应的全部输出计入分母，包括答错和长度终止。TTFT到首非空正文，TPOT=(响应结束−首正文)/(输出token−1)，逐请求等权；P95/P99采用最近秩法。HTTP/SSE采集开销保留。
- AI Core利用率按同轮单调时钟中点、左保持采样作时长加权；逐卡及观测空档见JSON。采样均值不是物理精确值或置信区间。

## 连续窗口

| 方案/阶段       | 请求 | token/s  | TTFT P95/P99 (ms)   | TPOT P95/P99 (ms) | 失败 |
| --------------- | ---- | -------- | ------------------- | ----------------- | ---- |
| B0 4096-formal  | 153  | 126.2931 | 2789.30 / 6410.66   | 59.78 / 61.51     | 0    |
| B0 16384-formal | 73   | 60.0049  | 16786.59 / 27618.84 | 118.22 / 119.35   | 0    |
| B1 4096-formal  | 408  | 341.3732 | 1535.84 / 2577.37   | 21.95 / 22.65     | 0    |
| B1 16384-formal | 183  | 153.8416 | 8936.14 / 10090.46  | 46.26 / 46.95     | 0    |

同原4115/16407实际输入token，C8闭环、out256，分别60秒预热、300秒正式发起并完整排空。
B0复用已封存原件；B1预热24/40、正式408/183，共655请求无失败，独立准入655、reused_tokens=0，两个rank实际活动均经核验。16K尾时延高于4K，完整分窗口记录保留；零失败不等于五任务全范围稳定性通过。

## 成本依据与范围

沿用[天翼云类似双卡910B报价](https://www.ctyun.cn/document/10029787/10047957)：pak3.8xlarge.8，36908元/月；[计费范围](https://www.ctyun.cn/document/10029787/10030140)。来源核验日期2026-10-29，30天为分析约定。
此报价包含对应SKU的CPU/主机内存/加速卡，云盘和公网另计；报价主机256GB内存不等同本机64GiB容器，本结果为类似机型同费率估算，不是本机账单。
费用=每卡月费×实际卡数×测量秒数/(30×24×3600)；单位成本=费用×10⁶/完成输出token。固定费用不按利用率打折。 另列包含初始化、预检及退出观测的完整控制器生命周期成本：B0
25.996557元、B1 23.010912元；不与请求区间混合，也不把生命周期差异用作新增否决条件。

## 原件、复算与历史

- [V1.03上线与归档回执](delivery-v103.json)：合并/CI、生产中英文/桌面移动/明暗浏览器、Release下载与存储SHA回验；保留本机HTTPS传输失败，不扩大线上字节核验范围。
- [线上发布清单原始响应](production-manifest-v103.json)：独立生产浏览器留存的完整字节，采集流程与SHA见交付回执；不是本地重建替代。
- [逐题标量及配对汇总](joint-summary.json)：完整200×2标量、每题原SSE哈希、实际版本、配置、双卡采样、报价与公式。公开标量由独立原SSE重建/原评分器复核产生，不包含题文、gold或全文回答。
- [连续逐请求/窗口统计](continuous-summary.json)：包括原B0、当前B1、两rank激活和P95/P99。
- [未改写的原始采样与回执](original-telemetry.tar.gz)、[成员校验](telemetry-manifest.json)、[文件校验](manifest.json)。这些是选定原件，不是完整私有响应包。
- [8GiB候选原失败](retained-r3-failure.json)：24预热成功、正式16成功8失败，rank1的32MiB锁页分配207001；16K未执行。关停后的EngineDeadError不替代原始错误。
- [本批次全部尝试](attempt-history.json)：保留r1监控保护中断、r2模型启动前端口拒绝、r3驱动分配失败和r4新配置；没有从多轮完整回答结果中择优。
- [V1.02原连续失败和out256评分](../acceptance-completion-20261029/README.md)。原21/200对19/200不能借本轮完整回答评分。
- 私有完整响应、输入及实现源码保存在授权存储；不公开其内容。冻结源码包、原响应和控制回执均有SHA绑定；公开部分可按下列命令复算，原内容复核须通过私有保管入口。

```sh
python scripts/recompute_full_answer_joint.py reports/full-answer-joint-20261030/joint-summary.json
```

## English summary

This is one same-response MMLU-Pro fixed200/full-answer pair, capped at8192 tokens with EOS enabled.
Fixed B0 uses0.18/0.18; the candidate uses the disclosed0.28.1/0.25.1rc2 development stack,
per-object pinned packing and a4GiB/rank host-State budget. Both reserve the same two910B2 cards.
Quality, latency, utilization and cost come from the same responses, with all length stops and
per-task regressions retained. The4K/16K continuous formal cohorts completed153/73 requests for
reused B0 and408/183 for the new candidate, with zero failures. This does not establish five-task
stability. Prior allocation failures remain separately archived. Pricing is a public
analogous-machine estimate, not an invoice. New full-answer measurements do not supply missing
F/T/reference evidence for the historical M01 workload.
