# 第八版引擎上的 Request Throttling Controller 实测

## 1. 结论与范围

SWE-smith长度/工具等待微基准的完整批次耗时从302.0508秒降到217.6042秒，缩短27.9577%。两组各112个三轮程序实例、336次请求，全部成功，输入token总数和输出token总数分别相同。缓存输入token命中率从40.4253%提高到62.6310%，prefill计算token从3,569,962减少到2,239,301。

这是56份长度/等待模板各两个独立合成token-ID实例；续轮继承各自模型真实输出。它不是112道原始题目、原始文本回放或完整Agent任务解决率。仅一次固定顺序配对，不证明最优或稳定收益。

## 2. 系统、版本与改动

对照路径为客户端→优化引擎；候选路径为客户端→Request Throttling
Controller限流层→同一优化引擎。模型Qwen3.5-35B-A3B、BF16、Ascend910B2×2、TP2、MTP2、FULL图。引擎基于`full-answer-b1-20261117-r8`；保留官方执行上限36，两组共同覆盖任务状态驻留数54→80、逐请求强制重算→跨轮缓存复用、业务timeout→禁用。80是自定义状态容量，不是vLLM默认值或80个并行请求。

引擎host State为4GiB/rank、KV预算20GiB/rank；Controller由独立4CPU/16GiB工作区中的RIDE
local管理，使用原版ThunderAgent，调度周期0.5秒。模型独立40CPU/384GiB工作区。配置与引擎匹配回执见`matched-engine.json`；实际来源摘要见`execution-and-analysis-source-final-sha256.json`。

vLLM实际版本为`0.28.1.post1.dev143+gf18cf803c`，vLLM-Ascend为`0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`，CANN9.1.0。目录名称不作为版本证据。客户端、转发、限流等待和恢复均不设有限业务timeout；运维资源截止独立记录。

## 3. 输入与公平性

多轮微基准112个程序同时到达、无客户端并发槽位限制、每程序三轮，共336次请求，随机种子17，稳定program身份及隔离的跨程序cache
salt。输出共43,516token，两组相同；续轮继承各自真实输出，167/336个动态prompt token序列完全一致，不能称所有动态输入逐字节相同。

四项固定官网负载另建运行目录：MMLU-Pro200/C200、FrontierScience160/C16、SWE-bench
Pro公开轨迹32任务/64请求/C8、Terminal-Bench公开轨迹32任务/48请求/C8。每臂472次solver请求；Humanity’s Last
Exam按用户要求暂缓。原始提示词、样本顺序、生成预算、并发和原评分保留。请求正文与引擎prompt
token逐case相同，见`fixed-profile-pairs.json`及逐请求SHA。共同加入program身份、稳定salt、完成后release和流式usage记录；不宣称整个原始HTTP协议未改。

两臂分别冷启动并完成194次官方任务预热。每臂内按Terminal、SWE、MMLU-Pro、FrontierScience顺序运行，共用这一臂的模型；不是每个数据集重新冷启动。固定官网问答使用Chat
Completions且不设reasoning parser；微基准使用token-ID Completions入口。这是分别冻结的两类负载，不构成仅改变一个参数的消融。

## 4. 完整批次性能

| 负载                         | 每臂请求 |    直连秒 | 加限流层秒 | 耗时缩短 | 输出token变化 |
| ---------------------------- | -------: | --------: | ---------: | -------: | ------------: |
| SWE-smith三轮长度/等待微基准 |      336 |  302.0508 |   217.6042 |   27.96% |         0.00% |
| MMLU-Pro固定200题            |      200 |  496.9815 |   515.4949 |   -3.73% |        -3.36% |
| FrontierScience              |      160 | 1190.9711 |  1229.3562 |   -3.22% |        +3.02% |
| SWE-bench Pro公开轨迹        |       64 |   61.7606 |    60.5573 |    1.95% |        +0.29% |
| Terminal-Bench公开轨迹       |       48 |   63.6571 |    56.7506 |   10.85% |       -16.01% |

负值表示变慢。两轮共1,616次solver请求均成功；四项固定负载两臂均无release错误。评分请求另计，未混入solver耗时或吞吐。

Terminal-Bench输出减少16.01%，不能把10.85%耗时缩短解释成同等工作量效率提升；其整批输出token/s反而下降5.79%。MMLU-Pro输出减少仍变慢，保留负结果。FrontierScience输出增加3.02%、整批输出token/s约下降0.20%，不称明显性能收益。SWE公开轨迹差异小且未重复，不称稳定收益。

## 5. 质量、截断与执行完整性

MMLU-Pro原提取器下直连148/200（74.0%），限流145/200（72.5%）；达到8192输出上限的请求26→22。单对生成结果有变化，不证明质量差异由限流算法单独造成。

FrontierScience使用官网同款Qwen替代judge，不是官方judge分数：Olympiad达标61/100→64/100，两边各3条评分未解析；Research达标4/60→4/60，评分未解析0→1，有效评分均分2.509→2.48797/10。所有未解析保留原计划分母，没有按有效样本改写成功率。两边均无judge传输失败；输出截断、审计标记和逐项统计见完整JSON。

公开轨迹没有执行真实工具或完整代码任务，不提供任务解决率。多轮微基准不是质量实验，不把API成功率称作Agent任务成功率。

## 6. 分层机制与跨层代价

多轮微基准：104个程序变快、8个变慢。224个续轮中零缓存命中88→13；第二轮输入token命中53.47%→91.88%，第三轮66.30%→93.92%，首轮两组均0%。对应prefill计算减少，是缓存复用改善的直接证据。引擎KV池采样峰值100%→99.83%，池占用包含可淘汰的已完成前缀，不能仅凭峰值断定活跃状态或硬件饱和。

同336次业务请求的引擎均值完整覆盖得到核验：引擎端到端77.79→30.31秒，其中排队47.07→10.46秒，prefill16.36→7.87秒，decode13.68→11.75秒。客户端均值78.02→49.63秒，包含约18.74秒平均放行等待。该等待代价没有从端到端收益中删除。Controller事件缺少逐请求身份，只支持同批次聚合核算，不提供虚构的逐请求放行P95。

四项固定官网负载没有本批三轮输出继承结构，且逐case隔离后缓存输入token均为0；它们不能代替多轮缓存竞争验证。吞吐、质量、输出量和时延完整分列，不把引擎B0/B1倍率乘到限流层收益上。

## 7. 限制与变更记录

仅一次固定顺序配对，独立重复、最优参数、完整Agent任务解决率和Humanity’s Last
Exam未执行。没有连续设备AICore/带宽、CPU/RSS时序，不报告硬件MFU。引擎preemption为0不等于任务状态从未被淘汰；13个续轮未命中的底层原因仍开放，issue不以本次正收益关闭。

保留第六版30.06%历史配对，未重标为第八版。两版执行上限和引擎组合不同，不作单变量因果对比。第八版覆盖驻留80、跨轮缓存和无业务timeout的原因、代价与批准记录已同步issue
#5。全部固定负载、负结果、评分未解析均发布，不只发布正收益项。

## 8. 原件与复算

- `multi-turn-paired-result.json`：多轮完整配对结果。
- `multi-turn-diagnostics.json`：请求、程序、引擎指标及来源SHA。
- `fixed-profile-pairs.json`：四项逐负载对照、输出量、时延与原评分。
- `direct-requests.json`、`controller-requests.json`：各472条公开逐请求指标、请求正文SHA和原件SHA；不含题目或答案正文。
- `input-manifest.json`、`matched-engine.json`、`import-source-sha256.json`：输入与运行身份及导入来源。
- `controller-archive-receipt.json`：Controller侧客户端/模型原件归档摘要。

导入与公开投影可离线复算：

```bash
python scripts/export_controller_product.py \
  --multi-turn /absolute/path/to/myc-b1-delay-r8-1118-r2 \
  --profiles /absolute/path/to/myc-b1-delay-r8-profiles-1118-r1
python -m pytest -q tests/test_controller_product.py
```

这里的导入脚本不会发模型请求。真实运行及资源守卫脚本、完整原始响应流和部署记录保存在原实验归档；复跑需新run及实际资源身份，不能向历史提交目录重复投递。空白SDK环境的一键重建未声称通过。

完整原始响应流与运行归档另存[私有证据Release](https://github.com/vLLM-HUST/request-throttling-controller/releases/tag/controlplane-r8-evidence-20261118)，需要仓库权限；公开数据不包含题目和答案正文。模型、客户端工作区均已停止，原持久卷同UID保留，见`cleanup.json`。
