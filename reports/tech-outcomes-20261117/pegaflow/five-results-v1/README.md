# PegaFlow：完整五项 MOD 实验与闭合审计

已经完成要求的双卡 B0/B1 五项测试和共享 Native judge，无需补跑覆盖范围。五项吞吐均低于 B0，本轮没有证明 PegaFlow
加速或缓存恢复收益，也没有通过四目标验收。保存链路已经真实触发，不能描述为“根本未触发”。

实验在此前获授权的 ascend-91 上于 2026-11-14 完成；2026-11-16 仅经当前可用 yanbo-81 读取持久卷并进行 CPU 审核，没有访问或恢复
91，没有启动新模型、调优、构造复用、跨机实验或重新测量。原始实验状态直到本次恢复 NAS 证据后才确认全部闭合。

## 结果与逐请求记录

[机器可读比较](COMPARISON.json)｜[完整通用报告及图表](report/REPORT.md)｜[HTML 报告](report/report.html)｜[原始数字索引](summary.json)｜[独立核验](audit/closure.json)。五项逐请求数值分别是本目录
`native-<dataset>-requests.json` 和 `pegaflow-<dataset>-requests.json`，包含时间、usage、finish、状态和受控记录/SSE
哈希，不含题目、答案或对话。

| 数据集   | B0 tok/s | B1 tok/s | 吞吐变化 |        TTFT均值秒 B0/B1 |    TPOT均值秒 B0/B1 | solver length B0/B1 |
| -------- | -------: | -------: | -------: | ----------------------: | ------------------: | ------------------: |
| mmlu     |  631.681 |  619.995 |  -1.850% | 524.452639 / 470.061462 | 0.023994 / 0.024107 |             19 / 19 |
| hle      |  781.985 |  765.200 |  -2.146% |     1.381152 / 1.409020 | 0.020440 / 0.021108 |           122 / 122 |
| frontier |  776.049 |  756.743 |  -2.488% |     1.184481 / 1.215446 | 0.019947 / 0.019719 |             35 / 40 |
| swe      |   34.964 |   28.595 | -18.216% |     7.818178 / 7.795531 | 0.237631 / 0.250095 |               2 / 1 |
| terminal |   74.057 |   73.092 |  -1.302% |     7.216352 / 7.603369 | 0.115517 / 0.120315 |               2 / 2 |

吞吐为所有有效输出词元总数除以该批最早请求提交至最后响应完成的墙钟区间，TTFT 为首个 content 事件；TPOT 使用
`(end-first_content)/(completion_tokens-1)`，单请求等权平均，分位数按 `(n-1)q` 线性插值。所有
mean/P50/P95/P99、逐请求词元数及排除数见 COMPARISON.json 和逐请求文件。五项保留 EOS，输出长度可变，因此短 trace 的 tok/s
变化不能直接当作相同长度服务时间变化。单次有序队列未提供总体效应置信区间或因果归因。

## 冻结负载与公平性

使用正确 MOD 提示词 SHA256
`43bc925a6ad749971a64e5244b86451f5d4cf3361e87423b1eda8ea0b0d04461`，参见[绑定](mod-binding.json)与[执行计划](execution-plan.json)。每臂
MMLU200/C200/max8192，HLE200/C16/max16384，Frontier160/C16/max16384，SWE32任务64请求/C8/max2048，Terminal32任务48请求/C8/max2048；每臂672、两臂1344
solver。MMLU seed0，其他 seed17，temperature0。HLE Gold文64/图10、Revision文113/图13；Frontier
Olympiad100/Research60。原始题目、顺序、提示、预算、EOS和 grader 保持冻结；trace 仅推理回放、tool_choice=none，不执行工具、不声称任务解决率。

独立新建 Native 服务对两臂 HLE/Frontier 完成720个 judge 输入，max8192，不计入 solver 吞吐。全14个阶段实际 client wait
均为0，2064/2064个正式记录逐一从原始 SSE 重放并核对冻结请求、时间、usage、判分和缓存计数；4份原 grader 汇总由 CPU 无HTTP重算逐字节一致。所有1344个
solver传输成功；judge 的语义判分未决保留为未决。

两臂同用 Qwen3.5-35B-A3B（revision712cf74392b05026a6db2bf213d343747d1f6d45）、BF16、2×Ascend910B2
64GiB、物理2/5→逻辑0/1、TP2/PP1/DP1，16GiB/rank KV、ctx65536、seq16、batch4096、APC、Mamba
align、async、MTP2、FULL_AND_PIECEWISE图[3,6,12,24,48]及默认CPU binding=True。B0关闭连接器和sidecar；B1增加 e2cdc2f7
PegaKVConnector read_write 和本机30gb sidecar，并保留其必要 HMA本地prefix索引安全限制。该差异计入MOD成本，不能关闭B0优化来消除。

## 质量、未决与截断

MMLU原extractor：B0 145/200=72.5%，B1 151/200=75.5%；单次固定子集结果不证明MOD提高质量。分类明细在summary/COMPARISON中。

| 数据集/分层        | 固定分母 | 成功 B0/B1 | 已判 B0/B1 | 未决 B0/B1 | solver length B0/B1 | judge length B0/B1 |
| ------------------ | -------: | ---------: | ---------: | ---------: | ------------------: | -----------------: |
| hle/gold/text      |       64 |      5 / 6 |    30 / 27 |    34 / 37 |             40 / 39 |            34 / 37 |
| hle/gold/image     |       10 |      0 / 1 |      7 / 7 |      3 / 3 |               1 / 1 |              3 / 3 |
| hle/revision/text  |      113 |    21 / 20 |    48 / 57 |    65 / 56 |             78 / 80 |            65 / 56 |
| hle/revision/image |       13 |      1 / 1 |      4 / 7 |      9 / 6 |               3 / 2 |              9 / 6 |
| frontier/olympiad  |      100 |    64 / 65 |    98 / 97 |      2 / 3 |             29 / 32 |              6 / 6 |
| frontier/research  |       60 |      2 / 4 |    55 / 58 |      5 / 2 |               6 / 8 |             8 / 12 |

HLE 总成功27/200→28/200；judge未决111→102，均由原 strict_json
和固定预算保留，不能删除后重算成“准确率提升”。仅固定已判结果的条件缺失上下界分别13.5%–69.0%和14.0%–65.0%，不是置信区间，也不表示未决已被确认为错误。Frontier分别保留原Olympiad/Research计分；Research有效判分均值2.539273→2.618103只覆盖55→58个有效判分，完整60任务的成功下界为2/60→4/60。所有原汇总含学科分组、truncated-with-grade及审计标志，均在
`quality-*-summary.json`；judge无效111/102和7/5已经纳入分母。HLE校准误差沿原冻结grader保留null及原始说明，没有替换上游分箱口径。SWE和Terminal的任务解决质量未测。

## PegaFlow 实际触发与零复用

正式五项 engine save-success 分别12/8/0/402/390（共812），load-success、load/save failure均为0；各阶段本地和外部prefix
hit增量为0。Frontier阶段没有save，不外推为该阶段有保存收益。逐阶段初始化计数、sidecar字节和容量/常驻量见[机制](connector-mechanism.json)及[开销补充](connector/PEGA-CONNECTOR.md)。

本轮fresh
cache_salt要求零复用，因而没有从CPU/SSD取回KV的事件，不能把变慢归因于SSD回读。实际配置和容量观测为本机30GiB主机池，容量32212254720B；保存开销和连接器安全限制是可能的代价，但本试验没有分别隔离其因果占比。此标准负载没有创造缓存恢复收益的机会，故不据此否定其他复用场景，也不更换负载追求正数。

## C8 资格与可复算数据

[配对摘要](c8/paired-summary.json)｜[逐请求数值及phase边界](c8/requests/manifest.json)｜[独立SSE检查](c8/independent-verification.json)。Native343、Pega341条均完成，按原4K/16K、C8、60s预热/300s正式提交窗口，固定256输出并排空；该合成资格与五项EOS指标分开。4K正式194.791542→195.060650
tok/s（+0.138%）；16K正式66.410510→65.283506（−1.697%）。这些tok/s分母包括drain，不能改称固定\[0,300)产出。

C8实际保存2476次、sidecar累计保存116256702464B、恢复0、失败0。该证据证明保存链路工作，不能用接近零的吞吐差或保存调用证明复用收益。逐请求记录仅含数值和哈希，受控索引绑定原件。

## 准备、内存和成本

| 服务                           |  准备秒 | 准备双卡秒 | sidecar就绪至stop秒 | 正式容器memory.current最大观测B | 正式各卡HBM最大观测MB    |
| ------------------------------ | ------: | ---------: | ------------------: | ------------------------------: | ------------------------ |
| service-native                 | 475.685 |    951.371 |              不适用 |                     18075115520 | {"2": 56584, "5": 56583} |
| service-pegaflow               | 504.742 |   1009.485 |   6499.852251291275 |                     49357201408 | {"2": 56669, "5": 56668} |
| service-judge                  | 460.100 |    920.199 |              不适用 |                     15418048512 | {"2": 56436, "5": 56430} |
| qualification-service-native   | 467.674 |    935.349 |              不适用 |       见分开的startup telemetry | null                     |
| qualification-service-pegaflow | 486.616 |    973.232 |  1181.3420205116272 |       见分开的startup telemetry | null                     |

准备包括源码准入、模型装载、图编译和binding；编译日志分项原文由受控索引保留，不能把重叠日志时间简单相加。主机池、sidecar常驻时间、startup-model/startup-sidecar各自采样、judge及额外HLE图像资格成本分别保留于
[runtime-overheads.json](runtime-overheads.json.gz)、[diagnostic-overheads.json](diagnostic-overheads.json)及COMPARISON。HLE额外红/蓝资格每臂2请求、max32，不计入672/720；这4份原始SSE另行独立重放通过，见[audit/diagnostic-raw-sse-verification.json](audit/diagnostic-raw-sse-verification.json)。

上述内存是留存采样区间内整个容器和设备的总观测；不是精确分配峰值，也不能把两臂总字节差称为MOD独占增量。相同费率下 solver
双卡秒/输出词元是吞吐倒数的资源代理；真实账单、有效FLOP、MOD独占主机/HBM增量均未测，保留null。无证据声称更低总体成本。

## 身份证据与边界

实际历史资格/服务前CPU子进程import、source/compiled-extension字节校验和跟踪diff证明 stock vLLM0fc695f、scheduler9ae149（无Bid
bridge），Ascend5cb98ca及共同fe00/b7载体1264资源（含804个CANN文件）。三次正式服务各自runtime-identity和command/ready/PID/wait保留；每服务startup/steady两份有效worker/thread
affinity观测未见warning/skip，quota800000/100000、cpuset0–191。包为torch2.10.0+cpu、torch-npu2.10.0.post4、vLLM0.23.0+empty、Ascend0.23.0、Transformers5.5.4、tokenizers0.22.2；重复使用已验证e2
wheel（SHA df687c8d…，7481020B），本轮没有重新编译。

这些import是与服务环境相同的准入子进程检查，不是对正在运行Engine进程做loaded-import RPC或独立/proc
argv采样；此类直接观测没有补造。Popen前命令、实际PID、健康、worker日志、affinity与运行期保存计数构成各自独立证据层。绑定文件中的 preparation_only、旧81
actual_import_tree 等历史准备描述保留原样，不当作本轮实际路径；本轮实际载体路径见runtime-overheads。

14权重分片沿用此前完整逐字节哈希凭据，并核对当时当前文件size/mtime；不是本次重新读取70GiB权重的证明。原main
manifest原字节SHA8e4ef474…，formal内controller重新序列化副本SHA见COMPARISON；JSON值相等但原字节不同，两个哈希都保留。C8绑定原main8e4，不以formal副本替换。

## 原件、失败历史和可追溯性

[正式原件索引](controlled-originals-index.json)｜[全campaign受控索引](controlled-campaign-index.json)｜[实际等待与释放](audit/historical-receipts-supplement-v2.json)｜[原grader与计划哈希](original-grader-sources.json)｜[CPU核验源码](audit-sources/audit_worker.py.source)。原问题、答案、图像、对话、私有Engine/评测源码和原始SSE留在获授权NAS；公开数值、源码与哈希供复算，不能用缺少公网原件下载链接冒充不存在原件。

2026-11-14首次纯CPU预生成准入失败（RC1、模型启动/生成0）及后续修正RC0均保留，不能删除失败。正式outer/c8/formal真实wait都是0、无signals，5个owned服务均已释放双卡；Pega
sidecar退出−15为owned
stop的SIGTERM，不能写成其自身exit0。旧通用receipt导出器对outer文件名的查询不匹配，在新的supplement中按实际文件名补足，缺失的SDK外层wait仍为null。

本次CPU审核逐一验证原formal
manifest4585成员，并对metadata/qualification/logs/formal全库存前后校验一致；所有原件未变。公开包若因体积使用gzip，原字节哈希仍对应解压内容，见发布侧compressed-members映射；压缩不改原manifest。

可用于最终条目的表述：完成PegaFlow在Qwen3.5-35B-A3B、双910B2上的五项MOD对照及共享judge，保存链路实际触发且全部正式请求完成。五项吞吐相对Native分别下降1.85%、2.15%、2.49%、18.22%和1.30%；fresh-salt标准下零恢复，未证明缓存复用或总体成本收益。质量未决与截断按原规则完整保留，结果作为负面边界公开。[结果报告](README.md)｜[逐请求结果索引](summary.json)。

## 公开文件存储说明

超过单文件体积限制的 JSON 以 `.json.gz` 原字节压缩保存；解压后校验原 manifest 的大小和 SHA256。完整映射见
[storage-map.json](storage-map.json) 与 [compressed-members.json](compressed-members.json)。执行源码以
`.py.source` 保留真实运行字节，复现时复制回原名；原传输清单位于 `audit/original-transfer-manifest.json`，描述发布前路径，格式化后的报告由项目根
manifest 校验。公共哈希及数值不等同公开原始 SSE。
