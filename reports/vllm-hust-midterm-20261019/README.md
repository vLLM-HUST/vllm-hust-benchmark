# vLLM-HUST中期测试记录

当前定义：B0固定为vLLM 0.18 + vLLM-Ascend
0.18；B1是当前已实测最优优化配置，基座版本不限但明确标注。其他版本Native仅为版本对照，不称B0。原始归档中的历史运行ID保持不变。

主报告：[中期测试报告](%E4%B8%AD%E6%9C%9F%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A.md)，当前V0.75采用[项目里程碑大纲V1.5课题四](milestone-v15/README.md)。旧报告快照及原始证据保留，当前整体验收尚未完成。

**指定0.18吞吐对照已存在：** C16 SWE-prefix-reuse 900秒窗口，B0 247.2656、B1 885.9611
token/s，3.5830倍（+258.30%）。见[复用审计](../c16-baseline018-reuse-20261022/README.md)。该比值描述两套完整部署；客户端运行时版本和B1设备占用原始证据仍待补核，不能据此宣布完整验收或把收益归于单个MOD。MMLU-Pro等其他工况的0.18配对仍分别判断。

第五章列出数据集性能结果。下述合成长输入实验采用不同运行栈及缓存配置，作为独立诊断归档，不纳入第五章方案的验收判定。

## M01 同工况参考补测

当前V0.75按[V1.5课题四](milestone-v15/project4-v1.5.txt)报告`C=F/(k×T×10¹²)`和`η=T参/T×100%`，有效工作量、匹配无插桩计时及模型依赖参考仍未闭合。见[当前方法与状态](milestone-v15/method-status.json)。旧经验参考仅作历史插桩结果；[75种通信图参考](m01-recovery/collective-graph-20261023/README.md)已完成，不再列为未测。

V0.64纳入GDN递归、WY/KKT、卷积、HBM及同接口通信参考，18类计算参考和设备最终预处理映射已完成汇总核对。
完整证据见[参考补测与汇总](m01-recovery/reference-closure-20261022/README.md)。
V0.65完成本次插桩请求窗口的经验参考核算：达成率30.2443%–30.2505%，未达到内部50%阈值。不外推为未插桩服务结果，也不替换正式B1。见[请求窗口核算](m01-recovery/request-assessment-20261022/m01-request-window-assessment.json)。含请求前后开销的[设备包络核算](m01-recovery/accounting-assessment-20261022/README.md)仅作诊断保留。

## 恢复环境性能与成本配对

V0.59新增5.10节：两臂各完成200请求、零失败。请求阶段吞吐提升29.20%，平均TTFT降低10.23%，平均TPOT降低4.43%，请求阶段资源成本降低22.60%；含启动、编译与退出的全流程成本增加57.13%，该全流程增幅不是V1.5当前成本否决条件，货币费用与同区间利用率判定仍待核验。P99
TPOT增加121.48%。单次配对不推断长期摊销或统计显著性。

完整原始记录、独立预检查、空编译缓存、退出证据及首次协议偏差结果保存在[m06-paired-20261021](m06-paired-20261021/archive.json)。本组不覆盖既有环境结果。可运行
`python render_m06_pair.py` 复算展示表；仓库测试 `tests/test_midterm_m06_pair.py` 从归档请求与计时复算指标。

## 当前B1长输入补测

V0.60新增7.3节独立接口修复确认：启用原生`qwen3`思考解析器并兼容当前`reasoning`字段，原六题、完整请求及严格正文评分器不变。资格请求通过后六题各执行一次，正文严格匹配6/6、零失败，思考字段独立保存，四卡已释放。原0/6及补充解析6/6记录不改写。原始SSE、源文件、固定计划与校验清单见[m05-parser-confirmation-20261021](m05-parser-confirmation-20261021/archive.json)。该结果是合成检索与接口确认，不是主数据集质量或吞吐收益。

V0.57补充恢复环境下E24/R36、KV materialization与Microflow的六题合成检索。实际输入16,457或32,843
token，6/6完成、零传输失败；正文严格匹配0/6，按唯一`</think>`边界补充解析的最终答案为6/6。原始判分不改写，不将思考文本未分离解释成六题检索码错误。完整请求、SSE、首次零请求控制器失败、第二次空缓存运行和资源释放证据见`long-context-current-b1-20261021/archive.json`。

## 数据集配对补测

B1定义为“当前已实测最优的vLLM-HUST优化方案”。按同工况、同指标从已测MOD或兼容组合选择；本章保留具体候选的配对证据。候选范围、选择复算及补测依据见[B1候选结果审计](B1-CANDIDATE-AUDIT.md)。

新增MMLU-Pro同机配对沿用原200条输入及学校栈优化配置，开发栈对照与优化配置各200成功、0失败。吞吐、平均TTFT、平均TPOT及请求区间资源成本相对开发栈对照达到数值阈值，不是相对固定B0的验收结果；P99
TPOT与完整生命周期成本未改善。完整配置、统计边界和结果见主报告5.2节，数值由`aligned-mmlu-results.json`生成。

E24/R36与KV
materialization的同工况补测见5.4节。`b1-current-best.json`按指标记录当前已测最优配置；`publish_mmlu_followup.py`从原始结果摘要生成网页矩阵，吞吐与时延分别选择，不拼接为同一部署。原始输出与设备采样保留于两个`b1-*-screen-evidence.tar.gz`归档。`b1-admission-attempts.tar.gz`保留设备锁占用及端口尚未释放时的两次未启动回执，不计为测量结果。

在本报告目录执行以下命令，可使用CPU在独立临时目录中复算：

```bash
work=$(mktemp -d)
tar -xzf aligned-mmlu-common.tar.gz -C "$work"
tar -xzf aligned-mmlu-b0.tar.gz -C "$work"
tar -xzf aligned-mmlu-b1.tar.gz -C "$work"
python verify_aligned_pair.py "$work/20261019-midterm-aligned" mmlu-b0-r2 mmlu-b1-r2
```

`aligned-profile-probes.tar.gz`保留计数探针原始轨迹；`aligned-profile-control.tar.gz`保留整模型补采的失败回执和离线恢复索引。双rank算子明细已恢复，覆盖见`aligned-profile-coverage.json`；自定义算子形状及物理流量计量仍不完整，未给出有效算力验收结论。整模型大轨迹保留于主报告列明的本机路径，未完整上传Git。

10月20日的[M01计量补证](m01-recovery/README.md)恢复了包含FULL图重放的矩阵形状执行次数和通信API载荷，并完成物理HBM采样探针校验。矩阵形状算量、通信API载荷和物理流量分别保存，不互相替代；新增脚本支持CPU复算与采样边界校验。

10月21日的[恢复环境同工况计量](m01-recovery/model-runtime-20261021/archive.json)包含两轮MMLU-Pro完整模型运行，每轮200请求全部成功。第二轮360秒硬件采样覆盖请求、末次前向事件及全部导出同步三个区间。原始硬件数据库、请求记录和源码保存在benchmark
Release，逐文件校验清单与便携复核包入库。插桩吞吐不替换第五章性能数据；完整有效F、跨流依赖及匹配参考容量仍有证据缺口。V0.56 对 M01
的判定为证据受限、未通过本轮验收，不等于实测低于阈值。

新增的[投影算子参考标定](m01-recovery/projection-references-20261021/archive.json)覆盖恢复环境两卡的 121 种密集投影形状、36,708
个专家分组分布和 45 种词表投影形状。各卡原始重复计时、数值检查、 设备状态及退出记录均保留；原始大文件由索引定位至 Release。组件参考不等于完整有效算力达成率。

[词表通信参考](m01-recovery/logits-collective-reference-20261021/archive.json)覆盖 76 种 TP2 BF16/INT64
载荷，每种三组、每组 512 次；全部通过数值与物理计数器校验。归档保留重复计时与离散度，不将组件通信参考冒充整模型 HCCL 覆盖或 M01 达成率。

[恢复环境完整回答质量复核](quality-regression/recovered-20261021/archive.json)保留 200 题全部
SSE、固定评分器、输入核对、源码和清理证据，完整归档分片随仓库提供。结果为 146/200、0 请求失败、22 条达到输出上限；相对既有 B0 和 Microflow B1 分别为
-0.50、-1.50 个百分点，不作质量等价声明，不替换原吞吐结果。

新增的[GDN 元数据发布与预处理图重放验证](m01-recovery/preprocess-publication-20261021/archive.json)覆盖两卡、两个元数据
bank、四档图容量及变更元数据重放，保存原始张量和独立 CPU 复算证据。该项为局部链路验证，不等于完整模型并发执行流闭合或 M01 验收通过；具体配置、误差和限制见主报告。

## 合成长输入诊断

2026年10月19日完成Qwen3.5-35B-A3B BF16、TP2、C8合成长输入B0补测： 每条实际输入16,384 token，60秒资格检查后测量300秒，窗口内输出79,437
token， 264.79 token/秒。312条请求全部成功，其中304条在窗口内完成、8条在排空完成。 B1也已完成：窗口内57,585 token、191.95
token/秒，230条请求全部成功。 本轮吞吐下降27.51%，平均TTFT、TPOT及资源系数均退化，如实保留。
这是不同运行栈的整体部署对照，不是单组件消融、主测数据集质量结果或900秒Frontier点。

`long-input-verification.json`是B0独立复核结果，`long-input-comparison.json`是B0/B1对照； `results.json`是既有报告复算，
二者不可拼接成一次配对实验。计时来自AsyncLLM输出事件，不是HTTP端到端计时。

`synthetic-b0-evidence.tar.gz`保留三次启动的原始日志、输入、脚本、逐请求事件、 设备采样、配置及SHA256清单，不含编译缓存。前两次失败均保留。成功运行退出码0，
自建进程已释放，析构与共享内存清理告警详见主报告。

`synthetic-b1-evidence.tar.gz`保留B1同类原始证据及组件加载检查，
`calibration-evidence.tar.gz`保留双卡BF16计算、逻辑内存复制和HCCL标定。 合成输入输出序列并不完全一致，不能宣称质量无损；标定不等于整模型有效算力。

在空目录解压归档后，可只用CPU执行复核，无需启动模型：

```bash
tar -xzf synthetic-b0-evidence.tar.gz
python verify_long_run.py midterm-run-20261019-r4
cd midterm-run-20261019-r4
sha256sum -c SHA256SUMS
```

在报告目录解压B0和B1归档后运行：

```bash
python compare_long_runs.py midterm-run-20261019-r4 midterm-b1-20261019 comparison.json
```

原始运行脚本中的绝对路径用于追溯本机环境，不应直接在其他机器执行。 主报告列出的既有来源依赖原报告仓库或保留的私有证据，未声称全部包含在本归档中。

- Terminal原答案恢复评分：[规范结果](../terminal-openssl-model-recovery-20261026/README.md)。

- Navidrome增量：[规范结果](../swe-native018-navidrome-20261027/README.md)。

- Terminal部分产物恢复评分：[规范结果](../terminal-cancel-async-partial-recovery-20261027/README.md)。
