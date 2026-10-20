# B1候选结果审计

日期：2026年10月19日，北京时间。

B1定义：**当前已实测最优的vLLM-HUST优化方案**。按同工况、同指标选择，可使用任意适配MOD及兼容组合，不限定为BetterScale。吞吐最优与尾时延最优可以属于不同方案，不拼接为一次部署结果。

## 2026年10月20日扩展审计

vSpec
MTP局部argmax已完成学校0.28受限适配及三组同工况OFF/ON补测。六轮共1200请求、0失败，双rank模型头一致性、真实草稿图捕获与重放、Microflow及KV控制、设备释放均通过校验。该候选未超过现有B1，保留Microflow吞吐/TTFT及E16
KV的TPOT最优配置；完整中位数、范围、配对变化和原始证据见报告5.6节与`vspec-school028/summary.json`。不据此推断vSpec其他组件或其他工况无效。

当前MMLU-Pro吞吐B1为E24/R36 + KV recompute + FA Microflow三次运行中位数824.894609
token/秒。固定200请求、原始输入哈希、inf/out256、BF16、TP2、MTP2、APC、async、FULL及每卡20 GiB状态预算、8 GiB host缓存。

本次重新读取22个仓库的默认分支、归档状态、完整commit、README、可发现的manifest及选定实现。原始文本与SHA256见`b1-mod-source-audit-20261020.json`；前次9仓库回执保持不变。下表为源码筛选，不表示22个MOD均已完成设备测试，也不把未触发控制动作的安装视为优化。

| 候选                                                                 | 本工况审查结果                                                                                                                                   | 后续资格条件                                                                                                   |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| vSpec MTP局部argmax                                                  | 已支持Qwen3.5，不能继续归类为仅Qwen2.5。当前Ascend proposer仍走完整logits收集，Qwen3.5 MTP模型已有LocalArgmaxMixin；存在减少TP词表通信的具体机会 | 只接入局部argmax，不启用动态gamma、refill或其他patch；双rank原生/局部token一致、FULL图实际重放、独立OFF/ON配对 |
| Hybrid Spec GDN元数据快速路径                                        | 已有真实eager机制证据，不能概括为完全没有设备实现。但当前try_build要求gamma1、eager、单请求、APC关闭、async关闭；MTP2直接返回None                | 当前BetterScale已替换Builder.build并使用owned元数据，须先适配其状态所有权；不能靠删除guard启用                 |
| Split-Batch / Cascade                                                | 学校0.28/Qwen3.5已进入支持范围，但speculative_config仍使Cascade和dual-pad拒绝；共享前缀阈值也不匹配本轮短输入                                    | MTP验证行及图桶实现与一致性验收                                                                                |
| KVCompress / PyramidKV                                               | 压缩实现存在；当前短输入/out256不满足所审查推荐gate，学校运行栈仍需专门适配                                                                      | 原负载出现真实压缩机会、无损或质量约束验收；不缩减预算人为制造压力                                             |
| PegaFlow / KV tiering                                                | 已有存储与缓存实现；当前BetterScale调度器要求connector为None，不能直接叠加其offloading connector                                                 | 缓存所有权协同或同预算独立方案验收                                                                             |
| BidKV                                                                | 有typed victim选择实现；前轮页池占用约3%至4%，没有抢占压力证据                                                                                   | 原负载真实抢占与策略动作计数                                                                                   |
| DLA / Request Throttling                                             | 前者候选缺predicted_length时弃权，后者普通无program身份请求旁路                                                                                  | 在不改变原请求语义的前提下接通输入和真实控制路径                                                               |
| OPset                                                                | 两项优化限定batch-invariant Graph/Triton路径，当前组合不属于该路径                                                                               | 单独实现与现有算子路径的适配；不把仅设置开关视为生效                                                           |
| OPfuse                                                               | 默认分支仍为patch模板，保留TODO和示例资源声明                                                                                                    | 真实目标函数、实现及数值设备验证                                                                               |
| Unified Comm / Layered Prefill / QoS Scheduler / Activation Sparsity | 可安装的合同、规划或策略实现存在，但README与manifest仍明确宿主接入未完成                                                                         | 对应HOST_CONTRACT、宿主实现和设备验证                                                                          |
| StateAxis async KV transfer / automatic prefix cache                 | 独立StateAxis协调器与所有权合同；组件结果不能直接归入当前vLLM/BetterScale服务组合                                                                | StateAxis provider接入、实际restore/hit与同预算端到端对照                                                      |
| LatchMoE                                                             | 专家卸载/驻留缓存实现存在，不是当前全驻留权重路径的可直接替换算子                                                                                | 保持本轮资源和APC/FULL配置的MoE集成与传输成本验证                                                              |
| ADM                                                                  | 当前DP1走原生旁路，TP2不等于DP2                                                                                                                  | 不以改变DP拓扑产生同工况结果                                                                                   |
| SimLLM                                                               | 新增真实Qwen3.5混合状态恢复，旧“只有重放”概括不再适用；其测量建议关闭chunked prefill/APC，相近提示复用属于近似路径                               | 精确前缀/状态所有权与当前APC组合接通；近似复用须独立任务质量约束                                               |

首选接入点为vSpec的`_sample_mtp_local_argmax`。同工况源码审计未证明其性能优于Microflow；B1数值仅在设备一致性与配对结果通过后更新。

校验：`pytest -q tests/test_mod_source_audit_20261020.py`复算全部来源摘要，并执行固定Hybrid
Spec函数的MTP2拒绝分支，确认拒绝发生于任何张量或设备访问之前。

## 已发布选择复算

`b1-selection-audit.json`记录全部已发布候选列表的33个单元格、候选值、来源和文件哈希，包含新增MMLU-Pro同工况的3项服务性能指标。按各单元格指标方向重新选择，33项均与已发布选择一致。该检查验证选择算术，不替代原始实验质量审查。

统一SWE 900秒工况的输出吞吐选择为：C1 KVCompress，C2/C4/C8/C16
PegaFlow。BetterScale和Pipeline的独立工况保留各自基线，不跨工况竞争。BidKV的TP4/EvoScientist结果不填入TP2/MMLU-Pro。

## MOD证据与补测适用性

本次检索范围为网站当前性能登记的13个MOD，以及数据集矩阵中的Request
Throttling。登记来源固定于网站提交`89b351f343c6259c0bddb6c4e8ed752bbb50affe`；具体观察ID、报告URL和来源文件SHA256见审计JSON。这不是全组织各仓库最新源码兼容性认证。

| MOD                                | 已有证据                                 | 对当前MMLU-Pro补测的用途                                                          |
| ---------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------- |
| BetterScale                        | Qwen3.5数据集性能、SWE宽度实验           | 优先补测同输入的执行宽度候选；旧631.6 tok/s与隔离529.8 tok/s分别保留              |
| KVCompress                         | Qwen3.5统一SWE、Qwen2.5 Coder长上下文    | 有真实控制动作；须在MMLU及学校运行栈适配后测量，不能搬用SWE增益                   |
| PegaFlow                           | Qwen3.5统一SWE                           | 多并发吞吐较高；与State缓存组合须先验证缓存所有权，不能直接叠加插件               |
| KV materialization arrival control | Qwen3.5统一SWE、Qwen2.5控制实验          | 缓存复用/重算控制；学校栈已测逐请求重算路径，不改变外部到达负载                   |
| BidKV                              | Qwen3.5 TP2/SWE、TP4/EvoScientist        | KV压力控制候选；需匹配拓扑和压力，不能直接填MMLU                                  |
| DLA、KV tiering                    | Qwen3.5统一SWE                           | 保留实测值；相关观察未记录控制动作，不作机制收益归因                              |
| Pipeline microbatch                | Qwen3.5 PP2/SWE                          | 拓扑不同，保留独立工况                                                            |
| ADM                                | Qwen3 MoE W8A8 DP4                       | 模型、精度和DP不同；不直接替换当前TP2/DP1结果                                     |
| vSpec                              | Qwen2.5及Qwen3.5 MTP                     | 当前局部argmax候选见10月20日扩展审计，不移用其他数据集收益                        |
| LatchMoE                           | Qwen3                                    | 模型不同；不直接替换                                                              |
| SimLLM                             | Qwen3.5状态复用                          | 已有真实恢复路径；与当前APC和独立提示组合的边界见10月20日审计                     |
| DiffSpec                           | Qwen3.8                                  | 模型不同；不直接替换                                                              |
| Request Throttling                 | MMLU-Pro同输入、0.23运行栈，三次正式重复 | controller中位360.525 tok/s，普通单轮转发；非本轮0.28同栈配对，不声称调度机制收益 |

## 候选筛选方法

首轮选择BetterScale E32/R36，作为当前E16/R36的执行宽度候选。选择依据为已有SWE宽度实验，而非预先宣称MMLU收益。保持同一模型、BF16、TP2、学校运行栈、20 GiB
State/rank、8 GiB host cache/rank、MTP2、APC、FULL、async及200条MMLU输入；外部无限到达率、256输出上限不变。执行席位与相应图捕获形状为候选变量。

输入SHA256为`fb79b6fa98f232282a948e4a17f404f80c03dc2bae05af57b95b9cc0651b3506`。每次运行保留启动命令、脚本哈希、服务日志、原始客户端结果、逐卡采样、退出及资源释放回执。筛选结果单独记录，不替换第五章既有配对；优胜候选需独立确认，并保留质量与尾时延约束。

## 复算

首轮筛选已完成，运行`mmlu-b1-e32-r36-screen-r1`：200成功、0失败，51,200输出token，106.980744636秒，478.590798506
token/秒。相对原E16/R36的529.808250753
token/秒下降9.667167730%，不选作吞吐B1。运行实际使用E32、R36、MTP2、FULL及APC；全部设备释放。原始证据和派生统计见`b1-e32-screen-evidence.tar.gz`、`b1-e32-screen.json`。

后续E24/R36和E16/R36 + KV materialization均已完成测量，分别为561.509092499和695.250621960
token/秒。完整结果、各指标选择与控制证据见报告5.4节和`b1-current-best.json`。原输入没有复用锚点，KV路径的200次策略决策均为逐请求重算；不把该结果解释为分段复用收益。

## 2026年10月19日源码准入记录

本次继续审查9个有明确实现或近期更新的仓库默认分支。`b1-mod-source-audit.json`保存审计时间、完整commit、README、相关manifest/宿主合同与部分实际实现的内容及SHA256。以下是当前MMLU-Pro工况的筛选结论，不是MOD的一般性有效性结论，也不是所有组织仓库的完整兼容认证。

固定输入为200个独立单轮请求，实际输入925至2,502 token，输出均为256 token。保持既有每卡20 GiB状态、8 GiB
host缓存、TP2和原始到达时序。不能通过增加program身份、重复前缀、缩减缓存或延长输出，把其他工作负载结果冒充本工况补测。

| MOD                   | 最新实现/约束                                                                                      | 本工况决策                                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Prefill Microflow     | 已完成学校0.28的FA-only适配、双rank同分片一致性与真实target图重放核验                              | 已完成三组同负载OFF/ON；中位数及完整证据见5.5节与`microflow-school028/summary.json`。GDN移植及任务质量评分未完成 |
| Split-Batch / Cascade | Cascade和dual-pad在speculative_config存在时拒绝多查询行路径                                        | 当前MTP2组合不触发；需要支持MTP验证行后再作为同组合候选                                                          |
| KVCompress            | 当前主线宿主范围0.29，另有精确0.23旧栈；推荐Qwen配置输出guard为2,048、KV预算4,608                  | 本栈需适配，且本轮短输入/out256不触发推荐压缩；不改负载或强压缩制造测点                                          |
| PyramidKV             | 依赖KVCompress共享适配器；其MMLU五样本配置已明确低于4,096 gate                                     | 当前输入最长2,502，缺少该配置的控制机会；不作为预计提升候选                                                      |
| PegaFlow              | 外部KV存储/connector；当前实测BetterScale `LiveStateScheduler.__init__`明确要求`connector is None` | 不能直接叠加；后续需协同缓存所有权或独立同预算部署验收                                                           |
| BidKV                 | 当前typed preemption API在学校宿主存在；只改变池满后的victim选择                                   | 既有KV测量日志页池占用约3%至4%，没有抢占瓶颈证据，暂缓设备补测；不是宣布其不兼容或无效                           |
| DLA                   | 当前README明确真实候选缺少predicted_length，策略弃权                                               | 需预测输入与候选传递真正接通后再验证控制；本轮不填优化值                                                         |
| Request Throttling    | 默认实现普通无program身份请求直接旁路                                                              | 原请求未提供program身份，不能新增身份改变工作负载后比较；本轮不宣称调度收益                                      |
| Hybrid Spec           | 安装包仍为import_only；另有显式eager实验工具，执行计划execution_ready=false                        | 已有有界设备机制证据，但当前Graph/async/APC/MTP2组合未获资格；见10月20日扩展审计                                 |

设备补测选择E24/R36 + KV
materialization逐请求重算，原因是执行宽度与KV控制分别已有同输入的实测依据，且不引入缓存所有权冲突。该项是已适配MOD的组合补测，不代表上述9个MOD均完成运行测试。原E16 +
KV结果保留，所有候选按指标独立比较；本轮实测结果以报告5.4及原始归档为准。

在仓库根目录执行：

```bash
python reports/vllm-hust-midterm-20261019/audit_b1_candidates.py \
  --root leaderboard-data/dataset-validation \
  --website-root /path/to/vllm-hust-website \
  --output /tmp/b1-selection-audit.json
```
