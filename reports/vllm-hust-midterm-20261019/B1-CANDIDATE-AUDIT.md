# B1候选结果审计

日期：2026年10月19日，北京时间。

B1定义：**当前已实测最优的vLLM-HUST优化方案**。按同工况、同指标选择，可使用任意适配MOD及兼容组合，不限定为BetterScale。吞吐最优与尾时延最优可以属于不同方案，不拼接为一次部署结果。

## 已发布选择复算

`b1-selection-audit.json`记录全部已发布候选列表的30个单元格、候选值、来源和文件哈希。按各单元格指标方向重新选择，30项均与已发布选择一致。该检查验证选择算术，不替代原始实验质量审查。

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
| KV materialization arrival control | Qwen3.5统一SWE、Qwen2.5控制实验          | 调度候选；须保持外部输入和到达负载，计入全部排队时间                              |
| BidKV                              | Qwen3.5 TP2/SWE、TP4/EvoScientist        | KV压力控制候选；需匹配拓扑和压力，不能直接填MMLU                                  |
| DLA、KV tiering                    | Qwen3.5统一SWE                           | 保留实测值；相关观察未记录控制动作，不作机制收益归因                              |
| Pipeline microbatch                | Qwen3.5 PP2/SWE                          | 拓扑不同，保留独立工况                                                            |
| ADM                                | Qwen3 MoE W8A8 DP4                       | 模型、精度和DP不同；不直接替换当前TP2/DP1结果                                     |
| vSpec                              | Qwen2.5-14B                              | 模型不同；不直接替换                                                              |
| LatchMoE                           | Qwen3                                    | 模型不同；不直接替换                                                              |
| SimLLM                             | Qwen3.5重放                              | 重放不等同当前真实HTTP性能测量                                                    |
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

本轮未发现可直接替换当前MMLU-Pro隔离配对方案的更高结果。后续候选为E24/R36的中间宽度，以及学校运行栈下KV
materialization调度适配；后者需保留相同外部到达负载和完整排队计时。PegaFlow或KVCompress与State组合须先核验缓存所有权，不能仅拼接单MOD结果。本轮没有完成这些后续候选的测量。

在仓库根目录执行：

```bash
python reports/vllm-hust-midterm-20261019/audit_b1_candidates.py \
  --root leaderboard-data/dataset-validation \
  --website-root /path/to/vllm-hust-website \
  --output /tmp/b1-selection-audit.json
```
