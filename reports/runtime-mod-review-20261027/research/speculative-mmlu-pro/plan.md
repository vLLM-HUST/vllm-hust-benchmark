# ascend87：vSpec / DiffSpec 严格 MMLU-Pro 评审计划

2026-10-26。Fletcher明确授权：允许打兼容性补丁调通两种插件，在ascend87（用户称8卡）运行MMLU-Pro。此授权替代此前“完全不修制品”的本任务限制，不授权改变验收语义、伪造/选择性丢弃证据或修改共享环境。原始准入失败继续保留。

## 当前状态

用户纠正目标为 ascend87（原 ascend81 的两次 Connection refused 不属于该宿主）。已连接 ascend87：8×910B2 64 GiB，首次检查无 NPU
进程；native runtime 映射 logical0..7 → physical0..7。共享环境 torch 2.10.0+cpu / torch_npu 2.10.0.post4；Core
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，Ascend `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。已有
Qwen2.5-14B-Instruct 与 Qwen3.8-27B；draft 与固定revision数据已下载并校验。14题pilot及对照已完成；当前阶段以35B
vSpec为主，实时占卡与退出状态见progress及运行记录；用户已批准固定280题阶段（不自动扩全量），运行状态与补丁见 [progress.md](progress.md)。

## 评审问题与证据边界

目标是严厉检验既有收益主张，不以得到负结果为成功条件。准确率、输出语义与吞吐如实记录；正结果及不确定结果同样保留。MMLU-Pro只能决定此负载适用性，不能以短输入/输出未加速就否定DiffSpec的长上下文主张，也不能用不同target的绝对分数横向判断两插件优劣。

- vSpec：严格验收下adaptive是否优于target-only和同proposer固定gamma；refill分开控制，禁止confidence margin。
- DiffSpec：收益是否超过target-only及可比原生Eagle3；检索/working-cache改造是否抵消接受收益。记录每位置与总体draft接受率、提交tokens与阶段耗时。原生Eagle3若不可比/不可用明确缺失，不伪造消融。
- 制品迁移失败、算法负收益、任务得分下降分别报告，不把兼容失败当作性能反证。

## 模型与补丁

先核查宿主软件/卡型、已有模型、tokenizer与draft兼容性，再冻结模型和revision。vSpec历史Qwen2.5-14B+0.5B与DiffSpec已验收Qwen3.8-27B/Eagle3并非同一合同，不能为“公平”强塞不支持的模型。

在任务私有checkout/venv修补；每项保留原始异常、最小diff、对应回归测试。API、导入顺序和张量接口的修复不得删除功能门禁；修改门禁必须以实际等价支持证据解释，不强制绕过。任何更改target数学、采样或拒绝验收的补丁退出性能比较。DiffSpec
runtime超过1500行，若需深入改造混合职责而非局部兼容修补，应先提出结构/投入边界。

## MMLU-Pro协议

采用官方TIGER-AI-Lab/MMLU-Pro的生成式5-shot CoT思路，固定dataset
revision、验证集示例、prompt模板、答案提取和计分实现后运行；不能以multiple-choice
loglikelihood计时冒充投机解码性能测试。解析失败/截断/服务失败均明确统计；不使用随机答案兜底掩盖解析失败。

同一target各臂使用相同prompt顺序、temperature0、tokenizer、输出上限、停止条件、dtype、TP、缓存和graph/async条件。先做native重复一致性，避免把后端本身非确定性全归插件；严格验证仍禁止近似接受。

选择两类对照：匹配栈target-only用于机制归因；实用原生配置用于部署价值。不得通过关闭原生优化制造弱baseline，也不能将两类对照混为单因素实验。

准确率报告完整分母、各学科/总分、逐题得失和配对不确定性；性能报告TTFT、TPOT、请求E2E、总wall/output
tokens/s、输入/输出长度、截断/失败率、实际draft计数。输出变化时不将总耗时变化全解释成计算加速。

## 资源预算与阶段门

1. 只读环境与物理/逻辑卡映射，确认空闲和协作锁；不得默认八卡都空闲。
1. 补丁与CPU/小型验证先行。首轮调通最多按插件必要TP占卡，单次服务启动/小探针设15分钟上限，不遗留进程。初始调通预算每插件2小时；重复同一失败无新证据则停，不无限修课题。
1. 首次设备实验前固定分层小样本ID、输出预算与超时，仅用于调通/估算成本，不将其当最终评测或据得分筛选题目。
1. pilot后按真实速度冻结正式预算、全量质量运行与固定性能子集3轮配对（交错次序）；在正式输出前写明设置。如果全量三臂/四臂时间或模型下载远超合理预算，先向Fletcher报具体成本再决定。
1. 完成、失败或外部阻塞均保存证据与状态，释放自有进程及锁。禁止只选最快轮或因结果正向而追加直到负向。

官方入口：https://github.com/TIGER-AI-Lab/MMLU-Pro 。已完成部分14题调通pilot，尚未进行正式MMLU-Pro评测；不要把pilot当全量结果。

### Pilot 冻结（首次 MMLU-Pro 推理前）

按 category 字典序、question_id
升序，选每学科首题，共14题：2804,70,3526,10356,6826,11285,6001,4669,866,7687,5059,10774,9044,1986。输出上限4000，temperature0、top_p1；Qwen2.5-14B原始tokenizer单user
chat template，五个同学科示例；TP1 BF16、B4、max_model_len16384，APC/async关闭，FULL_DECODE_ONLY。每臂先16-token
warmup，不计入pilot；分开保存warmup后指标快照。pilot只估算调通与预算，不能用14题宣称正式准确率。卡间同时进行调通会有主机干扰，pilot耗时不作正式速度结论。后续正式性能必须隔离运行。

Adaptive pilot 使用 online policy、gamma1..4、refill0、margin未设置、entropy
stop关闭；不用B128历史profile外推B4性能。固定gamma4与native后续须同设置。

DiffSpec pilot 沿用相同14个ID、5-shot CoT、4000输出上限与同样B4/context16384；target为宿主Qwen3.8-27B、TP4
BF16，budget2048/chunk64/interval4、depth3与原制品auto verification。先通过四rank
RPC断言实际proposer和diffspec_cache，未生效运行不得进性能表。该target的默认chat
template（含其thinking行为）在各臂完全相同，不与Qwen2.5绝对分数比较。

## 2026-10-26：用户批准的 280 题阶段（推理前冻结）

Fletcher选择先做每学科20题、共280题配对对照，再决定是否值得全量；**未批准自动扩到12,032题**。

- 用 `freeze_screen.py` 对 pinned 数据按 `SHA256(lumi-mmlu-pro-screen-20261026-v1:category:question_id)`
  排序，每科取前20；排除已经查看结果的14道pilot题。SHA256仅用作跨Python版本稳定的抽样排名，不作逐文件校验仪式。具体ID在
  `evidence/screen-manifest.json`，在正式推理前写入。学科round-robin交错提交；每科前4个排名组成56题性能子集，不依据输出挑题。
- 正式准确率每臂跑一次280题。性能子集每臂三次、每次新engine，三轮顺序采用
  native/fixed/adaptive、fixed/adaptive/native、adaptive/native/fixed；DiffSpec同样轮换
  target-only/native-Eagle/DiffSpec。性能轮与质量轮分开，不挑最快轮。
- B32而非调试B4；TP1的三臂固定物理0，TP4三臂固定物理0,1,2,3；所有NPU实验串行。启动前映射、锁和占用重新验证。其余4卡不占锁；若发现其他任务共享主机/设备，应标注干扰而非宣称独占。
- 其余模型、prompt、temp0、4000输出上限、context16384和draft参数沿用pilot。APC与async关闭是匹配栈归因对照，**不是声称最优部署baseline**；B32不能替代原生完整调优。FULL_DECODE_ONLY
  capture buckets在`screen.py`，同family一致。vSpec margin未设置、refill0、online gamma1..4、trace关闭。DiffSpec
  budget2048/chunk64/interval4/auto，四rank witness与运行后cache metrics单独留证。
- 预热用原14道pilot，每题最多16输出tokens，不计时；每个engine相同预热流程。质量/性能计时均不含加载、graph捕获、RPC
  witness；包含真实generate过程和每个完成请求一次JSONL落盘。输出文本/长度差异单列，不把wall比直接称作kernel加速。保留完成请求，超时不得伪装成完整分母。
- B32先做不进入正式汇总的capacity
  smoke；若不支持，只依据报错做最小兼容/容量处理，在正式输出前重新冻结配置。每个正式质量臂最多2小时、每个56题性能臂最多30分钟；遇失败停止campaign，不无脑重试。总计6个质量臂及18个性能臂，不自动增加实验。
- 报告subject-balanced宏平均、逐题得失、配对不确定性、解析失败和截断。该分层抽样不是原始全量学科占比；不把280题得分叫作完整MMLU-Pro成绩。

### 正式DiffSpec输出前的机制边界补充

B32容量检查发现default
auto/linear不产生检索：见progress与AST门禁验证。仍按推荐auto配置测default实现的实际开销/分数，但不称为已验证完整attention-score
retrieval。追加witness只读取cache状态；不在此轮改变linear/tree选择、检索门禁或预算。tree路径源码只支持B1，不拿B1替代本轮B32生产负载，也不偷偷换目标合同。

## 用户纠正后的vSpec主实验：Qwen3.5-35B-A3B

此前14B质量轮降为补充，停止其性能重复，不改写/删除已有结果。35B用同一280题和56题性能子集，从 pinned vSpec
`qwen35-35b-a3b-frontier-mtp2.toml` 直接生成环境，不沿用14B draft_model设置。

三臂为35B原生target-only、原生固定MTP2（gamma2）、vSpec推荐adaptive
MTP（gamma2/4/6，refill8）。目标checkpoint内置MTP，不另配0.5B草稿。采用作者推荐TP2/B16、APC+async+chunked-prefill、FULL_AND_PIECEWISE、EP、BF16、memory0.9、block128、language-model-only；仅将256K上限收至本数据协议需要的16384，generation_config=vllm，采样仍temp0/top_p1/max4000。confidence
margin禁用，不用synthetic acceptance，不开trace。各臂按自身verify宽度生成原作者capture buckets，graph mode相同。

这项比较评估推荐adaptive+refill整体，不把收益或开销单归gamma控制器。对照保留原生MTP2，可区分MTP本身与插件策略增量。B16来自该35B推荐部署合同，不是依据此次MMLU-Pro成绩调参。实际engine/environment参数冻结于
`q35-engine-configs.json` /
`q35-environments.json`；同一family固定物理0,1且串行，依旧先capacity再质量再三轮56题性能。现有DiffSpec质量轮不抢停；其性能轮先延后，腾出下一阶段核验35B。

35B纠正后的容量准入例外：为避免整个DiffSpec质量阶段结束后才发现35B制品无法启动，允许35B
capacity在空闲物理4,5上与DiffSpec质量轮（0..3）重叠。此例外仅用于16输出token的启动/graph/接口检查；**所有受影响的DiffSpec质量轮wall都不作为性能结论**。后续56题三轮正式性能必须串行、同卡组；35B正式质量/性能仍用0,1。该变化发生在35B设备推理前，不依据得分选择配置。

35B正式输出前补充正确性门：出现完成请求至少16tokens且>90%为token0时，保存输出后终止，标记链路退化，不给未完成题目补分或从分母删除坏题。原因是27B原生Eagle与DiffSpec的旧B32
capacity共享这一症状；三臂35B capacity未触发。27B大规模比较暂停，先做小范围定位；其原生Eagle 280题结果不是有效DiffSpec算法反证。

## 27B/Eagle C04 revision (2026-10-27, before new quality runs)

The original 0.7-memory native Eagle path has a semantic failure independent of DiffSpec. Reverse
ordering and no-warmup probes show a cumulative request-position boundary; reducing the cache budget
to 0.4 removes the observed 56×16-token failure. This is a bounded workaround, not a proven upstream
root-cause fix. Shared source/packages remain unchanged.

For the existing authorized 280-question outcome, freeze a new **C04** comparison: all three arms
use `gpu_memory_utilization=0.4`, preserving target, draft, B32, TP4, original graph settings,
gamma3, prompt/parser, temperature0, seed0, 4000-token cap, question IDs, warmup and rotated
three-repeat perf56 order. Do not compare C04 arms against the original 0.7 native score/timing.
`screen.py` accepts only the explicit 0.4/0.7 cache budget alternatives and records resolved engine
config; `campaign27.py` uses C04 exclusively and distinct labels.

Admission: native Eagle and DiffSpec must each complete the same perf56 subset at up to128 tokens,
with actual proposer witness and saved outputs; continuous token-0 degeneration fails closed after
persisting the affected request. Full 12,032 questions remain unauthorized. No new selection based
on scores. Default DiffSpec auto/linear behavior is preserved; the confirmed retrieval gate
limitation is not silently rewritten into a different algorithm. A passing C04 result evaluates that
artifact as executed, not an unobserved full attention-score retrieval implementation.

### Performance-only B16 amendment (before any C04 performance run)

While the B32 quality run was progressing, native Eagle sustained ~98% cache occupancy and admitted
about24 concurrent requests under C04. This is a capacity-constrained serving result, not an
isolated algorithm throughput comparison. Preserve the running B32 quality contract and its outputs.
For the still-unstarted independent perf56 comparison, set **max_num_seqs=16 for all three arms**,
retaining C04 and every other setting. Labels use `screen-performance-c04b16-*` to prevent
accidental mixing. Repeat order and three fresh-engine repetitions remain fixed. Record actual
occupancy/preemption and score/output-length variation in each performance run; do not transfer B32
quality claims to B16 as if they were the same execution. No further adaptive tuning within those
repetitions.

The C04 B32 native-Eagle quality run ultimately reported **1488 preemptions**. Stop the queued
DiffSpec B32 quality run during initialization rather than spending another hour on the same
confound. Run only `performance-first` (one perf56 pass per arm at B16) next, with a zero-preemption
gate after each arm. Inspect that bounded result before purchasing `performance-rest` or a new B16
280-question campaign. Preserve incomplete scope explicitly: DiffSpec does not yet have a valid
completed three-arm280 comparison.

### 2026-10-27：Fletcher决定停止追加实验

Fletcher明确选择“停止追加，整理现有证据”。本轮收口：保留已完成的35B/vSpec 280题与三轮性能，以及DiffSpec
C04/B16三臂56题首轮和全部失败/诊断证据。不再补跑DiffSpec280题、第2/3轮性能或全量测试；未完成范围保持标明，不写成完整280题结论。此前等待投资选择的记录是历史状态，现已由本决定取代。上一轮收尾已释放设备，本次仅整理本地文档，未启动设备实验。
