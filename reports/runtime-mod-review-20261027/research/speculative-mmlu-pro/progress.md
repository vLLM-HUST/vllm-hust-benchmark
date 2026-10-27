# ascend87 实验记录（进行中）

## 固定输入

- vSpec `4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae`；DiffSpec
  `1294c95b6d9b2d9cbed0d0cffe31202fe5a6a254`（本地现有审阅快照，上传无 `.git`，不把副本视为新 upstream revision）。
- MMLU-Pro dataset `b189ec765aa7ed75c8acfea42df31fdae71f97be`。
- 官方 prompt/parser 源
  `TIGER-AI-Lab/MMLU-Pro@f418b116db00b065c2aea046518d8fcf74d39872/evaluate_from_api.py`。使用同学科
  validation 的五个 CoT 示例；parser 保持三级提取，但无随机猜答案兜底。
- Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775`。
- Eagle3 `e91a7235f077a49027105130f587c0bab4ab5b2d`，`compressed/model.safetensors` SHA256 与 DiffSpec
  原报告一致：`a57cefc45874197a24dd2a092cfd0d0f7d6a2f2cca156d09f2d2f4a56dc4e5be`。
- 宿主无法直连 Hugging Face；本机取得官方 metadata，宿主从 hf-mirror 下载上述固定 revision，所有 LFS 对象按官方哈希校验。没有更换为其他草稿。

## 隔离与安全

工作目录 `/root/lumi-speculative-mmlu-pro-20261026`，system-site-packages 私有 venv，未修改共享源码/包。`launch.py`
对所需物理卡上锁、重查占用并查询映射；`guard/sitecustomize.py` 在各 worker 建立 context 后以一元素分配确认自身 PID
的物理位置，再允许模型工作。单次启动/探针上限900秒，结束清理自身进程组。

初次 native smoke 的 launcher 错误覆盖了宿主 PYTHONPATH，导致缺失 `acl`；已改为 prepend，原始错误保留。这是我们的启动器缺陷，不是插件证据。

## 已观察到的兼容性

- vSpec 原始 doctor 重现 `DeviceOperator` circular import。ops-first bootstrap 后 adaptive ABI
  检查通过，但实际激活失败：宿主没有 `_sample_draft_from_logits`。
- `vspec_compat.py` 将宿主原有 `_sample_from_logits(self, logits, sampling_metadata)` 用旧名暴露，不更改 target
  或拒绝采样；适配后可激活 adaptive。
- Qwen2.5 target/draft 的 head padding 为152064/151936，宿主拒绝词表长度不同。两者 tokenizer.json **完全相同**，实际最大
  token ID151664。私有 shim 只对这两个明确路径名、逐对象完全相等的 tokenizer 与合法 ID范围放行；不 resize head、不 remap、不 mask
  target logits。若产生越过 draft embedding 的 padding token，仍须硬失败，不能掩盖为正常结果。
- DiffSpec 原始插件导入钩子通过。原始 CPU test collection 因宿主同一导入循环失败；ops-first 后 config/runtime 的25项测试通过。
- 首个 target-only Qwen2.5-14B eager smoke 成功，输出“7 times 8 is 56.”，卡0 placement 已记录。该冷启动单题耗时不是性能结果，更不是
  MMLU-Pro 成绩。

## 当前状态

两插件的设备 smoke 与后续 MMLU-Pro pilot 正在调通；尚无 MMLU-Pro 分数、接受率或可比较吞吐结论。兼容性失败不能当作算法否定证据。

### 后续调通记录

- vSpec 固定gamma4 eager smoke 已成功，与native上述输出tokens逐项一致；不能据单题证明全量等价。
- adaptive发现两个额外跨版本接口问题：初始化后cached reorder
  threshold3与所有实际builder的5不一致；以及老宿主没有`num_draft_tokens_per_pos`统计字段。shim只在所有builder一致时重算缓存，实际backend冲突仍拒绝；`vspec-stats-compat.patch`只跳过宿主未提供的逐位置draft尝试统计，不伪造该分母、保留total与accepted统计。
- DiffSpec TP4 的fork路径遇到PyTorch
  `ParallelOpenMP.cpp:64 Invalid thread pool!`。改spawn后四rank均确认落在物理2/3/4/5并完成加载/图捕获，继续检查实际生成。不是算法负收益证据。
- 首次MMLU-Pro
  native启动后因transformers5的chat-template返回字典导致输入类型错误，已显式`tokenize=True, return_dict=False`并断言每个token为int。这也是评测harness兼容修复，不归责插件。
- 首个有效native pilot：14题，10题正确，0解析失败、0截断，2083 output
  tokens，生成wall20.327秒。此时另有调通任务，不能作独占性能；14题准确率不能泛化为MMLU-Pro全量成绩。正在做native重复与插件对应pilot。

### 运行有效性与解释边界

- native第二次pilot与第一次14题的全部output token序列相同；固定gamma vSpec有10题token序列变化、2题答案变化（biology 2804 B→H，math
  7687 C→F），得分9/14。此样本不足以认定系统性精度下降，须原生spec/数值路径对照。
- native draft FULL graph失败于`AscendDraftModelProposer.update_stream`缺失；准备eager诊断，不把它冒充同graph性能基线。
- adaptive online的空scheduler step不带gamma字段，原逻辑却取该字段；补丁在原生update后丢弃空step反馈，不虚构gamma。进一步发现宿主D2H
  getter不裁实际draft宽度，将padding
  -1当作固定长度返回；私有shim按真实tensor.shape恢复宽度，且若尾部有真实ID则硬失败，绝不删有效proposal。此前因padding被迫eager的89.775秒pilot保留，但不能用不完整适配冒充算法负收益。
- 加入动态宽度修复后的14题pilot能实际切换gamma并积累online latency反馈；84.171秒、9/14、2131output
  tokens，仍开启逐步trace且并行有其他调通任务，不能当作正式独占性能。
- DiffSpec在显式早期bootstrap+四rank factory
  witness下，实际构造`AscendDiffSpecEagleProposer`并完成graph生成。普通entry-point启动没有足够运行有效性证据，排除出DiffSpec性能比较；下一步以RPC检查每rank
  cache非空。
- 本次所有评测进程的confidence margin未设置、refill0；宿主原始环境没有遗留HUST_VSPEC/VSPEC变量。不是历史margin5.25的近似验收实验。

### 精度归因补充

同一14题，native eager与native graph的全部token序列一致。未启用vSpec的native draft
eager也在biology2804出现B→H，得分9/14（target-only10/14）。因此至少这一题的变化不是vSpec独有，不能把pilot的净少一题都归责插件。固定gamma
vSpec的实际draft接受率约37.99%；修复动态宽度后的adaptive约53.34%，不能称这一组“接受率极低”。其低收益/额外耗时须从动态宽度、图回退和控制成本解释，并以无trace独占运行复核。

DiffSpec已由四rank named
RPC确认实际proposer为AscendDiffSpecEagleProposer、policy=diffspec且cache_initialized=true。首次RPC使用callable被宿主安全序列化拒绝；改为固定方法名，没有开启不安全pickle开关。此错误属于harness，不是算法。后续正式harness增加前后runtime_metrics快照；当前pilot只有启动有效性与vLLM接受计数，不宣称已测量检索阶段耗时。

### DiffSpec 有效14题pilot（非正式比较）

四rank proposer/cache断言通过后，DiffSpec完成14题：13/14、11,050 output
tokens、441.848秒、draft接受率13.57%；target-only同题12/14、8,751
tokens、183.502秒。得分没有下降，须保留此正结果；输出长度增长约26%，总耗时差不能全归执行效率。两个pilot不在独占同卡重复条件下，不宣称已证明2.4倍算法减速。原生Eagle3对照尚在进行，用于区分草稿质量与DiffSpec增量路径。

正式投资尚待Fletcher选择：先固定280题配对与性能复核，或全量12,032题。当前没有启动正式全量任务。最初只按Qwen2.5
pilot估算的5小时量级不适用于Qwen3.8：其CoT输出更长，现有B4
pilot线性外推全量原生约44小时、DiffSpec约105小时，均只是粗估，正式B32/更高并发可能明显改变成本。因此更应先做有界screen并冻结实用batch配置，而不是直接跨夜盲跑。

### Pilot阶段收尾 / 待投资决策

原生Eagle3也完成14题：接受率15.19%（4432/29178），12/14，14,165output
tokens，533.641秒，**2题触及4000-token上限**；parser虽然抽到字母，仍明确计为截断，不隐藏失败风险。DiffSpec为13.57%（3195/23547）、13/14、0截断。低接受率在不启用DiffSpec时同样出现，不能全部归因于DiffSpec的KV选择；更接近该target/draft组合本身的问题。两者pilot的raw
output tok/s约26.54与25.01，target-only约47.69。输出长度、并发尾部、非独占宿主均不同，不能从总时长直接推出纯算子加速比。

截至收尾，8张卡全部无运行进程，三组物理卡协作锁均可重新取得；没有遗留自有设备任务。私有环境、固定权重、原始错误与补丁留在宿主以便继续。所有已运行结果已回收到本目录`evidence/`，摘要为`pilot-summary.json`。未提交、推送或对外发布。

下一阶段等待Fletcher选择280题screen或全量投资，未偷偷启动全量。建议正式协议至少比较实用batch（例如B32）而不只沿用调通B4；先估算容量/成本，随后冻结。所有正式速度试验关闭trace并独占运行，加入重复及同卡顺序交错；准确率使用固定题目、明确截断与解析统计。未来本地harness新增逐题完成结果落盘及DiffSpec前后runtime
metrics快照（尚未在此轮设备pilot执行）；原始pilot没有这些额外证据，不能回填声称已有。

### 280题阶段启动（2026-10-26 14:49 UTC之后）

Fletcher批准先做固定280题，不自动跑全量。已在正式推理前生成
`evidence/screen-manifest.json`：每学科20题，固定seed的SHA256排名，排除14题pilot；每科前4个排名组成56题三轮性能子集。正式B32，各family内相同卡组串行运行；vSpec
trace关闭。新增 `screen.py`、`campaign.py`、`freeze_screen.py` 和
`analyze_screen.py`，launcher默认仍900秒，只有显式环境配置才允许正式阶段2小时上限。

读取完整prompt JSONL时发现Unicode行分隔符会被Python
`str.splitlines()`切断；正式路径改为只按ASCII换行拆分（JSONL真正分隔符），不修改题目。原14题pilot不受此问题影响。CPU四项协议测试通过。B32
native额外smoke已完成，六臂16-token容量检查正在依次进行；正式质量尚未开始。

### DiffSpec 默认 linear 路径的检索门禁发现

六臂B32容量检查均成功（56题，每题16输出token；不计入质量/性能结论）。但是四rank DiffSpec post-witness 的
retrieval_count=0，tree_cycles=0。不能因后者为0直接说整个插件没运行：该计数只记录tree；默认auto实际选linear。

进一步读 pinned `diffspec/proposer.py`：`_propose` 在非tree路径调用
`clear_proposed_tree()`；`prepare_next_token_ids_padded` 却以 `proposed_token_ids is None` 传入
`is_prefill=True`。`accept_target_queries` 遇这个标志在检索计数增加前直接返回，即使target
attention已捕获。`test_diffspec_linear_gate.py` 使用原方法AST、fake
parent/cache执行该门禁，确认这一控制流（不构造NPU算子，不声称数值验证）。原样default auto路径不能作为已执行attention-score检索的证据。

这不等于完全等同原生Eagle：没有selection时 `_history_indices` 使用最近history_budget的滚动尾部，compact
cache/pre-RoPE路径仍可能运行。后续witness补充已有cache内部状态（不改算法），区分“检索未运行”和“整个缓存未运行”。本轮保留仓库推荐auto配置，不把修算法门禁静默混入兼容补丁；正式结果明确叫作default-auto实现的测试，不据此否定完整tree/检索算法。vSpec质量对照已开始；DiffSpec正式质量在此次机制审计记录后开始。

### vSpec 280题质量轮完成（性能重复尚未完成）

在B32严格验收、trace关闭的固定280题上：target-only 186/280，固定gamma4 187/280，adaptive
189/280。固定相对原生7得6失，adaptive相对原生9得6失；配对分层bootstrap差值区间均覆盖0，不能声称准确率显著改善或下降。实际接受率fixed48.33%、adaptive62.86%，因此本负载不支持“adaptive接受率太低”的解释。

单次质量轮wall为native215.939s / fixed167.693s / adaptive238.502s；输出tokens分别72771 / 69282 /
71784，不能把wall比全部解释为等输出计算加速。原生有1截断、2解析失败；fixed无截断/解析失败；adaptive各1。这些是质量轮的初步计时，不替代56题三轮性能对照。正结果保留：固定深度在这一轮比原生快，不能将两插件一并定性为无收益。

### Fletcher纠正vSpec目标模型：必须35B

用户明确要求vSpec用35B。此前用Qwen2.5-14B+0.5B draft是Lumi未对齐目标；该280题结果完整保留为补充实验，**不作为用户要求的35B主实验结论**。已在远端设置
`hold-vspec-performance`，campaign会跳过旧14B性能三轮，避免继续错误投资。DiffSpec原27B合同未被本条指令改变，原串行quality/performance任务仍在运行。

ascend87存在
`/models/Qwen3.5-35B-A3B`，config为Qwen3_5MoeForConditionalGeneration，包含MTP权重（14个分片，text_config.mtp_num_hidden_layers=1）。vSpec固定源有推荐
`qwen35-frontier-mtp2`
协议：TP2/B16、MTP深度2/4/6、自适应+refill、APC+async+FULL_AND_PIECEWISE。35B主实验将从该已存在入口核对，不沿用14B的独立0.5B草稿或把“35B”仅替换模型名。冻结280/56题ID不变；尚未启动35B设备实验。

### 35B容量门与私有适配

物理4,5上的35B target-only和原生固定MTP2均通过56题×16token
capacity，支持TP2/B16/APC/async/FULL_AND_PIECEWISE。vSpec最初在模型加载前失败两处旧接口：`kv_cache_utils._annotate_eagle_groups`已被宿主删除，`gpu_model_runner.copy_kv_cache_blocks_inplace`与`_get_mamba_state_copy_funcs`也已移除。保留两份原始错误日志。

私有`vspec_compat.py`将原插件不变的Qwen MTP
group注解接到宿主`get_kv_cache_groups`返回处，保留原builder与所有原group对象；两个CPU测试验证调用顺序/对象身份/重复安装与未知签名拒绝。对于新版MambaBuffers宿主，校验新API签名与tuple
group返回合同后保留原生逐plane拷贝，不安装旧插件会改成dict-return的兼容shim。不是禁用Mamba状态搬运；原生MTP2已通过该新路径。修复后插件已进入35B权重加载、MTP
head共享及FULL draft graph包装；capacity生成尚待完成。

35B最终capacity `screen-capacity-q35-adaptive-native-mamba` 已成功，56题全部返回；原生MTP2与vSpec
adaptive均有实际draft/accepted计数。日志确认MTP head共享、FULL ACLGraphWrapper、vSpec
online候选[2,4,6]且初始gamma4；不是仅加载插件名。16-token
capacity耗时/分数不作主实验结论。35B三臂配置保持不变，补丁未修改采样/接受规则。当前DiffSpec质量轮仍占0..3，35B正式质量等它释放后串行开始；旧14B性能保持暂停。

### DiffSpec比较链路正确性门失败：不得据此否算法

27B target-only
280题：235正确、39截断、0解析失败、347965输出tokens。原生Eagle3：94正确、203截断、145解析失败、857733tokens。检查发现多题为连续token0（`!`），不是普通答案误差；尚未证明根因是NaN，不能把症状直接命名为数值NaN。

回看56题×16token
capacity，native-Eagle和DiffSpec恰有相同19题的token0占比>90%（例如8127），target-only没有；35B三臂也没有。原capacity只检查跑完/形状，没有检查退化输出，漏过了本可提前发现的正确性信号。将其纠正为“启动通过，语义准入失败”，不把后续大规模错误输出称为有效算法反证。vSpec35B正式runner增加完成请求token0退化门：先保存完整输出，再停止，不把错误样本静默过滤；CPU回归覆盖保存后报错。

已中止正在初始化的
`screen-quality-diffspec-r1`，SIGTERM后未退出，经精确核对自有PID/PGID1704330及四个worker后SIGKILL同一进程组；退出记录-9（campaign返回247）。只清理本任务这组进程；原始日志/已完成结果与reviewer-stop
receipt保留，未清理模型或他人任务。正式DiffSpec性能尚未跑，暂缓；后续需小范围定位B32/native Eagle的退化，不继续无效长跑。

35B三臂capacity已通过；DiffSpec释放0..3后开始35B正式quality，随后按计划三轮perf56串行。所有35B正式计时不再与其它NPU任务重叠。

35B native 280题完成：203正确、92截断（32.9%）、0解析失败、719282输出tokens，generation
wall1296.066s。大量截断意味着这轮是固定4000-token预算效用比较，不是充分推理后的完整准确率；保持冻结协议，不依据分数事后换题或调上限。汇总增加jointly-untruncated配对的描述性诊断（该交集是事后选择，不能冒充无偏总体推断），并单列实际答案变化与截断。

### 35B三臂质量轮完成，性能复测进行中

原生203/280（92截断）；原生MTP2 202/280（94截断）；vSpec adaptive193/280（93截断）。三臂均无解析失败、无>90%
token0退化。输出tokens719282 / 726871 / 713705；单次generation wall1296.066 / 984.338 /
822.165秒。实际draft接受率MTP2 82.63%、adaptive67.05%，本轮不支持“接受率太低所以无收益”的解释。

Adaptive相对原生19得29失，差-3.57pp，分层配对bootstrap95%区间[-8.21pp,+1.43pp]；相对原生MTP2
27得36失，差-3.21pp，区间[-8.57pp,+2.14pp]。均跨0。与MTP2共同未截断的172题上，adaptive12得10失（事后选择描述，不作总体推断）；总体分差集中在至少一侧截断的题上，不能包装成已证实的严格验收精度破坏。几乎全部输出序列变化也发生在原生MTP2对照中；等待三轮native
perf56重复确认本栈一致性。

APC按配置开启，但已完成native/MTP2质量轮的cache查询391218、命中0；不声称本任务观察到了APC收益。三轮perf56仍按冻结顺序、同物理0,1、无其它NPU实验并行执行。DiffSpec仅准备小型eager/exact-bucket/B4诊断脚本，尚未占卡运行；不会在正式35B性能计时期间插入。

### 35B三轮性能完成；DiffSpec只做小型定位

35B三臂全部完成280题与3×56性能复测，完整结果见 `results-qwen35.md`。中位generation时间原生293.763s、MTP2
224.864s、adaptive191.793s；adaptive分别1.53×/1.17×，不能按原先负面预期否认这组稳定收益。原生56题重复40/40/37正确，前两轮序列相同、第三轮全变；配置/安全环境（除label）/题序/输入长度一致，权重大小mtime不变，原因未定位。

DiffSpec后续容量诊断：native Eagle
eager与B4两次仍为相同19/56题连续token0退化，排除简单的graph-only或B32过大解释。原生Eagle与DiffSpec原capacity有54/56序列完全相同，不能把共同故障归给插件。CPU检查Eagle
embed/head/d2t与选定norm张量均finite（不是全checkpoint检验）。原生Eagle同步logits诊断在各rank第29次target采样处看到NaN，但此前已有请求输出0而已观测raw
logits没有argmax0，继续审计sampler边界，不提前把所有症状归为forward NaN。

`diag-native-eagle-sampler`首次审计因诊断wrapper未兼容host传入(logits, indices)
tuple而失败；这是Lumi的probe错误，不是被测系统新故障。修正为提取tuple首项并断言非reduce-sample，使用新label
`diag-native-eagle-sampler-v2`。审计只比较原输出和CPU严格greedy参考，不改采样结果；所有同步诊断计时均不作性能证据。

Sampler-v2审计完成：所有已审计greedy输出与CPU参考相符（kernel与whole-sampler均无mismatch），仍相同19题退化；async+eager对照也相同。**纠正上一段时间归因**：按实际perf56数组重算，坏题恰是0-based位置37..55，8127是位置37，不是23；14题warmup占request编号后它对应request51。首次NaN的request51..55正好对应这批坏题，先前“已有坏输出而raw
logits没有argmax0”的推断源于漏算warmup编号，不成立。现在证据支持target logits非有限后进入采样，不能据此怀疑sampler生成了凭空的0。

坏题是连续尾部19题，且B4/B32、graph/eager、sync/async都不变；新假设是请求顺序/状态生命周期而非19题独立内容异常。以reverse-eager新label反转同56题，仅作诊断、不得混入冻结性能/质量轮，区分尾部位置与题目内容。

### 缓存容量定位与修订合同的准入

reverse-eager仍只坏最后19题，但ID全变；cold-eager取消14题warmup后只坏最后5题。两者均从累计第52个真实请求开始退化，不是题目内容特异。将memory利用率0.7降至0.4，保留其它eager参数后，56题×16token
**全部无连续0退化**。此结果只说明容量相关绕行有效，不足以声称根因已修复。原host源码无未提交改动。当前怀疑大缓存的地址/状态布局边界，尚未证实具体算子或整数溢出。

metadata snapshot显示65个单层group共用一个原始大buffer，较小容量num_blocks=5508、buffer约8.83GB；target
attention每block与Eagle attention总元素相同（1×256 vs 2×128），不能仅因head_size不同断言布局冲突。Mamba为float32
SSM；需要精确算子证据才可声称溢出。首个snapshot
probe因为RPC返回torch.dtype不可序列化而失败，是诊断代码错误；保存记录并只停止已核对的自有PGID2079841，改成worker端JSON安全转换后v2完成。

为恢复授权的280题对照，先增加新的显式C04合同准入：原生Eagle与DiffSpec均使用memory0.4、原B32/graph/其它参数，固定56题×128token；连续token0语义门先保存结果再停止。旧合同与其失败结果不覆盖；若准入通过，所有三臂（包括target-only）都在相同C04预算下重新比较，不能把新旧预算数据拼成一组。当前仅短准入运行，不把eager诊断耗时当性能收益。

C04 graph路径准入完成：native
Eagle与DiffSpec各56题×128token，无连续0退化；人工查看包含原坏题8127/2135的输出，恢复连贯且针对题意的推理。DiffSpec四rank实际proposer/cache身份通过；post
witness仍retrieval_count=0、selected_slots=0、retrieval_cycle=0、proposed_tree_present=false。主质量与三轮perf已按
`campaign27.py` 串行启动，使用全新 `screen-*-c04-*` 标签。C04不是对宿主根因的修复宣称；不把低缓存准入短跑耗时当正式性能。

C04 target-only280完成：238正确、44截断、0解析失败、838.307秒。native
Eagle正常推进，无语义门失败，但B32时缓存长期约98%、实际约24请求运行；尚未观察到日志中的preemption，最终以计数器为准。为避免把此容量限制冒充算法纯开销，未启动的perf56三轮统一改B16（新label
c04b16），质量B32不改。该修订已在第一轮性能之前冻结并告知Fletcher；质量与性能合同分开报告。

### B32 C04容量痛点：停止继续消耗，先做B16短对照

原生Eagle C04质量完成：233/280、44截断、0解析失败、356146
tokens、3348.201秒，接受119241/705960=16.89%。**最终num_preemptions增量1488**，尽管普通日志没有preempt文字。先前“未观察到日志抢占”不能当作没有抢占。这个4×质量轮耗时混合了缓存容量/重算效应，不能声称全是Eagle算法开销。

因此停止尚在初始化的DiffSpec
B32质量轮（PGID2276583，先TERM、worker退出后仅剩本任务engine/driver，再按argv与PGID核对后KILL；receipt保存）。没有丢弃已完成原生/Eagle结果或将其包装成有效三臂对照。原顺序pipeline因非零退出停止，未启动旧性能队列。

下一步改为明确的
`campaign27.py performance-first`：同一perf56三臂B16/C04，首次重复，不先支付后两轮或新280题费用。增加完成后num_preemptions计数门，任何抢占保留结果但停止后续臂/重复。先验证无容量干扰，再估算是否值得继续280题；这是投资与合同边界，不宣称DiffSpec280已完成。35B主实验完全独立、已完成，不受影响。

### C04/B16首轮完成，抵达明确投资边界

三臂perf56全部完成且num_preemptions=0。原生44正确/14截断、86748tokens/373.716s；原生Eagle46正确/9截断、73219tokens/766.069s，接受24661/145623；DiffSpec46正确/12截断、84884tokens/929.285s，接受25927/176796。均无解析失败、无连续0退化。DiffSpec实际proposer/cache存在，retrieval_count/cycle/selected_slots仍0。

仅一轮，不声称稳定速度差；按输出tokens/s后两者约为原生41%/39%。没有启动第2/3轮或新的B16
280题。已向Fletcher请求投资选择：建议保存当前证据，不再额外花约三小时重跑完整三臂280题。完整范围与限制见
`results-diffspec.md`。35B质量及九轮性能另核对num_preemptions增量全部为0。

### 2026-10-27：Fletcher决定停止追加实验

Fletcher明确选择“停止追加，整理现有证据”。本轮收口：保留已完成的35B/vSpec 280题与三轮性能，以及DiffSpec
C04/B16三臂56题首轮和全部失败/诊断证据。不再补跑DiffSpec280题、第2/3轮性能或全量测试；未完成范围保持标明，不写成完整280题结论。此前等待投资选择的记录是历史状态，现已由本决定取代。上一轮收尾已释放设备，本次仅整理本地文档，未启动设备实验。
