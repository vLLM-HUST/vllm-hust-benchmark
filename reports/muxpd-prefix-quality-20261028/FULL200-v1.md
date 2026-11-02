# MuxPD prefix 布局：原200题完整复测，质量门仍失败

**两次 native、旧 raw、prefix 的正确数为165/200、165/200、166/200、167/200。prefix 修复旧三个失分，但新增一个“两次 native
均正确、候选错误”的案例07338，并有11个 length 截断；原严格质量门 FAIL。** 正确数净增不能抵消逐题退化。运行收集和生命周期 PASS 是独立结论，不能改写质量结果。

本候选[已公开发布](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-prefix-quality-full-20261028)。[匿名公开完整下载读回](public-readback/full200/final-status-v1.json)已
**PASS**：12个附件实际HTTP200下载，2,389,234字节；归档完整SHA、57个payload与1份嵌入manifest、gzip
EOF/CRC均核验。没有认证、未以本地payload替代下载、未解压或执行下载代码。公开可用性PASS不改变质量FAIL。只包含已终态的
`prefix-quality-full-01`、冻结quality-v1生产者/输入/CPU回执、全量独立审计及新loss上下文；不含current-main bridge结果，也未读取其live
child。

| run          | 正确/200 |    stop | length | 原 all_complete |
| ------------ | -------: | ------: | -----: | --------------- |
| 历史 native1 |      165 |     191 |      9 | false           |
| 历史 native2 |      165 |     191 |      9 | false           |
| 历史 raw     |      166 |     193 |      7 | false           |
| prefix候选   |  **167** | **189** | **11** | **false**       |

原严格检查中，`both_native_controls_complete=false`、`candidate_complete=false`、`no_new_wrong_on_both_native_correct=false`；非空共同正确集与正确总数条件为true。最终
`full_quality_gate_attempted=true`、`full_frozen_plan_passed=false`。两份native原有9个length继续保留；候选11个length同样按未完成、错误处理，没有扩大预算或将length改标为stop。

仍使用原冻结200案例、25个完整八题batch和原顺序。每批seed先产生恰好1个输出token，再释放其余7个peers，整批结束后开始下一批。原5-shot CoT输入、prompt token
IDs、gold、tokenizer、canonical `extract_answer`、greedy、seed42与**8192输出token上限**不变。原plan SHA为
`dd3e1f305475ec556dc824947f26d5fb5743242e82b5099155c95b8954bb2e65`。此候选确实收集全200题输出；“全量收集完成”不表示每题都在预算内自然stop。

旧raw的三项loss均恢复：

| case  | Gold | 旧raw                | prefix               |
| ----- | ---- | -------------------- | -------------------- |
| 03766 | B    | A；stop；6396 tokens | B；stop；1053 tokens |
| 10672 | A    | G；stop；1046 tokens | A；stop；2613 tokens |
| 01741 | E    | length；8192 tokens  | E；stop；590 tokens  |

但165个both-native-correct案例中新增loss **mmlu-pro-07338**：两次native均H、4723 tokens、stop；旧raw为H、5133
tokens、stop；prefix达到8192并以length结束，原评分answer=null、incomplete、wrong。prefix与native共有206个输出token，从第207个分叉（native
token318，prefix198）；与raw共有37个，从第38个分叉（raw7661，prefix26068）。

旧raw四个gain的去留也单独保留：

| case  | native1/2           | 旧raw         | prefix        | 旧gain去留 |
| ----- | ------------------- | ------------- | ------------- | ---------- |
| 06074 | length；8192        | E；stop；2013 | length；8192  | 丢失       |
| 06741 | length；8192        | E；stop；1869 | length；8192  | 丢失       |
| 00882 | C；stop；1351，错误 | G；stop；732  | G；stop；836  | 保留       |
| 08107 | length；8192        | C；stop；708  | C；stop；3760 | 保留       |

prefix另在11363答D、4761 tokens、stop，native答B错误、旧raw为length。相对native，prefix是3 gain与1 loss；相对旧raw，是4
gain与3
loss。报告没有用净正确数差值代替逐题门槛。prefix的11个length为03290、04497、10365、07338、11666、12028、06074、06741、01004、08833、09276（均为mmlu-pro）。

完整token序列与native1、native2各仅 **89/200**
完全相同，与旧raw为27/200。独立审计用原tokenizer重新解码四份run的全部800条完整token数组并核对保存文本，逐题保存解析结果、结束原因、完整token/text
SHA及首分叉。不能将旧三题恢复或早期logits恢复扩大为全体完整输出等同native。

07338位于原batch_id=7（零起，第8批），成员原序为07151、07213、07275、07338、07401、07461、07524、07585；07151为seed，07338为第4成员/第3个peer。四run在目标**第201–207个输出token**附近没有任何收据可见的请求退出：扩大到190–220，共124对schedule/retirement，均为原8成员按保存的scheduler字典/列表顺序调度及D退役。prefix最早结束的peer07213对应目标token302；prefix/raw的seed到目标435才结束，native的seed到288才结束。seed相对目标的输出领先为native
1、raw/prefix 10，额外9个token确实存在，但并非在207附近结束。peer07461已于第203个token对native分叉，这只是观察事实，不作请求间因果推断。

这些收据保存 `scheduled_tokens`、`phases_before` 和 `decode_retired_ids`，没有模型runner实际tensor行序、row
index、graph
bucket、KV/slot映射、logits或设备时间。保存的scheduler字典序不等于runner物理行序；finish步骤由最终D退役、保存的stop/length与后续缺席联合派生，没有直接的逐步finished_req_ids。该证据排除“207附近natural
finish造成scheduler成员减少”这一具体解释，**尚未确定新的数值分歧原因**。后续探针应保持原batch和预算，并先bridge观察器；不提出case特判、fallback或延长生成来过门。

生命周期单独核验通过。worker终态时间为2026-10-28 08:04:55.609785 UTC，controller于08:05:02.879711
UTC结束、exit0。25个batch各记录10次实际P层前进且D token退役的事务；每rank 50个closed cohort、25个treated cohort，各2025次treated
collective、2025次D0 bypass，四rank共8100次treated调用。实际prompt与原题SHA、seed D0/peer D1
admission行数、BF16、恰好一次既有P collective、原tensor alias及isolated P路由均独立核对。

终态执行空maintenance且aborted IDs为空；scheduler/worker无活请求或P所有权、KV恢复、三个layout
hook正常恢复、shutdown返回。controller明确记录owned_members_after_cleanup=[]，保存的前后物理卡0/2/3/4均idle，终态日志无Python
traceback或已知graph ownership异常。该记录不宣称“全局无信号graceful shutdown”，也不以cleanup成功推断质量成功。

候选限定于冻结6c插件上的外部诊断prefix模块，安装发生在native warmup之后。模块按真实admission D行数在实际P payload前补零行，使用原P
communicator一次，copy-back并返回原tensor；D0旁路。插件 `6c606f789450a45ef35afee398b0a3e516caa1a1` 与旧controls
f7已有20个runtime源文件逐字等价，core/Ascend保持；外部模块、新producer、源码位置和物理卡组合的差异明确披露。没有修改历史run字段来伪装通过原compare_provenance。结果不资格化current
main、async、group、SP、安装包或性能。

原 `score_quality.py` 字节不变，SHA
`157b52ffdbae6d7f0f990bb27379c463df1807daeebf7298fd396968e96b17fd`。显式provenance wrapper
`score_layout_quality_v1.py` SHA
`249762dd55dbc28e5b3f60b3219afbd02e11cc9fdf654a5fa579a65ca8f1255c`，本次调用原score/evaluate和mixed
coverage门。外部prefix模块SHA
`6431c4c0abdd869ef17e7122f681c457ad2b4d5741f86c6a0653e9fca44e02fc`。独立审计没有调用driver/controller/scorer/wrapper；只复用固定canonical
extractor和tokenizer，独立完成评分、逐题转换、完整token和生命周期核验，最后才交叉核对assessment。两者所有评分行、配对转换和严格门一致。

候选归档
[muxpd-prefix-quality-full-01-20261028-v1.tar.gz](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-prefix-quality-full-20261028/muxpd-prefix-quality-full-01-20261028-v1.tar.gz)
为 **2,313,081 字节**，SHA256：

`64a03ebe50e30e83bc7381a6274fa878a6eb271bd57c9c646f240821c2cba72c`

包含57个payload文件及1份逐字相同的嵌入manifest：28个candidate原件（含冻结producer/全200输入）、9个CPU/冻结回执、10个全量独立审计文件、9个loss上下文文件、1个历史引用索引。排除compiler-cache、pycache和工具缓存。本canonical报告在原旁附草稿基础上更新发布状态和证据链接；包内run、assessment、生产者及审计原件未修改，原暂存草稿保留于publication/full200。

新包**不附带历史native/raw的完整run或完整token输出**。assessment和独立审计包含历史派生评分/比较行；上下文包含选定历史scheduler回执原行及映射，不能代替历史完整原件。历史controls使用原冻结external引用，不新下载。[逐成员manifest](publication/full200/muxpd-prefix-quality-full-01-20261028-v1.manifest.json)列明安全相对路径、原来源、字节数与SHA；[完整本地验证](publication/full200/local-verification-v1.json)逐成员读取归档、复核原件SHA、精确成员集合、普通文件类型、安全路径及嵌入manifest，不解压、不执行。

| 原件                           | SHA256                                                             |
| ------------------------------ | ------------------------------------------------------------------ |
| candidate results/run.json     | `344f459193c04c75ad675dcd8c07d7f32eedef5a892dfe2af14271268206e192` |
| candidate assessment.json      | `5fa2b1be429abb35daae9494590688b3f8c7978bd31cb9643a0838f921a9ba62` |
| producer freeze                | `07a796c568e8cd69ae7157b3386269b1a624a94b6b8393bba232a5cec68a9345` |
| full independent audit summary | `28219b66045a319ff9f91f37b02704e03bd8ffdcf9f33a4c970ccf59b80813e5` |
| new loss context summary       | `ebe18d26d1f135eab05a122d196f94a3871088e4243e575be305e012727b48da` |

历史引用来自固定benchmark snapshot `b77e05b0d906b0e14b274533edf6750ffacd0170` 的canonical
v3/v5索引；本次没有网络请求或重新验证当前远端可用性。[historical-controls-reference-v1.json](publication/full200/historical-controls-reference-v1.json)原字节复用已冻结引用，列明历史archive、run
member、公开manifest/readback文档与SHA，且其run
SHA与本候选评分实际引用一致。旧prepared_not_uploaded字段不改写，历史发布状态由后来readback文件说明。

| 外部历史原件                                                                                                                                               | archive SHA256                                                     | 原run SHA256                                                       |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| [native1](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-followup-20261025/muxpd-native-quality-v2-hccl-true-01.tar.gz) | `5aa0cfebf7ae804100ba6a440512e4daefd823e062efcb7ef66fa660c91e2adf` | `cb7c36fb75a5f586c01d3e4f30a7369b5befa7bbc6021d1fcf0c5cb3a71a6a28` |
| [native2](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-followup-20261025/muxpd-native-quality-v2-hccl-true-02.tar.gz) | `3556615a84daf4b8f8b7abc762259f43cf845d8932d304f60fad6180d66f5033` | `14454401799a1745ca369891f177624ce1ff4348c2a29c8addb00f8b5d90500e` |
| [raw](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-muxpd-followup-20261025/muxpd-raw-quality-v2-hccl-true-01.tar.gz)        | `bbfdef527e51194205b7f7cf1e5d5613f2b63e3c7c2e3e8482e075dc8cfabf8a` | `75aa8495902fd795dbe845da35eb1bf06bb3adea7c81b515e8c3375b07a13f9e` |

公开复核可以只检验包，不访问原本地路径：

```sh
python verify_archive_v2.py \
  --archive muxpd-prefix-quality-full-01-20261028-v1.tar.gz \
  --manifest muxpd-prefix-quality-full-01-20261028-v1.manifest.json \
  --output reader-verification.json
```

生产机本地完整验证已额外比对所有原source字节；公开读者不使用
`--compare-sources`。包内脚本保留原绝对来源路径，不能假定重定位后可直接执行；完整重评分仍需外部历史controls、匹配tokenizer与依赖。本地custody、上传与实际匿名公开读回分别有证据；公开PASS不改变full
quality FAIL，也不构成根因闭环或性能结论。

本报告对应的小型独立证据见[全200审计](independent-audit/full200/README.md)及[新loss上下文](new-loss-context/README.md)。完整候选raw文件从固定Release归档取得，本目录不复制大型raw。
