# BidKV：2026-11-16 结束前证据核查

本目录只审计已经完成的正式模型实验，不是新一轮模型测试。根据最新 MOD 提示词（SHA256
`43bc925a6ad749971a64e5244b86451f5d4cf3361e87423b1eda8ea0b0d04461`），正式五项 `five-results-v1` 可保留为已完成的
**MOD 关闭/开启配对、机制未触发结果**。没有发现需要因为 0 次策略调用或收益不理想而重做该全量的依据。0 调用不能宣称 BidKV
获得机制收益，也不等于程序异常：本轮精确原生抢占计数的十个阶段增量同为 0，抢占策略没有进入执行条件。是否在有压力的场景触发，由另列诊断回答，不改变冻结正式工作负载。

已存在的固定版本：[完整结果报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9930a0b772b79c92a487509767991a9a476f110d/reports/tech-outcomes-20261117/bidkv/five-results-v1/report/REPORT.md)｜[逐请求数字数据与受控原件索引](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/9930a0b772b79c92a487509767991a9a476f110d/reports/tech-outcomes-20261117/bidkv/five-results-v1)｜[精确原生抢占计数](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9930a0b772b79c92a487509767991a9a476f110d/reports/tech-outcomes-20261117/bidkv/full-exact-preemptions.json)。后续发表必须保留固定
commit 链接或替换为已经核验的新固定 commit。

## 正式实验与指标

实际模型 Qwen3.5-35B-A3B BF16，revision `712cf74392b05026a6db2bf213d343747d1f6d45`；yanbo-81 两张
Ascend910B2、物理 6/7、逻辑 0/1、TP2/PP1/DP1。两臂共用 vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、Ascend
`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`、相同 prepared scheduler 和测量桥。B0 是目标策略关闭的共同基础配置；B1 启用 BidKV
`53eed4dd65bc4ba12efa37f9f1e334fa1ae51076`。它们不是完全未修改上游与成品集成 SDK 的比较。共同单芯片拓扑兼容补丁、初始化/编译、CPU binding
默认 True、APC/MTP2/async/FULL_AND_PIECEWISE 图均已披露，不把准备阶段改动归因于目标 MOD。

每臂 200/200/160/64/48，共 672 solver；另一 fresh Native 服务给双方统一完成 HLE/Frontier 720 条 judge。1344 条 solver
全有终态且 HTTP/传输成功；720 条 judge 全尝试完成，但可解析判分数不足 720，不能将其写为 720 条判分成功。

| 数据集（每臂 n）     | B0→B1 输出 tok/s | 吞吐变化 | content TTFT P95 秒 B0→B1 | TTFT P95 改善 | TPOT 请求等权均值毫秒 B0→B1 | TPOT 均值改善 |
| -------------------- | ---------------: | -------: | ------------------------: | ------------: | --------------------------: | ------------: |
| MMLU（200）          |  614.569→629.913 |  +2.497% |           929.079→951.498 |       −2.413% |               24.112→23.676 |       +1.809% |
| HLE（200）           |  793.276→794.148 |  +0.110% |               6.416→5.786 |       +9.832% |               20.362→20.791 |       −2.105% |
| Frontier（160）      |  761.031→762.812 |  +0.234% |               5.532→5.561 |       −0.512% |               19.559→19.617 |       −0.295% |
| SWE trace（64）      |    36.152→35.545 |  −1.679% |             20.774→23.336 |      −12.331% |             231.771→236.574 |       −2.072% |
| Terminal trace（48） |    85.393→87.391 |  +2.340% |             21.585→22.806 |       −5.658% |             111.313→110.181 |       +1.017% |

`metrics-independent.json` 对全部 1344 条公开数字记录独立重算：输出 usage 的 accepted completion tokens 总和 / 最早
submission 到最晚 response completion；TTFT 是首个非空 **content** delta，不用首 reasoning event；TPOT 每请求为
`(end−first_content)/(output_tokens−1)`，再请求等权求均值；分位数线性插值 `(n−1)q`。十阶段 TTFT/TPOT 有效样本各等于计划数，TPOT 排除数均
0。所有 P50/P95/P99 与报告一致，完整精度保留在 JSON。各臂 length stop 依次为 B0 `20/124/36/2/2`、B1
`20/121/43/2/4`；这些记录没有剔除。

质量使用原评分器和计划分母：MMLU 正确 `144/200→149/200`，两臂未解析均 0；配对不一致为 Native-only 24、BidKV-only 29，精确双侧检验
p=0.583132，不能作等价证明。HLE 已判正确 `32/200→33/200`，未决 `94→92`；Frontier 已判成功 `64/160→57/160`，未决
`8→4`。未决保留为非成功而非被证实答错。原 unchanged Frontier `mmlu_extract_first` 仍可能给 length-stop judge 解析出首个完整
VERDICT，不擅改评分规则。SWE/Terminal 只做 trace 推理性能，没有任务解决率。

| 质量层/轨道        | 计划数 | B0 已判成功 | B1 已判成功 | B0 未决 | B1 未决 |
| ------------------ | -----: | ----------: | ----------: | ------: | ------: |
| HLE gold/text      |     64 |           7 |           8 |      29 |      28 |
| HLE gold/image     |     10 |           2 |           1 |       3 |       6 |
| HLE revision/text  |    113 |          21 |          22 |      58 |      57 |
| HLE revision/image |     13 |           2 |           2 |       4 |       1 |
| Frontier Olympiad  |    100 |          63 |          55 |       5 |       2 |
| Frontier Research  |     60 |           1 |           2 |       3 |       2 |

未决全部取正/负、固定已判结果时，HLE 条件范围 B0 16.00%–63.00%、B1 16.50%–62.50%；Frontier B0 40.00%–45.00%、B1
35.62%–38.12%。这些不是统计置信区间，也未包含 judge 误判不确定性。统一使用本地 Qwen judge 的结果不等同官方排行榜复现。单批固定先 B0 后
B1，未建立重复批次总体效应区间或质量等价性。

## 提示词逐项核对及证据边界

| 项                                        | 核对结论与证据                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 提示词/标准/指南                          | SHA256 分别 `43bc925a…` / `66c050b7…` / `5db88d61…`，已按实际字节重算。完整 hash 在 `audit-receipt.json`。                                                                                                                                                                                                                                                                      |
| 原任务、数量、预算、EOS、无重试/补齐/挑选 | `driver-mod-v4/finite.py.source`、`execution-plan.json`、原始输入/评分代码 hash 与受控逐请求审计相互绑定。MMLU8192/C200/seed0；HLE和Frontier16384/C16/seed17；两种trace2048/C8/seed17；judge8192。正式五项无 `ignore_eos`。                                                                                                                                                     |
| 逐条实际输入绑定                          | 原 `verify_finite_records.py` 对受控实际 request 去唯一 salt 后与原 body 全字段相等；MMLU frozen 200 原顺序/源行/prompt hash；HLE/Frontier 原 contract+tasks+request digest；trace 原 histories/tools/`tool_choice=none`，不执行历史命令。                                                                                                                                      |
| 本轮实际 API tokenize 覆盖                | 每臂资格的 128 条 = 112 条 trace + 16 条合成 C8；正式 trace 阶段另有自己的全 112 条 API token-ID/hash 记录。不能写成对每臂所有 672 条题目均进行了当前服务 `/tokenize`。                                                                                                                                                                                                         |
| MMLU 历史 tokenizer 与输入文件            | 原 `mmlu_scope` 协议 hash `8124492d6c59f53817116e27b32cd86c7b06c5ce59a45d1d77df56c080772c87` 绑定 5 个冻结文件：request plan `f4ed4ce5…`、历史 tokenize preflight `5851407b…`、verified cases `708056e5…`、recovered `0fe20223…`、原 scorer `c2237c69…`。这属于冻结身份校验，不是本轮重新 tokenize200 的直接记录。                                                              |
| 公开记录的顺序证据                        | 312 条 MMLU/SWE/Terminal 的 submission_index 完整且双方 case 顺序一致。360 条 HLE/Frontier 的该字段全部 null，公开数字可直接核 membership；原冻结 task 顺序/实际 body 绑定和原执行器保持顺序的实现另作受控证据，不伪造实际提交序号。                                                                                                                                            |
| 零 KV 复用                                | 14 阶段均唯一 salt、engine prefix hit 增量 0；usage 的 cache detail 全部 null 时不将 null 冒充 0。请求级和阶段级联合核查收据保留。                                                                                                                                                                                                                                              |
| 真实启用与真实触发                        | formal `full-owned-lifecycle-policy.json` 有 enabled=true 的 EngineCore `initialized` journal 和同一策略类；十个 solver 阶段精确 `num_preemptions_total` 增量 0，policy calls/decisions/selections 0。仅初始化成立，没有机制收益证据。                                                                                                                                          |
| 实际命令与导入                            | `actual-execution-identity.json`（SHA `9920282253f8bbb44876b80184c70df091c4d2536d0e11fd90194fef986d6537`）保留 Popen 前实际构造 argv/选定 env、PID/ready/actual wait。每服务独立 `/proc argv` 与 loaded-module import RPC 均 null；不得把资格导入观测替代成正式进程直接导入观测。提示词未规定必须使用这两种特定采集手段，缺少它们是证据强度限制，不自动使已存实际请求/SSE无效。 |
| 默认优化/CPU binding                      | 两臂 defaultTrue。共同 fe00e47f… 准备脚本、b7a84571… CPU parser 与1264文件/804CANN资产 carrier inventory 已披露。资格有真实 worker/thread affinity 与无 warning/skip；正式服务的 default 参数、日志/采样/生命周期另存。不能把 True 参数单独写成所有 IRQ/迁移操作均成功。                                                                                                        |
| 720 条同一 Native judge                   | 四阶段 planned/attempted=200/200/160/160，传输失败均0；可解析数106/108/152/156，未决94/92/8/4。原 grader 四份 CPU 复算 hash 字节一致收据在 closure 文件，不用 token 等价诊断替代原质量分数。                                                                                                                                                                                    |
| 原 C8 资格                                | 两臂各 4K/16K，60秒预热/300秒正式提交后 drain，原合成配方固定256/ignore_eos=True；吞吐使用含 drain 的对应完整阶段区间。与五项真实任务 EOS 规则分开，不能当作五项质量或连续16K P95/P99长稳达标证据。附件同时要求保留原C8和EOS，须明确此合成资格范围解释；若要求所有范围均保留EOS，应另立EOS C8，不改动已完成五项。                                                               |
| 成本、内存、适用性                        | solver 同费率时间/输出token、三次 fresh load/compile 资源代理、10秒初始化cgroup/NPU采样和额外red/blue4条资格均另列；真实货币账单、精确分配峰值、MOD独占增量、有效FLOP、≥32K长稳未验证，不记0。                                                                                                                                                                                  |
| 原件、失败、清理                          | 已公开旧中止False批次与零生成native04失败及缺失返回码，不改写。正式14阶段实际wait0，三个API实际exit0，release无自有进程；2064记录与SSE及4份原评分复算的封存收据齐全。公共数字/hash不是可公开全文原SSE，不声称公共原SSE重新读取。                                                                                                                                                |

公开 `verify_full_five.py` 于本次观察返回 0：2064 条数字身份、1344 solver 指标、720 judge、2064 条受控 SSE 回放收据、4份原 grader
字节一致收据。完整 `verify_public_evidence.py` 本次实际进程返回 0，stdout/stderr 和 `audit-receipt.json`
在本目录。本地独立指标复算不访问私有SSE。

另于本次在81重新读取受控原件，使用原冻结校验器 `c72cffc310c16151cc2dc4f2ee221197e2c9bffbec11f4cc0327e0265dc49e57`
和原v4执行源，在全新NAS目录 `analysis/final-closeout-20261116/bidkv-audit01` 进行纯CPU复核：1344 solver +720
judge逐事件重放完成，四份 unchanged grader
报告再次字节一致，18个子进程实际wait返回0。4549份原件的当前字节/hash均与已公开索引匹配；整批原件before/after库存JSON同为SHA256
`af1b196b860f5aa6560eb51830e41cc633b58220d6082ff995dc2846938febf8`，没有修改旧数据。总耗时166.467059秒，模型请求0、无HTTP推理或SDK操作，未启动91、未设置保活/自动重启。详情和实际子进程命令见
`remote-original-evidence.json`；`public_original_sse_readback=false`、`controlled_original_sse_readback=true`
明确区分公开数字和受控原始SSE。

还以公开 `original-grader-sources.json` 的当前实际字节（SHA
`b725f8110c02bdb0b4a1017d192ee3913a2c0282733dacf8f0de1d40f073cb41`）为外部pin，在81直接hash核对98个原reference-tools文件和46个冻结plan/资产文件，全部逐项匹配；原
`DEPLOYMENT-TOOLS.json` 仍为
`1f0b34b2221312b81ce2a91b190d694dfb3a30662f94720e38a86e45c5f960b7`。显式Python3.12.13的CPU子进程实际返回0，模型请求0，逐文件字节数/hash及收据见
`current-input-grader-hash-check.json`。这补强当前持久化数据的身份，不把历史tokenizer文件变成本轮模型tokenize观测。

## 收尾与重做判定

正式 five-v4 无需为寻找正收益重跑或改负载。最终发布应使用本报告固定链接，并带五项逐请求数字、原件索引/协议/实际身份、质量分层与所有未决和负收益。不能写成“BidKV
完成正式提速验收”或“质量无损已证明”。

Evo48 ABBA、MMLU首8压力诊断、失败及 corrective 请求均单列；它们改变 KV/负载或策略阈值，不计作标准672/臂，也不作为调参最佳轮。Evo
成功触发但配对收益不稳定且有一次CPU干扰；MMLU8诊断见正确率8/8→6/8、总生成时间增加，不能用约7.5% token/s增长掩盖该质量代价。闭合数字与原件hash见
`diagnostics/evo02`、`diagnostics/mmlu8`，原始Evo旧export中的 `fully_validated_pair=true`
仅表示输入/策略/原件核验通过，后补CPU干扰更正使第二配对不能作为无干扰性能复测。此次只补闭合CPU数字导出，不对这些诊断追加推理。

在目前证据下，“已完成固定五项；正式负载未触发策略，未证明机制收益；另行压力诊断能触发但未建立稳定且质量无损的收益”是可复核的结论。四项目标的未测部分继续标为未验证，不因完成范围变成达标。

## 公开文件存储说明

超过单文件体积限制的 JSON 以 `.json.gz` 原字节压缩保存；解压后校验原 manifest 的大小和 SHA256。完整映射见
[storage-map.json](storage-map.json) 与 [compressed-members.json](compressed-members.json)。执行源码以
`.py.source` 保留真实运行字节，复现时复制回原名；原传输清单位于 `audit/original-transfer-manifest.json`，描述发布前路径，格式化后的报告由项目根
manifest 校验。公共哈希及数值不等同公开原始 SSE。
