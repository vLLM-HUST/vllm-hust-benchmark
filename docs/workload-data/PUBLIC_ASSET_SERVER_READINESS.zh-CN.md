# 公开原始资产：服务器准备与任务验证缺口

## 已验证的数据链路

2026-10-23 将此前仅本地验证的七个来源的选定原始文件传到当前授权服务器。 传输前对照固定来源目录核验原始摘要；服务器读取时再次核验文件摘要。 使用 #431 固定提交
[84b901f6](https://github.com/vLLM-HUST/vllm-hust-benchmark/commit/84b901f673c30ba8dce2c73038dc52e1c5cc9932)
的成功 CPU CI 39862575744 工件，实际安装于独立 site-packages 环境。
每源预选原始行0、1、2，完成原始读取、prepare、resolve，并分别比较输入和参考字段。 本记录不是模型执行或官方评分结果；原始文件、输入、答案、图像及完整数据库不随文档发布。

| 来源              | 选定原始文件                                      | 数据链路 | 运行类型            | 原生任务与评分                                                      |
| ----------------- | ------------------------------------------------- | -------- | ------------------- | ------------------------------------------------------------------- |
| BIGCODEBENCH      | data/v0.1.4-00000-of-00001.parquet                | 三条通过 | task_dataset        | 原三题NPU生成3stop；功能执行与评分尚需隔离环境                      |
| DS1000            | test.jsonl                                        | 三条通过 | task_dataset        | 原三题NPU生成2stop/1length；原版功能判定未运行                      |
| SWE_VERIFIED      | data/test-00000-of-00001.parquet                  | 三条通过 | task_dataset        | 未运行；需原仓库版本、修复 Agent、测试容器                          |
| SCIENCEAGENTBENCH | data/verified-00000-of-00001.parquet              | 三条通过 | task_dataset        | 未运行；需原任务资产、执行环境和判定                                |
| TOOLACE           | data.json                                         | 三条通过 | historical_dialogue | 原始发布方合成历史工具对话；不是可执行 Agent 环境，不生成假 episode |
| CHARTQA           | data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet | 三条通过 | task_dataset        | 未运行；当前服务为 language-model-only，不能据此宣称图像推理        |
| TRACELAB          | syfi_coding_trace.jsonl.gz                        | 三条通过 | invocation_metadata | 元数据读取不是原始请求重放；未调用模型/工具或运行采集分析           |

所有选定记录均通过，合计21条；每个来源只取表中一个原始文件。 ChartQA 的训练/验证分片未传到本次服务器，不将这三条结论扩展到其他文件。 来源版本、原路径和摘要由该固定工件中的
source_manifests_v1.json 定义， 使用者不能更换同名文件后沿用结论。三个代码/修复任务也没有因能读入就执行生成代码。

## 执行与证据边界

- [x] 七个选定原始文件按固定目录摘要核验并私有传入服务器。
- [x] 固定 CI wheel 安装，21条输入及参考的独立往返通过。
- [ ] 适用的模型生成、原生完整任务终止和官方评分。
- [ ] 各原生执行环境、模态能力和安全隔离逐项核验。
- [x] 固定数据版本653ad351的最终CI工件已在服务器重新安装并完成七源21条往返；研究端原PR
  #37亦完成双模式21条及旧来源9条回归，四项CI成功。见[固定版本消费记录](DATA_CONSUMPTION_VALIDATION.zh-CN.md)。

上述七来源数据链路检查的模型调用0、原生任务执行0、评分调用0。
后续同三条BigCodeBench原题已另完成NPU生成阶段，见[独立记录](BIGCODEBENCH_ORIGINAL_GENERATION.zh-CN.md)；未运行功能评分，不改变数据链路回执范围。
DS-1000同三条Pandas原题也完成[单独生成探测和入口复跑](DS1000_ORIGINAL_GENERATION.zh-CN.md)，其中一条原预算length；未运行功能评分，不证明科学闭环。
这里不改写题目、不注入解题证据，不将历史对话或元数据伪装成在线任务。 已有HumanEval隔离探测失败不能被绕过后裸执行生成代码；其他套件仍需逐项检查。
本批随[Draft PR #432](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/432)合并交付多个来源；不是逐数据集开PR。最终增量CI工件与消费端核验状态见PR，不将生成阶段或读取结果混作完整任务通过。

## 独立QASPER原发布版接入准备（2026-10-25）

这是登记来源`QASPER`的原v0.3任务包，不是已经交付的`LONGBENCH`派生Qasper200题。发布方固定[HF加载器](https://huggingface.co/datasets/allenai/qasper/blob/7bde7e4a9f805bed6c6e79e664c37057263dbc1d/qasper.py)给出原S3归档地址；服务器直接取得两个原归档，字节数和SHA与Hub记录一致，未使用镜像或改题。参见[获取与原题清点](validation/qasper-original-acquisition-20261025.json)。

原train/dev/test分别为888/281/416篇论文、2593/1005/1451个问题，合计1585篇论文、5049个问题。按原paper对象顺序和原qas顺序展开，一题一记录；输入保留完整title、abstract、full_text和figures_and_tables原caption/file字段，以及该题原question。所有答案标注、证据、yes/no、不可回答标签及高亮证据完整留在参考侧；不把兄弟题、参考答案或证据注入输入。保留原paper
ID、question ID、父/题位置和问卷元数据，不把论文数当问题数。caption/file不是图像字节，也不证明视觉推理可用。

安装包含此增量的匹配数据组件后，在服务器私有目录取得原归档：

```sh
python -I scripts/prepare_qasper_original_assets.py \
  --train-dev-archive /private/qasper-train-dev-v0.3.tgz \
  --test-archive /private/qasper-test-and-evaluator-v0.3.tgz \
  --output /private/fresh-qasper-v03
```

入口只核验并提取三个原JSON及归档自带`publisher-assets/qasper_evaluator.py`，保留原字节，拒绝错摘要、歧义成员及覆盖输出目录；不执行归档代码、不下载数据或调用模型。`workload-data catalog download-plan QASPER`列出归档URL和成员名；URL指向归档，不能把归档直接另存为同名JSON。再使用`iter_workload("QASPER", data_root=Path("/private/fresh-qasper-v03"), files=["qasper-dev-v0.3.json"])`读取指定split，或不指定files读取三个split；不指定limit就是全量。所有记录均拒绝伪造现成API请求体。

统一索引配置为`{"sources":[{"source_id":"QASPER","data_root":"/private/fresh-qasper-v03"}]}`，沿用pipeline
plan/run及resolve_input入口。服务器独立安装候选wheel后，33个包内payload逐字节一致；全部5049条输入、参考及父题关系与原JSON逐项一致，全部5049条索引完成，各split首尾各3条共18条回查通过，见[安装候选包CPU审计](validation/qasper-original-provider-cpu-audit-20261025.json)。这是本地构建的软件候选工件，不冒充成功CI工件或研究端已接通；没有在本地下载原数据。

随后补充固定提交`39556148d18a30bb1f6d3ec04cba663ab8d67db6`的成功CPU CI `40119103024`工件验证：wheel
SHA256为`7d2225b2efb376da1d64685fccd3f8ed2b91f1dcb2b6121294503b15e3316282`，服务器新独立环境中33个payload逐字节相符，指定研究消费源码`6ca466e9`的40份Python文件也逐字节相符。三个split各原前三题共9条分别完成公共prepare→resolve、公共索引→研究导入、研究直接prepare→resolve；全文、答案/证据标注及父题关系一致。见[固定CI研究消费回执](validation/qasper-ci-research-handoff-20261025.json)。本次没有更新研究仓默认依赖锁，不代表其默认安装自动升级；使用该增量须在独立环境显式安装上述固定工件。本项只验数据消费，不验研究特征、观测或科学结论；此前候选5049条的历史回执不改写成最终CI全量验证。

原图表也已按发布方README的两个S3地址在服务器取得，合计603112831字节；条件GET的ETag及S3版本标识一致，另记录完整下载SHA256，不把multipart
ETag误称全文件MD5。见[原图归档获取](validation/qasper-original-media-acquisition-20261025.json)。目录download-plan的`media_archives`单列两个图像归档及摘要，与原JSON/scorer的归档成员分开；它们不随wheel分发，也不是`prepare_qasper_original_assets.py`的输入参数。

原三split共11364个唯一图表文件引用均能定位到原归档并完成CPU解码，无缺图；复用已验证的8486张train/dev图，新解码2878张test图，均为PNG。归档另有1个未被原JSON引用的文件，不补入任务。train/dev原相对路径分别为`train_dev_figures_and_tables/train/{paper_id}/{file}`、`train_dev_figures_and_tables/dev/{paper_id}/{file}`，test为`test_figures_and_tables/{paper_id}/{file}`，不能把三个split的归档根目录假定相同。见[图像关联及CPU解码审计](validation/qasper-original-media-cpu-audit-20261025.json)。只读原字节，没有裁剪、缩放、替换图像或改写caption；此处不是视觉模型推理证明。

原任务同时评价答案与证据，归档中的发布方评分器已保留，但本轮尚未进行模型生成或官方评分。后续须采用适配模型的完整原任务执行与原评分；不能用LongBench派生提示及QA-F1替代原QASPER证据评价，也不能用诊断预测或读取通过充当原生执行。本来源保持未勾，企业数据及科研观测/分析均不在本增量中。

## RULER 原生成任务与完成式服务接入（2026-10-25，进行中）

使用 NVIDIA/RULER 官方固定提交
`c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`，这是发布方程序化生成的长上下文基准，不是原始企业请求，也不是本项目自造的family。原
`scripts/synthetic.yaml` 定义13个任务，分别覆盖8种检索、变量追踪、常见词提取、词频提取及2种问答；不同长度也须分别声明，跑通一个profile不代表整套完成。

在服务器隔离的 Python 3.12.13 客户端环境安装
`configs/workload-native/ruler-original-client-py312-dependencies.txt`，再取得原发布方源码和原任务需要的
NLTK/语料资产。此依赖文件记录实际安装版本，不是原 CUDA/NeMo 服务端 Docker 环境的原样复现；服务端仍使用既有双NPU Qwen3.5-27B BF16
TP2、131072上下文。Qwen是本次可部署替代模型，不称发布方指定模型。客户端加载其匹配的本地tokenizer，不能以不同tokenizer生成后宣称相同长度。

首个profile使用以下原生成器命令（在固定源码的 `scripts/` 下运行；所有路径均为服务器私有路径）：

```sh
python data/prepare.py --save_dir /private/ruler-generated \
  --benchmark synthetic --task niah_single_1 \
  --tokenizer_path /private/models/Qwen3.5-27B --tokenizer_type hf \
  --max_seq_length 4096 --model_template_type base \
  --num_samples 500 --random_seed 42 --subset validation
```

输入为生成的 `niah_single_1/validation.jsonl`，生成完成后先固定SHA256和物理行号选择。首个profile预定全500条，不能运行后缩小到通过子集。原字段
`index` 实际可能重复（该500条有440个不同值），不能假定是连续唯一任务ID，更不能按其去重；身份采用文件摘要与物理行号，原字段保留。

公共入口：

```sh
python -I scripts/run_ruler_original_profile.py \
  --source /private/RULER --task niah_single_1 \
  --data /private/ruler-generated/niah_single_1/validation.jsonl \
  --data-sha256 YOUR_GENERATED_FILE_SHA256 \
  --api-base http://127.0.0.1:19466/v1 --model YOUR_SERVED_MODEL \
  --context-window 131072 --output /private/fresh-ruler-run
```

默认执行全文件；大profile如需固定抽样，在运行前通过 `--rows 0,1,2`
显式指定物理行号并记录选择依据，不按结果补题或替换失败样本。入口校验5份发布方客户端、任务配置及评分文件摘要后，直接使用其 `VLLMClient`、原 `postprocess_pred`
及原指标；保留原input+answer_prefix、输出预算（NIAH为128）及采样设置。只绑定到本机完成式API，不加聊天模板、不删输入或参考、不缩减输出预算、不重试换题。依赖服务提供
`/tokenize` 及 `/v1/completions`，完整输入加输出预算超出容量即阻塞。原CLI的NeMo JSONL读取由stdlib逐行读取替代，因此不能宣称整份原CLI完全不变。

RULER原客户端要求返回生成后缀，不是输入加后缀；若误将含答案的输入拼回预测会污染评分，入口明确禁止该映射。合法空输出或length结果保留进入原评分，不把低分当不可用，也不改题求通过。私有输出保存每条原数据、实际请求响应、哈希、原预测格式和进度；新目录拒绝覆盖，原始数据与预测不进入仓库。

服务器已完成公共入口的5份源码核验、全500条读取、原客户端/后处理/指标加载及重复index保留检查，见
[CPU回执](validation/ruler-public-entry-cpu-20261025.json)，模型调用为0。另一个私有执行进程正在完成首个500条原生profile，尚未终态和审计；不能把该CPU回执或17项软件夹具测试当作原生完成。其他任务/长度profile、公共入口的真实执行确认、数据消费适配及研究侧接通尚待补齐，清单保持未勾。

另以已保存的前75条实际原生请求/响应完成公共入口的无网络CPU回放：逐条原数据与哈希一致，公共入口构造的请求、生成预测与原私有执行进程完全一致，原指标在该75条记录上为100.0，见同一CPU回执中的
`saved_native_transcript_replay`。没有重复调用模型，也未以回放冒充公共入口的新现场执行；这不是500条终态、整套RULER或性能结论。

统一读取也已接入：`iter_workload("RULER", data_root=Path("/private/ruler-generated"))` 默认只读取清单中固定的
`niah_single_1/validation.jsonl`
500条，不会下载或生成数据。生成配置随source保留，原input/answer_prefix在输入侧，outputs在参考侧，index/others/length/truncation在原注释侧。它不是现成chat请求，`request_body()`明确拒绝伪造API请求。其他生成profile必须先保留真实生成配置并绑定实际文件摘要，不能拿另一个tokenizer或profile的文件覆盖这个固定摘要。

服务器新隔离环境安装候选wheel后，33个payload逐字节一致，全部500条输入、参考、原index、生成配置均与原生成文件一致，500条公共prepare→resolve全部回查通过。指定研究源码的40份Python文件与此前已核验安装逐字节一致，其公共索引导入及研究直接prepare→resolve各原前三条也通过，见
[候选读取及研究接通回执](validation/ruler-reader-candidate-handoff-20261025.json)。该候选wheel不是最终成功CI工件，未改研究侧默认依赖锁，也未迁入观测/特征/分析。最初审计误用公共数据库读取函数读取研究数据库，已改用研究自身接口并复核既有索引，无模型请求重跑；这一审计脚本错误不计作原任务失败。

随后使用固定提交 `50474ec0cc62a7de83b8f75298eb01fa650d67cc` 的成功 CPU CI `40123881426`
工件，在服务器另一新环境重复核验软件工件及消费链路：wheel SHA256为
`65e498e3cc4ecb016e0d14decc82c4f9fdcdbe56853ab6ccb4016940e3140884`，33个payload、全部500条原数据及公共索引回查一致，研究侧原前三条索引导入和直接准备/回查通过。见
[固定CI工件消费回执](validation/ruler-ci-research-handoff-20261025.json)。该提交的lint/tests/CPU检查均通过；此前候选回执保留为历史，不改写其工件身份。研究端须显式安装这个固定wheel后使用，默认依赖锁仍未升级；本次不是模型执行或科研分析验收。

其他原任务准备已发现并排除两个实际环境缺口：

- `scripts/data/synthetic/json/english_words.json`
  在Git源码中是LFS指针，不能直接当JSON词表。按发布方指针的SHA256和8564991字节大小取回完整370101项词表，在独立运行副本中还原；原27份源码快照保持不变，运行副本的其余26份文件与原Git
  blob一致。见 [LFS资产回执](validation/ruler-original-lfs-hydration-20261025.json)。常规Git获取须安装Git LFS并对该路径执行
  `git lfs pull --include="scripts/data/synthetic/json/english_words.json"`，不能拿别的词表替换。
- 原FWE生成器还依赖SciPy，已在隔离客户端补装 `scipy==1.16.3`，pip
  check通过，依赖文件已同步。未改推理服务的模型、精度或启动参数。补齐后CWE与FWE原生成器各已生成完整500条，仍不是原生模型验证完成。

原 `data/prepare.py`
会捕获部分子进程错误后返回0，因此仅进程返回0不能算准备成功，必须检查实际输出文件、500条数、字段、摘要及日志。初次CWE/FWE失败日志保留，修复仅限真实依赖与原LFS资产，没有改生成逻辑或选择规则。后续验证范围已在模型结果之前固定：短输出VT/FWE以及原SQuAD
QA profile预估30分钟内，保留完整500条；其他较长输出或更长输入profile固定原前三条。原首个NIAH 4K全500条不因实际耗时超预估而缩减。

QA依赖只从发布方固定脚本列出的URL取得：SQuAD v2
dev原文件已取得，4370528字节、11873个原问题；HotpotQA的原HTTP地址和脚本指定的固定HF备用地址均以URLError失败，尚未取得，不改用自造文档或其他问答集顶替。依赖获取和原生成过程继续在服务器完成，不把原始数据放入软件包或公开提交；其他原任务、长度profile和最终原生评分仍按清单继续推进。
