# 公开原始资产：服务器准备与任务验证缺口

## Multi-News 原成对多文档摘要（原生抽样交付已验证）

使用[作者发布的 Multi-News](https://huggingface.co/datasets/alexfabbri/multi_news)，保留原六个成对文件及作者 loader。
train44972、validation5622、test5622 共56216条；每个原 source 行与同位置 target 行绑定，不把目标摘要拼进输入。 按作者 loader
原语义去除行边缘空白并仅在文档中还原 `NEWLINE_CHAR`，`|||||` 文档分隔符保持原样；原文件字节留在外部目录。
模型为[官方任务专用 PEGASUS-Multi-News](https://huggingface.co/google/pegasus-multi_news)，不是原论文训练复现。
固定数据、模型、原配置和评分源码见 `configs/workload-native/multinews-author-pins.json`。

```bash
# 原文件、权重和评分资源均下载到服务器外部目录，不提交到Git。
python scripts/prepare_multinews_original_assets.py --output /private/multinews-assets
# 必要时加 --hf-endpoint https://hf-mirror.com；中断后显式加 --resume。
```

获取器复用已验证的原文件下载和摘要校验组件，数据与模型身份单独绑定。 `--parts data model --verify-only` 可只读核验已有原文件，不重下数据和权重。
先在生成与评分客户端安装同一固定 `vllm-hust-workload-data` 工件；源码使用者可安装 `./workload-preparation`。 匹配硬件平台的
PyTorch/torch_npu/CANN 应事先配置；独立NPU客户端依赖见
`configs/workload-native/multinews-native-npu-client-dependencies.txt`。 独立CPU评分客户端沿用
`xsum-original-score-client-dependencies.txt` 中的通用HF评分依赖， 只复用评分组件，不载入XSum数据或BART模型，也不改共享服务环境。

```bash
# NPU客户端：默认固定原test物理行0、1、2。
python scripts/run_multinews_original.py --mode generate \
  --data-root /private/multinews-assets/data --model-root /private/multinews-assets/model \
  --output /private/multinews-original-generation --device npu:1
# CPU评分客户端：复用上述原生响应，不调用模型。
python scripts/run_multinews_original.py --mode score \
  --data-root /private/multinews-assets/data --source-root /private/multinews-assets/scoring \
  --reuse /private/multinews-original-generation --output /private/multinews-original-score
```

原配置为encoder1024、decoder max256/min32、beam8、length-penalty0.8、no-repeat-ngram0，未为当前机器缩减。
原1024窗口不等于原始多文档束全文长上下文评测；完整文档与原关系仍可由公共读取入口取得。 `--full` 执行所选split全部原题，`--indices 3,4,5`
是输出前固定的其他原位置，不能按失败结果换题。 本次原test全量执行保守预测约48971秒，超过30分钟，冻结前三条为主要抽样。
其原生NPU生成、未修改的官方HF后处理/ROUGE及额外公共入口未见样本已验证；候选读取已逐条对照作者loader全部56216条。
[固定CI工件和研究侧消费核验](validation/multinews-fixed-ci-research-handoff-20261025.json)已通过：
安装包39个文件与CI工件逐字节一致，公共索引及研究直接准备均保留全部56216条；每split公共回查6条、研究索引导入3条、研究直接回查6条通过。
研究40个源码文件未改，默认依赖锁未自动升级，观测和研究分析未迁移。
[最终交付回执](validation/multinews-native-delivery-20261025.json)记录固定入口原test6、7、8以及原空正文test4736的额外NPU生成/原评分。
原数据10条空正文全部保留，不填充或删除；补充空正文案例使用npu:0，主要冻结协议及固定入口未见样本使用npu:1，不合并解释为性能对比。
本来源已打勾，不宣称全5622题test推理、全文长上下文覆盖或论文分数复现。 数据使用遵守作者发布条件；原始文档、参考、详细响应及完整索引不公开提交。

## XSum 原文章摘要（原生抽样交付已验证）

数据使用[发布方原始 XSum parquet](https://huggingface.co/datasets/EdinburghNLP/xsum)，保留原
`document`、`summary`、 `id` 和 train204045、validation11332、test11334 三个 split。参考摘要不进入模型输入。
模型为[官方任务专用 BART-large-XSum](https://huggingface.co/facebook/bart-large-xsum)，不是原论文 ConvS2S 的训练复现。
固定版本、文件摘要、原生成配置和评分源码见 `configs/workload-native/xsum-author-pins.json`。

所有数据、权重和原评分资源在服务器外部目录准备，仓库不托管这些资产：

```bash
python scripts/prepare_xsum_original_assets.py --output /private/xsum-assets
# 网络需要时可指定 --hf-endpoint https://hf-mirror.com；仍核对原发布方摘要。
# 中断后使用同一目录加 --resume，只复用已经通过校验的原文件。
```

准备器支持 `--parts data model --verify-only` 对已有原文件只读核验；不能把准备成功当作原生任务完成。 匹配硬件平台的 PyTorch/torch_npu/CANN
应事先配置，独立 NPU 客户端依赖见 `configs/workload-native/xsum-generation-npu-client-dependencies.txt`；独立 CPU
评分客户端依赖见 `configs/workload-native/xsum-original-score-client-dependencies.txt`，不修改共享推理服务环境。

```bash
# 使用独立 NPU 生成客户端；默认固定原始物理行0、1、2。
python scripts/run_xsum_original.py --mode generate \
  --data-root /private/xsum-assets/data --model-root /private/xsum-assets/model \
  --output /private/xsum-original-generation --device npu:1
# 切换到独立 CPU 评分客户端，复用刚才的原生响应，不调用模型。
python scripts/run_xsum_original.py --mode score \
  --data-root /private/xsum-assets/data --source-root /private/xsum-assets/scoring \
  --reuse /private/xsum-original-generation --output /private/xsum-original-score
```

生成使用原 checkpoint 配置：encoder1024、decoder max62/min11、beam6、no-repeat-ngram3、原 pipeline 前缀。 1024 是该原
BART 评测协议的窗口，并非为当前服务器临时裁题；完整原文仍保留，不将此协议冒充全文长上下文服务验证。 `--full` 执行整个所选 split，`--indices 3,4,5`
是生成前固定的其他原位置；不能看结果后换题。 评分核对保存响应的原模型/配置及原文章/摘要身份，使用未修改的官方 Transformers 后处理函数、HF ROUGE、原英文 Punkt 和
stemming。 两阶段均拒绝覆盖已有输出目录。

当前原 test0、1、2 的 NPU 生成和原评分通过；公共入口另验证未见原位置并独立重算评分。 依据冻结规则估算原 test
全量推理约11870秒，超过30分钟，故原前三条保持为主要抽样，不宣称11334题全量推理。
原数据含34条空正文（train28、validation5、test1），适配器保留原样，不删除、不填充、不替换参考； 原 test11155 为空正文的补充边界验证，不替代主要三条抽样。
[固定CI工件与研究消费核验](validation/xsum-fixed-ci-research-handoff-20261025.json)已通过：
安装包38个文件与CI工件逐字节一致，三split全部226711条原关系保留；公共回查每split6条、研究索引导入每split3条、
研究直接准备全部原关系及每split6条直接回查均通过。研究源码40个文件未改，默认依赖锁未自动升级，观测与分析未迁移。
[最终交付回执](validation/xsum-native-delivery-20261025.json)记录固定入口原test6、7、8的额外NPU生成和原评分。
本来源已打勾，不代表全test推理或数据许可已获额外授权。 HF 数据卡未明确许可，使用与再分发前须确认原发布方及 BBC 条款；原文、参考、详细响应与完整索引不公开提交。

## LibriSpeech 原始音频原生入口（抽样原生交付已验证）

原始资产为 OpenSLR 12 的 `test-clean.tar.gz`（2620 条）和 `test-other.tar.gz`（2939 条）。 使用发布方 Whisper
LibriSpeech 示例选用的 `base.en` 模型，固定 `openai-whisper==20250625`； 这是一项英语语音识别原生链路验证，不是通用对话或顶尖识别质量的宣称。 依赖见
`configs/workload-native/librispeech-whisper-npu-client-dependencies.txt`。 模型参数及推理使用 NPU，FLAC
解码和原始英文规范化/WER 评分使用 CPU。

先按硬件平台提供匹配的 PyTorch、torch_npu 和 CANN，再在独立客户端安装上述依赖，不覆盖共享推理服务环境。 原始归档下载地址、字节数与 SHA256 在 `LIBRISPEECH`
source manifest 中；当前验证使用发布方列出的 EU 镜像。 模型发布地址与 SHA256 在
`configs/workload-native/librispeech-whisper-author-pins.json` 中。
使用者在服务器执行以下获取命令，仓库不内置数据或权重；入口会再次核验摘要：

```bash
mkdir -p /private/librispeech-archives /private/whisper-models
curl --fail --location https://openslr.trmal.net/resources/12/test-clean.tar.gz \
  --output /private/librispeech-archives/test-clean.tar.gz
curl --fail --location https://openslr.trmal.net/resources/12/test-other.tar.gz \
  --output /private/librispeech-archives/test-other.tar.gz
curl --fail --location \
  https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt \
  --output /private/whisper-models/base.en.pt
```

入口 `scripts/run_librispeech_original.py` 接受原始归档目录、独立准备目录、模型目录和新的私有输出目录：

```bash
PYTHONPATH=workload-preparation/src python scripts/run_librispeech_original.py \
  --data-root /private/librispeech-archives \
  --prepared-root /private/librispeech-prepared \
  --model-root /private/whisper-models \
  --output /private/fresh-librispeech-run --device npu:1
```

原始归档和模型按固定 SHA256 核对，音频按字节解包并保留原文与原 utterance ID 的对应关系。 默认在模型输出前固定每个 split 的原 ID 排序前 3
条及最长音频；`--full` 执行所选 split 的全部任务， `--indices 6,7,8` 为输出前指定的原始位置，不能根据失败结果换题。 每条完整 FLAC 以原 16kHz
单声道输入官方 `transcribe`，仅指定英文、转写任务，其他使用原默认值； 不做 30 秒裁剪、补造提示词、重采样或缩短原生流程。 输出保留原始参考和模型响应，必须留在私有目录，不提交到
Git。

已完成[固定工件与研究消费核验](validation/librispeech-fixed-ci-research-handoff-20261025.json)： CI 安装包 38
个文件逐字节一致，全部 5559 条原关系保留，两个 split 各 6 次公共回查、各 3 条研究索引导入、 全部原关系的研究直接准备及各 6 次直接回查通过。原始音频字节、参考和采样数均保真。
[最终交付回执](validation/librispeech-native-delivery-20261025.json)同时记录冻结原生 8 条及固定公开入口另 6 条未见原位置的 NPU
转写/评分。 研究消费使用该固定公共工件，不代表研究默认依赖锁已更新；观测与分析未迁移。 该来源现已计入完成，但不代表全 5559 条模型推理、流式 ASR、音频问答或全部音频类别完成。

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

### 已验证原生范围与其他原生成profile的消费入口

首个NIAH
4K全500条已执行和[终态审计](validation/ruler-niah4k-full500-native-audit-20261025.json)：逐条原数据、实际请求/响应哈希、原字段、token预算和finish
reason均核对，发布方原指标在该profile为100.0。500条均为原128
token预算的合法length结果，保留评分；没有因此加预算、换题或重跑。公共入口对这500条保存记录的CPU回放也一致，但这一回放不称新的现场模型执行。

公共入口另有[5组真实执行审计](validation/ruler-public-first5-native-audit-20261025.json)，各为预先固定的原物理行0、1、2：4K
multikey_2、4K multikey_3、32K/128K single_1及4K CWE，合计15条；实际128K组输入为130930至130939 tokens，完整输入加原128
token预算在部署容量内，未截断。其余VT/FWE/QA_1各全500条继续排队执行。此处是可执行性及原任务评分证明，不是整套RULER、完整长度网格、模型能力排名或引擎性能提升。

后续8组原生成文件共4000条的[准备回执](validation/ruler-eight-profiles-preparation-20261025.json)逐一给出摘要、长度、种子和固定选样规则。生成文件仍留在服务器，源码仓不附数据。RULER原13任务仍缺5个essay任务和HotpotQA任务，不以当前7种任务覆盖替代完整原任务集合。

其他原生成profile接入统一索引时，使用 `scripts/bind_ruler_original_profile.py`，不要手动改默认清单中的固定摘要。将真实原生成参数记录为私有
`generation.json`，例如前述原生成命令仅将长度改为32768时：

```json
{
  "task": "niah_single_1",
  "max_seq_length": 32768,
  "random_seed": 42,
  "model_template_type": "base",
  "tokenizer_type": "hf",
  "tokenizer_model": "Qwen/Qwen3.5-27B",
  "tokenizer_revision": "fc05daec18b0a78c049392ed2e771dde82bdf654", "revision_note": "pragma: allowlist secret - public tokenizer revision",
  "subset": "validation",
  "num_samples": 500
}
```

安装固定数据组件后执行：

```sh
python -I scripts/bind_ruler_original_profile.py \
  --data-root /private/ruler-generated-32k \
  --file niah_single_1/validation.jsonl \
  --generation /private/generation.json --output /private/fresh-ruler-manifest.json
```

入口不生成、不下载、不复制或改写原数据，只核验完整文件的真实行数、字段、摘要与必需生成声明，写入新manifest，拒绝覆盖。未匹配现有登记字节的文件明确标为调用者选择的本地生成快照，不能把声明当作独立验证过的发布方数据签名；相同已登记字节也不能被重新标成另一任务或长度。生成配置必须来自真实原生成命令，不是借此手编新题。

字段完整性修复：原生`length_w_model_temp`应保留为长度元数据；NIAH的`index`可能表示答案字符位置，故与`token_position_answer`一起归入参考答案侧，不发送给模型。绑定入口拒绝未知未映射字段，避免静默丢失。旧CI50474ec0与8组消费回执只验证当时声明的字段投影，不能替代修复后全字段重建核验。

QA2原生成器存在文档数量递减到必要文档数以下后无法退出的控制流问题。兼容入口只允许显式选择独立副本：

```sh
python scripts/prepare_ruler_qa_required_doc_compat.py \
  --source /private/immutable-ruler-source \
  --output /private/fresh-ruler-qa-runtime --allow-required-doc-floor
```

须保留生成的兼容回执，并把其中`generator_compatibility`所需声明随真实生成配置记录；不得称为官方生成器完全未修改。该修复不修改题目、必要材料、答案、随机种子、提示渲染或原输出预算，原checkout不变。它不是缩短任务以跑通的替代模式。

统一准备配置为
`{"sources":[{"source_id":"RULER","data_root":"/private/ruler-generated-32k","manifest":"/private/fresh-ruler-manifest.json"}]}`，沿用pipeline
plan/run、prepared_rows和resolve_input。准备输出目录须与原data_root分开，不能放在其内部。研究侧使用同一manifest导入公共索引或直接prepare/resolve，不需要复制原数据到研究仓。

服务器已用该绑定CLI候选和固定CI50474ec0数据组件完成[8组消费链路核验](validation/ruler-eight-profile-binding-chain-20261025.json)：全4000条绑定/索引、各首尾3条共48条公共回查、研究索引导入及直接prepare/resolve各24条。另独立对原生成文件逐条比较全部4000条输入、参考、原index/行号和生成配置，不只是把provider与自己比较。绑定CLI的实际字节摘要记录在回执中，不冒充它已在当时通过最终CI；研究默认依赖锁仍未改，也不迁观测/特征/分析。

### LongMemEval 原生生成及原评分入口

采用发布方 cleaned v1 的 oracle/S/M
三个原文件，版本与哈希见来源清单和[获取回执](validation/longmemeval-original-acquisition-20261025.json)。oracle只含原证据会话，不能冒充S/M全历史。数据获取到用户指定的服务器目录，不纳入Git仓库。

固定发布方代码后执行：

```sh
git clone https://github.com/xiaowu0162/LongMemEval.git /private/longmemeval-source
git -C /private/longmemeval-source checkout 9e0b455f4ef0e2ab8f2e582289761153549043fc
python scripts/run_longmemeval_original.py \
  --source /private/longmemeval-source --data-root /private/longmemeval-cleaned \
  --variant oracle --model your-served-model --context-window 131072 \
  --api-base http://127.0.0.1:19466/v1 --output /private/fresh-longmemeval-oracle
```

`--variant s`、`--variant m`分别使用完整原S/M文件；默认固定物理行0、1、2，`--full`跑该原文件全部500题。默认沿用发布方CoT生成预算800，显式`--direct`才采用发布方direct预算500。不得为了通过容量检查临时换模式、换题或截断历史。

入口核验原代码摘要，直接执行未改写的`prepare_prompt`，保留原json历史格式、日期排序及证据标签隔离。`topk_context`显式取全部原会话数，不隐式截成最近50次会话；保留发布方1000
token余量，超过其检索历史预算即报容量阻塞，不执行截断分支。通过本地服务tokenize取得实际模型token；原函数只读取token数组长度，接口层以列表代替CPU张量，不改变token内容。模型采用本地已部署模型，非发布方模型结果复现。

生成文件`hypotheses.private.jsonl`保留原`question_id`和模型回答。评分仍用发布方原脚本及其支持的judge：

```sh
python -m venv /private/longmemeval-eval-env
/private/longmemeval-eval-env/bin/python -I -m pip install \
  -r /private/longmemeval-source/requirements-lite.txt \
  -c configs/workload-native/longmemeval-eval-client-constraints.txt
/private/longmemeval-eval-env/bin/python -I /private/longmemeval-source/src/evaluation/evaluate_qa.py \
  llama-3.1-70b-instruct /private/fresh-longmemeval-oracle/hypotheses.private.jsonl \
  /private/longmemeval-cleaned/longmemeval_oracle.json
```

客户端约束仅将httpx固定为0.27.2，以兼容发布方openai1.35.1使用的proxies参数；不修改原评分脚本、judge模型或评分提示。服务器Python3.12.13隔离环境已完成原评分模块导入、12个提示分支夹具、无网络客户端构造及pip
check；不冒充发布方Python3.9整机环境，也不等于judge已执行。

上述本地judge要求按原脚本在`http://localhost:8001/v1`部署`meta-llama/Meta-Llama-3.1-70B-Instruct`；不是当前Qwen服务。发布方还支持GPT-4o及mini评分，需要用户自行配置已授权的API与费用，不自动调用。没有合适judge时保留生成结果并标记评分阻塞，不把Qwen自评或字符串匹配冒充原官方评分。只跑生成、只读数据、CPU夹具测试均不能勾来源完成。

### XL-Sum 原摘要任务、作者 mT5 与多语言评分（进行中）

[固定原资产](validation/xlsum-original-assets-20261025.json)、[原分词修正后的135条原生运行](validation/xlsum-original-native-samples-20261025.json)、[全字段与候选研究消费核验](validation/xlsum-candidate-raw-consumer-20261025.json)分别记录获取、原生运行和读取接通证据；不互相替代。

采用 `csebuetnlp/xlsum` 固定版本的45个 v2 语言/书写变体，原 test 共114198行；不是旧论文版本44种语言的同一统计口径。每行原 `text`
是输入，`summary` 是参考答案，`id/url/title` 保留为来源注释；不把标题或答案添加进模型输入。中文及塞尔维亚语书写变体不合并，来源文件摘要与物理行位置仍参与记录身份。

数据与作者推荐模型 `csebuetnlp/mT5_multilingual_XLSum` 均限 **CC-BY-NC-SA-4.0
非商业研究用途**，不可不加审查地用于商业验收或再分发。本仓仅登记元数据、准备器和运行入口，不包含原数据、模型权重或生成文本。固定数据文件及模型/评分代码摘要分别见来源目录和
`configs/workload-native/xlsum-author-pins.json`。

在有足够存储的服务器上，按发布方条款取得固定资产后执行：

```sh
hf download csebuetnlp/xlsum --repo-type dataset \
  --revision 30fece425f9a3866e04321773ca7a80056d55ca6 \
  --include 'data/*_XLSum_v2.0.tar.bz2' --local-dir /private/xlsum-archives
workload-data prepare-xlsum-original \
  --archive-root /private/xlsum-archives --output /private/fresh-xlsum-test
hf download csebuetnlp/mT5_multilingual_XLSum \
  --revision 2437a524effdbadc327ced84595508f1e32025b3 \
  --include config.json tokenizer_config.json special_tokens_map.json spiece.model pytorch_model.bin README.md \
  --local-dir /private/xlsum-model
git clone https://github.com/csebuetnlp/xl-sum.git /private/xlsum-source
git -C /private/xlsum-source checkout afcd803c8d7c98f3be60e9a7ce57fcbc3e729e8c
```

运行时必须使用独立的 NPU 生成/评分客户端，不能在运行中的引擎环境直接升级或降级 Transformers。客户端额外依赖固定在
`configs/workload-native/xlsum-author-npu-client-dependencies.txt`；Torch、torch-npu 和 CANN
需使用机器对应的兼容栈。服务器已测栈为
Python3.12、Torch2.10.0+cpu配套torch-npu2.10.0.post4、Transformers4.57.3；硬件为Ascend910B2/aarch64，CANN9.1.0，npu-smi报告版本26.0.rc1。CPU字样是Torch发行包标识，实际模型参数和生成输出均在NPU，不能据此误称CPU推理。初期继承系统包的客户端有vllm-ascend元数据冲突，不作为封闭依赖环境交付；已建立不继承引擎包的独立客户端，只读复用匹配的Torch/NPU库与完整UniDic，`pip check`通过，并以事前选定、此前未执行的英语原物理行3完成NPU生成和作者原评分。该条只验证环境，不替代、增加或改变主验收的固定前三条范围。

[固定CI工件与研究侧接通](validation/xlsum-fixed-ci-research-handoff-20261025.json)核验了数据提交f100a0a9的35个安装载荷、45个变体的公共首尾读取和研究侧消费；研究代码40文件与已有安装版本一致。没有升级研究默认依赖锁，也没有迁入研究分析或观测。独立客户端的原题探测见[环境验证](validation/xlsum-clean-client-environment-20261025.json)；与数据接口验证分开记录。

另按作者说明安装其固定源码下的 `multilingual_rouge_scoring/`，**不能以 PyPI 通用英语 rouge-score 替换**。准备显式 `NLTK_DATA` 目录中的
`stopwords/punkt/punkt_tab`，日语使用作者推荐的完整
UniDic；中文、泰语、孟加拉语、土耳其语等分别使用原评分器的依赖。作者原本没有专用词干器的语言，沿用其原回退逻辑，不伪装成英语评分。

```sh
NLTK_DATA=/private/xlsum-nltk-data python scripts/run_xlsum_original.py \
  --source-root /private/xlsum-source --model-root /private/xlsum-model \
  --tokenizer-python /private/xlsum-tokenizer-env/bin/python \
  --data-root /private/fresh-xlsum-test --language english \
  --output /private/fresh-xlsum-english
```

默认固定原物理行0、1、2；`--language` 可重复指定，`--language all` 选择45个变体，`--full` 执行所选变体全部原
test。小集按统一30分钟规则全跑，大集固定抽样；前三条预检仅用于检查可执行性和安排后续范围，不自动等于来源交付完成。`--reuse /private/prior-native-output`
校验原文件、模型、配置、分词输入、输出解码与私有记录摘要后复用已完成请求，损坏证据直接报错，不通过重复生成凑通过。

生成严格沿用作者模型卡：空白规范化、编码长度512、解码长度84、beam4、禁止重复2-gram，保留模型配置 `length_penalty=0.6` 和未绑定输入/输出权重，FP32/NPU
执行。**512是作者原协议，不是容量不足时临时截题；原文完整保留，也不能据此宣称验证了长上下文能力。** 评分直接执行固定作者源码中的 `calculate_rouge`、句子切分及多语言
ROUGE，保留四类 ROUGE 和 bootstrap；这是模型卡生成加原评分接口，不冒充整套训练/Trainer 流程或论文结果复现。

兼容性核验发现两处不可忽略的差异：Transformers5.5.4
会强制改变作者未绑定权重配置；4.57.3虽保留配置，但固定135条中有1条的分词输入与4.11.3不一致。因此生成客户端保留4.57.3/NPU模型后端，分词与解码通过本地管道调用独立的原4.11.3慢分词器，绝不修改原文或猜测替代
token。分词环境用 Python3.9，完整依赖见 `xlsum-original-tokenizer-py39-dependencies.txt`，以 `-I` 隔离环境启动，无
Torch、模型推理或网络服务；已通过该环境 `pip check`。旧Python只用于离线分词兼容，不作为公网服务运行环境。

修正后45变体固定135条已完成原分词、NPU生成和原多语言评分；134条有效输出复用，仅重新执行了输入不一致的那道原题。全114198行已核验五字段不丢失，45变体均完成公共索引首尾读取及研究侧接通；这是候选代码消费验证，不是固定CI工件交付或研究结论。按
`60秒 + 1.25 × 该变体前三条中最慢生成耗时 × 原 test 行数`
估计，10个变体不超过30分钟，需要全跑6068条，范围冻结后不因结果调整；其余35个固定抽样。已停止不兼容预检后的全量进程，保留其证据，恢复时逐条校验并复用有效请求，输入不一致者只重跑同一道原题，不换题。此耗时只用于排期，不是引擎性能结论。以上保留恢复阶段记录，以下为最终状态：
该全量执行现已结束。45变体共6173个主要位置（10变体全量6068，其余35各3）已逐条复核原字段、原分词输入和生成token解码，作者多语言ROUGE均完成；最终记录见[完整交付审计](validation/xlsum-final-frozen-selection-audit-20261025.json)及[交付回执](validation/xlsum-native-delivery-20261025.json)。固定CI工件、研究消费和独立客户端验证均已闭环。前述135条与候选回执是保留的阶段记录，不重复计覆盖；不是全114198条推理或长上下文服务验证。

`--rerun-tokenizer-mismatch` 是修复历史 tokenizer
输入差异的显式选项：只有同一原文件、同一行、同一模型和预算且私有证据摘要完整，但原编码输入不同的旧结果才失效并重跑原题。默认遇此差异报错；文件损坏、改题、配置不符仍报错，不被此选项忽略。保存生成
token 可用原解码器重解码，不需要再请求模型。

### STS-B原始句对连续相似度评价

官方依据：[EmbeddingSimilarityEvaluator说明与模型示例](https://www.sbert.net/docs/package_reference/sentence_transformer/evaluation.html#embeddingsimilarityevaluator)、[v5.1.2原STSBenchmark加载/评价示例](https://github.com/huggingface/sentence-transformers/blob/v5.1.2/examples/sentence_transformer/training/sts/training_stsbenchmark.py)。本接入只使用其原数据加载约定和原评价，不执行训练示例的微调。

`STSB`采用Sentence
Transformers官方提供的`stsbenchmark.tsv.gz`原分发，保留全部8628行及原8列，train5749、dev1500、test1379不混用。它是原STSBenchmark任务的框架分发格式，不声称与最初SemEval压缩容器逐字节相同；没有自行改题、生成替代句子或改成二分类。原句对只作为两个文本输入，原0到5分的连续标签在参考侧保留原字符串；仅在原评估器入口按作者示例除以5，不改写存储资产。

数据URL、SHA和体积见`workload-data catalog download-plan STSB`。模型绑定为官方评估文档示例采用的`sentence-transformers/all-mpnet-base-v2`；模型与Sentence
Transformers5.1.2源码和PyPI
wheel均固定在`configs/workload-native/stsb-author-pins.json`。model目录按pins保存全部13个列出文件；可用HF下载工具指定相同revision和文件列表取得，不下载到工作站或提交进Git。

```sh
python -I scripts/run_stsb_original.py --data-root /private/stsb \
  --model-root /private/all-mpnet-base-v2 --device npu:0 \
  --output /private/fresh-stsb-native
```

默认分别全跑原dev1500对和test1379对，用原`EmbeddingSimilarityEvaluator`计算模型默认cosine相似度的Pearson及Spearman；允许`--split test`或`--split dev`明确选择一个完整split，不支持改题、缩小原句或把train冒充测试。模型FP32/eager，batch16；384为原模型配置，不是人为容量截断策略，不构成长上下文验证。不是训练/微调复现，也不是生产检索流量或KV可复用证据。

服务器已验证的客户端复用本文件BEIR节的隔离环境和`beir-npu-client-dependencies.txt`完整依赖超集，Python3.12.13、Torch2.10.0+cpu、torch_npu2.10.0.post4、CANN9.1.0；其中BEIR与pytrec是该客户端现有额外依赖，不参与STS评分，不声称已验证一个另建的最小化依赖环境。不要替换服务环境的Torch。固定数据组件安装后用`-I`运行，避免全局PYTHONPATH污染；候选源码测试使用明确的数据组件路径，不能把源码测试冒充固定wheel交付。

官方PyPI5.1.2
wheel的6个相关运行文件是CRLF，tag源码为LF；已核验其原wheel整体SHA、安装字节及只转换换行后与Git原文件一致。入口验证PyPI原安装文件摘要，不修改已安装文件、不仅凭版本号放行。模型必须通过原`SentenceTransformer`类加载，验证代理不继承它以免触发其按具体类名判断模型类型的自动fallback；模型参数、原图最终向量均核验在NPU，不能回退到临时mean-pooling图。

保存的私有向量可通过同一原评估器复核，不重新跑模型：

```sh
python -I scripts/run_stsb_original.py --data-root /private/stsb --mode score \
  --reuse /private/prior-stsb-native --output /private/fresh-stsb-score-audit
```

入口核对原文件、模型与协议身份、编码文本序列摘要、参数及向量文件摘要；score模式不是新NPU运行。原句对、完整向量和索引不公开。原dev/test全量NPU评价、固定CI组件和研究消费、安装后的原指标重算均已通过，见[最终交付证据](validation/stsb-native-delivery-20261025.json)。当前只对上述原数据和具名协议勾选，不延伸到训练或其它句对数据集。公开来源定位不代表已取得任意商业使用或再分发授权，原始数据始终外部存放。

### BEIR 原始检索集合与重排协议

`BEIR_SCIFACT`和`BEIR_NFCORPUS`共享原集合，但有两个不同执行协议：原查询/全文档的dense检索，以及用该检索结果产生候选的top100交叉编码重排。qrels是原相关性评分参考，不是模型输入，更不能只用正例构造重排候选。数据组件完整保留corpus、queries和各split的qrels；不是将这些行转换为chat请求。

框架固定为`beir-cellar/beir`的v2.2.0，具体提交和运行文件摘要见`configs/workload-native/beir-author-pins.json`。模型选用其官方示例列出的TAS-B与MiniLM组合，分别固定为`sentence-transformers/msmarco-distilbert-base-tas-b`和`cross-encoder/ms-marco-MiniLM-L6-v2`的已校验revision。不是作者当前Stella示例或BM25第一阶段的基线复现，不使用聊天模型替代编码器。本次部署为FP32/eager、NPU后端；512为原编码器策略，不是为容量临时截短原任务，也不能据此宣称长上下文能力。

数据获取信息与逐文件SHA通过`workload-data catalog download-plan BEIR_SCIFACT`或对应NFCorpus命令查看。模型按pins中的revision下载其全部列出文件，放在外部路径，不打包到Git或wheel。HF的BEIR数据卡标注CC-BY-SA-4.0，原始来源的适用条款也须核对；登记不构成商业使用或再分发授权。

```sh
workload-data prepare-beir-original --source-id BEIR_SCIFACT \
  --data-archive /private/scifact.zip --output /private/fresh-scifact
python -I scripts/run_beir_original.py --source-id BEIR_SCIFACT \
  --data-root /private/fresh-scifact --encoder-root /private/tas-b \
  --cross-encoder-root /private/minilm --device npu:0 \
  --output /private/fresh-scifact-native
```

两来源均按原test全部查询执行，SciFact为5183文档/300查询、NFCorpus为3633文档/323查询。第一阶段cos_sim、batch16、corpus_chunk_size50000，第二阶段top100、batch128；指标调用原`EvaluateRetrieval`的NDCG/MAP/Recall/Precision/MRR。训练和开发集相关性标签保留为资产，不混入test。使用NFCorpus时替换source
ID、原zip和数据根即可。此入口当前面向这两个小集合的全量协议，不伪称已支持所有BEIR集合。

先建立独立客户端，安装`configs/workload-native/beir-npu-client-dependencies.txt`和固定数据组件wheel。已验证平台为Python3.12.13/aarch64、Ascend910B2、Torch2.10.0+cpu、torch_npu2.10.0.post4、CANN9.1.0。Torch/NPU采用服务器已经部署的配套二进制，只读复用；包名中的`+cpu`不代表本次模型在CPU上运行。不要让pip替换服务端Torch，更不要在服务环境安装依赖。此客户端依赖组合是已验证的NPU适配，不声称是BEIR作者完整历史环境。运行和`pip check`使用`-I`，避免CANN
shell中的全局PYTHONPATH混入其他工具依赖；当前隔离检查无冲突。

ARM如需构建`pytrec-eval-terrier==0.5.11`，使用其原sdist和`trec_eval`v9.0.8原源码，把后者放在原setup识别的`trec_eval/`子目录。不要改评分实现或改用简化Python指标；服务器遇到上游构建时下载超时后，已从官方codeload取得同一版本、核验全部111个原Git文件并用原setup构建通过。具体源摘要记录在BEIR核验回执中。可在隔离环境执行`python -I -m pip wheel --no-deps --no-build-isolation /private/pytrec-source`，再安装生成的本机wheel；不同架构不能直接复用ARM
wheel。

已有原始排名可离线重算，不需再次加载模型：

```sh
python -I scripts/run_beir_original.py --source-id BEIR_SCIFACT \
  --data-root /private/fresh-scifact --mode score \
  --dense-results /private/prior/dense-results.private.json \
  --rerank-results /private/prior/rerank-results.private.json \
  --output /private/fresh-scifact-score-audit
```

score模式会核验完整查询覆盖、文档标识、有限得分以及重排候选恰好来自dense
top100；它只证明这些排名可被原评分器评价，不能独立证明排名由某模型生成。真实NPU生成、原始排名SHA、固定数据组件及研究消费核验须分别取证。完整排名、原文和索引始终留在用户私有目录。原全量NPU执行、最终公开命令交接与固定CI工件/研究消费核验均已完成，见[最终交付证据](validation/beir-native-delivery-20261025.json)；只对这两个登记来源和上述协议勾选，不延伸到全部BEIR集合。

### 既有研究原资产复用（8来源、48文件；原生交付待补）

已按研究回执固定LongAlpaca、NarrativeQA、NQ-open、DuReader
demo、WildChat-1M、UltraChat、InfiniteBench、LMSYS-Chat的原始文件。仅迁原资产版本/URL/摘要及获取校验入口，不迁分析、观测、研究数据库或原文。总计48文件、9,509,254,890字节；这是历史取数记录，不表示这些字节已在新服务器重新核验，也不是全量原生任务执行证明。

```sh
# 默认仅输出离线计划，不创建目录、不联网。
python scripts/prepare_reused_research_assets.py WILDCHAT
# 在服务器已有原始目录上只读校验，不重新下载。
python scripts/prepare_reused_research_assets.py WILDCHAT \
  --mode verify --data-root /private/WILDCHAT/pinned-revision
# 确实缺失且权限允许时才下载；显式字节预算不足会在创建目录前拒绝。
python scripts/prepare_reused_research_assets.py WILDCHAT --mode download \
  --data-root /private/fresh-WILDCHAT --max-bytes 3400000000
```

其他七个来源使用同一入口替换ID；`--files`只允许显式原文件路径，不生成替代内容。`--resume`只重用同来源/同配置的完整已校验文件，不覆盖损坏文件。LMSYS等访问受限来源不自动登录或接受条款，应使用发布方已获授权客户端取得后执行verify；没有复制历史凭据。文件角色暂标`raw_source`，不会因为已下载就被当作可直接输入模型的请求。

范围须保持：NarrativeQA当前为公开QA/文档/摘要表，不是完整书籍；Natural
Questions当前为NQ-open；DuReader为demo；WildChat为原1M固定14分片，不是4.8M；UltraChat为发布方合成语料；InfiniteBench保留12原文件，答案类型和任务模板/评分需继续适配。来源级原生交付仍不勾选。当前只有无模型夹具测试，服务器复核待可信入口恢复后进行。

### FLORES-200 原始翻译资产（准备中，未完成原生接入）

公开资产入口已补充，以下命令应在有空间的服务器执行，数据与模型不放进代码仓：

```sh
python scripts/prepare_flores200_original_assets.py \
  --output /private/fresh-flores200-assets --parts data model
python scripts/prepare_flores200_original_assets.py \
  --output /private/fresh-flores200-assets --parts data model --verify-only
```

服务器无法直连HF时可显式加
`--hf-endpoint https://hf-mirror.com`，模型仍按原发布方固定版本及LFS/Git对象校验；原数据始终取Meta原地址，不用第三方改题版本。`--verify-only`只读、不联网、不写回执；`--resume`只允许匹配本入口来源及配置摘要的目录，复用完整已校验文件，不声称HTTP断点续传，也不覆盖已有错误文件。该公开入口目前仅有无模型软件测试，待恢复服务器后实测；不是原生翻译运行入口，不计来源交付。

官方短链连接重置后，已直接从Meta官方公开主机取得原归档，服务器检查411个普通文件：dev和devtest各204份语言/文字文件，每份分别997和1012条；隐藏test未取得，不能声称拥有隐藏集。原字节及观察SHA256保留，但尚未找到单独发布的校验和，不能把本次观察值冒称作者发布值。原文件仍只在服务器，见[获取回执](validation/flores200-original-acquisition-progress-20261025.json)。

发布方NLLB-600M的8个工件在断联前已取得并核验，独立环境准备终态待核。其语言特殊标记表缺少原归档的arb_Latn、min_Arab、sat_Olck；sat_Beng不是sat_Olck的同义代码，不得自行替换。该模型许可为CC-BY-NC-4.0，仅列非商业科研验证。原始语言关系、适配、原生执行和评分均待逐项补齐，当前不计交付。原[发布方说明](https://github.com/facebookresearch/flores/blob/main/flores200/README.md)主要评价chrF++，后续原评分协议需独立核验，不能拿文件检查代替。

### BurstGPT 官方缩放工具验证（历史执行截面，当前交付结论见本节补充）

第2份原20条已完成并通过保存响应/官方日志审计；第3份最后确认在运行，随后SSH拒绝连接，新入口指纹待核验。这里不能按断联推断进程终止、服务器回收或执行失败。恢复后核查原输出与进程身份，完成结果优先复用，不重复启动队列。

第1份轨迹的原速20条结果保留，不重复运行。第2、3份使用官方 `--scale 1000000`，仅用于工具可用性验证；官方源码将时间戳除以 scale，原 CSV
不改、不归零，选定的原物理行0—19全部保留，1024-token预算不变。原始首时间戳对应约61/225天，缩放后初始等待约5.27/19.44秒。这个配置压缩了到达间隔，不能用于原始流量性能、稳态能力或突发可靠性结论。

公共入口仍为 `scripts/run_burstgpt_original_example.py`，在原第1份命令上指定对应 `--trace /private/BurstGPT_2.csv` 或
`BurstGPT_3.csv`、`--count 20 --scale 1000000`
和全新私有输出目录。保留官方原始代码校验、原外部提示池校验及全部请求/响应日志。它是官方外部提示池模拟，不是企业私有原始请求回放。每份必须完成工具执行及保存响应审计后才更新进度；获准缩放或脚本准备好不算跑通。研究线程优先用卡，等待期间不发模型请求。

### BillSum 发布方 HF 原始法案文本与 Pegasus 协议（接入中）

来源为 `FiscalNote/billsum` 固定发布版本，保留 `text`、`summary`、`title`
全部原字段；train18949、test3269、ca_test1237，共23455条。这里接入的是发布方 HF 三字段版本，不是原论文的 `clean_text`/BERT
清洗流程，也不声称保留原归档的 bill_id/external_id。原归档公开下载尝试未成功，这个版本差异须保留说明，不以自行构造补齐。原始数据、生成摘要及全量索引均不入 Git。

运行环境依赖见 `configs/workload-native/billsum-native-npu-client-dependencies.txt`；仅原平台匹配的 NPU Torch
栈执行模型，CPU 可评分，不使用 CPU 代替 NPU 推理。采用 Google 发布的 `google/pegasus-billsum`，不是把其他摘要模型改名；保持原 checkpoint
generation_config、1024原生编码窗口、beam8、max256、min32、length_penalty0.6。长原文按该模型原生窗口处理，不据此声称全长上下文验证。

```sh
python scripts/prepare_billsum_original_assets.py \
  --output /private/billsum-assets --parts data model scoring
python scripts/run_billsum_original.py --mode generate \
  --data-root /private/billsum-assets/data --model-root /private/billsum-assets/model \
  --split test --device npu:1 --output /private/fresh-billsum-test
python scripts/run_billsum_original.py --mode score \
  --data-root /private/billsum-assets/data --source-root /private/billsum-assets/scoring \
  --split test --reuse /private/fresh-billsum-test --output /private/fresh-billsum-score
```

`--split ca_test` 为原加州测试集。默认固定物理行0、1、2；`--full` 可跑原 split 全量，`--indices`
在运行前明确原行号，不能失败后换题。原两测试集全量估时分别约9.16和2.75小时，超过30分钟规则，因此首批验证固定各前三条；不是全量成绩复现。模型输入仅原
`text`，标题留元数据侧、摘要留参考侧。评分复用保存响应，执行固定 HF 原后处理与
ROUGE，不新增模型调用，也不是原论文全部实验流程。下载、字段读取、临时推理、公共入口、固定工件和研究消费须分别验收；固定CI39文件工件与原包逐字节一致、两项CI成功、公共/研究全量prepare和首尾18条回查、18条研究导入已通过；现按此HF版本协议勾选，见[最终交付回执](validation/billsum-native-delivery-20261025.json)，不外推到其他版本或全量推理。

### GPQA 作者原发布包与原zero-shot协议

使用作者公开GitHub的`dataset.zip`，固定代码版本见[原资产回执](validation/gpqa-original-acquisition-20261025.json)。按该版本README公开提供的密码解压；不需要绕过HF门槛，也不把题目下载到本地工作站或提交进Git。main448、diamond198、extended546是有重叠的原split，不能相加称为1192个独立问题；`gpqa_experts.csv`是专家元数据，不是第四个benchmark。

```sh
git clone https://github.com/idavidrein/gpqa.git /private/gpqa-source
git -C /private/gpqa-source checkout 56686c06f5e19865c153de0fdb11be3890014df7
workload-data prepare-gpqa-original \
  --data-archive /private/gpqa-source/dataset.zip --output /private/fresh-gpqa-data
python scripts/run_gpqa_original.py \
  --source /private/gpqa-source --data-root /private/fresh-gpqa-data \
  --split diamond --model your-served-model --context-window 131072 \
  --api-base http://127.0.0.1:19466/v1 --full --include-index69 \
  --output /private/fresh-gpqa-diamond
```

运行依赖Python和pandas。发布方原环境为Python3.9/pandas1.5.2；服务器已建立独立Python3.9.25/pandas1.5.2/numpy1.23.5客户端，安装项见`configs/workload-native/gpqa-publisher-py39-loader-dependencies.txt`。[原版本复核](validation/gpqa-publisher-py39-verification-20261025.json)确认全部1192行输入/选项与固定CI组件一致，并对main447、Diamond197个保存结果重建原提示和重算评分，未重复调用模型。原NPU生成复用的服务器客户端为Python3.12/pandas3.0.5；独立固定CI消费环境为Python3.12/pandas2.2.3/numpy1.26.4（见`gpqa-original-client-py312-dependencies.txt`），三者边界不可混称。

上述验证只涉及原CSV加载、提示、解析和评分路径，不能冒充发布方完整训练、商业SDK或GPT-4模型环境。原代码顶层会初始化商业API客户端，因此入口在逐文件核对摘要后只载入所需原函数/类，不初始化外部凭据或调用商业API。

统一数据入口为`GPQA`及三个原CSV，按同一source
manifest执行pipeline/resolve。输入只包含原`Question`与按作者seed0打乱的匿名`choices`；答案索引、原答案标记、解释、专家注释和canary全部在参考侧，所有原列均保留，物理文件SHA+行号标识案例。不是手编题目或自造选项。服务器[固定CI工件与研究消费核验](validation/gpqa-fixed-ci-research-handoff-20261025.json)核对34个安装payload、三split全部1192条原字段及作者选项顺序、公共回查18条、研究索引导入与直接prepare/resolve各9条；既往候选回执仍保留，不用候选冒充固定交付工件。

当前绑定原`load_examples`、seed0选项打乱、`zero_shot_prompt`、`select_and_call_model`的系统/用户消息与原`parse_sampled_answer`，输出预算1000不改，仅将模型传输指向已声明的本地模型。原主循环对索引69有显式跳过逻辑，本入口如实保留为未生成且答错，计入原分母；不能隐藏该行为后声称每题都发出了模型请求。结果不是作者GPT-4基线复现，也不代表CoT、检索或self-consistency协议已验证。

为验证所有原题确实可运行，提供显式`--include-index69`兼容模式，只取消本入口对原物理行索引69（从0计数，即第70条）的跳过；原题、选项、提示、1000预算及解析器不变，不改官方源码。应配合`--full --reuse`复用已有等价请求，单独保存兼容模式回执与分数，不覆盖或冒充作者原循环结果。仅“原循环已结束”不足以证明被跳过题的执行能力；兼容验证未完成前不据此声称全题生成已验证。

服务器已完成[全部原位置兼容审计](validation/gpqa-allrows-compatibility-audit-20261025.json)：三split的1192个位置均有原提示下的模型响应和原解析评分，其中1189个保存响应逐字节复用，仅为三个原索引69新增3次请求。原循环分数与兼容结果分别保留。[交付回执](validation/gpqa-native-delivery-20261025.json)记录固定数据工件、入口CI、环境和消费边界。不带`--include-index69`仍按作者原循环保留跳过行为，不能将其结果标为全位置执行验证。

`--split main`、`--split extended`分别选原split；不带`--full`为固定物理行0、1、2，不能在看到失败后换题。新输出目录拒绝覆盖，保存请求/响应和原解析结果；缺失答案按原逻辑记错，不重试凑分。`--reuse /private/completed-original-run`只复用已完成同协议、同服务模型且私有记录摘要一致、实际请求完全等价的保存响应；题目相同但选项不同不会误复用，不能只按题目ID命中。复用后仍重新执行原解析与评分，不再调用模型。来源主勾不因下载、CPU测试或运行中的文件提前勾选。

# 历史准备能力补迁：LongAlpaca、NQ-open 与 DuReader demo

2026-10-25恢复后的实测补充：BurstGPT第3份原20条保存终态已找回并逐条审计，未新增模型调用；三份各20条均已核。随后固定CI提供者33份安装文件与研究消费端40个Python文件安装哈希通过，三份各原0/1/2行的公共直接读取、公共索引解析、研究索引解析全部相等，且均拒绝把元数据变成模型请求。该来源现可勾选，见[交付回执](validation/burstgpt-native-delivery-20261025.json)。授权scale缩放工具验证不是原始时序性能复现；外部池不是原私有用户prompt，研究默认依赖锁未更新。
FLORES英→法、英→简中各原devtest:0/1/2已在NPU完成生成，原始句对及chrF++复算一致；是3个平行行号、2个方向、6次生成，不是6个独立句子或全语言覆盖。
首次读取因原tar成员带`./`前缀而在模型导入前失败，日志保留；修复只规范成员路径查找，不改归档/题目，回归覆盖原前缀形式。真实运行回执见[双方向审计](validation/flores200-two-profile-native-audit-20261025.json)。
当前客户端整体`pip check`仍非零，报告可见CANN编译/分析工具的依赖缺失。已完成的生成/评分保留，但不能宣称整个环境合格；不为清零盲装全部工具或替换服务栈。原模型非商业许可、3个不支持语言代码、spBLEU和数据提供者/研究消费缺口均保留。

InfiniteBench 的历史准备映射已分数字答案和字符串答案补迁，均去除研究feature选择器。数字组仅
code_run.jsonl、math_calc.jsonl、math_find.jsonl；另9份为字符串组。按原记录保留数字类型和字符串前导零，不拼造新题。

```bash
python scripts/prepare_reused_source_records.py INFINITEBENCH \
  --data-root /path/to/original-infinitebench --files math_calc.jsonl \
  --output /path/to/fresh-math-calc-preparation --limit 3
python scripts/prepare_reused_source_records.py INFINITEBENCH \
  --data-root /path/to/original-infinitebench --files number_string.jsonl \
  --output /path/to/fresh-number-string-preparation --limit 3
```

必须显式选择登记文件；不允许混合两类答案的文件共用一份类型映射。跨多文件时limit是准备入口的总记录上限，不是每文件上限，因此逐任务验证应各建独立输出目录，不据此宣称12任务已消费。
历史按首行猜类型的逻辑不迁入；文件范围依据历史全文件映射回执固定，绑定时仍检查完整文件哈希和每行字段类型。当前只有软件夹具通过，真实原数据、固定工件研究消费和原生评分仍待完成。

私有产物防误提交：`.private.json`、`.private.jsonl`、`-private.json`、`-private.jsonl`及`.private-error.txt`后缀已加入Git忽略规则；现有`workload-runs/`仍整体忽略。此规则不清理、不取消跟踪已有文件，也不替代权限审查；原始数据仍须外置，禁止强制提交私有产物。
InfiniteBench运行入口逐原始位置先保存完整请求及预检token数，再保存完整响应和两者摘要；尝试与完成位置分开记录，API失败时保留请求且响应摘要为null。文件名为`row-N.request-private.json`、`row-N.response-private.json`，仅供服务器私有复核，不代表新增原生运行证据。

DS-1000历史结果复用（不执行code_context或生成代码）：

```bash
python scripts/prepare_ds1000_official_samples.py \
  --generation-dir /path/to/existing-ds1000-generation \
  --data-root /path/to/original-ds1000-assets \
  --publisher-inference /path/to/fixed-ds1000/run_inference.py \
  --output /path/to/fresh-private-ds1000-score-input
```

固定作者版本`b39aab71da6d23ef8d3cac59a7c5f834516ab334`，CLI校验原始数据和推理源码，核对原0/1/2问题的原始completion请求、终态、usage及已保存后处理结果。`length`保留，不能更换题目或增加原1024预算。
输出 answers-private.jsonl
保留作者推理marker后处理结果，original-dataset-private.jsonl保留全部原始行和code_context；不额外修代码。旧回执缺少逐文件历史哈希时只记录本次观察摘要，不追溯补造原有证明。
作者`test_ds1000.py`还有独立postprocess及test_execution/test_string步骤，须在核验的隔离环境里按原顺序执行，每题独立进程。作者默认驱动遍历1000题，不能将三条answers直接放进去或给其余题填占位答案；固定原始前三题的抽样绑定仍需补齐并验证，不得因此宣布整条执行链完成。

2026-10-25实测补充：原1000题逐条与作者归档相等，原前三题上下文仅需copy/NumPy/Pandas。已在Python3.10.22、NumPy1.26.4、Pandas1.5.3的私有命名空间/chroot中，使用作者原postprocess与eval_ds1000完成评分；原120秒超时、16worker未改，汇总3题均值0.667，失败保留，无新模型调用。抽样通过将作者模块原ds1000全局列表显式绑定为原前三题，未修改作者源文件、code_context、test_execution或适用test_string，未为其他997题填答案。见[实测回执](validation/ds1000-original-pandas-namespace-scoring-20261025.json)。这不是七类库环境或1000题完整运行；公共复用启动器和消费收口仍待补，来源不勾。依赖求解中tqdm4.70.1不支持Python3.10，故仅将作者未固定的tqdm选为4.67.1，不更改作者Python/NumPy/Pandas版本。

随后已将该流程整理为公共入口并完成服务器验证，固定数据组件与研究侧全部1000条原prompt、reference_code、code_context和metadata直接读取相等，原0/1/2行索引解析相等；见[公共Pandas抽样回执](validation/ds1000-public-pandas-profile-20261025.json)。上段消费/入口待补是先前截面，不再作为当前缺口。

```bash
conda create -y -p /private/ds1000-py310 --override-channels -c conda-forge \
  python=3.10.22 numpy=1.26.4 pandas=1.5.3 tqdm=4.67.1 pip
python scripts/run_ds1000_isolated_scoring.py \
  --author-root /private/DS-1000-b39a \
  --prepared-dir /private/ds1000-official-prepared \
  --python-runtime /private/ds1000-py310 \
  --work-root /private-accessible-parent/new-ds1000-scoring \
  --sandbox-uid 65534
```

先按本节准备命令取得私有`prepared-dir`；作者源码固定`b39aab71da6d23ef8d3cac59a7c5f834516ab334`，原Python文件、environment.yml及完整gzip由`configs/workload-native/ds1000-author-scoring-pins.json`核验。需获授权root启动器、专用非0宿主UID及可供该UID遍历的父目录；不要改变共享目录权限。入口复制指定评分环境到私有任务目录并只读挂载为`/opt/python`，不暴露宿主`/data`、模型文件或NPU设备；需要相应服务器磁盘空间，不下载到开发机。公共输出不含候选和错误原文，日志及作者完整汇总留在私有目录。NumPy/Pandas可能来自conda的egg-info，元数据检查兼容标准egg-info/dist-info，不导入候选代码。

**当前DS1000已完成通用原题范围入口及七库抽样交付**，见[最终回执](validation/ds1000-native-delivery-20261025.json)。前述Pandas三题和环境阻塞为历史截面：现复用原0/1/2题，新增其余六库各原前三题18题，并增加原3/4/5连续范围回归，共24个不同原题已生成并完成作者评分。新增模型调用21次，评分只复用保存结果；失败、length终态及原1024预算保留，不合成全量正确率或性能结论。

生成入口保留默认前三题，同时支持互斥的`--indices`、`--start/--count`、`--full`。准备入口默认复用该次生成的完整选择，也可用`--indices`取预先规定的库分组，必须包含在父生成选择中，父选择一并留在回执，不以失败为由换题。评分通过原始ID字典及原任务视图对齐作者索引，未修改作者postprocess、eval_ds1000、code_context、test_execution或适用test_string；保持16
worker和120秒测试超时。全1000选择仅通过软件检查，未运行全套推理／评分；大规模评分应显式增大外围`--deadline`，该参数不改作者每题120秒。

原作者归档内`data/ds1000.jsonl.gz`逐字节解压后的SHA256正是生成入口要求的`test.jsonl`身份，已实测相等；不需要改题、重构prompt或另造数据。使用新的外部数据目录：

```bash
set -e
mkdir /private/new-ds1000-data
gzip -dc /private/DS-1000-b39a/data/ds1000.jsonl.gz > /private/new-ds1000-data/test.jsonl
conda env create -p /private/ds1000-full-arm64 \
  -f configs/workload-native/ds1000-author-arm64-port.yml
python scripts/run_ds1000_original_generation.py \
  --data-root /private/new-ds1000-data \
  --publisher-inference /private/DS-1000-b39a/run_inference.py \
  --output /private/new-ds1000-generation \
  --model workload-qwen35-27b-128k-nonthinking --context-capacity 131072 \
  --indices 0 1 2 291 292 293 511 512 513 666 667 668 711 712 713 817 818 819 932 933 934
```

若生成返回码2，表示原预算length终态已保存，不代表请求流程损坏；检查私有回执后按原准备入口继续，不增预算或重跑凑分。准备时指定该生成目录及新的外部输出目录；评分沿用上述`run_ds1000_isolated_scoring.py`，将`--prepared-dir`指向准备结果、`--python-runtime`指向独立评分环境。推理仍由独立NPU服务完成；CPU
PyTorch/TensorFlow只运行作者原数据科学代码评分，不能替换模型服务的Torch/NPU环境。

ARM环境是作者版本对齐移植，不是原环境字节级复现：原conda渠道无ARM的PyTorch2.2.0，故在独立评分客户端用同版官方PyPI
wheel；`tensorflow-cpu2.16.1`无目标ARM包，按[TensorFlow官方安装路线](https://www.tensorflow.org/install/pip)使用同版`tensorflow2.16.1`。新YAML整理自实测分步安装的前缀，非另行声称全新一次性clean-build通过。XGBoost2.0.3的conda包残留x86_64
wheel标签，隔离`pip check`因此返回1；原元数据未改，ARM导入和隔离预检已通过，不能称全部包元数据无告警。

隔离运行时只放行null/zero/random/urandom标准字符设备、只读CPU拓扑与cpuinfo，并提供明确标注的命名空间入口exe/cmdline兼容文件；不挂宿主进程树、宿主数据或NPU设备，不将兼容文件当观测数据。只读根、输入、作者源码及运行库，私有tmp/output可写。进程／线程上限256，并在候选代码前验证原16-worker的Manager/子进程结构；不因早期64上限耗尽而把环境错误计成模型答错。TF
intra/inter-op及OMP为1；保留所有失败环境尝试，不作隔离安全认证声明。
其他任务范围、其余库上下文及作者完整七库环境尚未接通，不把硬编码三题入口当成整个DS1000来源已完成。CI只跑无害准备/错误拒绝夹具，不跑生成代码。
代码库功能测试按作者环境在CPU上执行，与模型生成使用NPU是两个环节；不得为了运行测试改动模型服务的torch/torch-npu栈。当前只完成软件夹具验证，真实文件复核与官方功能评分均未完成，来源不打勾。

BigCodeBench 历史生成结果复用（不执行生成代码）：

```bash
python scripts/prepare_bigcodebench_official_samples.py \
  --generation-dir /path/to/existing-bigcodebench-generation \
  --data-root /path/to/original-bigcodebench-assets \
  --official-code /path/to/fixed-bigcodebench-source \
  --output /path/to/fresh-private-bigcodebench-score-input
```

输出 raw-solutions-private.jsonl
使用作者`solution`字段保存原始完整脚本回应，而不是`completion`，后者会让作者评分器另行拼接complete_prompt。 所有1140条原数据的全部字段保留到
original-dataset-private.jsonl；原Parquet校验身份，逻辑字段不改写。可将该文件设为作者支持的
`BIGCODEBENCH_OVERRIDE_PATH`，供固定版本的原loader读取，避免评分时漂移到另一个数据版本。
固定原始位置0/1/2、原提示格式化器、终止与usage元数据均核对，不自行剥离围栏或修代码。此前回执没有逐私有文件哈希；本入口记录的是本次观察摘要，不追溯宣称历史字节已被密码学绑定。
下一步为固定作者版本`09dd993f46c3fbf3a799465bb96d524edcb0b199`的官方sanitize，再在已核验沙箱里执行官方evaluate（instruct/full，固定三个task_id，pass@1，local执行backend）。保留raw与sanitize派生产物及摘要；官方sanitize是公开协议步骤，不是任意修补输出。
禁止直接采用作者evaluate默认的远端Gradio服务发送产物，亦不能在宿主进程执行`--execution local`。完整隔离、依赖、官方sanitize与功能评分均尚未验证；当前仅软件夹具测试，不打交付勾。

JSONSchemaBench
环境前检：`python scripts/probe_jsonschemabench_environment.py`。只读已安装分发版本，与固定作者requirements分别列出“缺失”“版本不同”“版本一致”，不安装、不给版本不同直接下不兼容结论，也不执行可选后端或调用模型。
当前本地核对缺少核心路径需要的dacite、prettytable，另缺工作流依赖stopit；因此此前模拟Engine测试不能升级为实际作者模块调用链通过。此结论仅针对本地软件环境，不代表服务器环境现状。
作者core初始化会自动导入其他后端，完整环境还须按原安装方式核验可选引擎依赖，不能只装几项核心包就宣称bench可跑。需在服务器独立客户端环境处理，禁止为匹配作者CPU客户端torch版本而替换正在服务的Ascend
torch/torch-npu栈。

JSONSchemaBench 作者Engine扩展（尚非完整运行入口）：`scripts/jsonschemabench_local_engine.py` 的 `create_engine`。
遵循[作者自定义后端接口](https://github.com/epfl-dlab/JSONSchemaBench/blob/8003e8405c4d8d8b327b1eb472c9856297d75493/docs/custom_engine.md)，接收固定版本
core.engine/core.types、显式本地模型配置、已禁用自动重试的loopback客户端及真实消息分词计数器。
不重写原始formatter、Dataset、bench或evaluator；adapt_schema只深拷贝，不补type、不改required、不强制additionalProperties=false。完整消息和schema原样进入后端，上下文不足不截断，必须收到原生终态及服务usage才设置成功状态。
固定作者OpenAI后端会为其供应商限制改写schema，且其本地base_url路径的分词器和max_context_length有模型名限制，因此不能直接把本地Qwen模型名填进去就宣称原生运行成功。该扩展按明确的“原schema、本地传输”配置处理，不宣称与作者OpenAI供应商配置完全相同。
Engine扩展当前只通过模拟接口/流式软件测试；原始core.bench→evaluator完整链路与真实NPU仍待接通。
旧guidance-ai三份schema的validator结果不能替代这一流程，原资产及回执不覆盖成新数据身份。

2026-10-25已通过官方HF元数据固定`epfl-dlab/JSONSchemaBench`版本 `5bd0f4640badc6f3f02df796421d21cb0ca0b141`，清单见
`configs/workload-native/jsonschemabench-official-assets.json`：10个具名配置及default汇总，
覆盖配置引用的train/val/test共33个原始Parquet及原README，共34文件、40827041字节。
default是汇总，不将其与具名分区重复累计成独立题目；仓库内未被这些配置引用的旧合并文件 不冒充官方test分区。清单的字节数/SHA256来自固定HF元数据，尚不表示服务器已取得文件。

在服务器外置私有目录准备，不在本地下载：

```bash
python scripts/prepare_jsonschemabench_original_assets.py --output /private/jsonschemabench-official
python scripts/prepare_jsonschemabench_original_assets.py --output /private/jsonschemabench-official --verify-only
```

入口复用既有固定资产获取/校验逻辑，默认只取data，不安装或下载模型；支持明确的 `--resume`及同摘要镜像，不覆盖不匹配文件。原始HF snapshot根为输出目录下的`data/`。
新入口仅有小型软件夹具证据，未在当前服务器运行。下一步须把原版`core.dataset.Dataset`
显式绑定到这个固定本地snapshot，核验`json_schema`字段与原test行，再运行官方bench与
evaluator；上游默认`load_dataset`没有revision参数，不能放任其重新拉取变化的main。
本轮未改schema、未换历史失败任务、未声称结构化输出后端或完整原生流程已跑通，不打完成勾。

官方bench桥接入口（软件实现，尚未实跑）：

```bash
python scripts/run_jsonschemabench_official.py \
  --official-source /private/JSONSchemaBench-clean-checkout \
  --snapshot-root /private/jsonschemabench-official/data \
  --output /private/fresh-jsonschemabench-native \
  --origin http://127.0.0.1:19466 \
  --model workload-qwen35-27b-128k-nonthinking \
  --max-tokens 1024 --context-capacity 131072 --limit 3
```

源checkout须为上述固定8003e840版本且Git干净，数据34文件须匹配固定摘要。 在独立官方客户端环境中安装原requirements和OpenAI
SDK；上游core会急切导入可选后端， 因此缺依赖必须先修复，不能替换成简化的假core，也不能为客户端匹配torch版本而改共享NPU服务。
模型ID、1024输出预算和上下文容量是示例本地profile，须与实际NPU部署核对，不冒充官方推荐模型配置。

入口只把官方Dataset的数据位置绑定到原始本地snapshot，保留原test顺序、原json_schema、
默认few-shot消息构造和原bench/evaluator；不shuffle、不改schema或补required字段。
默认10个具名分区各原前三条；`--tasks`可明确选分区，`--full`替代`--limit`运行全分区。
default汇总需用`--tasks default`单独运行，不与具名分区混算唯一任务覆盖。
各分区预期数量由官方Dataset读取，再核对实际返回数；每次原始输入和返回结果即时存到私有attempt文件。 中断/服务异常不自动重试、换题或重开，先复核已保存结果，不把失败条目从覆盖分母删除。

本地后端遵循作者custom Engine接口，原样发送schema。只有HTTP400且服务结构化错误码 明确为invalid/unsupported
schema时，才记录官方UNSUPPORTED_SCHEMA状态；断连、500、 限流、未知模型或不明400继续报错，不伪装成模型/数据质量失败。该分类比上游OpenAI后端的
catch-all更严格，属于明确的本地适配；原型性能字段不证明引擎独占耗时或性能提升。 新入口现有夹具联调证据，未在服务器完成真实HF加载、结构化后端、官方评分及研究消费验证， 来源总勾保持未完成。

安装工件核验：`python scripts/audit_workload_data_wheel.py --wheel /path/to/provider.whl`。
该检查将包内数据提供者模块、目录和schema资源逐字节与当前源码核对，并拒绝额外包载荷、重复成员及伪装在dist-info中的数据文件。 CI构建后生成 `PAYLOAD_AUDIT.json`
并随软件工件发布。它不执行安装、不校验真实数据或模型，也不证明研究仓已切换依赖；历史固定wheel回执保持冻结，新工件必须重新记录身份并核验消费链路。

HumanEval 历史生成结果复用入口（不执行代码）：

```bash
python scripts/prepare_humaneval_official_samples.py \
  --generation-dir /path/to/existing-humaneval-generation \
  --data-root /path/to/original-humaneval-assets \
  --output /path/to/fresh-private-official-score-input
```

核对固定原始HumanEval/0、/1、/2及历史生成回执、prompt哈希、完整请求配置和结果哈希；输出作者评分器需要的 task_id/completion
JSONL。completion逐字符保留，不去代码围栏、不修复、不替换失败样本，不重新调用模型。 后续需在独立核验的代码执行沙箱中使用作者固定版本的
evaluate_functional_correctness
和原始问题文件评分。不能在模型服务宿主进程、本机或未验证隔离的环境中执行生成代码。现有机器曾缺少可用Docker/Podman且PID/mount隔离失败，此历史阻塞不能视为新实例当前检查结果。

2026-10-25补充：替换实例已通过低权限宿主UID65534、独立user/mount/net/PID命名空间、chroot、只读运行库和清空capability探针；不暴露宿主数据、root、proc或NPU设备。使用`unshare --propagation unchanged`避免其默认传播处理失败，不再以缺少Docker/Podman判定不能隔离执行。已复用原HumanEval/0、/1、/2完整生成文本，按原作者`evaluate_functional_correctness`、3秒、单worker评分：前两题通过，第三题原样保留语法错误，样本pass@1为2/3；无新模型调用，不是164题全套成绩。见[实测回执](validation/humaneval-official-namespace-scoring-20261025.json)。一次性launcher误沿用了探针标签，实测回执已明确更正，不据错误标签否认实际代码执行；可复用公共评分启动器和消费闭环尚未交付，因此仍不勾选。

HumanEval随后已完成公共入口与固定消费收口，当前状态以[交付回执](validation/humaneval-native-delivery-20261025.json)为准，可勾选；上段“尚未交付”是先前一次性验证的历史截面。完整164题原字段已逐题比较，公共与研究侧直接读取一致；原前三题公共/研究索引解析一致。完整164题准备不等于完整164题推理评分。

公共隔离评分入口（仅Linux，不占NPU）：先使用上面的`prepare_humaneval_official_samples.py`生成私有准备目录；取得固定作者源码`b9453e8f1865df0a0057a1383767caa400acfbcf`，模块及原始gzip身份由`configs/workload-native/humaneval-author-scoring-pins.json`核验。另建Python3.12.13客户端，按`configs/workload-native/humaneval-isolated-scoring-dependencies.txt`安装NumPy/tqdm，不修改模型服务环境。已验证Linux
aarch64；其他发行版先验证隔离能力，失败不允许宿主执行回退。

```bash
python scripts/run_humaneval_isolated_scoring.py \
  --author-root /private/human-eval-b945 \
  --prepared-dir /private/humaneval-prepared \
  --package-root /private/scoring-client/lib/python3.12/site-packages \
  --python-runtime /usr/local/python3.12.13 \
  --work-root /private-accessible-parent/new-humaneval-scoring \
  --sandbox-uid 65534
```

由获授权操作员启动root入口；`--sandbox-uid`应使用专用、非0的低权限宿主UID，65534仅为已验证实例示例，不能盲目与其他任务共享。`work-root`必须不存在且不与输入/运行库重叠；它的父目录必须允许该UID遍历，不修改现有共享目录权限。`python-runtime`是包含`bin/python3`和`lib/`的专用基础Python前缀，不是整个`/usr`或`/`。挂载、网络、PID、capability或只读检查失败即停止；运行日志、全部题目、候选、测试结果保存在私有工作目录，公共回执只给任务ID、通过布尔值及分类结果。原作者测试超时仍为3秒、单worker；`--deadline`只控制外围总运行时间，默认600秒，不改变原题预算。此入口没有下载、没有模型调用，CI只运行无害软件夹具，不运行候选代码。

FLORES另已补充英语→德语devtest原3/4/5行NPU生成及原chrF++复核，累计三方向9次生成、6个不同平行行号；见[第三方向回执](validation/flores200-third-direction-native-audit-20261025.json)。这是新增方向验证，不要求把全部语言组合跑完才可交付；当前仍缺公共数据组件/研究消费和环境可复现收口。
本入口只验证安全的数据准备；尚未核验真实保存结果，也不证明功能测试完成。输出包含生成代码，保留私有，不提交Git。恢复服务器后先复用已有生成物，不重跑模型来替代缺失评分。

FLORES-200 原生入口代码（仅软件夹具验证，真实 NPU 运行待补）：

```bash
python scripts/run_flores200_original.py \
  --archive /path/to/flores200_dataset.tar.gz \
  --model-root /path/to/original-nllb-600m \
  --output /path/to/fresh-private-flores-results \
  --source-language eng_Latn --target-language fra_Latn \
  --split devtest --start 0 --count 3 --npu-device 0
```

只在服务器运行，事先确认研究线程释放该 NPU。资产身份沿用固定配置，不重新下载已校验资产。
按照[作者评测说明](https://github.com/facebookresearch/flores/blob/a6c830c6e1051fb4ac1a44b32358f00463f332bd/flores200/README.md)，使用主要指标
chrF++（sacrebleu CHRF(word_order=2)）；spBLEU 尚未集成，不宣称全部指标完成。
保留原始平行行号与空句，不换题；默认原始0/1/2行，`--full`为指定方向指定公开split全部原句。检查完整归档的语言数量与行数，但不把归档检查称为全部语言已推理。 模型使用固定
NLLB-600M 工件，NPU执行且无CPU/CUDA降级，不修改其生成配置，只指定目标语言token；完整输入超出模型容量直接失败。arb_Latn、min_Arab、sat_Olck
缺少对应模型token，拒绝执行而不替换文字体系。 模型限非商业研究使用，不能直接视为企业部署授权。输出含原句、参考译文和预测，留在服务器私有目录。需服务器安装并核实
transformers、torch/torch-npu、sentencepiece、sacrebleu 的实际兼容版本；环境准备终态、真实推理及评分尚未验证，来源不得打勾。

WildChat、UltraChat、LMSYS-Chat 使用同一准备入口，分别指定 `WILDCHAT`、`ULTRACHAT`、`LMSYS_CHAT`。 复用现有
historical_dialogue 适配定义，不新造第二套对话映射。原始 conversation/messages 的全部角色、轮次及顺序保留，包括历史 assistant
输出；它们不是新模型的生成结果，也不是可直接发送的完整 API 请求。 准备输出只是原始记录引用，消费端通过 resolve_input
恢复；需要模型运行的后续步骤必须明确历史前缀与待生成位置，不能将最后一条 assistant 当作用户问题或拿历史答案冒充新预测。

数据包中的 `vllm_hust_workload_data.conversations.historical_dialogue_views(record)` 提供显式历史续写视图：按原始顺序枚举全部
assistant 位置，若最后一轮为未回答的 user 则额外返回无参考答案的续写位置。 每次请求只含目标之前的完整原始历史，不含当前参考回答和未来轮次；前面轮次的 assistant
仍为原始历史回答，不能替换为新生成结果后宣称原协议未变。 保留原始行号、消息位置和非 API 注释，只有 role/content/name 进入请求视图。工具调用、非文本内容、非交替角色或中途
system 消息直接拒绝，不能静默跳过或自动转换。调用者需要另行选择适用的原生协议，不得用该视图覆盖不支持的记录。 这是语料历史续写，不是在线多轮 Agent、原始 API
抓包或官方质量评测；返回的历史回答不是质量金标准。

执行入口复用上述准备结果，不重新生成数据，也不重新下载：

```bash
python scripts/run_historical_dialogue_original.py WILDCHAT \
  --prepared /private/prepared-WILDCHAT \
  --data-root /private/WILDCHAT/pinned-revision \
  --output /private/fresh-WILDCHAT-original-continuation \
  --model workload-qwen35-27b-128k-nonthinking \
  --max-tokens 2048 --context-capacity 131072 --count 3
```

模型别名必须与现有本地 NPU 服务一致，容量必须与真实部署一致；命令不负责部署模型或判定设备归属。 服务器执行前先确认研究线程释放资源。仅连接本机 `/v1`
接口；先完整分词再生成，不截断、重试或换题。 默认按准备清单中的文件顺序取前三条原始会话，对每条执行全部原始续写位置；`--full` 为准备清单选定文件的全部会话，
不是整个来源所有分片。未准备的分片、其他协议不纳入通过范围。完整数据文件身份在执行前后复核，失败保留私有记录。 `length`
终止单独标为预算耗尽，不作为质量通过。原始历史、参考回答、预测只进入服务器私有输出；公开汇报只提取必要计数与身份摘要。
UltraChat已完成真实test_gen文件全28304条字段/索引核验，固定前三会话全部7个原assistant位置完成双NPU续写，全部stop； 精确44e24f46 CI
wheel安装载荷核对及首尾6条公共/研究回查通过。见[原生续写及消费证据](validation/ultrachat-original-continuation-20261026.json)。
这是历史语料续写，无质量评分，不是在线Agent或原始API捕获；不覆盖其他7个UltraChat分片、全量模型生成或另外两个来源。
WildChat现也完成原首分片59857条绑定/索引，原前三会话全部4处NPU续写及3条原字段公共/研究回查；3条stop、1条达到声明2048输出预算，原length响应保留，不称质量通过。
见[WildChat证据](validation/wildchat-original-continuation-20261026.json)。原日期类型在读取器保留，私有JSON快照沿用既有ISO可移植表示；日期/其他非API标注不作为模型参数。
其他13个WildChat分片、全量生成及LMSYS原生续写仍未验证。声明预算不是对原API调用参数的复原。

UltraChat可使用相同入口，明确选择原发布文件，不转换对话内容：

```bash
python scripts/prepare_reused_research_assets.py ULTRACHAT --mode download \
  --data-root /private/original-ultrachat \
  --files data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet \
  --max-bytes 100000000
python scripts/prepare_reused_source_records.py ULTRACHAT \
  --data-root /private/original-ultrachat --output /private/fresh-ultrachat-prepared \
  --files data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet
python scripts/run_historical_dialogue_original.py ULTRACHAT \
  --prepared /private/fresh-ultrachat-prepared \
  --data-root /private/original-ultrachat --output /private/fresh-ultrachat-continuation \
  --model workload-qwen35-27b-128k-nonthinking \
  --max-tokens 2048 --context-capacity 131072 --count 3
```

服务器已验证独立Python3.12.13/PyArrow23.0.1读取客户端和Qwen3.5-27B BF16双Ascend910B2服务。
获取默认使用发布方官方站；若网络不可达，允许显式`--hf-endpoint https://hf-mirror.com`，仍核验同一原文件版本/长度/摘要。
已有原文件先`--mode verify`，不要重复下载；本仓和CI均不附带原始数据。换模型需核验真实上下文和端点配置，不改原会话。

新增历史视图后的数据包已与研究仓现有十组消费夹具检查兼容：87通过、5跳过。 使用精确新 wheel 的 ZIP 导入，核对实际导入路径并确认研究仓依赖锁未改；5项正式安装验收跳过，不能算安装完成。
具体工件、研究提交及验证摘要见
[新版数据包兼容性回执](validation/research-dialogue-candidate-fixture-20261025.json)。这不代替服务器原始数据或原生执行验证。

本轮验证小型 JSONL 与嵌套 Parquet 软件夹具的绑定、准备和恢复，使用已安装的可选数据依赖。CI 安装数据包的 data extra
同样覆盖此格式测试。UltraChat和WildChat上述具名范围已有真实证据；其他分片与LMSYS原始文件/续写仍待验证，不能以格式夹具代替。

ToolACE原始语料准备与核验（仅在服务器执行）：

原始历史续写入口（原前三会话全部15处续写、固定工件公共/研究消费已在服务器验证）：

```bash
python scripts/run_toolace_original_continuation.py \
  --data-root /private/original-TOOLACE --tokenizer /private/existing-model-tokenizer \
  --output /private/fresh-toolace-original-continuation \
  --model YOUR_DEPLOYED_LOCAL_MODEL --context-capacity YOUR_VERIFIED_CONTEXT --count 3
```

默认固定原前三条会话、全部原assistant位置；每次保留完整原system及已有user/assistant/tool文本，目标及后续不入输入。
采用所选本地模型自己的`apply_chat_template`生成原始文本补全，不给历史tool返回伪造`tool_call_id`，不补造`tools`字段或可执行工具。
请求不是原始API捕获，也不是在线Agent；tool消息是既有历史数据，不随本次预测动态执行。512为具名续写预算，可显式配置；不是原API预算复原。
本地tokenizer只读、禁止下载/remote-code；逐次与真实服务token计数一致性及完整上下文检查，前后核对原文件与tokenizer身份。
绑定包括外部`chat_template.jinja`、词表与merges等实际分词文件，不只核对tokenizer配置JSON。
空输出、预算终止和模型错误原样保留；没有该训练语料专属的官方质量评分，不将BFCL成绩或工具学习模型论文结论套用到本接通测试。
[发布方模型用例](https://huggingface.co/Team-ACE/ToolACE-8B)展示ToolACE-8B的聊天模板函数生成；优先用合适的发布方模型，资源/权重不具备时可声明具名本地模型回退，不称论文或模型效果复现。
运行客户端需兼容该本地模型的Transformers/Tokenizer；这里只做CPU分词，实际模型生成必须由获授权NPU服务完成，不能在CPU或宿主执行工具。

已验证环境为Python3.12.13/Transformers5.5.4，实际Qwen3.5-27B
BF16双Ascend910B2，声明512输出预算、131072上下文；15条均stop，无空输出。
11300条原system/对话字段与索引均核验，首尾6处精确安装工件公共/研究回查一致；6个分词运行文件与既有可部署工件一致，15个原保存prompt重新渲染一致，未重复NPU生成。
见[交付证据](validation/toolace-original-continuation-20261026.json)。这不是全11300条模型推理、独立质量评分、ToolACE-8B或BFCL复现；研究默认依赖未升级。

新增入口的CI只运行CPU软件夹具、模拟传输/进程和数据包安装检查，不执行下述真实数据或模型命令。 历史对话、QReCC、MBPP、Mooncake与ToolACE测试在data
extra环境覆盖格式路径；源码目录之外的安装核验会拒绝editable/egg伪装。 CI工件仅含软件wheel及校验元数据，不包含原始数据、权重、预测或研究观测；不能以CI通过代替服务器原生运行。

ToolACE现有原始语料审计已迁成可复用入口，不再依赖作者本机目录或旧临时脚本：

```bash
python scripts/audit_toolace_original_corpus.py \
  --data-root /private/original-TOOLACE \
  --output /private/fresh-toolace-corpus-audit \
  --fetch-missing-card
```

复用已有`data.json`，不重新下载37MB原始语料。默认不联网；仅显式开启`--fetch-missing-card`才取得缺少的1873字节固定README，
已有错误文件或残留`.partial`拒绝覆盖。固定说明的长度、摘要、许可和仓库文件列表已核对；现已在服务器核验1873字节说明的副本。
原始37MB语料复用，不重新下载；服务器镜像说明请求403后仅转移已核对的官方说明文件，不转移原数据或凭据，历史失败保留。 固定库只有数据和说明，没有在线工具环境，模型效果另由BFCL等任务评价。
入口复用原适配、索引与resolve，全量核对原始system及所有conversations字段和顺序，固定首尾各三条回查。
已知角色与消息布局计数只用于后续渲染合同核验，不是研究特征挖掘；未知角色不转换，历史返回不执行，不补造API调用ID或函数实现。
可加`--provider-wheel /path/to/exact-data.whl`核对真实安装包；无该参数只标当前导入，不宣称固定工件安装通过；editable或egg源码元数据拒绝冒充安装。
这只是把已有数据准备能力公开化，不是新的NPU运行或原生Agent评分；来源仍不打勾。

MBPP固定发布资产准备命令：

```bash
python scripts/prepare_mbpp_original_assets.py --output /private/fresh-mbpp-assets
python scripts/prepare_mbpp_original_assets.py --output /private/fresh-mbpp-assets --resume
python scripts/prepare_mbpp_original_assets.py --output /private/fresh-mbpp-assets --verify-only
```

固定 Google Research 发布的 HF 版本 `4bb6404fdc6cacfda99d4ac4205087b89d32030c`，完整保留 full/sanitized 两个配置各自
train/test/validation/prompt 及 README，9文件360549字节；不下载模型或运行代码。
[作者协议](https://github.com/google-research/google-research/blob/f46ca8374b4cddef97ca4208ad986049d74d296a/mbpp/README.md)
指定原任务ID 11–510用于测试、1–10作为提示候选，论文使用2/3/4作三示例。不能把 few-shot 任务并入测试，不能把 full的500题与sanitized的257题混报，
也不能因预测失败改换版本或任务。HF字段与GitHub原JSONL的逐项保真仍待服务器核验。 原始 full 测试集的生成入口如下，客户端需数据读取依赖
PyArrow，模型使用已确认空闲的本机NPU服务：

```bash
python scripts/run_mbpp_original_generation.py \
  --assets /private/fresh-mbpp-assets \
  --output /private/fresh-mbpp-generation \
  --model workload-qwen35-27b-128k-nonthinking \
  --max-tokens 4096 --context-capacity 131072 --count 3
```

这是作者README三示例文本模板的具名实现：原2/3/4号题完整解答作示例，每个示例与目标题都含三个原测试，块间两个换行； 目标参考代码不入提示。使用原始
completions，不套聊天模板、不删除Markdown围栏或修补语法，以原 `[DONE]` 标记停止。
温度0、具名输出预算用于接通验证，不声称复现论文模型、抽样参数或成绩。默认原test前三题，`--start`/`--count`为固定原顺序范围，
`--full`为完整500题；sanitized变体目前仅准备资产，不能宣称其原生验证完成。
完整提示分词超容量直接失败，不截断；空输出、语法错误、预算耗尽的原始结果保留，不换题凑通过。预测与完整原任务仅保存在服务器私有目录。
这里只生成，不执行代码或计分；功能评分仍必须在受验证的独立沙箱中，不能在宿主或推理服务环境直接exec预测。

准备后续沙箱评分的私有输入，不重新生成、不抽取代码围栏或修补语法：

```bash
python scripts/prepare_mbpp_scoring_inputs.py \
  --assets /private/fresh-mbpp-assets \
  --generation /private/fresh-mbpp-generation \
  --output /private/fresh-mbpp-scoring-inputs
```

核验原始题号/行号、资产版本、完整请求、参考代码、终态与保存结果；空输出和length终止仍保留。 输出只留在服务器私有目录，包含完整原任务与预测，不能提交到公开仓。准备成功不等于执行测试成功。
固定的后续执行器元数据见 `configs/workload-native/mbpp-scoring-pins.json`：复用BigCode CodeEval执行/计分函数，
不复用其不同的InCoder提示；原始三测试和setup保持独立，增强challenge测试不混入主结果。
独立隔离入口如下，已在服务器固定前三题验证真实Linux命名空间及功能评分；仍不能在宿主设置代码执行授权后直接调用：

```bash
python scripts/run_mbpp_isolated_scoring.py \
  --assets /private/fresh-mbpp-assets \
  --generation /private/fresh-mbpp-generation \
  --prepared-dir /private/fresh-mbpp-scoring-inputs \
  --author-root /private/bigcode-evaluation-harness \
  --package-root /private/mbpp-client310/lib/python3.10/site-packages \
  --python-runtime /opt/mbpp-python310-standalone \
  --work-root /private/fresh-mbpp-isolated-score \
  --sandbox-uid "$MBPP_APPROVED_SANDBOX_UID" --deadline 600
```

`author-root`需检出固定执行器提交`8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd`；
使用独立Python3.10运行时，评分依赖固定numpy1.26.4、tqdm4.70.1，启动器的数据读取环境另需PyArrow。
已验证服务器aarch64/Python3.10.22、numpy1.26.4、tqdm4.70.1；读取客户端为独立Python3.12.13/PyArrow23.0.1。
模型服务为Qwen3.5-27B/bfloat16、Ascend910B2双卡，输出预算4096、上下文131072；不改服务全局Torch/NPU环境。
独立评分运行时必须位于沙箱UID可遍历的专用前缀，例如`/opt/mbpp-python310-standalone`；不要放在私有0700父目录下，
也不要放在`/data`、`/root`、`/proc`下，否则绑定该路径会与隔离检查冲突。不得通过放宽原私有目录权限或取消检查绕过。
`MBPP_APPROVED_SANDBOX_UID`需操作者设置为专用、非零且无其他任务使用的主机UID；启动器需操作者授权root，
降权后进入独立user/mount/net/pid命名空间与chroot，只读输入和运行时、无NPU设备、无外网、能力清空及no_new_privs检查通过后才导入执行器。
隔离设置复用现有HumanEval命名空间代码，不改变其原有评分路径；这是有界本地隔离，不是安全认证。
原参考解先按原setup与三个测试自检，未通过直接保留失败，不改测试。随后原预测按每题3秒、单worker、pass@1计分。
500题完整范围外围deadline至少4060秒，示例600秒只适合固定小样本，不缩短题内时间。
无隔离条件不回退，异常保留目录和日志，不自动清理或重试。来源完成仍需真实NPU生成、独立评分及固定工件消费闭环。 替换为EvalPlus增强测试需另列协议，不得当作原三测试同配置。
当前原题11/12/13生成、参考解自检及CodeEval评分已完成，预测2通过/1失败；原结果保留，不重试改答案。
见[原生抽样及消费回执](validation/mbpp-original-native-sample-20261026.json)。统一full配置适配、精确CI安装工件及研究消费均已核验，来源按full原生抽样协议勾选。
四个full分片共974条原记录全量映射/建索引，14个跨分片位置公共与研究回查一致；研究默认依赖未升级，不声称全500题模型评价。
完整full配置可从已准备资产的`data/`根目录读取四个原Parquet；输入仅含`text`和三个`test_list`，
参考`code`、`test_setup_code`、`challenge_test_list`分离到outcome；题号单独保留。sanitized格式不走该适配，不能静默当full读取。

InfiniteBench 作者代码身份与静态前检命令：

```bash
git clone https://github.com/OpenBMB/InfiniteBench.git /path/to/official-infinitebench
git -C /path/to/official-infinitebench checkout --detach 51d9b37b0f1790ead936df2243abbf7f0420e439
python scripts/verify_infinitebench_original_code.py --official-checkout /path/to/official-infinitebench
```

后续服务器原生验证入口（当前只有软件夹具测试，尚未完成真实环境验证）：

```bash
python scripts/run_infinitebench_original.py \
  --official-checkout /path/to/official-infinitebench \
  --data-root /path/to/original-infinitebench \
  --output /path/to/fresh-private-results \
  --task passkey --model YOUR_LOCAL_MODEL --template gpt4 \
  --context-limit VERIFIED_SERVICE_CONTEXT_LIMIT --start 0 --count 3
```

这是作者 gpt4 提示模板加本地模型传输的显式配置，不是 GPT-4 论文复现。直接调用固定作者版本 create_msgs（tokenizer=None，禁用其中的截断分支）、get_answer
和 get_score，不重写评分规则。 通过服务 `/tokenize`
对完整消息计数；完整输入加作者生成预算超限则失败，不改题、不截断、不缩预算。默认固定原始位置0/1/2；`--full`执行所选任务全部原题，不能与其他范围组合。
OpenAI客户端禁用自动重试，失败保留已写结果，不换题凑通过。输出私有目录中包含答案和预测，禁止提交；即使抽样与评分成功，也不自动将整个来源打勾。 需要服务器隔离环境安装作者
requirements及其实际导入依赖、OpenAI SDK和数据包，准备固定ROUGE指标资产，并记录版本；本入口尚无已验证的完整依赖锁，因此不能据此宣称开箱即用交付完成。

上述克隆及后续依赖安装只在服务器执行。固定配置为
`configs/workload-native/infinitebench-author-code-pins.json`，核对六份作者代码/说明文件的 Git blob 及字节数，不导入作者模块。
前检从作者代码静态读取各任务文件名与生成预算；不是依赖安装成功、原生推理或评分成功证据。作者评分模块导入时会加载 ROUGE，须在服务器另行固定指标资产；不能用静态前检替代评分验证。 作者历史
GPT-4 消息构造分支包含中间截断；本轮不以截断适配当前模型。不满足完整上下文要求时保留资源阻塞。历史数字型/字符串型字段映射亦不能直接覆盖全部子任务及评分。

NarrativeQA 另提供数据关系解析接口
`vllm_hust_workload_data.narrativeqa.original_summary_tasks(data_root, catalog_root=None)`。 它校验原始
documents.csv、qaps.csv、third_party/wikipedia/summaries.csv，按作者 QA 顺序关联 document_id 并核对 split。 返回的
input 仅含原始问题、document_id 和原始摘要；两份答案留在 outcome，三表原始行留在 original_relations 供核查，后者不得整体发送给模型。
此接口只支持明确标识的作者摘要设置，不代表长篇原文设置；缺失原文不能以摘要替换后宣称原文任务完成。 重复文档 ID、关联缺失、split
不一致或文件身份变化均拒绝，不自动改题或联网补料。目前仅通过小型软件夹具验证，尚未完成真实数据及原生执行验证。

以下入口复用已审查的原始字段映射，只准备可解析的原始记录引用；不执行模型、评分、观测或研究特征提取。原始数据留在服务器外部目录，不进入 Git。

安装本仓 `workload-preparation` 数据包后，在仓库根目录运行：

```bash
python scripts/prepare_reused_source_records.py LONGALPACA \
  --data-root /path/to/original-assets --output /path/to/fresh-preparation --limit 3
python scripts/prepare_reused_source_records.py NATURALQUESTIONS \
  --data-root /path/to/original-nq-open --output /path/to/fresh-nq-preparation --limit 3
python scripts/prepare_reused_source_records.py DUREADER \
  --data-root /path/to/original-dureader-demo --output /path/to/fresh-dureader-preparation --limit 3
```

文件路径须与登记的原始资产相对路径一致；先使用 `prepare_reused_research_assets.py` 获取或校验资产。输出目录必须全新且与数据目录分离。省略 `--limit`
会准备全部记录引用；绑定阶段始终校验全部选定文件的身份和字段，不因 limit 跳过文件校验。

LongAlpaca 保留 instruction/input 与独立参考 output；NQ-open 保留 question 与独立 answer 列表。NQ-open
是开放域单轮问答，不具有原始检索轨迹或多轮关系，因此不沿用历史映射中的 rag-followup 标签。三份映射均不带研究 feature 选择器。

DuReader 当前范围仅为作者 demo，不等于完整数据集。保留嵌套文档标题及段落；答案、答案位置、相关文档标注分别留在 outcome，不进入模型输入。未构造多轮关系，因此归为文档问答而不是
rag-followup。

本轮仅通过小型软件夹具验证绑定、准备、消费恢复、答案隔离、文件篡改拒绝及输出防覆盖；尚未在恢复后的服务器核验真实资产。此入口不是原生 benchmark
执行入口，不增加交付打勾数，原生执行及适用评分仍是独立待补项。

## StructEval 已保存非渲染结果的复用与官方重评分

历史回执已记录19个非渲染协议、共950道原题；这些结果应复用，而非重新请求模型。
`scripts/audit_structeval_saved_nonrender.py`把历史私有复核流程整理为可复用CPU入口：

```bash
python scripts/audit_structeval_saved_nonrender.py \
  --data-file /private/structeval/data/train-00000-of-00001.parquet \
  --official-archive /private/structeval-litellm-official.zip \
  --run-dir /private/completed-run-1 /private/completed-run-2 \
  --work-root /private/fresh-structeval-rescore
```

沿用原StructEval评分客户端环境，依赖PyArrow及已安装的固定官方StructEval源码；
入口核对原始2035条Parquet摘要、固定源码zip和已安装全部27个Python模块，再导入评分函数。
每个run须保留receipt.json、original-selected.private.json及official-workflow内的
inference.json、evaluation.json，并声明原始预算未覆盖、原生流程已结束。输入、输出、 原始位置/ID及字段逐项对应后，调用未修改的官方非渲染评分器和最终分数计算；
重新评分必须与保存结果一致。只输出计数及文件摘要，格式文件和详细日志留在私有目录。

重复或重叠run、尚未结束的run、渲染/VQA任务、原始字段/输出变化及评分不一致均拒绝； 不改题、不换失败任务、不调用模型。新入口目前只有软件夹具验证，未据此重报950题为新增实跑。
剩余1085道渲染/VQA及最终原生交付、研究消费仍待完成，StructEval来源总勾不变。

## ScienceAgentBench verified：复用官方评测，不替换科学任务

当前已取得102条verified元数据及固定源码的历史证据，不代表已取得完整科学数据。 官方固定版本`c26e151ed601ba109dc4d35e057ff8e73fec469d`的
`evaluation.harness.run_evaluation`已支持`--split verified`和`--dataset_name`； 无需为选择verified
split另写评测器。`run_infer.py`和旧`run_eval.py`是不同入口， 不能把它们的validation默认值误当成Docker harness不支持verified。

准备好兼容环境、私有原始科学资产和原生Agent输出后，官方评测入口形态为：

```bash
python -m evaluation.harness.run_evaluation \
  --dataset_name /private/pinned-ScienceAgentBench-dataset \
  --split verified \
  --benchmark_path /private/benchmark-verified \
  --pred_program_path /private/original-agent-programs \
  --log_fname /private/fresh-evaluation.jsonl \
  --run_id unique-private-run \
  --cache_level base --max_workers 4 --force_rebuild False
```

以上仅展示官方参数，不是已验证环境或可直接照抄的完成记录。dataset目录须事先取得并
固定到登记的`9c6e96c9e74572e979b0930ee735041cef528cb7`版本，并确认本地HF加载确实
返回verified原102条。上例省略instance_ids，表示官方全量流程；抽样验证时须追加
`--instance_ids`及固定原行对应的真实ID，不因失败改选，不能假设行号等于instance_id。 预测程序必须由完整原生Agent流程产生，不能拿参考程序冒充模型输出。
任务科学数据、gold和评测脚本只进私有执行环境，不进Agent输入。

还须处理以下真实缺口，未解决前不执行、不打勾：

- 按[固定官方README](https://github.com/OSU-NLP-Group/ScienceAgentBench/blob/c26e151ed601ba109dc4d35e057ff8e73fec469d/README.md)
  取得`benchmark_verified.zip`，核验解包后的科学数据/gold/评测器版本；该发布方明确要求 不在线重新分发解包资产，因此公共仓只提供索引、获取说明及执行代码。
- 旧`run_infer.py`硬编码validation；其模型工厂把非gpt/non-o1名称路由至Bedrock，
  并依赖LiteLLM模型成本/上下文登记。不能直接填Qwen名称、伪装成GPT或借别的模型tokenizer绕过。
  优先采用官方README推荐的OpenHands集成，核验其固定版本是否支持verified与本地NPU服务； 否则必须显式记录适配，保持原题、工具反馈、预算及评分，不把旧裸宿主代码执行作为后备。
- 官方科学评分涉及CodeBERT及适用任务的视觉裁判；未验证其模型、NPU路径和依赖， 不得改成字符串相似度或省略裁判来凑“跑通”。官方Python科学程序/CPU指标本身与LLM推理区分。
- Docker评测会构建任务环境、安装任务依赖，并按cache_level清理镜像；应在独立执行根、 新run_id及明确空间预算下使用，不共用其他任务目录。官方硬件下的参考耗时不是本服务器实测。

本轮只核实固定官方源码参数和模型路由，不下载科学数据、不调用模型、不启动容器； 102条原字段准备、真正Agent执行、科学评分与研究消费的证明仍分开保留。
