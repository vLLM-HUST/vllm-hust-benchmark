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
