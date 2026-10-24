# Natural Instructions 原始任务输入（下一批候选）

当前登记五任务交付已收口，见[来源级回执](validation/ni-native-delivery-20261024.json)：分类102题、翻译250题、抽取261题、改写1178题全部原题覆盖，摘要只执行原首尾六题，共1797唯一实例。改写[1178题去重审计](validation/ni-rewriting-complete-audit-20261024.json)及[旧352题完整任务复核](validation/ni-reused-small-tasks-audit-20261024.json)均用原collator重建完整请求、原scorer重算分数并核验保存响应，无新答案生成。此状态覆盖下文阶段性“待补”表述；不是整个NI套件、全部8284实例或底层数据集原协议的全量结论。当前模型/服务固定配置见[双NPU环境](BBH_ORIGINAL_INPUTS.zh-CN.md#%E6%9C%AC%E8%BD%AE%E5%B7%B2%E9%AA%8C%E8%AF%81%E7%9A%84%E5%8F%8Cnpu%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E9%85%8D%E7%BD%AE)。

task1453实体抽取已完成[全261题去重覆盖审计](validation/ni-extraction-complete-audit-20261024.json)：复用原0–2与258–260，仅新增3–257共255题，所有原题号恰好覆盖一次。三个批次的完整prompt经原collator重建、响应包SHA及终态核对、原评分器重算均一致；审计0新增生成调用，261次tokenizer请求只作原拼接器长度校验。分批分数保留，不以题号覆盖宣称全NI或跨环境统一准确率。

## 2026-10-24固定工件复核

新增通用入口`scripts/run_ni_original_range.py`复用摘要所固定的旧核心及改写/抽取任务配置，支持五个已登记文件的`--start/--count`原题连续区间和`--full`完整文件。必须显式选择`--task-files`，评分目录必填；核心collator/scorer、完整输入、1024生成预算不变。示例：

```sh
python scripts/run_ni_original_range.py --task-files task1290_xsum_summarization.json --start 6490 --count 3 --data-root /data/ni-original-tasks --collator-source /data/upstream/ni_collator.py --scorer-root /data/ni-original-scorer --output /private/ni-xsum-tail
```

同一命令把区间参数换成`--full`即可选择该原文件全量，不与区间混用。执行前在输出目录旁独占创建`*.selection.private.json`冻结原题号；完成后`range-receipt.json`核验实际题号和核心回执摘要。核心历史回执的旧selection标签可能仍为first_three，不能据此判断本适配入口范围；以冻结选择、实际samples和range回执共同核对，核心回执字节不重写。原题、响应及完整题号只留私有目录。

新增入口已完成三个任务各末三条共9条原题的NPU生成与评分，见[区间原生审计](validation/ni-native-range-audit-20261024.json)。使用原collator重建九条完整prompt，均与实际请求一致；原scorer重新计算指标，均与保存分数一致。审计没有新增生成调用，仅有九次模型tokenizer请求用于原拼接器长度校验。抽取task1453按本轮实测成本适合全量，正在补原中间3–257共255条，与历史0–2和本轮258–260合成完整题号覆盖，不为凑全量重跑首尾六条；该批终态和评分未收口前仍不勾选。

[本轮复核回执](validation/ni-fixed-ci-audit-20261024.json)覆盖已登记五个原任务文件（另含改写task275及实体抽取task1453），共8284实例：完整文件SHA、每行原Definition/input及output参考隔离一致；固定CI数据组件逐文件原前三条prepare→resolve输入和参考一致。已保存的两组原生回执共15条交接样本，其原始实例ID、请求prompt摘要、响应包摘要、usage和终态逐条核对，原评分保留，没有新增模型调用或重算分数。这15条不是累计原生覆盖总数；355条主任务与6条改写/抽取的历史覆盖仍按原回执分别引用。

当前公开入口仍有明确缺口：主入口默认前三条，`--full-small`只覆盖不超过250实例的选中任务；摘要、改写和实体抽取不能通过通用原题区间或全量入口运行。后续应补齐事前选择的原题区间及全任务文件选择，保留固定上游collator/scorer、不缩短预算、不重跑已有ID；旧脚本身份与历史回执须保留，不能改写成新入口实跑证据。来源本轮暂不打勾，也不把五个登记任务说成整个NI发布集。

固定来源：`allenai/natural-instructions@55a365637381ce7f3748fa2eac7aef1a113bbb82`。本次读取的是发布方
instruction-task 变体，不是自行构造的 family，也不宣称与底层原版 Amazon、PAWS-X 或 XSum 评测等价。

| 原始任务文件                                                | 类别     | 实例数 | 当前验证                                        |
| ----------------------------------------------------------- | -------- | ------ | ----------------------------------------------- |
| task1575_amazon_reviews_multi_sentiment_classification.json | 情感分类 | 102    | [x] 全102条生成、终止、原版评分；前三条索引回查 |
| task778_pawsx_english_french_translation.json               | 英法翻译 | 250    | [x] 全250条生成、终止、原版评分；前三条索引回查 |
| task1290_xsum_summarization.json                            | 摘要     | 6493   | [x] 原前三条抽样闭环；[ ] 全量                  |

每个文件按源目录固定 SHA。使用者自行按原许可取得原文件，存入仓库外。NI README 明确指令和实例许可不同；每个任务的 `Instance License` 原样保留为来源注解，不以仓库
Apache-2.0 自动覆盖实例权利。task1575 指向 Amazon 条款，task778 指向 PAWS 专用许可，task1290 元数据写
MIT；后者不是对全部底层新闻内容权利的独立保证。原文、实例和参考不公开提交。许可 Unknown 的 task1355 不纳入本批文件目录。

## 输入与索引

从原 JSON 的 `Instances` 逐行读取。`root_fields` 显式带入同文件的 Definition、Categories、语言和 Instance
License；发现字段碰撞或缺失立即失败，不猜测、不合成、不混不同任务文件。

`record.task.input` 只有原 `Definition` 和实例 `input`；原 `output` 单独放入
`outcome_record()`。Positive/Negative Examples 不自动加入输入，不生成 prompt 模板或模型事件。拼接格式、示例选择和正式模型执行协议须另行明确。

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

records = iter_workload(
    "NATURAL_INSTRUCTIONS", data_root=Path("/data/ni-original-tasks"),
    files=["task778_pawsx_english_french_translation.json"], limit=3,
)
```

通用 pipeline 支持同样的 files/limit 显式选择，冻结的 adapter 中包含 records_path/root_fields；resolve_input 校验原文件 SHA
后用同一规则恢复输入。来源索引不保存正文或参考答案。

## 实际验证与未完成项

提交前精确注释三个公开数据SHA以消除秘密扫描误报后，最终脚本SHA-256为`4c46f9ea6e862d91771049424221e0c69b67eefbc610f64e25fb82dd6a3e87ae`。已在服务器新私有目录再次执行同九条，全部stop，输入哈希及官方指标与前次一致，回执SHA-256为`7b383e34f58448fb0621343852f8f20e0b6aa370e267c9520c72bed569d8fc83`。前次5639脚本原字节另行保留；各次交接复跑仍不增加355个唯一实例计数。

可复用执行入口为
`scripts/run_natural_instructions_original.py`，单独在下述Python3.9/Transformers4.18环境运行；它不导入需要Python>=3.10的数据准备包，不将模型部署到CPU。原版上游代码和数据由使用者放在仓库外，参数显式指定：

```sh
python scripts/run_natural_instructions_original.py --data-root /data/ni-original-tasks --collator-source /data/upstream/Tk-Instruct/src/ni_collator.py --scorer-root /data/ni-original-scorer --output /data/private/ni-run-001
```

`scorer-root`布局为原版NI的`evaluation.py`、固定Tk-Instruct的`rouge/`模块，以及`gpt2/vocab.json`、`gpt2/merges.txt`、`gpt2/tokenizer_config.json`。后者只含`{"tokenizer_class":"GPT2Tokenizer","model_max_length":100000}`，前两文件为下述OpenAI原始资源的原字节。准备该目录时保留原源码版权和许可；脚本逐项核验代码/资源SHA，评分目录不包含模型权重。隔离环境额外依赖absl-py2.3.1、nltk3.9.2、six1.17.0，tokenizers0.12.1，已验证的numpy为2.0.2；不是完整训练依赖安装要求。

默认三个任务各前三条；`--task-files`显式选任务，`--full-small`只允许实例数不超过250的已选文件，不会悄悄把摘要截成小集。输出目录必须全新并在数据目录外。只允许本机无凭据endpoint；原实例或资源校验失败即停止，tokenizer要求截断则拒绝。原请求和回答只写私有目录；CI只运行入口契约夹具，不下载数据、不加载模型。

该公开代码入口已在服务器同一环境干净输出目录复跑同九条，全部stop并原版评分结果与下表一致，不算新增九个唯一实例。脚本SHA-256为`5639b3fc40fd186f607c7c41f9602ea166e41db77ccffdb5e25bd79e5ca0b6ed`，交接回执SHA-256为`bf55f9d70beecf08a9cad9126f885dcdb83438a6f675785aa71fd3e1bcdce57d`。

本地三个文件各前三条读取、原 Definition 保持、参考字段隔离及索引回查通过。服务器独立 Python 3.12 环境实际安装候选 wheel，`-I` 确认从 site-packages
导入，三个原文件各三条的准备/索引/输入和参考回查全部一致。候选wheel
SHA-256：`674d74ac20edc85c6b8f32a8bde8c33e2e62776f39ab75a6d1643067a854457b`。这只是软件和实际数据读取验证；0模型调用、0评分，不是公开CI工件。

原版 `eval/automatic/evaluation.py` 提供 normalized exact match 和最大参考 ROUGE-L，并在跨语言路径使用 GPT2
tokenizer。模块即使评分英文也初始化该tokenizer；该依赖最初缺失，后按下述原始资源和隔离环境补齐，未替换为Qwen
tokenizer或修改评分函数。两个小任务的完整文件已跑完，摘要和整个NI仍非全量。

研究端还需在回查中转发冻结的 root_fields，并明确消费 Definition/input；当前没有该批研究端验收回执。不改变首批 #379 的固定发布工件，不为单任务开PR。

### 原版输入拼接预检

已固定发布方链接的 `yizhongw/Tk-Instruct@7c0e4000901e7c148fbb8964042ebfc433f8d98e` 中
`src/ni_collator.py`，文件SHA-256
`d170d2230bba4a1f250ee5f3ecf16131f0e334855602115d95e4d16befdb7a29`，未修改原文件。新版Transformers环境首次导入因原代码依赖的Optional导出缺失失败，0模型调用；随后独立Python3.9环境按发布方固定Transformers4.18.0、tokenizers0.12.1安装，仅用于原版数据拼接，不安装训练/DeepSpeed或CPU模型。

原始DataCollatorForNI的instruction-only/text_only配置（Definition开启、任务名/正负示例/随机混合关闭）已对三个任务各前三条预检通过。tokenizer接口调用现有本地Qwen服务`/tokenize`返回实际token
IDs，只作长度校验；不是GPT2评分tokenizer，也不是模型生成。原拼接器的strip及标点规则保留，原任务文件不改；逐条证明没有触发其截断分支。非chat prompt
token计数分别为分类102/135/127、翻译61/70/63、摘要719/637/246。

此预检只证明官方输入拼接器在限定预处理环境可运行，不证明完整Tk-Instruct训练环境、论文模型或官方评分复现；不将9次tokenize算作9次模型任务。

### 三类原始任务的真实NPU生成

预检后在现有双910B2、Qwen3.5-27B/BF16/TP2/graph/非思考服务上执行同九条原始实例，temperature=0、max_tokens=1024。将未修改的instruction-only官方collator字符串放入模型原生chat传输；这是替代模型接入，不是Tk-Instruct论文模型成绩。原collator自带strip/补标点规则保留，不改原任务文件、不裁剪输入、不将labels放入模型消息。

| 任务             | chat输入token（三条） | 输出token（三条） | 终止   |
| ---------------- | --------------------- | ----------------- | ------ |
| task1575情感分类 | 113 / 146 / 138       | 2 / 2 / 2         | 全stop |
| task778英法翻译  | 72 / 81 / 74          | 22 / 32 / 28      | 全stop |
| task1290摘要     | 730 / 648 / 257       | 64 / 38 / 40      | 全stop |

九次真实生成完成响应完整性和原文件SHA不变检查。私有回执SHA-256为`55db75f62afffb5f1390a3a941c4276cc48629d642b3feb9a6c2696aadcb9087`；原文、参考和模型回答仍只留服务器私有目录。原版评分函数随后对同九条结果补评分，不将各三条写成全量完成。

### 原版评分函数补验（0新增模型调用）

加载固定NI源码的`evaluation.py`（SHA-256
`aac89279abb16c8e9acfaac90ad612f51d5ca27ae75259cb9524276b5dd7dbb3`）及固定Tk-Instruct的原始rouge模块，源码未改。直接调用其compute_metrics，传入原始参考列表和实际模型回答；不做答案提取、改写或选择更优候选。英文分类/摘要使用default模式，英法翻译显式使用xlingual模式。不是完整论文训练复现，也不是底层XSum/PAWS-X原版评测。

GPT2原始encoder.json和vocab.bpe来自OpenAI公开端点，SHA分别为`196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783`、`1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5`，与`openai/tiktoken@4e71bbe0c078468e00fefbf94b39849389f346e5`所列资源hash一致。只在隔离工作目录将原字节放到GPT2Tokenizer标准文件名，并用本地类选择配置加载，评分器的`from_pretrained("gpt2")`不改；离线模式、无模型权重。Qwen
tokenizer只用于前述生成输入长度，不用于评分。

| NI原任务     | 样本数 | normalized EM（%） | ROUGE-L（0–100） | 模式          |
| ------------ | ------ | ------------------ | ---------------- | ------------- |
| task1575分类 | 3      | 100                | 100              | default       |
| task778翻译  | 3      | 0                  | 65.9355          | xlingual/GPT2 |
| task1290摘要 | 3      | 0                  | 34.5394          | default       |

ROUGE-L不是正确率，翻译/摘要的EM=0不等于执行失败。上述三条抽样不产生全量或性能提升结论。评分回执SHA-256
`502f8de9f62f034e1aae2e06a37023ae47eef3b41ddbb272398b327d06205635`；评分未新增模型请求。隔离环境依赖检查通过，已记录并补齐环境暴露的decorator缺项；没有放宽原函数检查。

### 两个小任务完整文件验证

另启动可恢复的原始实例队列，按文件原顺序完成分类102/102、翻译250/250，所有请求均正常stop、输入未截断、原文件SHA保持。校验每个原始索引和ID唯一且无缺失、私有回答哈希匹配后，调用同一未修改官方评分函数：分类normalized
EM/ROUGE-L均92.1569，翻译EM=0、xlingual ROUGE-L=63.193。结果保留答错，不进行重采样或替换。

原始全量生成回执SHA-256 `170159f3d3874382f1363931c085530af3b932d8eaaac3b58f4284bb00cdb12b`，全量评分回执SHA-256
`44176e6b22191e18cfbe3fef3400db48ed7e8f8cbd83b41120b9c8d2e55cd2d8`。评分0新增模型调用；初始及交接复跑不重复计唯一实例，本次三个任务合计355个唯一原始实例（102+250+3）。仅两个NI任务文件的全量，不是全NI或底层数据集全量。见[脱敏汇总](validation/natural-instructions-original-npu-20261023.json)。研究端本批消费仍未闭环，不能因此勾整个交付批次完成。
