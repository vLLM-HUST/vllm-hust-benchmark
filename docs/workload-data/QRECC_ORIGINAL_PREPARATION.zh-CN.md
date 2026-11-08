# QReCC 原始数据准备与原生检索问答运行

当前状态（2026-11-03）：完整原passages包、56003成员CRC、54573064原行及Java空行规则均已核验，实际完整Lucene索引为54241550文档；早期“下载/构建中”记录不再是当前缺口。固定原0/1/2完整模型RAG、原检索/QA评分和6处公共/未修改研究回读均已通过，全16451原记录及其参考/关系保留。声明的本地LLM协议已交付，不代表全test推理、论文训练模型复现或性能验收。[运行与勾选清单](DATASET_CHECKLIST.zh-CN.md#qrecc%E5%8E%9F%E6%A3%80%E7%B4%A2%E9%9B%86%E5%90%88%E8%8E%B7%E5%8F%962026-11-03)，[最终回执](validation/qrecc-original-native-delivery-20261103.json)。

## 来源与任务范围

官方来源：[apple-aiml-research/ml-qrecc](https://github.com/apple-aiml-research/ml-qrecc)，固定提交
`1d174464290790c0198b474ad022952ecd83b76b`，原成员 `dataset/qrecc_data.zip`。 归档 SHA-256
`6ed52a0b0e495f752424b15742119a5f8f33ca7a029c07122be1b849de19a308`；test 成员 `qrecc_test.json`
有16451条、17685054字节，SHA-256
`a4d8ef686827feb00c90997e0d6ff135928bf6f4bcb26fb523b5fac025d58f84`。train原成员有63501条，本次准备接口不输出train。

发布方分别列出问题改写、段落检索、阅读理解子任务。`Context` 是发布的会话历史，不是已检索证据库；仅有任务JSON不能宣称原版检索问答执行通过。固定问题和历史必须原样保留，不使用
Rewrite/Answer/Answer_URL 作为待预测任务的输入。

发布方 README 声明数据为 CC BY-SA 3.0，代码为 Apache-2.0，两者不混同。本仓仅提供元数据、准备代码和使用入口，不公开复制原始数据或答案。使用者自行按许可取得归档。

## 已验证的运行与评分环境

本地LLM执行示例使用Python3.12.13、Java11（实测Temurin11.0.32.1+1）和Pyserini0.10.0.1发布包内的原Anserini0.10
JAR；不要求其Python检索包装器。JAR路径由`--jar`显式给定，执行入口核验其固定SHA256，不能改用较新0.44或SCAI0.12的JAR。索引须保留原`storePositions/storeDocvectors/storeRaw`，公开入口只读搜索，不重建或缩小集合。

原评分客户端独立安装，避免把旧nltk依赖写进模型服务环境：

```bash
python -m venv /data/qrecc-scorer-client
/data/qrecc-scorer-client/bin/python -m pip install \
  setuptools==75.8.0 wheel==0.45.1 packaging==24.2
/data/qrecc-scorer-client/bin/python -m pip install --no-build-isolation \
  -r configs/workload-native/qrecc-original-scorer-client.txt
```

模型实跑绑定为Qwen3-14B固定权重版本、BF16、单NPU、131072上下文、YaRN4（原32768）、eager、并发1、batch-token上限4096、显存预算0.9；每个模型请求固定非思考和温度0，改写128/回答512输出上限。原服务额外保留已获授权的cached-token报告，不据此报告性能。完整版本、模型revision和真实样本结果见[最终回执](validation/qrecc-original-native-delivery-20261103.json)；软件包标签中的`+cpu`不是LLM执行设备，本次LLM明确在Ascend
NPU，Java检索和非神经评分在CPU。

该环境用于公开入口的功能验证，不是作者训练模型复现或引擎性能验收；换模型、预算或检索参数应单列配置，不与此样本成绩混报。

## 准备和读取

安装本批根包和数据组件后：

```sh
workload-data prepare-qrecc-test --data-archive /data/qrecc_data.zip --output /data/qrecc-original-test
```

输出目录必须不存在；准备前校验原始归档与成员哈希，原样复制test成员，不构造题目、不改写JSON、不执行模型、检索或评分。

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

records = iter_workload("QRECC", data_root=Path("/data/qrecc-original-test"), limit=3)
for record in records:
    inputs = record.task["input"]  # 原始 Context、Question
    historical_references = record.outcome_record()  # 不传入模型
```

通用 `workload-data pipeline` 可按同一目录准备来源引用，并用 `resolve_input` 回查原行；准备数据库不包含正文和答案。Context
是历史数据，不伪装为实时工具执行或真实到达时间。

## 当前证据与缺项

### 作者已发布的完整段落集合（优先获取路线）

作者[Zenodo发布5760304](https://zenodo.org/records/5760304)，版本2021-12-06，提供完整`passages.zip`，27512656777字节，发布方MD5为`16c48fb9c4b9603b8ccc8ba3bd7d69c1`。<!-- pragma: allowlist secret - public publisher MD5 checksum -->
资源地址：`https://zenodo.org/api/records/5760304/files/passages.zip/content`。作者[随包说明](https://zenodo.org/api/records/5760304/files/README.txt/content)明确这些段落由原`paragraph_chunker.py`流程产生；不必把重抓Common
Crawl数十TB作为唯一获取路线。

公共准备入口默认只显示计划。取得完整原包后可只读校验其发布方MD5和完整成员清单，再显式解压整个集合；不抽取小候选库冒充完整检索空间。

```bash
python scripts/prepare_qrecc_original_collection.py
python scripts/prepare_qrecc_original_collection.py --action verify \
  --archive /external/passages.zip
python scripts/prepare_qrecc_original_collection.py --action extract \
  --archive /external/passages.zip --output /external/qrecc-prepared \
  --max-uncompressed-bytes 86557816928
```

原成员及层级保存在`/external/qrecc-prepared/collection/`；回执与其分开，索引时只传`collection/`，不把准备回执读成检索文档。解压前核验56007个成员及完整容量，拒绝路径穿越、重复成员、链接和非JSONL；逐成员解压检查ZIP
CRC，开始前及每个文件写入前检查64GiB磁盘余量，其他任务占用空间时停止并保留失败现场。已有输出拒绝覆盖，不自动删除或重做。`verify`只校验整包身份和目录，不冒称已解压或逐成员CRC通过。16项人工ZIP控制在Windows及服务器Linux均通过，服务器公共plan命令exit0，回执为`qrecc-public-collection-entry-20261103-002/qualification-public.json`。真实完整原包仍在获取，公共入口的实际全库验证尚未完成。

已在服务器保存并检查完整原ZIP目录：56007个成员，含56003个JSONL和4个目录，原文件解压合计86557816928字节，最大单成员271773680字节；这是空间计划，尚非整包MD5或成员内容校验。原包现以8MiB/s上限通过内存管道写入服务器，至少保留64GiB共享盘余量，本地不落原包；本服务器直连失败记录保留，不能把它写成发布方资产不存在。原包完整校验、解压、全库索引、原生检索/阅读与评分仍待完成。

该版本另提供`qrecc-test.json`，19546617字节，发布方MD5为`7ad29296a1dfd2f75dafd653a8822269`。<!-- pragma: allowlist secret - public publisher MD5 checksum -->
服务器已核验两个原包身份并逐行比较全部16451条：会话/话轮身份及顺序完全一致，每条新增`Passages`参考字段；16427条全部原字段相同，另外24条仅`Answer_URL`不同。问题、会话历史、改写参考和答案未变，但两个文件不是可互换的同一版本。原文件分别保留，没有删除不同字段或覆盖现有Git版。
比较回执为服务器任务目录`qrecc-author-test-release-comparison-20261103-001/qualification-public.json`；该核验没有模型调用、检索或评分。新增段落标签只能作参考，不能作为模型检索结果或输入；后续使用发布版检索参考时须明确其版本及24条URL差异，不能静默混用。

### 从原网页重建集合（官方备选流程，尚未执行）

固定版本的[collection说明](https://github.com/apple-aiml-research/ml-qrecc/blob/1d174464290790c0198b474ad022952ecd83b76b/collection/README.md)提供以下顺序：取得Common
Crawl网页和Wayback网页、按原脚本切成段落，再用Pyserini建立Lucene索引。官方明确提示Common
Crawl下载涉及数十TB量级；这不是已经取得的任务JSON大小，也不能把网页集合容量直接等同最终索引大小。开始前必须分别核算下载流量、原网页、段落副本和索引的峰值空间。当前没有批准或开始这批大规模下载。

以下为作者给出的命令结构，应在仓库外的固定官方源码副本内按作者相对路径执行；保留完整原集合，不为抽样评测缩小候选语料。后述固定Java入口已完成独立软件控制，但完整Pyserini
Python包装环境与QReCC实际语料尚未验证。

```bash
# 官方 collection/ 目录；--workers 是作者示例，不是本机已核定配置。
python download_commoncrawl_passages.py --output-directory collection/commoncrawl --workers 8
python download_wayback_passages.py --inputs '../dataset/*.json' --output-directory collection/wayback --workers 4
python paragraph_chunker.py --input-directory collection --output-directory collection-paragraph --workers 8
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
  -threads 76 -input collection-paragraph -index index-paragraph \
  -storePositions -storeDocvectors -storeRaw
```

上述流程之后还缺原问题改写、原集合检索、阅读生成及检索/答案两类评分的完整运行入口。必须保留问题和原对话历史，原`Rewrite`只作改写参考，原`Answer`只作答案参考，不用参考字段代替预测。抽样只固定待评任务，不能将参考URL页面拼成小候选库后宣称端到端QReCC验证。官方的段落切分/索引流程与历史发布集合是否可逐项复现，也须在取得资产后核验，不能仅凭相同脚本名称假定相同语料。

### 已核验的原始锁版本索引环境

原Apple仓库固定提交的[requirements.txt](https://github.com/apple-aiml-research/ml-qrecc/blob/1d174464290790c0198b474ad022952ecd83b76b/requirements.txt)明确锁定`pyserini==0.10.0.1`，同时声明nltk3.5、beautifulsoup4
4.9.3与warcio1.7.4。当前原始协议采用该Pyserini版本，不用较新版本替换后冒称原环境。服务器Linux
aarch64已取得并按发布方SHA-256核验以下软件包，复用独立Java11，不修改系统环境或现有模型服务。

| 软件                                   | 原发布包                                                                                                                                                   |    字节数 | 发布方SHA-256                                                      |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------: | ------------------------------------------------------------------ |
| Pyserini 0.10.0.1                      | [PyPI原包](https://files.pythonhosted.org/packages/19/5e/11247de00294454a0c3c6abb187e808d7f87f401dbaa753bf75675b2c108/pyserini-0.10.0.1.tar.gz)            |  63065880 | `67ded994108530d17db74512460347cadd84d0f9ef8ffcbf0f6459f925117abd` |
| Temurin JDK 11.0.32.1+1，Linux aarch64 | [Adoptium原包](https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.32.1%2B1/OpenJDK11U-jdk_aarch64_linux_hotspot_11.0.32.1_1.tar.gz) | 192356586 | `f27033e6f7523c1b0b2565a78e9c0e0abe5596a854ce00ca04ec1b06ece7a935` |

原包包含`pyserini/resources/jars/anserini-0.10.0-fatjar.jar`，本次解包实测SHA-256为`b2f40e9dc56091364814e4146515d34a7a05af2be2a1cba5c88e3d25a50276f6`；该内部摘要不是另一个发布方声明。未修改Java索引/查询入口已完成3个人工文档的建库、1条BM25查询及顶一结果核验，k1=0.82、b=0.68；没有执行QReCC原题或神经模型。当前只证明原锁版本的Java入口可用，完整Python包装依赖、全库索引及原生RAG仍未完成。回执：`qrecc-original-pinned-index-runtime-20261103-001/qualification-public.json`。

11份原作者软件和依赖声明共53913字节，固定身份见`configs/workload-native/qrecc-author-source.json`；不含数据集或权重。公共源码准备与只读核验入口：

```bash
python scripts/prepare_jsonschemabench_author_source.py --source qrecc \
  --author-root /external/qrecc-author --action prepare --max-bytes 53913
python scripts/prepare_jsonschemabench_author_source.py --source qrecc \
  --author-root /external/qrecc-author --action verify
```

在仓库外解包并校验上述软件后，Java入口的命令形态如下。`/external/qrecc-passages`必须是完整校验的作者集合，不能填验证样本目录；实际全库构建前仍须核算索引与临时空间、内存及共享资源窗口。

```bash
/external/jdk-11.0.32.1+1/bin/java -Xmx4g \
  -cp /external/pyserini-0.10.0.1/pyserini/resources/jars/anserini-0.10.0-fatjar.jar \
  io.anserini.index.IndexCollection \
  -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
  -threads 2 -input /external/qrecc-passages -index /external/qrecc-index \
  -storePositions -storeDocvectors -storeRaw
```

`-threads 2`与`-Xmx4g`是保守的待全库验证配置，不是作者的76线程实测配置，也未证明足以完成全库。尚未执行上述QReCC建库命令、Python包装器或原生RAG。必须核查原日志和实际索引/排名文件，不能只凭进程退出0判断成功。

原锁版本的`io.anserini.index.IndexUtils`主类探针实查返回ClassNotFound，不能以此断言已有索引损坏，也不以新版本工具读取原索引后混报。独立只读计数入口使用本仓固定、非模型生成的Java源码和原JAR，通过Lucene
`DirectoryReader`读取实际索引文档数，不创建或重建索引：

```bash
python scripts/probe_qrecc_original_index.py \
  --index /external/qrecc-index --java /external/jdk-11.0.32.1+1/bin/java \
  --jar /external/pyserini-0.10.0.1/pyserini/resources/jars/anserini-0.10.0-fatjar.jar \
  --output /private/fresh-qrecc-index-probe --execute
```

不加`--execute`只显示计划。上例只返回实际文档数；完整集合的真实条数另经核验后，可传`--expected-documents`进行精确比较，不能把论文约5400万直接当成精确值。默认Java堆上限1024MiB，可用`--heap-mib`显式配置；这不证明全库内存需求。输出在索引外的私有新目录，失败保留日志，计数相符也不自动证明原集合来源、检索/阅读执行或任务质量。

7项人工子进程控制通过；服务器公共plan/execute实际复用原锁版本的3文档软件索引，读回3，16个原索引文件哈希均未改变，没有重新建库或模型调用。回执：`qrecc-public-index-counter-control-20261103-001/qualification-public.json`；这不是QReCC全库索引验证。

历史Pyserini0.44/Java21软件对照及错误CLI参数日志保留在`qrecc-original-index-runtime-20261103-001`，不将其索引与原始锁版本混用。另核查到[竞赛组织方SCAI基线](https://github.com/webis-de/SCAI-QReCC/tree/8020bbe28127ca6af5fe99534908b4b33f331731/code/simple-baseline)锁定Pyserini0.12.0，属于不同软件协议；其测试集/评分修订也不能静默替换本方案的固定Apple任务。该组织方简单基线的回答阶段是启发式句子选择，不能拿它冒充NPU阅读模型或原端到端神经基线。

### 已保存检索结果的作者函数评分入口

`scripts/score_qrecc_original_retrieval.py`使用原11份软件身份校验及未修改的`evaluate_retrieval.py`、`evaluate_qa.py`和`span_heuristic.py`。独立Python3.12客户端固定nltk3.5，附加依赖见`configs/workload-native/qrecc-original-scorer-client.txt`，不是根包默认依赖，不安装神经框架。安装示例：

```bash
python3.12 -m venv /external/qrecc-scorer
/external/qrecc-scorer/bin/python -m pip install setuptools==75.8.0 wheel==0.45.1 packaging==24.2
/external/qrecc-scorer/bin/python -m pip install --no-build-isolation \
  -r configs/workload-native/qrecc-original-scorer-client.txt
python scripts/score_qrecc_original_retrieval.py \
  --data-file /data/qrecc-original-test/qrecc_test.json \
  --author-root /external/qrecc-author \
  --rankings-file /private/qrecc-original-rankings.jsonl \
  --output /private/fresh-qrecc-retrieval-score \
  --python /external/qrecc-scorer/bin/python --workers 1 --execute
```

不加`--execute`只显示计划，不读取任务或运行评分。保存的rankings每题一行JSON，顶层只含`original_row_index`、`Conversation_no`、`Turn_no`、`retrieved_passages`；每个真实命中只含`docid`、`content`、`rank`、`score`，保留原检索顺序、唯一文档ID及从1开始连续排名，不接受参考答案字段。所有实际返回排名均保留；Recall@100是指标截点，不是将MRR输入截断到100条或拒绝更多命中的理由。参考答案仅从已核验原test加入独立评分文件，不回传给检索或阅读模型。

该入口调用原作者函数API，不宣称原CLI未经包装。原CLI只从命中行建立查询分组，无法表达零命中查询；适配层按全部保存任务预建分组，零命中仍参与分母，原MRR/Recall函数直接返回该查询的零贡献，不丢题、不补造段落。原作者的抽取上界函数对空列表未定义，存在零命中时上界记为`null`，不伪造值；无零命中时其结果也由原函数计算。抽取上界是参考答案驱动的诊断值，不是阅读模型输出或质量成绩。

15项人工保存结果/子进程控制通过，含完整保留第101位命中的检查；服务器原nltk3.5、原作者函数、4个正例/次位命中/空答案/零命中人工查询控制通过，零命中保留。另复用先前原CLI的相同3查询输入和结果，对照5项指标完全一致，没有重新执行该原CLI。回执：`qrecc-public-retrieval-score-control-20261103-002/qualification-public.json`。首次依赖探针误用了被其他模块覆盖的`nltk.util`属性，失败日志保留，改用原源码相同的`from nltk.util import ngrams`后通过；没有修改nltk或作者代码。以上均不是QReCC原题、真实检索或模型验证，公共完整`--execute`原题链路仍待运行。

### 已保存答案的原作者评分入口（不是端到端运行）

固定版本的 `utils/evaluate_qa.py` 是函数库，没有命令行主程序。新增入口直接调用其原始 `compute_exact` 和 `compute_f1`，不重写指标；要求该文件
SHA-256 为 `f8fe4c1c163004953791983fe141363566b741851f42d7f3fa473646ed672165`。

```bash
python scripts/score_qrecc_original_qa.py \
  --data-file /data/qrecc-original-test/qrecc_test.json \
  --scorer-file /path/to/pinned-ml-qrecc/utils/evaluate_qa.py \
  --predictions-file /private/qrecc-original-predictions.json \
  --output /private/fresh-qrecc-qa-score
```

预测文件是 JSON 数组，每项只能含 `original_row_index`、`Conversation_no`、`Turn_no`、`prediction`； 行号为原 test
中从零开始的位置，其他任务身份必须与原行一致，预测为模型原始文本，不能填入原参考答案充数。 原题在运行前固定抽样，评分入口要求至少三项、原始顺序、无重复，不丢弃错误答案；空预测允许按作者指标计分。
该工具不验证预测生成过程，因此不能凭评分成功宣称原始检索或模型执行完成。原题、作者函数及预测文件身份在评分前后校验；已有输出目录拒绝覆盖。
输出仅含身份、哈希和指标，不复制正文与参考答案。原始检索资产不足仍是端到端阻塞，不使用 Context 或答案URL临时拼接替代检索库。
当前封装与准备回归11项通过；另核验原作者固定函数的三个人工评分夹具（完全匹配、部分匹配、空答案），均不是原始QReCC任务验证。

### 原有数据准备与运行证据

- 原始归档已由官方 Git blob API 取得，并复制至授权服务器，双方归档SHA一致。
- 本地真实test前三行已完成原样读取、参考字段隔离、准备索引及输入回查；没有模型调用。
- 单元测试覆盖未知归档拒绝、成员原样复制、其他成员不输出和已有输出目录拒绝覆盖；这些是人工夹具，不是原始任务执行。
- 服务器独立 Python 3.12 环境已安装候选数据wheel（SHA-256
  `43832634c5678f10a9a9a8e65ec47b091209d8466f8f4f645d3e2281d276bc6d`），`-I`
  确认site-packages导入；原归档准备、原test前三条读取、pipeline索引和resolve_input逐条一致，参考字段保持分离。首次私有核验脚本定位归档目录错误，0模型调用，纠正脚本路径后通过，未放宽数据身份检查。
- 发布方固定README说明端到端任务使用包含1000万网页、5400万 passages
  的集合，并分别提供检索和阅读评估脚本：[`evaluate_retrieval.py`](https://github.com/apple-aiml-research/ml-qrecc/blob/1d174464290790c0198b474ad022952ecd83b76b/utils/evaluate_retrieval.py)、[`evaluate_qa.py`](https://github.com/apple-aiml-research/ml-qrecc/blob/1d174464290790c0198b474ad022952ecd83b76b/utils/evaluate_qa.py)。当前取得的16451行test成员不是该检索集合；仅凭Context、Rewrite或答案字段不能宣称检索/端到端通过。
- 原始检索集合尚未取得，检索运行、阅读链路和发布方评分均未执行，模型调用0次。现有本地与服务器读取/准备回执可复用；不得为填空改用另一套语料或用QReCC的Context伪造检索结果。
- 早期能力曾由叠加草稿PR #426 承载；该编号是历史开发记录，不用于判断当前交付或PR状态。当前进度只读取唯一机器清单的`delivery_inventory.QRECC`，本批统一在PR
  #544维护，不另开单数据集PR。原归档准备、安装读取和三条消费记录可复用，端到端检索与阅读缺口保持未完成。
