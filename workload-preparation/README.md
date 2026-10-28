# vLLM-HUST Workload Data

## 按来源找到准备与执行入口

分类导航（只查元数据，不下载数据或宣称可运行）：

```bash
python -m vllm_hust_benchmark.workload_data classes --scheme delivery-plan
python -m vllm_hust_benchmark.workload_data class-plan W11 --scheme delivery-plan
python -m vllm_hust_benchmark.workload_data delivery-assets
python -m vllm_hust_benchmark.workload_data source-plan BURSTGPT
```

`delivery-plan`覆盖本次40类任务及118个登记来源；其中15项作为服务轨迹、受控探针、 跨任务关系或企业私有请求资产单列，不硬造语义任务分类。企业仅提供索引和边界，
不包含原始请求。类下来源是候选关联，不表示该来源全部记录属于该类，也不表示已适配、 取得数据或原生跑通。研究默认38类入口保持不变；当前交付状态以机器清单为准。

`source-plan SOURCE_ID`同时返回已索引的准备、生成、原生profile、隔离评分或数据读取入口，
按role区分。路径相对于benchmark仓库根目录，执行脚本不会随纯数据wheel打包；先按相应 文档准备外部原始资产和独立环境，再查看脚本帮助。数据读取入口不是原生benchmark执行器。
索引包含尚未验证的实现，空列表也不表示已经审计为“没有实现”；完成状态仍只看机器清单， 不能从脚本存在推断原生验证通过。

检索/重排新增入口：`BEIR_SCIFACT`、`BEIR_NFCORPUS`的原zip准备、原文档/查询/qrels读取，以及`scripts/run_beir_original.py`的全量test检索和dense
top100重排，见[原生协议与环境](../docs/workload-data/PUBLIC_ASSET_SERVER_READINESS.zh-CN.md#beir-%E5%8E%9F%E5%A7%8B%E6%A3%80%E7%B4%A2%E9%9B%86%E5%90%88%E4%B8%8E%E9%87%8D%E6%8E%92%E5%8D%8F%E8%AE%AE)。原集合不是chat请求，qrels只作评分参考。原全量NPU运行、最终公开入口交接和固定CI工件/研究消费均已验证，范围限定为上述两集合的原test检索及dense
top100重排。

| 场景                   | 入口与验证范围                                                                                                                                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 长文档理解             | [QuALITY 原始问题](../docs/workload-data/QUALITY_ORIGINAL_QUESTION_INPUTS.zh-CN.md)：问题级读取、完整文章、MC 执行与评分；不是发布方训练基线复现                                                                                                                   |
| 结构化输出             | [JSONSchemaBench 原始约束](../docs/workload-data/JSONSCHEMABENCH_ORIGINAL_SCHEMA_INPUTS.zh-CN.md)：不改 schema；三条生成结束，原约束符合仅1/3，失败保留                                                                                                            |
| 独立推理               | [MATH500 原始任务](../docs/workload-data/MATH500_ORIGINAL_TASK.zh-CN.md)：500条固定 held-out 问题按来源 ID 读取；problem 独立作为输入，answer/solution 保持在 outcome；三区分测试样本、H4 Inspect 配方与 PRM800K 确定性评分器                                      |
| 检索跟进、文本任务     | [QReCC 准备](../docs/workload-data/QRECC_ORIGINAL_PREPARATION.zh-CN.md)、[原始 NI](../docs/workload-data/NATURAL_INSTRUCTIONS_ORIGINAL_INPUTS.zh-CN.md)、[编辑与实体抽取](../docs/workload-data/NI_EDITING_EXTRACTION_ORIGINAL.zh-CN.md)：分来源区分准备与实际评分 |
| 到达、尺寸、匿名块标识 | [Azure、Mooncake、BurstGPT 原始轨迹](../docs/workload-data/ORIGINAL_SERVING_TRACES.zh-CN.md)：仅元数据，无原始 prompt，不生成伪回放或推导实测 KV 命中                                                                                                              |

代码类新增入口见[BigCodeBench 原始生成阶段](../docs/workload-data/BIGCODEBENCH_ORIGINAL_GENERATION.zh-CN.md)： 三条原题
NPU 生成与公开入口复跑完成，功能执行/评分未完成。
代码、科学、图表与历史来源的[服务器准备记录](../docs/workload-data/PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)
仅证明七源21条读取/准备/回查，不能代替其原生任务执行。

[LoCoMo原始长会话问答](../docs/workload-data/LOCOMO_ORIGINAL_INPUTS.zh-CN.md)保留完整会话和参考隔离，原生完整上下文文本QA已验证108个唯一原题，含显式缺键兼容；不是全1986题或RAG/图像任务完成，须遵守CC-BY-NC-4.0条款。

[独立QASPER原发布版准备](../docs/workload-data/PUBLIC_ASSET_SERVER_READINESS.zh-CN.md)保留三split共5049个原问题、完整论文与全部答案/证据标注；提供归档提取、读取和索引，原生模型及答案/证据评分尚未完成，不能与LongBench派生200题混计。

上表是使用导航，不代表全来源或全类别完成。精确已验证范围见各来源说明、
[类别执行清单](../docs/workload-data/WORKLOAD_EXECUTION_CHECKLIST.zh-CN.md)和对应 PR 的固定工件记录。

## 其他已接入来源与边界

OASST1 原始消息树的显式历史续接视图见[使用与验证边界](../docs/workload-data/OASST1_ORIGINAL_CONTINUATION.zh-CN.md)。
它保留原始父子关系、排除目标回复；不是实时交互环境。通用索引可显式选择 `history_targets`；对应固定工件的研究消费回执见类别清单，不能把历史续接等同实时会话。

BFCL v4
simple_python的原始JSONL读取和独立静态评分入口见[接入与验证](../docs/workload-data/BFCL_V4_SIMPLE_PYTHON_ORIGINAL.zh-CN.md)。
显式格式贯通原行回查；评分参考不进入任务输入，原三题NPU/官方AST评分不代表真实工具执行或其他类别。

LongBench v1 Qasper原文准备和执行见[使用与验证](../docs/workload-data/LONGBENCH_V1_QASPER_ORIGINAL.zh-CN.md)。
`workload-data prepare-longbench-qasper`只复制并核验原始归档成员，任务输入与评分参考分离；仅Qasper原三题完成NPU/官方评分，不代表其他子集或全量。

AgentBench FC
ALFWorld原生引用入口见[接入与验证](../docs/workload-data/AGENTBENCH_ALFWORLD_ORIGINAL_PREPARATION.zh-CN.md)。`workload-data agentbench-native-tasks`只核验官方索引/游戏字节并输出元数据引用；Python>=3.10数据组件和独立Python
3.9原生任务环境分离，不把闭环任务退化为静态请求，也不迁入科研观测。

数据来源目录、原始文件读取适配、可校验输入引用和显式请求回放入口。独立发行包 `vllm-hust-workload-data==0.1.0`，导入名
`vllm_hust_workload_data`；不覆盖科研仓库的 `llm_serving_workloads`。

不包含观测钩子、采集、研究特征计算、机制干预或 QBI
历史执行入口。目录登记、读取适配与模型执行/官方评分是不同状态，见[逐项验证清单](../docs/workload-data/DATASET_CHECKLIST.zh-CN.md)。

## 安装与来源查询

从本仓库根目录运行：

```sh
python -m pip install -e .
python -m pip install -e "./workload-preparation[data]"
workload-data identity
workload-data sources
workload-data classes
python -m vllm_hust_workload_data.catalog source GSM8K
python -m vllm_hust_workload_data.catalog download-plan GSM8K
```

数据由使用者按发布方条款单独取得，放在仓库外。此组件不分发企业载荷、gated 数据、模型或第三方环境。下载计划是来源信息，不保证已经下载或具备执行器。

## 固定本地文件与准备输入引用

```sh
python -m vllm_hust_workload_data.pipeline bind GSM8K --data-root /data/gsm8k --files test.parquet --revision ORIGINAL_REVISION --output /data/bindings/gsm8k.json
```

编辑配置（路径相对于配置文件；输出目录必须不存在）：

```json
{"sources":[{"source_id":"GSM8K","data_root":"/data/gsm8k","manifest":"/data/bindings/gsm8k.json","limit":3}]}
```

```sh
python -m vllm_hust_workload_data.pipeline plan --config /data/input.json
python -m vllm_hust_workload_data.pipeline run --config /data/input.json --output /data/prepared/run-001
```

输出 `workload.sqlite` 与
`manifest.json`，仅保存可回查输入引用、来源/适配器身份和准备状态，不保存原始请求、答案、事件或研究特征。默认失败即停止；`--allow-partial`
才允许明确标注的部分结果。文件 SHA 校验只固定本地字节，不证明发布方认证。

## 下游直接读取

```python
from pathlib import Path
from vllm_hust_workload_data.pipeline import prepared_rows
from vllm_hust_workload_data.consumer import resolve_input

database = Path("/data/prepared/run-001/workload.sqlite")
for row in prepared_rows(database, source_id="GSM8K"):
    record = resolve_input(database, row["record_id"], data_root=Path("/data/gsm8k"))
    original_input = record.input_record()
    # 交给相应任务的原版执行器；不要把参考答案拼进输入。
```

静态任务不是完整 Agent episode；任务执行器负责原生环境、工具、终止和评分。AgentBench、GAIA、tau2 当前明确要求官方执行器适配，不回退到 QBI
派生任务。真实请求回放仅接受原始请求体；显式 `--model` 只绑定服务端模型并记录差异，不修改消息、预算和流式标志。流式响应必须收到
`[DONE]`。`python -m vllm_hust_workload_data.execution --help` 查看入口，单个 HTTP 响应不代表任务评分通过。

科研消费者保留自己的观测、特征和分析代码，只从本组件读取原始输入。参考[消费契约](docs/consumer_contract.md)、[迁移计划](../docs/workload-data/MIGRATION_PLAN.zh-CN.md)。

## 软件检查

```sh
python -m pip install -e "./workload-preparation[dev,data]"
python -m pytest -q workload-preparation/tests tests/test_workload_data_boundary.py tests/test_workload_gap_catalog.py tests/test_workload_validation_checklist.py
```

上述测试是 CPU 软件正确性检查，不加载模型，不证明真实数据集完整执行或性能提升。

独立知识问答任务的原版候选似然执行示例见 [AI2-ARC Challenge/Easy 抽样复现](../docs/workload-data/ARC_CHALLENGE.zh-CN.md)，可用
`--suite` 选择子集，与聊天生成协议分开。

通用单轮指令语料的原始字段生成入口见
[Dolly 接入说明](../docs/workload-data/DOLLY_ORIGINAL_INSTRUCTIONS.zh-CN.md)：原instruction/context不改，参考response隔离，提示格式与模型替换显式声明；只验证生成完整性，不冒充官方质量评分。

长输出运行维度见
[LongWriter-6k 原消息接入](../docs/workload-data/LONGWRITER_ORIGINAL_MESSAGES.zh-CN.md)：保留原始user消息，隔离最后assistant参考；不要将训练语料生成接入称为LongBench-Write官方评分或长输出能力验收。

## LiveCodeBench 原始资产准备（原生运行尚未交付）

复用统一准备入口，不在仓库内下载数据。三个原始数据仓库分别固定 revision；代码仓库 revision 不替代数据 revision。先打印离线计划：

```bash
python scripts/prepare_reused_research_assets.py LIVECODEBENCH
```

只获取代码执行及测试输出预测的两个完整原文件（合计144,814字节）：

```bash
python scripts/prepare_reused_research_assets.py LIVECODEBENCH \
  --mode download --data-root /private/livecodebench-originals \
  --files execution-v2/data/test-00000-of-00001.parquet test_generation/test.parquet
```

已有上述目录时，将 `--mode download` 换成 `--mode verify`，即可按相同固定身份只读复核，不重复下载。显式镜像传输可加
`--hf-endpoint https://hf-mirror.com`，仍核验原始 SHA256。下载失败保留现场；仅同一目录、同一固定 profile 可加 `--resume`
继续，不能把失败文件当成功。

完整代码生成原文件为9,375,644,586字节，默认预算明确拒绝；需要充足服务器空间并显式选择：

```bash
python scripts/prepare_reused_research_assets.py LIVECODEBENCH \
  --mode download --data-root /private/livecodebench-original-code \
  --files code_generation/test.jsonl --max-bytes 10000000000
```

完整原文件校验后，可用固定原文件顺序抽样准备原题（默认先打印计划，增加 `--execute` 才写入外部新目录）：

```bash
python scripts/prepare_livecodebench_original_tasks.py \
  --data-root /private/livecodebench-original-code \
  --output /private/livecodebench-original-tasks --sample-count 3 --execute
```

每条原题的所有字段、完整测试字符串和元信息均保留，不替换失败题、不减少每题测试。该准备入口不解码私有测试、不构造提示、不运行模型、不执行生成代码或评分。私有测试可能包含
pickle，只能在合格隔离容器中交给官方加载流程，不能在宿主机反序列化。数据卡仅标注通用
`cc`，具体许可待核实，不公开再分发原文件。原生代码生成、自修复、代码执行、测试输出预测的执行环境、预算和评分闭环尚未完成；不得把下载/读取成功作为 benchmark
跑通。进度以[唯一清单](../docs/workload-data/DATASET_CHECKLIST.zh-CN.md)为准。

### LiveCodeBench 代码生成原生链路

以下是**代码生成单场景**的入口，不代表自修复、代码执行、测试输出预测也已交付。实际验证状态见[准备与运行回执](../docs/workload-data/validation/livecodebench-original-readiness-20261028.json)。所有路径均为服务器数据盘路径示例，CPU隔离评分节点和NPU推理节点可分开；不要在本地下载原始数据或模型。

在CPU节点下载固定作者源码压缩包至新的外部路径；工作进程会核验固定SHA256，不接受任意源码版本：

```bash
test ! -e /private/lcb-source.tar.gz && curl --fail --location \
  --output /private/lcb-source.tar.gz \
  https://codeload.github.com/LiveCodeBench/LiveCodeBench/tar.gz/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
```

构建固定Python3.10.22/aarch64的CPU API客户端与原生评分环境，只将两份公开环境文件放入构建上下文，不放数据、模型或整个仓库：

```bash
LCB_CONTEXT=$(mktemp -d)
cp configs/workload-native/livecodebench-api-cpu.Dockerfile \
  configs/workload-native/livecodebench-api-cpu.lock.txt "$LCB_CONTEXT/"
DOCKER_BUILDKIT=0 docker build \
  -f "$LCB_CONTEXT/livecodebench-api-cpu.Dockerfile" \
  -t workload-native/livecodebench-api-cpu "$LCB_CONTEXT"
LCB_IMAGE=$(docker image inspect --format '{{.Id}}' workload-native/livecodebench-api-cpu)
```

这是API/评分专用环境，不声称安装上游全部CUDA/vLLM/云供应商依赖；模型由独立NPU推理服务提供。Anthropic包是官方提示模块的直接导入依赖，不代表调用外部服务。

将前文准备的原题交给隔离容器生成官方generic API提示，保持实际模型标识，不冒用GPT名称。此步骤不请求模型：

```bash
python scripts/run_livecodebench_container_scoring.py \
  --prepared-dir /private/livecodebench-original-tasks \
  --author-archive /private/lcb-source.tar.gz --image "$LCB_IMAGE" \
  --work-root /private/lcb-prompts --model Qwen/Qwen3.5-9B \
  --prepare-prompts --execute
```

将 `/private/lcb-prompts/native-output/results/requests-private.json`
通过服务器间传输放到NPU节点；协调好设备使用窗口、核验服务身份与空闲后，在该节点运行：

```bash
python scripts/run_livecodebench_original_generation.py \
  --prompts /private/lcb-requests.json --output /private/lcb-generation \
  --endpoint http://127.0.0.1:19466/v1 --disable-thinking --timeout 1200 --execute
```

每题保留官方10个候选、2000生成token、temperature0.2和top_p0.95。`--disable-thinking`
是本次Qwen非思考配置；不适用其他模型时不要盲目沿用。默认HTTP等待90秒，本次因通信等待超时显式设置1200秒，**不是提高生成预算或官方评分时限**。首次90秒失败证据保留，不替换原题；任何请求失败停止，不隐藏重试、不换题凑通过。全部请求和原始响应保存在外部输出目录，生成代码此时不执行。

将完整 `/private/lcb-generation` 目录传回CPU节点，按保存响应执行官方提取器与评分器，不重新推理：

```bash
python scripts/run_livecodebench_container_scoring.py \
  --prepared-dir /private/livecodebench-original-tasks \
  --generations-dir /private/lcb-generation \
  --author-archive /private/lcb-source.tar.gz --image "$LCB_IMAGE" \
  --work-root /private/lcb-codegen-scoring --model Qwen/Qwen3.5-9B --execute
```

工作进程会重建官方原始提示并核对保存请求、原题ID、候选索引和完整测试绑定；原CLI的12个评分worker和每测试6秒设置不变。生成代码与私有测试解码只在无网络、非root、只读根文件系统容器内运行。容器使用32GiB内存上限及独立输入只读/输出可写卷；仅新建输出卷的所有权初始化使用最小CHOWN/FOWNER权限，不在宿主机执行生成代码。需有授权的Linux
Docker节点和足够内存；不满足时标阻塞，不能退回宿主机执行。评分不要求模型答对；原始失败与截断输出保留，不能将抽样成绩解释为论文复现、全量质量或引擎性能。

### LiveCodeBench 自修复：真实反馈与显式兼容模式

自修复承接上述官方评分输出，不重新生成初始候选、不人工编写错误反馈。作者流程会保留原本已通过的候选，只对其余候选各生成一个修复版本（`n=1`），随后交给原评分器重新评分。

固定上游版本有一处已实测兼容问题：评分器的一类`error_code=-4`反馈只有`error_message`，但自修复模板强制读取缺失的`inputs/expected/error`，导致`KeyError`。默认`strict`仍保留上游行为。经明确允许，可以选择`--repair-feedback-compat error-message-only`：只对这些缺字段异常直接展示原有`error_message`，不添加输入、期望输出或其他虚构字段，不改原题、完整测试、生成预算或评分器。该配置必须标为**显式兼容模板**，不能宣称未经修改的官方自修复模板已通过。

在CPU隔离节点，使用真实初始评分文件准备修复提示：

```bash
python scripts/run_livecodebench_container_scoring.py \
  --prepared-dir /private/livecodebench-original-tasks \
  --prior-evaluation /private/lcb-codegen-scoring/native-output/results/original-eval-all-private.json \
  --author-archive /private/lcb-source.tar.gz --image "$LCB_IMAGE" \
  --work-root /private/lcb-repair-prompts --model Qwen/Qwen3.5-9B \
  --scenario selfrepair --repair-feedback-compat error-message-only --prepare-prompts --execute
```

准备回执给出`request_count`、原题ID、原候选索引、复用已通过候选数和原反馈文件哈希。原始反馈JSON不会重写。将准备出的`requests-private.json`在服务器间传到NPU节点后，先协调并核验独占使用窗口，再执行：

```bash
python scripts/run_livecodebench_original_generation.py \
  --prompts /private/lcb-repair-requests.json --output /private/lcb-repair-generation \
  --endpoint http://127.0.0.1:19466/v1 --scenario selfrepair \
  --disable-thinking --timeout 1200 --execute
```

每个请求保留原题ID和原候选索引，辅助`request_id`仅用于区分同一道题的不同失败候选，不冒充新的原始题目。本次初始3题的30候选中，7个已通过候选不再请求模型，23个失败候选使用真实反馈生成修复。若准备回执的请求数为0，则不调用模型，评分时可省略`--generations-dir`并沿用全部原输出。

将完整修复输出目录传回CPU节点，用同一原反馈、同一兼容模式核验提示与候选关系，并按作者流程组合修复结果及原已通过输出，再调用原评分器：

```bash
python scripts/run_livecodebench_container_scoring.py \
  --prepared-dir /private/livecodebench-original-tasks \
  --prior-evaluation /private/lcb-codegen-scoring/native-output/results/original-eval-all-private.json \
  --generations-dir /private/lcb-repair-generation \
  --author-archive /private/lcb-source.tar.gz --image "$LCB_IMAGE" \
  --work-root /private/lcb-repair-scoring --model Qwen/Qwen3.5-9B \
  --scenario selfrepair --repair-feedback-compat error-message-only --execute
```

这里的接口、夹具测试和准备成功不等于实跑完成；修复推理、后续评分与保存审计的实际状态以[唯一清单](../docs/workload-data/DATASET_CHECKLIST.zh-CN.md)和[分阶段回执](../docs/workload-data/validation/livecodebench-original-readiness-20261028.json)为准。

### LiveCodeBench 代码执行预测与测试输出预测

公共数据包另提供`livecodebench_original_tasks`字段适配，可通过现有`iter_workload`与索引准备入口读取三个原始文件。它保留代码生成的完整不透明私有测试、执行预测的原生`id`、输出预测的题号与`test_id`关系；答案单独返回，不生成API请求。Parquet时间戳显式保存类型与ISO8601值。自修复不是第四份数据集，必须额外接入真实的先前输出和评分反馈。此适配读取通过不能替代下文原生运行与评分。

例如，已按公共获取入口放置原文件后，可以通过明确的文件选择读取执行预测：

```python
from pathlib import Path
from vllm_hust_workload_data.adapters import iter_workload

records = iter_workload(
    "LIVECODEBENCH",
    data_root=Path("/private/livecodebench-original"),
    files=["execution-v2/data/test-00000-of-00001.parquet"],
    limit=3,
)
for record in records:
    original_inputs = record.input_record()
    original_references = record.outcome_record()
    # Pass through the original scenario formatter/scorer, not request_body().
```

`limit`仅限制读取条数，仍校验整个原文件的固定摘要，不表示已完成这些任务的推理。省略`limit`读取选中文件的全部case；选择其他场景时显式更换原文件路径。

这两个场景使用固定发布版的原始
Parquet，分别有479和442个case。默认选择原始行序的前三条；`--full`选择全量。准备入口复制完整原文件并校验固定摘要，原字段与日期类型由隔离容器中的作者数据类读取，不把任务改写成自定义问答。代码执行场景按原始`id`区分case，测试输出场景按`(question_id,test_id)`区分；不能按题号去重。

以下对两个场景分别执行，`SCENARIO`取`codeexecution`或`testoutputprediction`。沿用上文固定作者归档与已经构建的`LCB_IMAGE`，不同场景使用不同的全新输出目录：

```bash
SCENARIO=codeexecution
python scripts/prepare_livecodebench_original_tasks.py \
  --data-root /private/livecodebench-original \
  --output /private/lcb-$SCENARIO-prepared --scenario "$SCENARIO"
python scripts/run_livecodebench_container_scoring.py \
  --prepared-dir /private/lcb-$SCENARIO-prepared \
  --author-archive /private/lcb-source.tar.gz --image "$LCB_IMAGE" \
  --work-root /private/lcb-$SCENARIO-prompts --model Qwen/Qwen3.5-9B \
  --scenario "$SCENARIO" --prepare-prompts --execute
```

将生成的`native-output/results/requests-private.json`复制到模型服务所在节点，在仓库外部目录保存回答：

```bash
python scripts/run_livecodebench_original_generation.py \
  --prompts /private/lcb-requests-private.json \
  --output /private/lcb-generation \
  --endpoint http://127.0.0.1:19466/v1 --scenario "$SCENARIO" \
  --disable-thinking --timeout 1200 --execute
```

两种场景均保留原生每case十个候选、2000-token预算、temperature=0.2和top_p=0.95；代码执行保留官方非CoT模板。`--timeout`只设置HTTP等待时间，不改变原生评分时间限制。将完整回答目录传回CPU节点后评分：

```bash
python scripts/run_livecodebench_container_scoring.py \
  --prepared-dir /private/lcb-$SCENARIO-prepared \
  --generations-dir /private/lcb-generation \
  --author-archive /private/lcb-source.tar.gz --image "$LCB_IMAGE" \
  --work-root /private/lcb-$SCENARIO-scoring --model Qwen/Qwen3.5-9B \
  --scenario "$SCENARIO" --execute
```

官方评分器的单位不同：代码执行为百分数，测试输出为比例，不应直接混用。评分涉及生成内容的解释执行，必须保留上述离线、非root
Docker隔离，不能在宿主机直接评分。命令支持情况、真实运行和统一接入完成情况分别核验，以唯一清单为准。

## AppWorld ARM核心环境准备（不是原生任务交付）

官方ReAct运行器的软件文件可单独准备，固定版本中包含8份ReCoMa模块、原始提示和原始Llama3配置，不下载任务或模型，也不运行Agent：

```bash
python scripts/prepare_jsonschemabench_author_source.py --source appworld-react \
  --author-root /private/appworld-react-author --action prepare --max-bytes 34698
python scripts/prepare_jsonschemabench_author_source.py --source appworld-react \
  --author-root /private/appworld-react-author --action verify
```

该原配置仍指向作者的Llama3-70B服务，不能直接冒用其模型名连接本地模型。其提示长度限制是22000**字符**，ReAct交互历史输出上限20000字符，每次生成400
tokens、最多100次LLM调用和500次环境调用；这些原生规则不能为了跑通自行降低。实验依赖安装、实际模型绑定、隔离环境和官方评分需要另行核验。当前ReCoMa0.0.4的精确依赖LiteLLM1.37.19在PyPI入口返回404；官方Git标签仍可用，已以同版本固定源码构建并通过实验依赖及离线非root导入检查，没有换版本；140项安装分发中的原核心83项版本不变。其用户缓存使用专用可写卷，根文件系统、任务和代码仍只读。实际模型绑定、Agent任务执行与评分仍未完成，不以依赖安装给来源打勾。

固定作者版本`0.1.3.post1`的官方Docker服务镜像经元数据核查仅有amd64，不能直接作为当前ARM节点的合格环境。新增核心配方沿用该版本官方Dockerfile的Python/uv安装路径，固定已核验的官方ARM64基础镜像，使用此前实际安装观察到的83项核心版本；这是可复现核心环境配方，不是作者依赖锁、完整Agent环境或代码执行隔离资格证明。

以下只复制三份软件文件到新的仓库外目录，不把任务数据、模型或企业资产放进构建上下文：

```bash
CONTEXT=/private/appworld-core-build-context
mkdir "$CONTEXT"
cp configs/workload-native/appworld-core-cpu.Dockerfile "$CONTEXT/Dockerfile"
cp configs/workload-native/appworld-core-client-py311-observed.txt "$CONTEXT/"
cp scripts/verify_appworld_core_cpu_environment.py "$CONTEXT/"
DOCKER_BUILDKIT=0 docker build --network=host --memory 2g \
  --cpu-period 100000 --cpu-quota 100000 \
  --iidfile /private/appworld-core-image-id.txt \
  -t workload-native/appworld-core-cpu:local "$CONTEXT"
```

构建中调用官方`appworld install`，受保护代码仅留在本地私有镜像，不发布或复制到Git；不会获取任务数据或执行生成代码。配方校验Python/83项版本、核心环境和评分函数能否导入，并执行`pip check`，不调用它们运行任务。镜像默认非root，后续仍需单独配置离线隔离、可写任务状态、原生Agent依赖与流程、NPU模型请求及官方评分；不能以构建成功给来源打勾。真实构建状态以[既有回执](../docs/workload-data/validation/appworld-core-client-preparation-20261028.json)为准。

公共原生调用入口为`scripts/run_appworld_original_react.py`。默认仅输出计划；它校验固定原始配置SHA256，只绑定实际模型名与provider，不改原任务、ReAct预算或原评分调用。执行时要求非root、只读根文件系统、network-none容器及单独环境Unix
socket；这些基础检查不能替代完整隔离配置审计。`--local-unbilled`只用于明确不收API费用的本地服务，零API计费不表示NPU成本为零，也不修改原作者调用次数、生成预算和费用上限。`--service-context`须与实际部署一致，不是单次生成预算；收费provider不可启用零计费选项。示例（先不加`--execute`）：

```bash
python scripts/run_appworld_original_react.py \
  --profile /author/experiments/configs/react_llama3_test_normal.jsonnet \
  --model openai/Qwen/Qwen3.5-9B --api-base http://127.0.0.1:14001/v1 \
  --environment-url http+unix://%2Fenvironment-io%2Fenvironment.sock \
  --experiment original-react-check --limit 3 --local-unbilled --service-context 262144 \
  --model-queue /transport
```

`--model-queue`在离线Agent容器内启动回环代理，把原始请求交给单独的可信模型worker。操作者须预先建立服务器私有队列目录，并且仅把该子目录挂入Agent容器为`/transport`；生成代码的环境容器不得获得队列或Docker
socket。可信worker使用同一队列的宿主机路径及已有模型服务（或事先授权的回环转发），不自动启停模型、建立SSH或抢占研究资源。仅获得空闲验证窗口后启动：

```bash
python scripts/appworld_model_transport.py --queue /private/appworld-queue \
  --api-base http://127.0.0.1:19466/v1 --disable-thinking \
  --watch --max-jobs 300 --idle-timeout 600
```

不带`--watch`或`--execute-one`时只输出计划。`--disable-thinking`是本Qwen绑定的显式选项，不是其他模型的默认设置；它不覆盖原请求中冲突的设置。worker保留原请求、实际转发请求和原始响应，已返回或已认领的请求不会自动重复发送；超时或失败保留现场，不能删掉标记盲目重试。原始内容仅保留服务器私有目录，不提交Git。此传输当前通过回环协议夹具测试，尚不是新公共入口的真实原题推理证据。

执行前仍需完整实验依赖、原始数据、原作者软件导入路径、模型worker和环境服务；核心镜像本身不足以执行。此公共入口已经通过服务器离线非root只读容器的原配置哈希、依赖导入及隔离前置检查，但尚未完成该入口的原生推理实跑和研究端联调，不把私有驱动三条原生任务的证据冒充此入口验证。运行输出包含原始任务与生成代码，必须留在服务器私有目录；失败不替换任务。

完整实验镜像配方为`configs/workload-native/appworld-react-cpu.Dockerfile`。在新的外部软件构建目录中准备`author/`（上述固定10文件核验后的原作者目录）和`litellm-original/`，以及该Dockerfile。LiteLLM必须取BerriAI/litellm原始提交`f20184cd49c77ee66da21001a960f50b8875c321`对应的v1.37.19源码；已取得归档的SHA256为`61ce6448db7d0d26e26670910c5830ac56ef7539db6593f328a0fe36215ffde5`。校验归档后在服务器解包完整源码（保留原符号链接及其目标），不要改用新版本以绕过PyPI旧版本缺失。构建时传入已核验核心镜像的实际ID：

```bash
docker build --network=host --build-arg APPWORLD_CORE_IMAGE="$(cat /private/appworld-core-image-id.txt)" \
  --iidfile /private/appworld-react-image-id.txt \
  -t workload-native/appworld-react-cpu:local /private/appworld-react-build-context
```

这份公共实验配方已在服务器独立构建成功，并通过离线非root只读容器中的原配置、核心83项版本及原生依赖导入检查。新解析环境有152项依赖，不等同先前140项完整环境，也不是原作者锁定依赖。默认原作者70B模型配置不变地归档；9B是资源适配替代，不能宣称复现70B结果。原始任务下载与目录核验、模型代理和完整隔离启动仍是独立步骤；不能仅靠镜像构建或入口计划输出打总勾。

`--network=host`只用于本专用CPU工作区的软件依赖构建，以避开嵌套Docker桥接网络的PyPI连接问题；原始数据和任务均不进入构建上下文，任务执行必须保持`--network=none`，不能照搬构建网络配置。读取模块保留全部732项任务、14,643个任务文件、46个共享数据库/API文档引用及原始划分/版本/许可文件。统一绑定与索引回查已在服务器原数据上验证，安装产物及研究端联调仍待完成，不视为该来源已交付。

隔离启动示例：先由操作者准备本任务专用的可写输出、缓存、队列及socket子目录，确保UID65534可访问；原始输入只读暂存，并预建空的`world/experiments`嵌套挂载点。不要递归改动原始共享数据目录、其他人的工作区或模型权重。`/private/appworld-agent-software`只放本仓库两个公开入口脚本；下面镜像ID文件分别来自前述构建。Docker
daemon与CLI不在同一容器时，先核实绑定源在daemon侧的实际位置，不能假定CLI侧`/root`可直接绑定。

```bash
docker run -d --name appworld-environment --network none --user 65534:65534 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --memory 4g --cpus 1 --pids-limit 1024 --tmpfs /tmp:size=512m \
  --mount type=bind,src=/private/appworld/world,dst=/work,readonly \
  --mount type=bind,src=/private/appworld-output,dst=/work/experiments \
  --mount type=bind,src=/private/appworld-env-socket,dst=/io \
  --entrypoint /opt/venv/bin/python "$(cat /private/appworld-core-image-id.txt)" \
  -m uvicorn appworld.serve.environment:app --uds /io/environment.sock

docker run --name appworld-agent --network none --user 65534:65534 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --memory 4g --cpus 1 --pids-limit 1024 --tmpfs /tmp:size=512m \
  --mount type=bind,src=/private/appworld/world,dst=/work,readonly \
  --mount type=bind,src=/private/appworld-output,dst=/work/experiments \
  --mount type=bind,src=/private/appworld-agent-cache,dst=/nonexistent \
  --mount type=bind,src=/private/appworld-agent-software,dst=/entry,readonly \
  --mount type=bind,src=/private/appworld-env-socket,dst=/environment-io,readonly \
  --mount type=bind,src=/private/appworld-queue,dst=/transport \
  --workdir /author --entrypoint /opt/venv/bin/python \
  "$(cat /private/appworld-react-image-id.txt)" -B /entry/run_appworld_original_react.py \
  --profile /author/experiments/configs/react_llama3_test_normal.jsonnet \
  --model openai/Qwen/Qwen3.5-9B --api-base http://127.0.0.1:14001/v1 \
  --environment-url http+unix://%2Fenvironment-io%2Fenvironment.sock \
  --experiment original-react-check --limit 3 --local-unbilled --service-context 262144 \
  --model-queue /transport
```

Agent命令默认仅检查计划；实际执行需环境socket、模型worker及资源窗口就绪后显式增加`--execute`。上例不挂载Docker
socket、不暴露NPU设备、不让生成代码接触模型队列；失败容器和日志先保留，不能为“跑通”替换题目。公开启动组合尚待真实环境联调验证，既有私有驱动原生结果与该组合的验证状态分开记录。

绑定已有原始数据（不会下载、改写或执行任务）：

```bash
python -m vllm_hust_workload_data.pipeline bind APPWORLD \
  --data-root /private/appworld/world/data \
  --files datasets/train.txt datasets/dev.txt datasets/test_normal.txt datasets/test_challenge.txt \
  --revision private-original-cli-acquisition-YYYYMMDD \
  --output /private/appworld-delivery/binding.json
```

`--revision`应如实标识本次原始数据快照；本地哈希绑定不是发布方签名认证。将以下内容保存为外部配置，再用通用`pipeline plan/run`入口准备仅含引用的索引。父目录须事先存在，输出文件/索引目录必须全新，不能放进Git：

```json
{"sources":[{"source_id":"APPWORLD","data_root":"/private/appworld/world/data","manifest":"/private/appworld-delivery/binding.json"}]}
```

```bash
python -m vllm_hust_workload_data.pipeline plan --config /private/appworld-delivery/config.json
python -m vllm_hust_workload_data.pipeline run --config /private/appworld-delivery/config.json \
  --output /private/appworld-delivery/prepared
```

真实数据的统一绑定/索引/三条回查已通过：732项任务及全部14,696个原始文件身份得到固定；`consumer.resolve_input`回查时校验该任务及共享/参考资产，返回原生环境任务而非可直接发送的请求。原生Agent仍使用前述独立环境与执行入口，不把索引成功当作原生推理完成。安装产物及研究端联调仍待核验。

## StructEval Matplotlib 渲染组件准备

这是原作者Matplotlib渲染函数的隔离环境，不是完整StructEval执行入口。原27模块已经复核；其他浏览器/CDN/npm/TeX/Typst依赖、原题生成及完整VQA仍待验证。19类950条原非渲染结果保留，不重跑。

最小镜像只包含观察并冻结的12项绘图依赖，符合原作者`matplotlib>=3.8`要求；不是作者依赖锁，不含其他benchmark运行时、受保护代码、原始任务或模型。已有服务器构建及原函数的可信准备夹具出图通过，**不是原题成绩**。

```bash
CONTEXT=/private/structeval-matplotlib-build-context
mkdir "$CONTEXT"
cp configs/workload-native/structeval-matplotlib-cpu.Dockerfile "$CONTEXT/Dockerfile"
cp configs/workload-native/structeval-matplotlib-py311-observed.txt "$CONTEXT/"
cp scripts/verify_structeval_matplotlib_environment.py "$CONTEXT/"
docker build --network=host --iidfile /private/structeval-matplotlib-image-id.txt \
  -t workload-native/structeval-matplotlib-cpu:local "$CONTEXT"
docker run --rm --network none --user 65534:65534 --read-only --cap-drop ALL \
  --security-opt no-new-privileges --memory 2g --cpus 1 --pids-limit 256 \
  --tmpfs /tmp:size=256m "$(cat /private/structeval-matplotlib-image-id.txt)"
```

默认命令仅校验版本与依赖导入，不执行生成代码。真实原题必须使用固定原源码、完整生成输出和原始VQA问题，不能因为PNG生成成功就宣称任务通过。不得在宿主机运行原渲染器中的`os.system`，也不得为离线出图删掉其他格式的CDN脚本。[现有阶段回执](../docs/workload-data/validation/structeval-nonrender-checkpoint3-20261025.json)区分组件夹具与原生任务证据。

公开组件worker为`scripts/run_structeval_matplotlib_worker.py`。它只接收原工作流提取的完整`parsed_code`，不重新提取、不修复代码、不添加替代答案；调用固定原`render_matplotlib.py`，保留其0/1结果。原工作流实际使用`safe_unicode_decode`与`extract_renderable_code`，不能用其他便捷提取函数代替。调用方须保留原生成输出、原提取结果及对应关系，worker本身不证明该关系，也不执行完整语法评分或VQA。

输入与原模块须只读挂载，输出只允许写入不超过256MiB的显式tmpfs（`/tmp`或控制器的`/output`）。额外检查非root容器PID1、只读根、无外网接口、无能力/提权、无Docker/NPU设备及常见凭据环境变量。此为本任务有界执行隔离，不是安全认证。代码文件不得超过16MiB，单文件输出受64MiB限制；超出资源时保留阻塞，不截断原输入或当作错误答案。先不加`--execute`检查底层worker计划：

```bash
docker create --name structeval-matplotlib-case-001 --network none \
  --user 65534:65534 --read-only --cap-drop ALL --security-opt no-new-privileges \
  --memory 2g --cpus 1 --pids-limit 256 --tmpfs /tmp:size=256m \
  --mount type=bind,src=/private/structeval-worker-software,dst=/software,readonly \
  --mount type=bind,src=/private/structeval-one-case,dst=/input,readonly \
  --entrypoint /opt/venv/bin/python "$(cat /private/structeval-matplotlib-image-id.txt)" \
  -B /software/run_structeval_matplotlib_worker.py --task-id ORIGINAL_TASK_ID \
  --renderer /software/render_matplotlib.py --code-file /input/parsed-code.txt \
  --code-sha256 ORIGINAL_PARSED_CODE_SHA256 --output /tmp/renderer-output
```

上面的直接worker示例只供底层计划/诊断；`/tmp`在容器停止后消失，不能据此安排事后导出。实际执行使用公开控制器，它不依赖CLI与Docker daemon共享宿主路径：

```bash
python scripts/run_structeval_matplotlib_isolated.py \
  --image "$(cat /private/structeval-matplotlib-image-id.txt)" \
  --task-id ORIGINAL_TASK_ID --renderer /private/original-source/render_matplotlib.py \
  --code-file /private/original-parser-output/parsed-code.txt \
  --code-sha256 ORIGINAL_PARSED_CODE_SHA256 \
  --output /private/structeval-runs/fresh-case --timeout 120
```

先查看计划；确认专用Linux
Docker服务、原提取结果与新输出父目录就绪后加`--execute`。控制器以`docker cp`传输文件，只在新专用卷中处理所有权，不改原输入字节或600权限。生成代码只在非root、无网络、CPU1/RAM2GiB/PIDs256的worker中执行；单独的可信只读持有进程保持256MiB输出卷，worker结束后校验并导出指定PNG、原渲染日志和组件结果，再释放持有进程。持有进程不执行生成代码、不读取评分参考；正常结果保存为外部私有文件，容器/卷不自动删除。

执行超时后仅停止ID、操作标签、名称全部匹配的本次worker，返回`BLOCKED`而不是伪造模型0分。失败或超时时暂留输出持有进程供检查（上限1小时），回执记录其ID；确认已保存证据后再停止本次持有进程，不重跑有效模型输出。不要以`timeout docker wait`替代停止容器，也不要在观察连接失败时盲目重启任务。原工作流即使把渲染异常折算为0，也不能覆盖控制器的资源阻塞状态。公开链路已验证正常1、语法失败0、超时不填分及退出后导出，**仍不是原题与VQA验证完成**。

原工作流的分派适配点为`scripts/structeval_render_queue.py`中的`MatplotlibRenderQueue.render`，参数与原`render_matplotlib_and_screenshot`一致。它只传输原`parsed_code`与校验后的返回图片；按原任务/代码/固定镜像身份复用已保留结果，不覆盖已有图片。每次独立实验使用新的私有队列目录，只有恢复同一次运行时复用旧目录，不跨模型/实验混用。实际CPU隔离分派已验证有效代码1、语法错误0及恢复不重新启动；原工作流固定前三题的生成、原解析器、隔离渲染和完整VQA也已通过，其他渲染协议仍未验证。

接入时保留原`THREAD_RENDERERS`的其他项，仅把`matplotlib`调用位置绑定到该分派器；原提取器与任务内容不变。原`render_file_async`返回后必须调用`require_ready_for_judge()`，通过后再进入原`evaluate_file`。这道门禁防止原框架捕获异常后填0，掩盖资源或通信失败。官方2035行数据中该协议有50题，现已全部完成：固定原位置79/80/81、ID000802/000803/000804保留复用，新增部分从`--start 3 --count 47`继续，不重新生成前三题。新实验可用`--all-selected`选择该完整协议；已有有效结果不为展示命令重复跑。

原生入口在原命令后指定`--output-type Matplotlib --render-queue /private/queue --render-image sha256:IMAGE_ID --model Qwen/Qwen3.5-9B --disable-thinking`。队列须预先创建，输出目录须为新目录；同时在隔离CPU端处理该队列请求。原作者27个Python模块仍逐字节核验，只在内存中切换渲染调用位置；生成和裁判均保留原`max_tokens=None`、温度0及完整任务内容。实际前三题各保留10字段、7/8/8个VQA问答对，3次生成与3次裁判均stop；实际裁判图片与原渲染图片逐字节一致。9B是现有NPU服务的替代模型，不宣称复现作者推荐模型或据此证明引擎性能。模型请求和响应保存在私有运行目录，不入Git。

隔离CPU端单任务入口为`python scripts/run_structeval_render_queue_backend.py --request /private/queue/JOB.request.json --renderer /private/original/render_matplotlib.py --work-root /private/render-work --image sha256:IMAGE_ID`，默认只检查计划；确认专用Docker与固定镜像就绪后加`--execute`。请求文件名必须匹配内容哈希。执行仅调用上述隔离控制器，先发布原图片、后发布结果；恢复时核验并复用同一输入的已保存控制器结果。已有未完成或阻塞目录必须先检查实际容器，不自动重启，不把超时变成模型0分。请求及原图片只保存在操作者指定的私有目录；跨机器需另行安全传输，不要求共享SSH凭据。此链路已完成全部50道Matplotlib原题，共50次生成、49次视觉裁判，99次调用均stop；000815的原生成代码AttributeError使原渲染器返回0，按作者缺图规则跳过该题裁判，未发现缺模块或资源不足，不换题或重生成。此结论只覆盖该协议，其他格式进度见后续小节及唯一机器清单，不能据此勾选整个StructEval来源。

完整VQA须调用原`vqa_eval_async`并保留全部原问题、期望答案及对应原图；答案进入作者定义的裁判输入，不进入生成模型输入。原评测器会把部分API/解析异常折算为0或None，因此必须另核实际裁判请求、响应、图片身份和异常日志；不能只凭一个0分文件就认定裁判成功运行，也不能把资源/接口失败当作模型质量结论。

### StructEval Typst 环境补充

`configs/workload-native/structeval-typst-cpu.Dockerfile`以已合格的最小Matplotlib软件镜像为基础：将自己的基础镜像身份传给`--build-arg RENDER_BASE=...`，构建目录放入该Dockerfile、`scripts/verify_structeval_typst_environment.py`和从[Typst官方v0.13.1发布页](https://github.com/typst/typst/releases/tag/v0.13.1)取得的ARM64
musl二进制`typst`。不要把原始任务、模型或生成代码放入构建目录。配方核验观察到的二进制SHA256；此旧发布的GitHub元数据无厂商digest，不能把观察哈希称为厂商签名。

服务器已实际构建并在非root、无网络、只读根文件系统中核验Typst0.13.1、ImageMagick、Ghostscript和字体。为执行原作者PDF转PNG流程，仅开启PDF读权限，PS等规则不放宽；发行版配置的嵌套注释保持不变。原Typst模块在隔离容器中已通过正常文本1/非法语法0两个控制；公共分派入口又通过Typst正常、语法错误、多页缺图和Matplotlib兼容性四项控制，并确认退出与有效结果复用不重新启动。OS软件包版本记录于现有阶段回执，但配方尚未固定apt快照，不能称为作者环境锁或位级可复现环境。

使用原生入口的`--output-type Typst --render-queue /private/queue --render-image sha256:TYPST_IMAGE_ID`；其他输入、模型与预算参数同上。隔离端仍用`run_structeval_render_queue_backend.py`，将`--renderer`指向原`render_typst.py`，它按请求中固定的渲染类型核验源文件与镜像。单独调用控制器或worker时显式加`--renderer-kind typst`，默认仍为Matplotlib，已有Matplotlib请求身份与缓存不改。Typst多页输出可能不生成作者指定名称的PNG：必须保留原渲染器返回1及缺图状态，由原VQA规则处理，不能补造单页图片或改返回值。编译器捕获的stderr只保存在私有运行日志；命令、返回值与异常保持原样，不自动重试，不修补生成代码。

实际固定前三题为原位置234/235/236、ID001401/001402/001406。9B的三次生成均stop，但包含不能编译的Markdown式标题，原渲染分数均0、原最终分数均0.1；原缺图规则使视觉裁判调用数为0。原10字段、8/8/9个问答对、生成提示、原提取结果和27个作者模块已核验。诊断只复用三份失败代码重新编译以保留原先丢失的stderr，没有重新生成、替换题目或覆盖原评分。\*\*这些结果证明失败分支可执行，不证明Typst实际视觉裁判已覆盖，不计来源或新增原题验证完成数。\*\*更强模型配置尚未启动，需在获准且研究空闲后单独验证；不能把控制夹具当原题或用模型生成错误冒充环境成功。

### StructEval SVG 浏览器入口

原生入口支持`--output-type SVG --render-queue /private/queue --render-image sha256:BROWSER_IMAGE_ID`，其余原始数据/源码、模型和预算参数同上。隔离端仍使用`run_structeval_render_queue_backend.py --renderer /private/original/render_svg.py`，并要求同目录原`render_utils.py`；两个模块均固定哈希核验并只读装入，不重写原SVG包装页面或原解析器。原异步渲染位置只做隔离CPU分派，原评分和全部VQA问题不变。

浏览器基础软件配方为`configs/workload-native/structeval-browser-cpu.Dockerfile`：构建上下文只放配方、`structeval-browser-py311-observed.txt`与`scripts/verify_structeval_browser_environment.py`，指定已合格Matplotlib基础镜像`--build-arg RENDER_BASE=...`。七个扩展依赖版本来自已取得并验证的环境，Playwright按官方`install --with-deps chromium`准备浏览器；不是作者环境锁，apt系统包快照未固定。公共配方已实际构建退出0，并在非root/离线/只读容器通过依赖及浏览器资产核验和四项隔离控制；原题使用的准备镜像与公共配方镜像分开记录，没有为配方重复运行有效原题。历史Matplotlib脚本文件名保留以兼容，实际渲染类型由明确的selector及原模块哈希决定。

执行仍要求非root、只读根文件系统、无网络/凭据/NPU设备及明确CPU/RAM/PID/tmpfs/时间边界。内联SVG及空输入保持原行为；若浏览器实际请求未取得的HTTP/S资源，则在原视觉裁判前返回资源阻塞，私有日志记录缺失资源，不将缺图或白图静默记成模型0分，也不擅自开放网络或删改原资产。该门禁及Matplotlib兼容性已通过服务器控制验证。

原位置94/95/96、ID001200/001201/001202各10字段与7/8/10个原问答对保留，原最终分0.9/0.65/1.0如实记录。其后按`--start 3 --count 47`执行其余原题，保留前三题结果不重生成；现在全50题已完成，共50次生成、50次实际视觉裁判，100次调用均stop，原提取代码、397个原问答对、裁判提示及PNG字节均独立核验。新实验可用`--all-selected`选择完整协议，但不要为展示命令重复跑已有有效结果。使用现有Qwen3.5-9B、显式关闭thinking、温度0和原`max_tokens=None`，不是作者模型复现或性能结论。其他渲染协议仍未完成，不据此勾选整个StructEval来源。

### StructEval HTML、Canvas、Markdown 接入

复用同一原生入口与隔离分派器：指定`--output-type HTML`、`Canvas`或`Markdown`，并指定与该格式匹配的固定渲染镜像及私有队列。HTML的原输入协议包括`--input-type Text`、`Markdown`、`React`、`Vue`；Canvas和Markdown当前对应原Text输入协议。准备、选题及原评分仍使用固定原始数据和原源码，不构造替代题目。

HTML、Canvas直接复用已合格的浏览器镜像。Markdown使用`configs/workload-native/structeval-markdown-cpu.Dockerfile`，以该浏览器镜像传入`--build-arg RENDER_BASE=...`，仅增加观察到的`Markdown==3.11.2`，不重下浏览器。该扩展已实际构建退出0。当前配方为已验证的Linux
ARM64环境，不据此宣称其他架构已运行。

隔离端`--renderer`须指向对应原`render_html.py`、`render_canvas.py`或`render_markdown.py`；同目录保留原`render_utils.py`，Canvas/Markdown还须有原`render_html.py`，均固定哈希、只读装入。Canvas原函数把输入嵌入其JavaScript包装，Markdown按作者默认转换器生成HTML；不增加代码修补、替代包装或Markdown扩展。资源门禁覆盖被内部调用的HTML浏览器工厂，不仅检查外层函数。

三种格式的正常1、空输入0、外部HTTP/S资产阻塞，以及SVG兼容性，共10项CPU隔离控制已通过。其后六组协议各固定前三题实际完成18道原题、36次模型调用、158对完整VQA；Canvas三题VQA均0，原最终分0.29/0.3/0.3如实保留。Text-to-HTML另执行剩余47题，与前三题合计50题、100次stop调用、345对原VQA，原字段、提示、提取代码及实际裁判图片字节独立核验。其他五组剩余220题尚未计为完成。控制夹具不计原题，既有有效原题不重跑，整个StructEval来源仍未勾选。

### StructEval LaTeX/TikZ 编译入口

固定原数据中的协议名称大小写不同：Text→`Latex`有50题、Text→`Tikz`有50题、`Matplotlib`→`TikZ`有100题。使用原生入口的对应`--input-type`和`--output-type`，同时指定私有`--render-queue`及固定`--render-image`。不把Matplotlib输入改写为文字任务，也不把三组混成重新生成的题目。

`configs/workload-native/structeval-latex-cpu.Dockerfile`以已合格最小Matplotlib镜像为`RENDER_BASE`；构建目录仅放配方及`scripts/verify_structeval_latex_environment.py`，不放题目、模型或生成代码。配方沿用基础镜像的`/uv`安装观察版本`pdf2image==1.17.0`，增加原作者已有回退路径所需的pdfLaTeX、LuaLaTeX、Poppler及TeX包，不增加Tectonic替代实现。系统包快照未固定，不称为作者环境锁。

隔离端把`--renderer`指向原`render_latex.py`；手动调用worker/控制器时使用`--renderer-kind latex`。同一原函数用于三个协议，按原签名`render_latex_to_png(latex_code, output_path, task_id)`调用，保留默认dpi300、45秒预算、原解析器、片段包装、颜色前导、原引擎回退及第一PDF页转换，不修补生成代码。TeX临时缓存留在有界`/tmp`，生成代码仍用非root、只读根、断网及CPU/RAM/PID/时间限额。

公共配方已实际构建，非root/只读/断网软件核验与五项隔离控制通过：TikZ片段、完整LaTeX文档、fontspec/LuaLaTeX、缺失文档类0且缺图、Matplotlib兼容。原始200题及完整VQA尚未运行，不能以这些控制代替。若专用Docker的bridge下载失败，先对比同一官方资产在工作区与构建容器的可达性；本次仅对可信软件安装构建使用经核验的`--network=host`，不修改全局网络、不允许生成代码使用该网络。首次bridge获取失败及第二次无pip失败均保留，公共配方已使用现有uv修正。

## Video-MME 原生视频任务

这是发布方原始Video-MME与固定VLMEvalKit的两个已有配置：`Video-MME_8frame`和`Video-MME_8frame_subs`，不是把64帧配置临时减成8帧。原2700题/900视频全部保留；默认每种duration取最先3个不同视频并保留每视频全部3题，每配置27题。选题在推理前固定，不因失败换题；`--full`才选择全部2700题。

Video-MME-v2是另一个原评测协议，不适用本节旧版8帧配置或评分；其资产准备与未完成项见下一节及阶段回执，不能复用旧预测冒充v2已验证。

数据限学术研究，禁止把本入口视为商业使用或再分发授权。原媒体、字幕、题目、抽帧及模型输出均放在外部私有目录；准备完整原发布版须预留至少500GiB磁盘，首次获取/完整校验耗时不等同原题推理耗时。

以下命令从仓库根目录运行。已有合格ChartQA/MVBench HTTP客户端与固定工具包时直接复用，不重新安装。新Linux ARM/Python3.12客户端先装官方CPU
Torch与直接依赖，再构建Decord/FFmpeg；CPU库只用于图像/视频处理，模型推理仍在单独NPU服务：

```bash
python3.12 -m venv /private/vlm-client
PY=/private/vlm-client/bin/python
$PY -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install -r configs/workload-native/chartqa-native-client-py312-direct-requirements.txt
python scripts/prepare_chartqa_native_runtime.py \
  --output /private/vlm-runtime --python "$PY" --execute
$PY -m pip install -r configs/workload-native/chartqa-native-client-py312-verified-lock.txt
$PY -m pip check
$PY -m pip install --no-deps --target /private/videomme-subtitle-overlay \
  -r configs/workload-native/videomme-subtitle-requirements.txt
python scripts/prepare_chartqa_native_toolkit.py \
  --output /private/VLMEvalKit --archive --execute
```

原版库的字幕版本使用显式隔离overlay，不升级在用模型环境。运行客户端时设置其私有FFmpeg库路径，不修改模型服务进程。原数据获取和作者原解包：

```bash
python3.12 -m venv /private/videomme-download-client
/private/videomme-download-client/bin/python -m pip install huggingface_hub==0.36.0
/private/videomme-download-client/bin/python scripts/prepare_videomme_original_assets.py --cache /private/videomme-cache \
  --action download --accept-academic-only --max-bytes 101002235118
SNAPSHOT=/private/videomme-cache/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b
$PY scripts/prepare_videomme_native_media.py \
  --official-checkout /private/VLMEvalKit --snapshot "$SNAPSHOT" \
  --output /private/videomme-media-receipt.json --execute --accept-academic-only
```

原站不可达时，获取入口可显式增加`--hf-endpoint https://hf-mirror.com`；固定版本、全24文件大小/SHA256和原内容检查不变，不替换原题。再次获取会复用SDK缓存；已具备原文件时先用`--action verify`，不要盲目重下。只需字段/索引的研究输入可再执行`--action prepare-data --data-root /private/videomme-annotations`，它不等于媒体或原生任务已完成。

使用现有支持8图像输入的NPU视觉模型服务。示例模型为实际验证采用的`Qwen/Qwen3.5-9B`；必须与服务端声明ID一致。以下保留原2048输出预算、temperature0、显式nonthinking和原exact-matching评分；无外部付费judge。不同profile必须用不同新输出目录：

```bash
mkdir -p /private/videomme-frames
for PROFILE in Video-MME_8frame Video-MME_8frame_subs; do
  LD_LIBRARY_PATH=/private/vlm-runtime/install/lib "$PY" scripts/run_videomme_original.py \
    --official-checkout /private/VLMEvalKit --snapshot "$SNAPSHOT" \
    --media-receipt /private/videomme-media-receipt.json \
    --frame-cache /private/videomme-frames \
    --subtitle-overlay /private/videomme-subtitle-overlay \
    --model Qwen/Qwen3.5-9B --profile "$PROFILE" \
    --output "/private/videomme-native-$PROFILE" --execute --accept-academic-only
done
```

默认endpoint为本机`http://127.0.0.1:19466/v1/chat/completions`；可用`--endpoint`明确绑定合规本机转发入口。若先检查抽帧/字幕/原提示，增加`--prepare-prompts-only`并使用单独的新输出目录：该模式0模型调用、无评分，不能算benchmark完成。去掉`--execute`仅输出计划。

字幕配置仅提取原作者在采样帧时间戳处选取的字幕，缺失字幕按原逻辑处理，不补全整段转录。请求失败、空答案、帧数或原题关系不一致会保留失败并停止；不得缩短视频题、改预算或换题凑通过。验收实际进度和已运行范围见[唯一清单](../docs/workload-data/DATASET_CHECKLIST.zh-CN.md)与[实测记录](../docs/workload-data/validation/videomme-original-readiness-20261028.json)，不是全量质量或性能结论。

## Video-MME-v2 原资产准备（原生执行仍待补）

复用现有获取器，不另建下载框架：

```bash
python scripts/prepare_reused_research_assets.py VIDEOMME_V2 --mode plan
python scripts/prepare_reused_research_assets.py VIDEOMME_V2 --mode download \
  --data-root /private/videomme-v2-original --max-bytes 10000000
python scripts/prepare_reused_research_assets.py VIDEOMME_V2 --mode verify \
  --data-root /private/videomme-v2-original
```

这里只注册固定官方版本的四份小资产，共9046844字节：原README、eval.yaml、全部test.parquet与subtitle.zip，不包含视频。已有原文件直接用只读verify，不重复下载。全部3200题、800个四题组、11字段与800份字幕身份/对应关系已在服务器核验；原始文件不进入代码仓库。新环境须自行具备合规网络可达性，不能以本文命令已列出推断当地下载成功。

官方源码固定为`MME-Benchmarks/Video-MME-v2@b2834bd4091d6234ccaf0fa1f149a7ef4eac5875`；原评测脚本为`evaluation/test_video_mme_v2.py`。原分组非线性rating与accuracy均需保留，不以旧版逐题正确率替换。抽样按两个原group_type各最先三个完整四题组，保留每组全部四题和原顺序，不取孤立三题；该固定计划共24题，尚未推理。

目前仅完成原资产和原评分器参考/CPU控制。仍缺视频、干净的原生读取/抽帧环境、64帧原提示、合适NPU服务、实际推理评分及固定工件消费。当前32图上限的服务不能声称执行过官方64帧配置；不为跑通缩帧或删减字幕。具体证据与限制见[阶段回执](../docs/workload-data/validation/videomme-v2-original-preparation-20261029.json)，来源总勾保持未完成。
