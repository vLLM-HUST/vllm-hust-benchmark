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

原作者配置启用`use_cache=true`，温度为0时Recoma会使用`~/.cache/litellmcalls`磁盘缓存。保持原配置不变，但需要实际模型调用的核验应使用本任务新的专用缓存目录，不能复用其他实验的缓存后把缓存命中算作实时推理。复用已验证响应进行入口核对时，须单独记录缓存复用与新增API请求；真实工具反馈不一致就停止精确重放，不改反馈或套用不匹配的旧响应。此类功能验证不构成推理引擎性能结果。

```bash
python scripts/appworld_model_transport.py --queue /private/appworld-queue \
  --api-base http://127.0.0.1:19466/v1 --disable-thinking \
  --watch --max-jobs 300 --idle-timeout 600
```

不带`--watch`或`--execute-one`时只输出计划。`--disable-thinking`是本Qwen绑定的显式选项，不是其他模型的默认设置；它不覆盖原请求中冲突的设置。worker保留原请求、实际转发请求和原始响应，已返回或已认领的请求不会自动重复发送；超时或失败保留现场，不能删掉标记盲目重试。worker须与Agent使用相同UID，或具备仅对本专用队列响应文件转交属主的权限；响应在原子发布前交给请求文件属主，不放宽整个目录权限。原始内容仅保留服务器私有目录，不提交Git。通用worker经过协议/权限夹具验证；本次真实运行使用同一公共Agent代理及专用跨节点受控传输，并非通用worker
CLI的整套原题实跑。

执行前仍需完整实验依赖、原始数据、原作者软件导入路径、模型worker和环境服务；核心镜像本身不足以执行。此公共入口已在服务器离线非root只读容器完成固定原三题ReAct及官方评分，141步中28条精确缓存响应复用、113次新增NPU调用全部HTTP200。三题各1/5测试通过且任务均失败，如实保留，不换题。实际CI71运行载荷与当前提供方一致，未修改研究消费者完整回读三题通过。详见[交付证据](../docs/workload-data/validation/appworld-native-delivery-20261030.json)；不代表全732题、原70B质量或性能结论。运行输出包含原始任务与生成代码，必须留在服务器私有目录。

完整实验镜像配方为`configs/workload-native/appworld-react-cpu.Dockerfile`。在新的外部软件构建目录中准备`author/`（上述固定10文件核验后的原作者目录）和`litellm-original/`，以及该Dockerfile。LiteLLM必须取BerriAI/litellm原始提交`f20184cd49c77ee66da21001a960f50b8875c321`对应的v1.37.19源码；已取得归档的SHA256为`61ce6448db7d0d26e26670910c5830ac56ef7539db6593f328a0fe36215ffde5`。校验归档后在服务器解包完整源码（保留原符号链接及其目标），不要改用新版本以绕过PyPI旧版本缺失。构建时传入已核验核心镜像的实际ID：

```bash
docker build --network=host --build-arg APPWORLD_CORE_IMAGE="$(cat /private/appworld-core-image-id.txt)" \
  --iidfile /private/appworld-react-image-id.txt \
  -t workload-native/appworld-react-cpu:local /private/appworld-react-build-context
```

这份公共实验配方已在服务器独立构建成功，并通过离线非root只读容器中的原配置、核心83项版本及原生依赖导入检查。新解析环境有152项依赖，不等同先前140项完整环境，也不是原作者锁定依赖。默认原作者70B模型配置不变地归档；9B是资源适配替代，不能宣称复现70B结果。原始任务下载与目录核验、模型代理和完整隔离启动仍是独立步骤；不能仅靠镜像构建或入口计划输出打总勾。

`--network=host`只用于本专用CPU工作区的软件依赖构建，以避开嵌套Docker桥接网络的PyPI连接问题；原始数据和任务均不进入构建上下文，任务执行必须保持`--network=none`，不能照搬构建网络配置。读取模块保留全部732项任务、14,643个任务文件、46个共享数据库/API文档引用及原始划分/版本/许可文件。统一绑定、实际CI组件与未修改研究消费已验证；来源交付勾仅覆盖固定原三题原生执行和评分，不把732条读取当作全量推理。

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
socket、不暴露NPU设备、不让生成代码接触模型队列；失败容器和日志先保留，不能为“跑通”替换题目。公共入口及独立构建的152依赖实验镜像已完成固定原三题实跑；早期私有驱动与精确重放尝试的历史记录分别保留，不混算调用数。

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

真实数据的统一绑定/索引/三条回查已通过：732项任务及全部14,696个原始文件身份得到固定；`consumer.resolve_input`回查时校验该任务及共享/参考资产，返回原生环境任务而非可直接发送的请求。[研究消费回查证据](../docs/workload-data/validation/appworld-research-consumer-20261030.json)复用了实际历史CI组件的66个运行载荷和已有732项索引，固定原三条任务的公共与未修改研究消费完整记录一致、各6份参考资产保留；没有重跑Agent或评分，也没有修改研究仓默认依赖。原生Agent仍使用前述独立环境与执行入口，公开执行组合及其余保存日志异常原因尚待核验，不把这次读取兼容性验证算作原生推理完成。

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

三种格式的正常1、空输入0、外部HTTP/S资产阻塞，以及SVG兼容性，共10项CPU隔离控制已通过。其后六组协议各固定前三题实际完成18道原题、36次模型调用、158对完整VQA；Canvas三题VQA均0，原最终分0.29/0.3/0.3如实保留。Text-to-HTML和Markdown-to-HTML各另执行剩余47题，每组累计50题/100次stop调用，分别345/439对原VQA；React-to-HTML另执行剩余42题，累计45题/90次stop调用/408对原VQA。原字段、提示、提取代码及实际裁判图片字节独立核验，原转义警告与回退保留，不修补生成。其他三组剩余131题尚未计为完成。控制夹具不计原题，既有有效原题不重跑，整个StructEval来源仍未勾选。

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

## Spider官方Test Suite评分环境（尚不代表原生交付）

统一provider已注册四个原划分。安装本项目wheel后，可直接读取原解压目录； schema和`database`/`test_database`目录必须保留原布局，不把同名数据库合并。
已在服务器独立安装环境核对全部11840条，原SQL、AST及query tokens只在参考侧。

```python
from pathlib import Path
from vllm_hust_workload_data.adapters import iter_workload

for record in iter_workload("SPIDER", data_root=Path("/private/spider_data"), files=["dev.json"]):
    original_input = record.input_record()
    scorer_reference = record.outcome_record()  # 不送入模型
```

此读取不运行SQL或替代原生模型/官方评分。固定提交366f8323的CI工件12876710598已独立安装核对67份载荷、全11840条原记录和12次公共/研究回查；研究默认依赖未改。

官方评分器要求sqlparse和NLTK但未提供版本锁。本仓库提供独立CPU软件配方
`configs/workload-native/spider-test-suite-cpu.Dockerfile`，其中版本为明确披露的集成选择，
不是作者环境锁。原题、数据库和评分器源码均不打入公开仓库或软件镜像。

在服务器创建独立构建目录，复制该Dockerfile及 `scripts/verify_spider_test_suite_environment.py`；从NLTK官方固定提交获取
[punkt_tab.zip](https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt_tab.zip)
至同一目录。资源SHA256必须为 `e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106`，配方会校验。
不要把此资源或数据库提交到Git。

```bash
docker build -f spider-test-suite-cpu.Dockerfile -t spider-test-suite-software .
docker run --rm --network=none --read-only --user=65534:65534 \
  --cap-drop=ALL --security-opt=no-new-privileges --cpus=1 --memory=512m \
  --pids-limit=64 --tmpfs=/tmp:rw,nosuid,nodev,size=32m spider-test-suite-software
```

已在专用ARM64 CPU工作区构建并通过上述非root/只读/断网软件检查；这一步没有执行SQL、 原题或模型。后续仍须隔离运行未修改的官方评分器，使用公开dev对应的完整测试套件，
再验证实际NPU生成和公共入口；不能用本检查替代Test Suite Accuracy。
详见[阶段证据](../docs/workload-data/validation/spider-original-preparation-20261029.json)。

原数据和测试套件已按作者目录解压后，可用以下入口准备固定原始dev前缀；默认前三题， 全量使用`--count 1034`。此命令校验完整原套件ZIP及选中数据库的全部SQLite成员，
不执行SQL，也不生成提示。输出目录必须是与原文件分离的新私有目录。

```bash
python scripts/prepare_spider_original_dev.py \
  --data-root /private/spider_data \
  --suite-root /private/test-suites/database \
  --suite-archive /private/testsuitedatabases.zip \
  --output /private/spider-dev-first3 --count 3
```

`inputs.private.jsonl`只含原问题、原schema和位置；`references.private.jsonl`保留七个原字段，
`gold.private.txt`是官方评分格式的原SQL和db_id。参考文件不得送入模型。 服务器已用原前三题和完整32份数据库完成该准备，并在非root/只读/断网容器运行原评分器的
参考答案控制（1.000）；这不是模型成绩。后续已另完成真实NPU原前三题和公共评分入口验证： 三次原响应均stop、原输出未经SQL修补，完整官方32库Test Suite
Accuracy为1.000。 这仅证明固定三题的执行链路，不代表全dev模型质量；固定CI工件与研究消费闭环已完成，原推理结果复用未重跑。

公开生成入口如下（默认只计划，执行须加`--execute`并指定新私有输出目录）：

```bash
python scripts/run_spider_original_generation.py --prepared /private/spider-dev-first3 \
  --model Qwen/Qwen3.5-9B --max-tokens 4096 \
  --output /private/spider-generation --execute
```

该提示与4096输出预算是明确披露的零样本接入配置，不是官方统一LLM模板或预算。 原问题/schema保持不变；参考文件不读取；多行或异常输出保留并阻塞，不剥代码块或换题。
默认只允许本地`/v1`模型服务，服务部署与资源窗口由操作者确认。

评分资产目录须包含`author/`（官方`taoyds/test-suite-sql-eval`固定提交
`e97acc546ecbee8fa27fa8dbf025ef61493a876c`）、`inputs/`（上述准备入口输出）和
`database/`（原完整测试套件）。未选数据库可以保留；选中数据库不得删减或增加变体。 官方源码不随本仓库分发，入口会校验五个运行模块的哈希。使用上节构建并核验的本地镜像ID：

```bash
python scripts/run_spider_official_scoring.py --assets /private/spider-scoring-assets \
  --predictions /private/spider-generation/predictions.private.txt \
  --image sha256:YOUR_QUALIFIED_LOCAL_IMAGE_ID \
  --output /private/spider-scoring-output --execute
```

不加`--execute`只读核对并输出计划。执行仅支持具备权限的Linux Docker环境，容器非root、
断网、只读，原SQL不在宿主执行；保留原值和DISTINCT，不注入gold值。低分正常保留； 资源/输出契约失败不冒充模型0分；容器与证据保留，无自动清理或宿主执行回退。

## SWE-bench旧格式原任务链路（两来源分别固定抽样验证）

SWE-bench原版12字段、Verified13字段的旧格式原任务的准备需要匹配harness版本：现有新版配置要求额外的镜像/评测脚本字段，不能为跑通自行补造。公共软件准备器新增`--profile v4.1.0`，固定官方发布版87份原文件；默认`current`保持不变。可先运行`python scripts/prepare_swebench_author_source.py --profile v4.1.0 --author-root /external/swebench-v410`查看固定源码归档地址；取得原软件归档后加`--archive /external/official-source.tar.gz --action prepare`核验准备。Verified原顺序前三题的ARM仓库镜像、原Agent执行与官方评分已完成，完整范围和失败结果见下文；两来源各自固定原样本均已完成，交付各有独立回执，不互相套用完成状态。

官方Agent软件复用相同准备入口，增加可选`--profile mini-swe-agent`，固定原78份源码/配置及完整性身份，不自行编写替代Agent：

```bash
python scripts/prepare_swebench_author_source.py \
  --profile mini-swe-agent --author-root /external/mini-swe-agent
# 从上述计划给出的固定官方归档地址取得软件后：
python scripts/prepare_swebench_author_source.py \
  --profile mini-swe-agent --author-root /external/mini-swe-agent \
  --archive /external/official-mini-agent-source.tar.gz --action prepare
```

这只准备原软件，不安装、不启动容器、不调用模型。服务器实际原文件核验、独立客户端差量安装/依赖检查、原CLI帮助及原Docker环境类的隔离控制通过；原题Agent和评分的固定抽样结果见下文，不能将软件准备等同任务执行。后续按[官方SWE-bench入口](https://mini-swe-agent.com/latest/usage/swebench/)及[原Docker环境配置](https://mini-swe-agent.com/latest/reference/environments/docker/)接入；ARM镜像不能静默套用默认x86镜像名。本地NPU模型是部署适配，不是官方默认Claude的等价质量复现；原题、模板、步骤预算与测试应保持不变，安全隔离参数需单独核验。可信软件构建所用联网方式不得继承到生成代码的评分容器。

原生执行分为原仓库镜像准备、官方Agent、官方功能评分三个入口。原v4.1.0镜像CLI没有架构参数且默认x86_64；本入口显式使用其原生`make_test_spec(..., arch=...)`与原构建器，不改Dockerfile、安装脚本或任务。示例使用同一份原始Parquet，不要求自行抽取问题或构造prompt：

普通SWE-bench的固定原测试文件已单独登记：2,294条、12字段、12,097,227字节；它不是Verified的500条。获取、复核沿用现有公共入口，原始文件仅下载到调用者指定的仓库外目录：

```bash
python scripts/prepare_reused_research_assets.py SWE_BENCH --mode plan
python scripts/prepare_reused_research_assets.py SWE_BENCH --mode download \
  --data-root /external/swebench-original --max-bytes 13000000
python scripts/prepare_reused_research_assets.py SWE_BENCH --mode verify \
  --data-root /external/swebench-original
```

已存在的同版文件直接用`verify`，不重复下载。SWE_BENCH固定原顺序前三题为Astropy11693/12057/12318，均不在本次Verified原生样本中；后续镜像、Agent、评分命令使用`--source-id SWE_BENCH`及该来源的原Parquet，不能借用Verified的任务结果。该来源原三题镜像与参考补丁环境对照已完成，对照3/3通过；实际CI工件已完成全2294条字段/索引与固定3条研究回读。固定同一原三题的原Agent及官方评分已完成：首题非法补丁保留不重跑；原两题连接故障恢复后完成官方测试但均未修复，旧失败尝试全部保留。原字段、顺序、模板和250步预算不变，组合验证不冒称单轮官方榜单结果，非全2294题评测。来源交付与最新已核验代码CI见[独立回执](../docs/workload-data/validation/swebench-classic-native-delivery-20261030.json)。

```bash
python scripts/prepare_swebench_original_images.py \
  --source-id SWE_VERIFIED --records-file /external/original.parquet \
  --harness-root /external/swebench-v410 --arch arm64 \
  --docker-url unix:///external/private-docker.sock \
  --output /external/new-image-preparation
```

默认仅计划；追加`--execute`才准备镜像，并复用已有镜像，不强制重建或清理。仅在可信软件构建确有网络需要时指定`--build-network host`，不改变Docker全局网络，更不用于生成代码的运行容器。服务器原镜像冷构建已完成；该公共入口又实际复用三份镜像通过检查、构建API调用为0，不把缓存检查冒称再次冷构建。

```bash
python scripts/run_swebench_original_agent.py \
  --source-id SWE_VERIFIED --records-file /external/original.parquet \
  --agent-root /external/mini-swe-agent --harness-root /external/swebench-v410 \
  --docker-url unix:///external/private-docker.sock \
  --endpoint http://127.0.0.1:19501/v1 --model Qwen/Qwen3.5-9B \
  --arch arm64 --output /external/new-agent-results
```

默认仅计划，不导入原Agent、不创建输出或容器、不调用模型。`--records-file`支持原Parquet与原字段JSON数组，旧`--records-json`参数仍可用；不是抽取问题后重写的prompt。默认取原物理顺序前三题，`--all`保留全部输入题目。Parquet入口已实测全部500条/13字段与原文件逐值一致，前三题导出字节与实际原生运行的JSON快照完全相同。执行前须准备对应原仓库镜像、确认研究任务空闲并取得模型独占窗口，再追加`--execute`；输出目录必须是仓库外新目录。

该入口直接调用固定官方`process_instance`，保留原模板、工具循环、250步预算、提交与轨迹，镜像绑定采用原框架支持的部署字段；参考补丁和测试不拼进模型任务。使用本地兼容模型接口；远端自有引擎可由调用方授权的回环转发接入，不内置平台凭据。价格信息缺失不当作已测量成本。真实执行发现固定LiteLLM版本还需可选`mcp`运行时，服务器补入官方`mcp==2.3.0`及必要新依赖后，原有依赖版本不变、`pip check`通过；CLI帮助通过本身并不能发现这一运行时缺口。

跨机连接应在Agent执行主机上由独立终端或服务管理器持续托管，不要依赖会随单次自动化调用结束的临时进程。以下`npu-serving`为调用者自行授权并核验主机密钥的SSH别名；目标端口必须是调用者有权使用的服务。共享设备若已有租约或验证代理，应连接该代理的指定端口，不绕过资源互斥：

```bash
# 在独立终端持续运行，或由本机服务管理器托管；只监听回环。
ssh -N -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:19501:127.0.0.1:19466 npu-serving
# 在另一终端确认连接与服务模型身份；这不是模型任务验证。
curl --fail --max-time 10 http://127.0.0.1:19501/v1/models
```

连接故障与模型质量失败必须分列。已有完整原生终态的任务，包括未修复或非法补丁，不为提高分数而重跑。若连接中断使任务未完成，保留其原轨迹、失败尝试与已完成任务结果；仅恢复同一固定样本中的故障原题，原ID、全部字段、顺序、模板和预算不变，使用新的外部输出目录。优先使用原框架的恢复能力；若需重新执行完整原流程，必须记录这不是无损续跑及实际重复调用成本，不能把未完成题换成更易通过的题。

生成补丁后使用同一原数据与其原参考测试评分，不重新生成模型输出：

```bash
python scripts/run_swebench_original_grade.py \
  --source-id SWE_VERIFIED --records-file /external/original.parquet \
  --harness-root /external/swebench-v410 --arch arm64 \
  --docker-url unix:///external/private-docker.sock \
  --predictions /external/new-agent-results/preds.json \
  --output /external/new-original-grading
```

默认仍仅计划，追加`--execute`才执行未修改的官方`run_instance`及报告逻辑，保留原1800秒逐题测试时限。评分SDK边界仅增加容器隔离和限额：断网、cap0、NNP、无宿主挂载/设备、8GiB/2CPU/512PID；原题、补丁应用策略、测试、容差和判分不改。环境正对照可将`--predictions`替换为`--gold-control`并使用另一新输出目录，使用原参考补丁核验环境，必须与模型结果分开。

Verified验证范围：固定原前三题全部由原Agent提交，实际NPU请求144次；原参考补丁对照3/3通过。Qwen3.5-9B实际结果为一题修复、一题未修复、一题被官方工具拒绝非法补丁，失败输出未修补、未换题或重新生成。已从保存的原日志重新计算评分，并核对初始模板、任务文本、提交、原测试脚本和参考测试ID；这不是完整500题准确率，也不是引擎性能提升数据。SWE-bench原版有单独的原数据、三题原生结果及交付回执，不能互相借用完成状态。

可复用环境清单：[评分客户端76包](../configs/workload-native/swebench-original-core-py312-aarch64.lock)、[Agent完整客户端120包](../configs/workload-native/swebench-original-agent-py312-aarch64.lock)，均为本次真实CPU客户端使用的官方PyPI包及SHA256，目标是Python3.12/Linux
aarch64。请在独立虚拟环境使用`pip install --require-hashes -r <所选lock文件>`，再设置上述固定原作者源码根目录；不要安装进现有NPU服务环境。120包版本已逐项与实际导入环境核对，原生运行使用只读基础环境加独立差量层；未额外重跑一次全新扁平环境安装。原仓库镜像内Python3.9及其依赖仍由原安装脚本管理，不套用宿主客户端lock。

## DRAGON模型客户端与原生执行（固定前三题已验证）

原作者独立示例含CUDA初始化，不能直接在NPU节点照搬执行。检索、原提示与评分核心可复用， 但必须另行验证NPU
embedding与现有reader服务的薄传输层，不能换成CPU或随机embedding。固定原前三题已经通过实际执行和保存核验，范围及限制见下文；软件导入本身仍不代表原生验证。

已验证的Python3.12/aarch64客户端包记录在
`configs/workload-native/dragon-npu-model-client-py312-observed.lock.txt`（56项、PyPI原始工件哈希）；
这是观察到的集成版本，不是作者原始锁。先在已有合格NPU Python运行环境核对保护版本， 只向仓库外的新虚拟环境安装，不能在服务环境直接pip安装：

```bash
python scripts/probe_dragon_native_environment.py \
  --runtime-profile configs/workload-native/dragon-npu-inherited-runtime-py312-observed.json
python -m venv --system-site-packages /external/dragon-client
/external/dragon-client/bin/python -m pip install --require-hashes --no-deps \
  -r configs/workload-native/dragon-npu-model-client-py312-observed.lock.txt
```

版本不符时应在自己的隔离客户端重新解析，不能为复用这个观察锁而更改运行中的服务。 原`lib/requirements.txt`保持不变。HuggingFace接入需full
extra，不能漏掉sentence-transformers； NumPy1.26保护配置下使用已实测可导入的Datasets4.8.5/Arrow22组合。继承环境pip check仍有11项
已有SDK/工具依赖问题，修复后客户端未新增问题，不声称整个服务SDK依赖检查通过。

需自行解析时，`scripts/prepare_dragon_model_client_lock.py`可将pip报告转为带哈希的确定计划，
会拒绝保护包替换、非官方PyPI工件、凭据URL及缺失embedding依赖；该工具不执行安装。
详细失败、修复及证据见[阶段记录](../docs/workload-data/validation/dragon-original-preparation-20261028.json)。

### DRAGON原生执行入口

`scripts/run_dragon_original.py`复用固定作者核心，不是重新实现RAG：完整542篇语料，原CLI的字符分块1000/重叠100、Chroma
MMR/top-k5，原俄语提示、模型自带chat模板和原评分器。默认原顺序前三题，`--full`执行全部600题；参考QA仅用于原评分，不加入检索池或模型提示。所选E5原工件固定在`configs/workload-native/dragon-e5-original-artifact.json`；保留原Transformer/Pooling/Normalize及512-token编码器窗口。执行使用原standalone
CLI的无额外doc/query指令配置，不混称另一套YAML的k20/chunk500配置或发布方模型分数复现。

原数据使用前述公共获取/核验入口；模型可用已验证客户端自带的官方Hub CLI取得固定版本，排除重复ONNX导出，运行时逐文件核对清单中的11份原文件。已有相同外部模型目录时直接只读复用，不重复下载：

```bash
/external/dragon-client/bin/hf download intfloat/multilingual-e5-large-instruct \
  --revision 274baa43b0e13e37fafa6428dbc7938e62e5c439 \
  --exclude 'onnx/*' --local-dir /external/models/multilingual-e5-large-instruct
```

在独立、已核验的NPU客户端中运行，先取得模型/API所有者授权的空闲窗口。模型目录、原始数据和输出均放在仓库外。下面默认仅展示计划；加`--execute`才加载NPU编码器并发送现有服务请求，`--engine-pid`和`--context-window`必须填写实际值。代码不启动、停止或改配模型服务，也不自动抢卡或重试失败题。原reader生成预算按该模型的原`generation_config`解析，缺失时沿用作者512上限；原模板提示直接通过`/v1/completions`传输，不再次套chat模板、不截断、不减少语料。

```bash
/external/dragon-client/bin/python scripts/run_dragon_original.py \
  --author-root /external/dragon-author \
  --data-root /external/dragon-original \
  --embedding-root /external/models/multilingual-e5-large-instruct \
  --reader-root /external/models/reader \
  --output /external/runs/dragon-first3 \
  --model ACTUAL_SERVED_MODEL --device npu:0 \
  --endpoint http://127.0.0.1:19466/v1 \
  --engine-pid ACTUAL_ENGINE_PID --context-window ACTUAL_CONTEXT_WINDOW
```

使用可见设备掩码时`--device`是该进程中的逻辑NPU编号，不是物理芯片编号。作者日志默认写入当前目录，入口将工作目录定位到新私有输出，结束时恢复，不在源码目录或HOME写作者日志。生成配置和分词器在NPU加载前检查：若原模型没有独立`generation_config.json`，显式使用Transformers的`from_model_config`从原`config.json`派生并记录来源，不补造发布方文件；已有但损坏的独立配置仍失败，不做静默回落。

固定原前三题已在Ascend NPU完成：完整542篇/1157原分块，E5编码结果直接核验为NPU张量，现有Qwen3.5-9B
reader完成3次原提示请求和官方评分。3次均为原512预算下的length终止，原样保留，不代表正确作答或模型质量复现；这不是未经修改的CUDA
CLI。完整请求/响应、检索ID、原生编码结果及官方评分保留服务器，未纳入Git。提交675b7da2的两项CI均成功；实际工件67份载荷、安装导航、全部600行原字段/索引及固定3条当前研究回读通过，原生入口和保存核验器与服务器执行源码字节一致。来源已按固定原生抽样交付打勾，不是全600推理、质量或性能验收。

无推理复核入口为`python scripts/verify_dragon_original_saved.py --native-root /external/runs/dragon-first3 --author-root /external/dragon-author --data-root /external/dragon-original --reader-root /external/models/reader --output /external/audits/dragon-first3.json`。它逐字节匹配原完整模板、问题、检索分块及ID，核对全部原生编码输入/向量、原响应与原答案，调用原评分器并与保存评分比较；不会重跑嵌入、检索或reader，也不把length输出改成通过。

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

## SWE-bench／Verified 共用新版官方软件准备（仅软件核验，不替代旧版原生链路）

两来源复用同一套官方 `SWE-bench/SWE-bench` 软件，不与本批仅分类的 SWE-bench Pro 混同。固定版本及605份文件身份见
[软件清单](../configs/workload-native/swebench-author-source.json)。准备入口只获取并核验软件，不获取任务数据、仓库快照或镜像，不执行生成补丁。

在仓库外准备目录，下载原官方软件包后执行：

```bash
mkdir -p /private/swebench-downloads
curl -fL https://api.github.com/repos/SWE-bench/SWE-bench/tarball/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e \
  -o /private/swebench-downloads/author.tar.gz
python scripts/prepare_swebench_author_source.py \
  --author-root /private/swebench-author --action plan
python scripts/prepare_swebench_author_source.py \
  --author-root /private/swebench-author --archive /private/swebench-downloads/author.tar.gz --action prepare
python scripts/prepare_swebench_author_source.py \
  --author-root /private/swebench-author --action verify
```

入口逐文件校验原Git对象、SHA256和字节数，复用已匹配的文件；不同文件或已有部分文件不覆盖。只提取固定软件成员，不使用不受控的tar解压。`verify`是只读核验。

隔离核心客户端的76项依赖已在ARM/Python3.12服务器按确定哈希安装，`pip check`和未修改原评测CLI的帮助探测通过。以下是本次观察到的客户端配置，不是发布方全环境锁，也不包含任务镜像或推理extra：

```bash
python3 -m venv /private/swebench-client
/private/swebench-client/bin/python -m pip install --require-hashes --no-deps \
  -r configs/workload-native/swebench-core-client-py312-observed.lock.txt
/private/swebench-client/bin/python -m pip check
PYTHONPATH=/private/swebench-author HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /private/swebench-client/bin/python -m swebench.harness.run_evaluation --help
```

发布方推荐x86_64、至少120GB可用存储、16GB内存、8核CPU；ARM64支持仍为实验性。本节新版605份软件的准备与只读核验已完成，但它要求的数据字段与原版12字段及Verified13字段原任务不匹配，不能据此声明本节新版流程完成原生任务。已验证的Verified固定前三题使用上节v4.1.0及原mini-SWE-agent，而不是补造字段套用新版。模型推理外接NPU，不能安装CUDA推理extra后默认为NPU可用。原数据复用、不重复下载；环境、模型失败及来源边界见[回执](../docs/workload-data/validation/swebench-original-readiness-20261029.json)。

## LJSpeech 1.1 原始 TTS 语料准备（原生合成未完成）

本来源是原文本、规范化文本与单说话人波形的配对语料，不是已有自动语义评分器的 benchmark，也不以 ASR 转写代替 TTS 合成。
固定[原始文件身份](../configs/workload-native/ljspeech-original-assets.json)，下载和核验只使用 Python 标准库：

```bash
python scripts/prepare_ljspeech_original_assets.py --data-root /private/ljspeech --action plan
python scripts/prepare_ljspeech_original_assets.py --data-root /private/ljspeech \
  --action download --max-bytes 3000000000
python scripts/prepare_ljspeech_original_assets.py --data-root /private/ljspeech --action verify
```

仅在确认同一目录属于该获取器且先前传输已结束后，才用 `--action download --resume --max-bytes 3000000000`
续传。入口要求匹配的私有来源记录及精确HTTP范围；服务器忽略续传范围、返回HTML或校验失败时保留原部分文件，不自动重试、覆盖或更换来源。完整文件已存在时只读核验，不重复下载。

服务器已取得官方1.1完整原包2,748,572,632字节，并核对官方TorchAudio固定源码中的SHA256；这不是只验证下载链接。全部13,100条三字段元数据与13,100份WAV配对、RIFF长度和原音频头已核验。原包留在外部存储，不写入本机或Git；没有重复解压一份语料。

公共适配器直接流式读取该归档，保留CSV原行顺序，原`text`和发布方`normalized_text`分别保留；参考WAV以归档成员引用放在`outcome.audio`，不是模型输入或合成结果。TTS前端应按原模型协议明确选用哪一列，适配器不自行展开数字、清洗文本或构造API请求。

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

for record in iter_workload("LJSPEECH", data_root=Path("/private/ljspeech"), limit=3):
    original_text_fields = record.task["input"]
    reference_wav = record.outcome["audio"]  # 不传入TTS模型
```

归档读取会先流式核对完整的文本/音频一一对应关系，即使`limit=3`也需要扫描原压缩包；因此不要在逐请求循环内重复创建迭代器。读取不会解码WAV、执行TTS或评分。实际CI工件安装、完整13,100条原索引和固定前三条公共/未修改研究消费入口的原数据回查已通过，见[原数据交接证据](../docs/workload-data/validation/ljspeech-actual-ci-handoff-20261030.json)；这不替代真正的NPU语音合成与适用输出核验。

VITS发布方提供对应LJ
Speech的模型及原推理示例，可作为候选，但其CUDA/PyTorch1.6环境不等于已适配NPU；尚未取得并固定检查点，不宣称原生合成已运行。没有自动MOS分数可凭空补造，波形非空或格式合法也不能代表语音质量。[当前证据与缺口](../docs/workload-data/validation/ljspeech-original-readiness-20261029.json)持续更新，来源总勾保持未完成。

## MultiAgentBench 原始场景获取（原生执行未完成）

官方MARBLE固定版本及原文件身份来自机器清单，获取入口不生成prompt、不改任务或关系。当前覆盖bargaining、coding、database、minecraft、research五份原JSONL，每份100条；不能据此声称包含官方全部环境或全部任务。先查看计划：

```bash
python -I scripts/prepare_multiagentbench_original_assets.py --data-root /private/marble-original
```

显式下载加`--action prepare --max-bytes 5369142`；已有原文件用`--action verify`，不重复下载。所有文件同时核对固定Git
blob、SHA256及大小，已有损坏文件或partial保留并报错，不覆盖重试；目录必须位于仓库之外。服务器已通过公共入口复用五份原文件验证，但尚未核验各场景完整输入/答案关系、官方Agent运行环境、模型/裁判绑定、原生执行及评分、公共/研究消费。来源仍不打勾。

原记录包含agents、environment、relationships、metrics等字段，Minecraft没有其他四份文件共有的task_id/scenario，database另含communication；不能强造同一静态问答格式或编造缺失ID。多Agent协作/竞争在当前分类中是跨任务关系，不另造第41类。

服务器已检查全部500条原记录的字段形状；这不等于完成语义适配或原生运行。Minecraft的原任务标识位于`environment.task_id`。database的`task.labels`、`task.root_causes`及`task.number_of_labels_pred`由原引擎传给数据库评估器，须保持参考/评分侧关系，不能将整个task字典拼入模型输入；各Agent私有配置也不能广播成共同prompt。恢复原生运行配置与构造模型输入是两回事。

统一读取及还原原配置：

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.multiagentbench import restore_original_configuration

for record in iter_workload("MULTIAGENTBENCH", data_root=Path("/private/marble-original")):
    original_config = restore_original_configuration(record)
    # 交给原MARBLE运行器；不要把整个配置当作共同聊天prompt。
```

公共适配器已在服务器对五文件500条原记录逐条还原相等，原生关系、未知扩展字段及数据库评分参考未丢失；`request_body()`明确拒绝此类记录。记录位置ID采用文件摘要与原行号，不冒充原始task_id。统一pipeline建立500条纯引用索引后，公共与未修改研究消费者逐条读取，完整记录相等、原配置还原相等，读取期间索引字节不变。研究代码及默认依赖未修改，也未恢复研究任务。随后实际CI运行41039829015的工件12970235878在服务器独立目录安装，72个运行载荷与Git及安装文件逐字节一致，复用原500条索引的公共/研究完整回读再次通过。仍没有原生Agent运行或官方评分，不得由数据消费打来源总勾；历史源码版证据保持原范围，详情见机器清单的`actual_ci_artifact_handoff`。

原生运行尚未就绪：固定源码要求Python`>=3.9,<3.12`及原`poetry.lock`。`scripts/research/run_simulation.sh`使用`--config`，但当前`marble/main.py`实际解析`--config_path`；不可直接把示例运行成功作为已验证事实。模型调用经LiteLLM，保留原默认512-token预算及温度0；本地端点绑定仍需独立验证。`metrics.evaluate_llm`未配置时评分器回退到外部`gpt-3.5-turbo`，通信和规划评分均会调用模型，不能在未授权时直接启动。embedding另走`litellm.embedding`，已有聊天服务不代表具备该能力。以上为源码核查，不是环境安装、模型兼容或评分实跑证据。

服务器已按原Git对象校验99份原软件文件（1420037字节），其中94份Python仅做静态解析，未导入或执行。固定原`marble/evaluator/evaluator.py`第324行的`except json.JSONDecodeError`缩进导致`SyntaxError`，服务器及直接原Git对象解析均复现；原文件完整保留；初始检查尚未安装依赖，批准的兼容副本及客户端最新进展见下文。须先找到与原任务兼容的正式可用版本，或明确审批并披露最小源码兼容修复，再验证原生运行；不能将该阻塞归咎于模型或靠删评分绕过。

官方软件的公共计划/校验入口如下；当前固定100文件的版本、大小、Git对象和SHA256在`configs/workload-native/marble-author-source.json`，源码目录必须与运行输出分离。原99份之外补入同一版本的951字节`multiagentbench/runjsonl2yaml.sh`，不改已有软件：

```bash
python -I scripts/prepare_jsonschemabench_author_source.py --source marble \
  --author-root /external/marble-author
python -I scripts/prepare_jsonschemabench_author_source.py --source marble \
  --author-root /external/marble-author --action verify
```

已有缓存直接校验，不下载、不导入。需要取得缺失软件时可显式使用`--action prepare --max-bytes 1420988`；发布方Git接口可能限流，失败保留已核验文件，先处理访问条件，不能删缓存盲重试或换浮动版本。当前公共`verify`已在服务器复用原99文件、仅补入上述官方脚本后通过100文件校验；不据此宣称公共下载命令已重跑或原生执行成功。

经明确批准，已增加独立的最小语法兼容副本入口，不再把这个语法错误视为无可处理的上游阻塞：

```bash
python scripts/prepare_marble_evaluator_compat.py \
  --source /external/marble-author/marble/evaluator/evaluator.py \
  --output /private/marble-evaluator-compat
```

默认只显示计划；加`--execute`才在全新独立目录生成`evaluator.py`、`COMPATIBILITY.patch`及回执。入口只接受已固定原文件SHA256，拒绝猜测修复；仅移除原324—326行各四个前导空格，原文件及其他字节不变。服务器对真实原文件完成静态解析，并仅提取原`parse_research_ratings`函数验证有效JSON、无JSON、非法JSON三个分支。完整模块尚未导入、原生Agent和官方评分未运行；运行配置必须显式披露兼容副本及其摘要，不能宣称“未经修改的官方代码”。此工具不自动替换原作者目录或升级环境。

客户端预检确认ARM64及Python3.10符合原版本范围；默认Python3.12不满足，不能直接混装。153包原锁的CUDA/Triton依赖仅适用Linux
x86_64，须按原平台标记选择。独立Poetry2.1.3工具及Python3.10.12客户端已准备：原锁`poetry install --only main --no-root`退出0，当前平台main组合实际93个分发包，版本均在原锁中，依赖元数据检查通过；没有CUDA/Triton包，原项目及锁文件未修改。安装后仍余约6.17GiB，未触发5GiB安装余量门限。

平台继承的`PYTHONPATH`会使普通解释器列出环境外的CANN包；已用客户端`python -I`核验隔离后的93包，没有删改平台包。后续启动应显式清理本进程的继承路径，或使用隔离解释器并只加入已核验作者目录，不能将平台包混入原锁环境。这里只证明安装及依赖元数据，不证明完整模块导入、模型/embedding兼容、Agent执行或官方评分；模型调用仍为0。

随后真实入口导入暴露`coding_utils/coder.py`依赖原锁未声明的`ruamel.yaml`。已单独固定[客户端补充依赖](../configs/workload-native/marble-client-supplements.txt)：`ruamel.yaml==0.18.6`及`ruamel.yaml.clib==0.2.12`，从官方PyPI取得，未改原93包版本或原锁。可在完成原锁安装后，针对显式客户端解释器执行`uv pip install --python /external/marble-client/bin/python --no-deps -r configs/workload-native/marble-client-supplements.txt`，再检查依赖元数据。当前客户端共95包，不冒称全部来自原锁。

应用明确披露的评分器兼容副本后，真实`marble.main --help`及模块导入已在服务器通过，退出0、无假模块、无模型调用；最初缺ruamel的失败日志保留。预检使用Python连接审计拒绝、HF离线标记和空模型缓存，观测到0次Python
socket连接尝试；这不是内核网络隔离认证或生成代码sandbox。原生Agent任务、embedding/裁判、各场景隔离与官方评分仍待验证，不得据此打总勾。

固定原题配置准备入口（用原客户端解释器，默认仅计划）：

```bash
/external/marble-client/bin/python -I scripts/prepare_marble_original_configs.py \
  --data-root /private/marble-original --author-root /external/marble-author \
  --output /private/marble-fixed-configs --limit 3
```

加`--execute`才在全新独立目录写出五文件各原前三条，共15份配置。仅序列化为原`yaml.safe_load`可读的JSON形式YAML，不改题、模型名、预算、字段或路径；全部原字段保真，并用固定原`Config.load`验证任务、环境、Agent、关系、评分与协调配置。服务器15份原配置已通过。原生配置仍含原模型名及输出路径，尚未完成本地模型/裁判绑定和隔离运行部署，不能直接在宿主机盲跑或意外调用外部付费服务；配置准备不是原生执行。

进一步的真实离线容器预检发现：其中12份原配置的`environment.type`与`max_iterations`为空；Minecraft三份为`Minecraft`、20次迭代。`Config.load`能读入空字段，不代表引擎能运行。准备回执现单列发布方清空的运行字段，不自动补值。官方`multiagentbench/jsonl2yaml.py`提供`fill_defaults`，其CLI默认却是Research环境、3次迭代，不能不分场景地套到bargaining、coding或database。须先取得各场景的官方配置依据，再显式绑定环境、模型、裁判和输出目录；尚无任何原生任务或评分完成结论。

同版本`multiagentbench/runjsonl2yaml.sh`明确针对`research/research_main.jsonl`指定Research环境、5次迭代、graph协调、BaseMemory、原模型gpt-3.5-turbo及judge
gpt-4o。这是Research场景的发布方配置依据，不代表其他场景，也不授权调用外部付费模型。后续资源替代模型及裁判须单独披露，原题、关系与5次迭代不得为跑通而缩减。该脚本已核验但未直接执行，尚未完成原生研究场景或评分。

同版本官方模板的服务器逐字段对照还发现：`coding_configs/config_1/2/3.yaml`与原JSONL前三题并非同题，不能按编号直接替换；`test_config_world/test_config_world.yaml`与前三条bargaining的任务、Agent及关系也不一致。模板中的Coding/5次或WorldSimulation/3次只能作为公开运行设置参考，不能伪称已找回这些原题的完整配置。对照未改写原题或创建替代测试，完整原生资格仍待验证。

对已经由原`official-research`流程准备好的固定前缀，可以显式生成本地模型部署绑定，不修改原准备目录：

```bash
python scripts/bind_marble_research_local.py \
  --prepared-root /private/marble-official-research \
  --output /private/marble-local-research \
  --model openai/Qwen/Qwen3-14B --judge openai/Qwen/Qwen3-14B
```

默认只显示计划；加`--execute`才核对原准备回执、原配置摘要和完整连续前缀，在新目录只改`llm`、`metrics.evaluate_llm`、`output.file_path`。题目、Agent、关系、5轮、规划器、评分维度及未知字段保持不变，拒绝含其他Agent级模型绑定的输入。当前原前三题已在服务器通过这个公开入口及官方`Config.load`，Agent数仍为5/5/3，无新模型调用。随后原Engine构造也通过，三题均处于第0轮，无网络尝试或Agent步骤。运行工作目录须具有私有`logs/`，并能从`evaluator/evaluator_prompts.json`读取原评分提示；服务器只绑定原资源路径，未改提示，初次缺目录和资源路径的失败保留。明确披露本地执行/裁判替代，不能宣称复现官方GPT-3.5/GPT-4o结果；替代裁判资格、模型请求传输和完整原生运行仍待验证。本入口不会启动服务、执行Agent或修改环境。

Research专用准备可在上述配置准备命令追加`--runtime-profile official-research`，并提供全新输出目录。默认仍只打印计划；加`--execute`后仅选原Research文件前N条，读取已固定官方脚本的字面量参数，通过原`fill_defaults`填入发布方清空的运行字段。入口检查原task、agents、relationships不变，不运行shell、模型、环境或评分。原官方模型与judge名称保留，因此生成配置不能直接盲跑。服务器原前三条已准备并在离线、非root、只读容器构造官方ResearchEnvironment成功，5次迭代及各自Agent数保留；这仍不证明检索工具、原生步骤或评分可用，后续须显式绑定已核验的本地模型、裁判、网络工具和私有输出。

软件隔离镜像配方为`configs/workload-native/marble-compat-cpu.Dockerfile`。构建上下文只允许两个软件子目录：`client/`为已核验的Python3.10客户端（原锁93包＋披露的2包），`author/`为99份原软件及明确披露的评分器缩进兼容副本；禁止带入原题、答案、模型缓存、结果或凭据。操作者须先核验已有Ubuntu22.04基础镜像的不可变ID，显式传给`MARBLE_BASE_IMAGE`，不能把浮动tag当身份依据。

Research关键词工具还会调用KeyBERT默认的`sentence-transformers/all-MiniLM-L6-v2`辅助模型。在要求模型只用NPU的部署中，不能让这个分支静默回落CPU推理。独立辅助客户端的实际版本约束见`configs/workload-native/marble-keybert-npu-client.txt`：先由平台提供匹配的厂商Torch/Torch-NPU构建，再在独立客户端安装其他依赖，不能升级现有服务或原MARBLE
CPU环境。该文件是已观测的软件兼容组合，不是完整CANN/驱动安装脚本。辅助模型固定版本与资产校验在机器清单记录，原始权重不进入仓库。

原`get_paper_by_arxiv_id`已在既有独立Python3.10客户端实测返回所请求论文；运行前核对固定原文件摘要，原请求及重试参数未改。此检查使用独立工具控制例，无模型调用，不属于原Agent题目，不代表其他检索工具、KeyBERT
NPU前向或Research原生执行通过；具体回执见机器清单。

```bash
python scripts/probe_marble_auxiliary_client.py \
  --client-python /external/marble-keybert-npu/bin/python
```

默认仅计划；加`--execute`才核对独立客户端11项版本、`pip check`及真实库导入，不安装、不下载或加载模型，不初始化设备。公共入口已在服务器已准备环境执行通过，虚拟环境解释器路径不解析回基础解释器。另一次独立控制测试已核验固定原MiniLM全部11份资产并完成真实FP32
NPU前向：模型参数与前向输入设备均核对，输出为NPU上的384维向量，保留原256-token模型上限。随后在同一专用环境补入[原锁版本的5项关键词依赖](../configs/workload-native/marble-keybert-npu-supplements.txt)，仅对这些包使用`--no-deps`安装，依赖检查通过；未升级原核心库或修改服务环境。KeyBERT原1—3
gram、英文停用词的独立控制调用也通过，共观察到3次NPU神经前向，返回5个关键词。控制测试不是原Agent题目，Research工具的远程辅助模型绑定尚未接通，因此不能据此认为原生Research任务或评分已完成。CI只检查计划和夹具，不导入模型库或占卡。

辅助模型的可复用单次工具后端如下。先完成上述独立环境和固定模型资产准备，并确认所选NPU空闲、取得共享执行窗口。`original-tool-query.json`只包含原工具收到的`query`字符串，不重写查询、不把参考答案拼入输入：

```bash
/external/marble-keybert-npu/bin/python scripts/run_marble_keybert_npu.py \
  --model-root /private/all-MiniLM-L6-v2
ASCEND_RT_VISIBLE_DEVICES="$OWNED_NPU_ID" \
  /external/marble-keybert-npu/bin/python scripts/run_marble_keybert_npu.py \
  --model-root /private/all-MiniLM-L6-v2 --execute < /private/original-tool-query.json
```

默认不读取输入、不加载模型；执行入口以[固定原文件身份](../configs/workload-native/marble-minilm-original-assets.json)核对模型，禁止远程代码和CPU神经前向，输出原KeyBERT关键词及实际前向次数。它是单次工具后端，不自行开放网络服务、分配卡或接管模型服务，也不等于Research工具已经接入此后端。跨机传输和原Agent调度仍须继续接通，不能以本命令代替原生多Agent流程。

原Research模块的显式队列绑定在`scripts/marble_keybert_transport.py`。在原工具调用所在进程中使用`keyword_transport(paper_collector, private_queue)`上下文，只临时替换该模块的默认KeyBERT工厂；退出上下文恢复原工厂，不修改发布方文件。原查询和1—3
gram/英文停用词设置保持不变，关键词及分数按原tuple格式返回。队列复用既有原子写入，每次调用一个UUID；可信跨机worker将该请求送至上述NPU入口，再将返回JSON写入同UUID的响应文件。超时保留请求并报错，不自动重发；worker不得把它接到公开端口或允许任务代码访问SSH凭据。

实际跨机控制已确认原`get_related_papers`消费了NPU返回的5个关键词，原查询两端摘要一致，无本地原始数据文件。同一进程的后续论文获取约423.85秒后返回1篇论文，未重发请求；后台启动器未保存退出码，因此只报告已观察到原工具返回和进程结束，不补造退出码。中继多打印的一枚末尾换行已披露，独立核验去除末尾换行后字节一致，JSON值和关键词分数不变；后续中继采用原字节写入。该控制不是原Agent题目，也不是Research场景完成，完整原生Agent与评分仍待验证。相关CI只运行夹具及协议检查，不访问arXiv或分配NPU。

本配方已在专用ARM64 CPU工作区构建，并通过非root65534、无网络、只读根、去全部能力、禁止提权的真实容器`--help`探测，退出0；未挂载任务数据、Docker
socket或NPU。任务运行还必须另行部署私有数据/输出、本地模型传输、各场景原环境和评分；镜像只证明软件导入底座可用，不代表Minecraft、数据库、研究工具等环境已齐备，也不是安全认证或原生任务完成。旧容器、镜像和结果未清理。

### Research原题运行入口（真实推理尚未启动）

`scripts/run_marble_original_research.py`要求既有独立Linux/Python3.10原锁客户端。默认只显示计划；软件和配置预检如下：

```bash
/external/marble-client/bin/python -I scripts/run_marble_original_research.py \
  --author-root /external/marble-author \
  --prepared-root /private/marble-official-research \
  --bound-root /private/marble-local-research \
  --output /private/marble-native-preflight --prepare-only
```

该命令核对固定100份原源码、批准的三行缩进补丁及原前三题部署绑定，创建独立软件副本、原资源路径、私有日志目录和运行配置；不会运行工具、模型或Agent。服务器已实际执行通过。源码/原配置不原地修改；输出必须是独立新目录，不得放入仓库或任何输入目录。

确认NPU服务和可信worker可用后，将输出改成另一个全新私有目录并使用`--execute`。入口依次调用原`marble.main`，保留原题、Agent、关系、5轮协议及原函数各自生成预算；不以单次问答替代Agent。`model-queue/`使用既有`appworld_model_transport.py`协议，`keyword-queue/`使用上述关键词协议，均由可信worker对接NPU。运行器本身不会分配设备、恢复模型、提供SSH凭据或启动这些外部worker；未连接worker时不要开始原题运行。模型proxy仅监听本进程的`127.0.0.1:14005`，端口须空闲，外部推理服务不能指回该proxy形成环路。

每题保留原stdout/stderr、结果JSONL及子进程PID/出生标记；正常失败保留并继续固定剩余原题，不补换题。任一CLI失败或缺失结果文件，运行器最终返回非零；低质量答案本身不等于CLI故障。运行终态仍需核验原评分、实际请求及研究消费，不能直接给来源打勾。当前仅准备入口实测通过；最新NPU入口连接拒绝，尚未启动这三题的真实推理。

## LongMemEval M 原检索与读者入口（问答评分尚未完成）

[固定官方版本](https://github.com/xiaowu0162/LongMemEval/tree/9e0b455f4ef0e2ab8f2e582289761153549043fc)区分全历史和检索增强问答；M过长，不把全历史超限当作原检索路径不可用。原数据在服务器仓库外按发布方方式取得，本仓库不打包原题或权重。原软件可独立准备和核验：

```bash
python scripts/prepare_jsonschemabench_author_source.py --source longmemeval --author-root /private/LongMemEval-author --action prepare --max-bytes 81752
python scripts/prepare_jsonschemabench_author_source.py --source longmemeval --author-root /private/LongMemEval-author --action verify
```

BM25客户端与模型服务使用分离环境。本次实际环境为Python3.12.13、平台vendor
Torch2.10.0+cpu/torch-npu2.10.0.post4，依赖见`configs/workload-native/longmemeval-bm25-client-requirements.txt`。先准备匹配的CPU/vendor
Torch，再安装该文件；不要覆盖已有推理服务环境，也不要隐式拉取CUDA依赖。此客户端中的神经库只是原脚本顶层导入依赖：固定`flat-bm25`、`session`、无扩展分支不执行神经模型。不是原全量CUDA环境复现，也未验证dense检索或索引扩展。

全量原生示例（示例输入是发布方完整M文件；以下`--full`与完整检索日志配套）：

```bash
mkdir -p /private/longmemeval/retrieval
PYTHONPATH=/private/LongMemEval-author python /private/LongMemEval-author/src/retrieval/run_retrieval.py --in_file /private/longmemeval/longmemeval_m_cleaned.json --out_dir /private/longmemeval/retrieval --retriever flat-bm25 --granularity session --index_expansion_method none --index_expansion_result_join_mode none --index_expansion_result_cache none --outfile_prefix original-m
python scripts/run_longmemeval_original.py --source /private/LongMemEval-author --data-root /private/longmemeval --variant m --retrieval-log /private/longmemeval/retrieval/original-m_retrievallog_session_flat-bm25 --full --api-base http://127.0.0.1:19468/v1 --model Qwen/Qwen3-14B --context-window 131072 --output /private/longmemeval/reader-preflight --disable-thinking --prepare-only
```

显式去掉`--prepare-only`并选择另一个全新输出目录，才执行生成。原全历史入口默认不变，不传`--retrieval-log`仍保留完整历史并禁止截断。RAG分支核验所有原字段、文档内容、时间戳和重复次数；保留原重复会话ID，不自行去重。读者调用原`prepare_prompt`的`flat-session`分支，top10、JSON、双侧消息、原CoN800-token预算不变，不允许额外截断。`--disable-thinking`仅是明确的兼容模型绑定，分词计数与生成使用相同设置，不是作者模型质量复现。

已验证范围：固定原前三题的482/472/473个完整会话经原BM25退出0；公开入口准备及分词通过，输入34366/31034/30916
tokens；同三题在NPU上各生成一次，全部`stop`。不是500题全量推理通过。三个问题都是助手侧证据题，官方用户侧索引的聚合检索统计排除它们；空正例`recall_all=1`不能解释为检索成功。原生问答评分仍待完成，Oracle/S既有六条生成保留复用。

发布方`src/evaluation/evaluate_qa.py`及`print_qa_metrics.py`同时按原字节提供。它的原CLI只列出指定GPT-4o及本地Llama3.1-70B评估模型；使用前应准备真实对应的模型/端点或授权，不能把Qwen服务伪装成70B。当前本地14B绑定在原提示/10-token预算的空答案负例中未通过自检，因此未接受其正式分数。以上生成结果不冒称已有官方质量分数。原题、检索日志、参考答案及生成结果均只放服务器私有目录。

## ScienceAgentBench 原科学数据资产核验（原生任务尚未执行）

HF上的原`verified`
Parquet是102题的任务元数据，不能用`dataset_preview`代替真实科学数据。配套文件须使用[发布方verified版说明](https://github.com/OSU-NLP-Group/ScienceAgentBench/blob/c26e151ed601ba109dc4d35e057ff8e73fec469d/README.md)链接的`benchmark_verified.zip`，不能混用旧validation归档。发布方要求不在网上再分发解压数据；本仓库仅保留身份、入口和说明，不放入数据、参考程序或临时下载授权URL。

当前在专用CPU服务器取得原前三题1/2/3对应的完整`clintox`、`mat_diffusion`、`crystalline_compound`目录，以及三份原参考程序、三份评价脚本、三份rubric，共14文件1006008字节。通过匿名发布页面给出的原下载地址取得文件，没有账号登录；每次Range读取绑定同一ETag并核对Content-Range，原ZIP
CRC和保存后SHA均复核通过。未下载整份1.77GB归档，数据文件未落入Windows或Git。只完成六份Python的静态解析，没有运行其中任何程序。

复用既有原资产时执行以下入口，不会下载、重构数据或执行代码：

```bash
python -I scripts/verify_scienceagentbench_original_assets.py \
  --metadata-root /private/sab-original-metadata \
  --asset-root /private/sab-original-assets --action verify
```

`metadata-root`保留原`data/verified-00000-of-00001.parquet`；`asset-root`保留原`benchmark/datasets/`、`benchmark/gold_programs/`、`benchmark/eval_programs/`、`benchmark/scoring_rubrics/`结构。省略`--action verify`仅显示计划。入口核对固定原Parquet及14份文件身份，并拒绝所选数据目录中缺失、额外或链接文件；完整官方归档中的其他任务目录可以保留，不计作已验证任务。服务器已实测公开入口退出0。

[身份清单](../configs/workload-native/scienceagentbench-original-fixed3-assets.json)中的成员SHA是实际取得后固定的摘要，不冒称发布方公布的SHA；其来源链路、原ZIP
CRC和归档目录身份另行记录。完整102题仍可从发布方原归档取得；当前只验证前三题资产，不是仅凭三个preview拼出的替代语料。后续还缺隔离科学环境、原verified推理入口、本地NPU模型与原评价流程；两个旧直接脚本仍写死validation，不能拿它们的默认结果当verified成绩，来源总勾保持未完成。

原既定版本的30份源码已在CPU工作区重新核验（206703字节、24份Python静态解析），不是切换到浮动上游。原`make_test_spec`已在Python3.10对前三题调用通过，12个原字段保留；任务ID仅沿用官方入口的整数转字符串处理，生成配方选择`linux/arm64/v8`。这只验证原任务规格及配方构造，不能据此宣称科学依赖已安装或Docker任务已执行。

[官方Docker说明](https://github.com/OSU-NLP-Group/ScienceAgentBench/blob/c26e151ed601ba109dc4d35e057ff8e73fec469d/docs/20250110_docker/README.md)建议峰值空间按`5 + max_workers * 25 GB`规划；单worker约30GB。当前专用Docker文件系统仅约10.6GB可用，CPU卷约5.9GB可用，未开始大镜像构建，也未删除旧镜像/结果或擅自扩容。不能用宿主卷空闲、镜像逻辑大小代替Docker实际文件系统余量。

另需完成部署隔离：原镜像配方包含API凭据ENV字段，不能把真实凭据交给生成代码；参考程序和评价资源也应与生成代码隔离。原`compute_scores`在执行程序前计算CodeBERT，须核验NPU度量后端，不能静默在CPU推理或跳过该度量来宣称跑通。当前预检清空凭据值且未执行代码，完整科学环境和评分仍未验证。

原`LLMEngine`会按模型名前缀选择OpenAI或Bedrock，`ScienceAgent`还直接查模型价格表，不能把Qwen名称填进去便宣称接通。显式本地通信模块为`scripts/scienceagentbench_local_engine.py`，仅实现原`respond(messages, temperature, top_p)`接口：使用真实本地模型标识及已拥有的回环端点，采用原GPT-chat分支的2000-token预算、采样参数和SDK重试设置，不修改消息，也不将隐藏推理字段冒充答案。该分支是明确选择的已发表生成配置，不宣称与其他原生模型分支或GPT模型质量等价。

[薄客户端版本](../configs/workload-native/scienceagentbench-sdk-client-requirements.txt)固定为OpenAI1.54.4、httpx0.27.2及backoff2.2.1；在独立Python3.10客户端安装并通过依赖检查，现有服务和客户端未升级。服务器使用真实SDK对本进程的合成回环HTTP服务完成两次控制调用，核验原消息、模型名称、2000预算、可选thinking绑定和实际返回的用量字段；这些是协议控制，不是原题或NPU推理。默认依旧保留发布方外层backoff，不能在模型不可达时盲目启动原题运行。

通信模块尚未接入原Agent的价格和tokenizer路径，不能用GPT别名、虚构单价或“成本为零”绕过价格表；未知价格应报告不可得，完整Agent绑定仍待完成。另检查了[官方OpenHands legacy入口](https://github.com/OpenHands/benchmarks/blob/542cb71c860c11d6a70238a1dd4b4ac7a3e0d4d2/legacy/scienceagentbench/run_infer.py)：它仍默认读取validation，旧镜像上下文还复制整个benchmark目录，不能直接当作本次verified数据与隔离部署已就绪的依据。未拉取旧镜像，也未切换到未经验证的替代流程。

## StreamingBench 原生流视频评测（仅软件准备，尚未完成原生验证）

官方源码固定到
`THUNLP-MT/StreamingBench@89299232c4cc1cc3721951173c5f107b48cf6877`，复用作者软件获取器，不把原始问题JSON、视频或发布方模型结果复制进仓库：

```bash
python scripts/prepare_jsonschemabench_author_source.py --source streamingbench --author-root /srv/benchmarks/StreamingBench
python scripts/prepare_jsonschemabench_author_source.py --source streamingbench --author-root /srv/benchmarks/StreamingBench --action prepare --max-bytes 74376
python scripts/prepare_jsonschemabench_author_source.py --source streamingbench --author-root /srv/benchmarks/StreamingBench --action verify
```

默认只计划；显式准备核验31份原始软件文件的Git
blob、SHA256和大小。服务器已取得全部74376字节、解析25份Python语法且公共verify命令exit0；未导入作者模型代码。回执：`/root/streamingbench-original-software-20261031-001/preparation-public.json`、`/root/streamingbench-public-source-verify-20261031-001/qualification-public.json`。软件准备不代表数据、NPU环境、推理或评分已通过。

原始问题文件获取：`python scripts/prepare_reused_research_assets.py STREAMINGBENCH`默认只计划，显式`--mode download --data-root /srv/datasets/streamingbench --max-bytes 9752155`取得固定作者修订的8份JSON；已有文件用`--mode verify --data-root /srv/datasets/streamingbench`只读核验。服务器已取得全部8份9752155字节，逐文件Git
blob/SHA256验证，公共verify命令exit0，见`/root/streamingbench-public-question-verify-20261031-001/qualification-public.json`。offline/stream及proactive子集有重叠，不把行数求和当独立任务数；SQA原嵌套组未展平。只有获取注册，原生适配及研究回读尚未验证，视频/音频仍需另备。

后续必须遵循[官方流程](https://github.com/THUNLP-MT/StreamingBench)：原数据按real/omni/sqa/proactive组织，依次使用`preprocess.sh`、`eval.sh`、`stats.sh`，保留原视频组与时间戳。SQA原实现把前序参考答案加入后续上下文；主动响应原实现逐秒查询到参考时间+4秒，不能简化成单次问答。官方Qwen2-VL-7B包装器采用自适应1/0.5/0.2fps和128输出预算，但写死CUDA与flash_attention_2；当前尚无已验证NPU绑定。完整组固定抽样、原始媒体、原生推理/评分和统一消费链路均待补，来源不打勾。

## OVO-Bench 原生在线视频评测（软件准备入口，尚未完成原生验证）

[官方仓库](https://github.com/JoeLeelyf/OVO-Bench)当前修订版区分回溯、实时感知和向前主动响应，包含12种原生任务。复用已有作者软件获取器，固定45份原Python、shell、依赖说明和许可，不复制注释数据、视频或发布方模型结果进仓库：

```bash
# 默认只显示计划；prepare显式获取小型原作者软件，verify只核验已有文件
python scripts/prepare_jsonschemabench_author_source.py --source ovobench \
  --author-root /srv/benchmarks/ovo-author --action plan
python scripts/prepare_jsonschemabench_author_source.py --source ovobench \
  --author-root /srv/benchmarks/ovo-author --action prepare --max-bytes 79170
```

原始注释入口是`data/ovo_bench_new.json`；固定原文件918443字节、1640个原始case对应644个视频名。向前case包含多次查询，因此case数不等于官方介绍的查询数。原视频由[官方数据页](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench)发布。官方说明提供约44GB原视频拼接包或约144GB预切分包，不能因体积大而用静态截图或自造prompt冒充完整任务。当前仅取得固定原ASI
case483/484/485完整片段（3/103），其余片段尚缺，不能按下述推理示例声称已跑通。

注释复用公共原始文件获取器，默认只显示计划，显式`--mode download`才下载，`--mode verify`核验已有原文件；不把数据放进仓库：

```bash
python scripts/prepare_reused_research_assets.py OVOBENCH --mode plan
python scripts/prepare_reused_research_assets.py OVOBENCH \
  --mode download --data-root /srv/datasets/ovobench --max-bytes 918443
```

适配器保留原始case及完整`test_info`顺序；阶段`realtime/step`为输入，`type/count`和顶层答案为参考，`restore_original_case(record)`可还原原字段。服务器已验证全部1640条、1567个向前阶段，公共`iter_workload`全量回读一致；候选源码引用索引及每种任务原前三条共36条公共/未修改研究端回读也已通过。这些不是实际CI工件安装或模型运行证据。使用`iter_workload("OVOBENCH", data_root=Path("/srv/datasets/ovobench"))`读取，不能调用`request_body()`将其冒充聊天请求。

在完整原数据及合格NPU模型环境具备后，官方原入口示例为：

固定抽样准备：`python scripts/prepare_ovobench_native_selection.py --data-root /srv/datasets/ovobench`只显示计划；显式增加`--output /srv/datasets/ovobench-fixed3`才写入一个全新外部目录。服务器已核验12类各原前三题，共36个完整case，对应103个原始片段。输出`original-first3-each-task.json`保留原字段、答案和全部阶段，直接作为官方`--anno_path`；`preparation-public.json`列出每个case及阶段的原片段名。此入口不下载视频、不裁掉阶段、不执行模型、不覆盖已有目录。片段未取得时不能据此声称可执行，回执`/root/ovobench-native-selection-20261031-002/qualification-public.json`。

实际软件交付验证：CPU
CI运行41151526268的工件13022928107已在专用CPU独立安装，75个Git/包内/安装文件身份一致；复用1640条索引、按12类各原前三条共36条公共/未修改研究端完整回读一致。回执`/root/ovobench-actual-ci-wheel-20261031-001/qualification-public.json`。这证明该版本注释消费链路，未运行视频模型，也未恢复暂停的研究任务。

```bash
cd /srv/benchmarks/ovo-author
python inference.py --model QWen2VL_7B --model_path /srv/models/Qwen2-VL-7B-Instruct \
  --mode offline --anno_path data/ovo_bench_new.json \
  --video_dir data/src_videos --chunked_dir data/chunked_videos
bash scripts/score/QWen2VL_7B.sh
```

这只是官方调用方式，不是本项目已验证的Ascend命令。原Qwen2-VL实现明确使用CUDA、FlashAttention2、最多64帧和128生成token；必须先核验保持原帧采样、提示词、生成预算及所有`test_info`阶段的NPU适配，不能直接替换成普通Qwen文本服务。回溯/实时任务用原`id.mp4`，向前任务按原`id_index.mp4`执行全部阶段，不能只取一个片段当整题。数据为CC
BY-NC-SA 4.0的README标注与Hub数据卡存在差异，见下述许可记录；不作为默认商业可再分发资产。

媒体固定身份和容量登记在同一作者profile的`media_archive_inventory`，计划CLI直接展示：原视频5分卷共46341744640字节，预切分15分卷共153277153280字节；仅为官方LFS身份登记，未取得或重算完整分卷哈希。2026-10-31专用CPU卷剩余5040644096字节，不能在该卷整包下载/拼接/解压；固定题目媒体还需要有界取得方案或足够的服务器存储，不在本地落盘。

许可标注存在待澄清差异：代码LICENSE为MIT，固定代码README声明数据CC BY-NC-SA 4.0，而官方Hub数据卡标注CC BY-SA
4.0。保留原视频来源条款，不自动选择更宽松声明，不将公开可见解释为商业再分发授权。当前不再分发原视频/注释，许可差异核验未完成。

OVO媒体范围计划可使用`python scripts/plan_ovobench_original_media_ranges.py --data-root /srv/datasets/ovobench --index /srv/private/ovo-captured-tar-index.json`。输入是对固定发布方归档捕获的原TAR头检查点；规划器核验捕获头哈希、原TAR校验和、连续偏移、分卷边界和已记录ETag，按原103片段生成不截断的跨分卷范围。它不对伪造索引提供原站认证，也不重新计算整分卷LFS哈希；获取时的固定URL/原版本/精确HTTP范围证据需另行保留。默认明确报告未定位项；`--require-complete`在目录未完成或固定片段未找全时失败，不换题补齐。真实368头快照已通过公共CLI，定位16片段共553531286字节；未下载视频、未完成全目录，也不视为模型可执行。

OVO原片段校验入口：`python scripts/verify_ovobench_original_media.py --data-root /srv/datasets/ovobench --media-root /srv/datasets/ovo-original-clips --decode`。默认仅检查profile已登记的原成员，当前为ASI原前三题483/484/485共3份、37469641字节；`--decode`使用已有PyAV逐帧核验，省略时只证明文件身份，不证明解码。服务器公共CLI已完整解码8556帧，未重编码或更改采样；这不是官方Decord/Qwen帧处理或模型验证。`--require-all`在103个原片段未全部登记时拒绝运行，不能把部分媒体校验说成全套可用。原文件、全部解码回执和范围证据仅保留在服务器，Git不含视频或任务原文。

| 获取参数`--member` | 完整原文件字节（`--max-bytes`至少此值） | 完整视频解码帧数 | 原帧率   |
| ------------------ | --------------------------------------: | ---------------: | -------- |
| `483.mp4`          |                                17464195 |             4076 | 2997/100 |
| `484.mp4`          |                                11205256 |             2352 | 24       |
| `485.mp4`          |                                 8800190 |             2128 | 2997/100 |

原片段获取入口：`python scripts/prepare_ovobench_original_media.py --data-root /srv/datasets/ovobench --index /srv/private/ovo-captured-tar-index.json --member 485.mp4`默认只显示计划。显式加`--download --media-root /srv/datasets/ovo-original-clips --max-bytes 8800190`才获取；仅支持profile已登记、原抽样集合内且已定位的原成员，HTTP读取每段不超过8MiB，必须匹配捕获分卷ETag、精确206/Content-Range和完整成员SHA。保留512MiB磁盘余量；已有完整原文件校验后复用，不覆盖不兼容文件或`.partial`。新文件只有完整身份通过才发布，失败保留partial供排查。

验证边界：公共transfer已在Windows
RAM完成该原成员实际网络读取，逐字节匹配服务器保留文件，未在Windows写数据或增加服务器原媒体副本；服务器公开CLI已验证原文件复用。服务器自身直连网络路径尚未通过，公共目录捕获入口已实现并验证RAM网络推进，但服务器直连仍失败；原主目录已沿用旧控制完成，不重复扫描已取得结果。因此当前不宣称完整103片段一键准备或原生视频评测完成。原模型、原帧处理和评分必须另行验证。

公共目录捕获入口（需要可访问发布方的网络）：

```bash
# 默认只做计划；明确capture才取目录头，不读取或保存视频载荷
python scripts/capture_ovobench_original_tar_headers.py \
  --data-root /srv/datasets/ovobench --capture --max-headers 4096 \
  --output /srv/private/ovo-index-pass1
# 保留上一检查点，继续写入另一个全新目录；不会重扫已有头部
python scripts/capture_ovobench_original_tar_headers.py \
  --data-root /srv/datasets/ovobench \
  --index /srv/private/ovo-index-pass1/checkpoint-private.json \
  --capture --max-headers 4096 --output /srv/private/ovo-index-pass2
```

每次严格取512字节原头部，跳过成员载荷，要求固定发布版本/精确HTTP范围/同分卷ETag一致；不支持的TAR扩展、重复或不安全成员直接失败，不静默跳过。失败保留最后成功步的检查点；不覆盖旧目录或回退到其他题目。真实RAM网络已验证两步推进2048→2050，服务器公开CLI因直连`Network is unreachable`失败但保留有效检查点。首个验证控制脚本输出路径放错而被边界保护拒绝，纠正后的网络失败另行记录；不修改保护规则或主运行索引。回执`/root/ovobench-public-header-capture-20261031-002/qualification-public.json`。完整目录、媒体取得、模型及评分分别判断，不因捕获入口存在就声称原生benchmark已可用。

完整目录的最新验证：沿用原运行控制取得3036个原TAR头，覆盖3035个视频成员和两个原结束标记；公共规划器`--require-complete`已实跑通过，103个固定片段全部定位，总3520157133字节，回执`/root/ovobench-complete-original-index-20261031-001/qualification-public.json`。这不等于服务器公共捕获CLI已全程直连成功，也不意味着103个视频已经取得。媒体队列正在复用已验证3份、补剩余100份；保持原case和全部阶段，完整解码后保留原文件及证据、仅移除本次重复分块缓存。公开profile目前已登记的仍是上述ASI三份，其余在验证后再登记。

传输故障处理：公共原片段获取器对临时连接中断、TLS
EOF、超时及408/部分5xx最多尝试3次，始终重取同一固定URL/字节范围，写入前才重试，保留原ETag与完整文件SHA约束；成功结果含`transport_retry_count`和失败摘要。403/401、429限流及身份/ETag差异直接失败，不关闭TLS验证，不换题或切换数据来源。最终失败仍保留原partial供排查，不发布不完整文件。重试逻辑经可控故障夹具测试；这些不是额外模型运行或质量证据。真实批量队列曾在5份通过后遇TLS
EOF，确认旧进程退出后复用5份和已验证缓存继续补98份。

## WavCaps 原始音频元数据（数据适配已核验，原生音频执行待完成）

同一校验器支持四个已核验组件，默认保持SoundBible；其他组件显式传`--component`和独立媒体目录。四组件各固定原前三条，共12个原FLAC已完整解码，原PCM
MD5及原时长均一致。只核验已取得范围和原成员，不声称整套归档/分卷哈希已重新计算。

| `--component`       | 固定原样本身份                                       | 媒体清单                                                                             |
| ------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `SoundBible`        | 原前三条`2219/2218/2217`                             | [固定身份](../configs/workload-native/wavcaps-soundbible-original-fixed3-media.json) |
| `BBC_Sound_Effects` | 原前三条，保留ID前导零                               | [固定身份](../configs/workload-native/wavcaps-bbc-original-fixed3-media.json)        |
| `AudioSet_SL`       | 原前三条，原`.wav` ID不改写为`.flac`文件名           | [固定身份](../configs/workload-native/wavcaps-audioset-original-fixed3-media.json)   |
| `FreeSound`         | 原前三条`180913/177462/176413`，保留原网站文件名字段 | [固定身份](../configs/workload-native/wavcaps-freesound-original-fixed3-media.json)  |

原JSON的`audio`字段实际为`wav_path`占位符，不能直接当作真实媒体文件。应保留原行，另按来源组件、原`id`和原发布包成员绑定。现已取得SoundBible固定原前三条`2219/2218/2217`的原FLAC；固定身份见[媒体清单](../configs/workload-native/wavcaps-soundbible-original-fixed3-media.json)。官方ZIP未整包下载，仅从固定版本、同一ETag取得原目录/局部头/压缩成员；校验原CRC、FLAC头和全部解码PCM的原MD5，原时长均完全一致，没有重采样、截断、改写原行或换题。完整归档SHA仅登记发布方身份，不能声称已重新验证整包哈希。

使用者也可从[固定版本官方SoundBible包](https://huggingface.co/datasets/cvssp/WavCaps/resolve/0930ec11ded28fa0eaa910fde2f6fc3538acbeac/Zip_files/SoundBible/SoundBible.zip)取得原成员，原完整ZIP约580MB，先核验清单中的发布方LFS身份；不能用另一个版本的文件凑合。将清单列出的原成员放在显式指定的私有媒体目录，以原ID命名；校验入口不下载、不执行模型，也不安装依赖：

```bash
python scripts/verify_wavcaps_original_media.py \
  --metadata-root /private/wavcaps-original-metadata \
  --media-root /private/wavcaps-soundbible-original-media --decode
```

`--decode`使用已有SoundFile/FLAC解码器逐块验证PCM16与原STREAMINFO
MD5；本次复用服务器已有SoundFile0.14.0、libsndfile1.2.2，没有安装或升级包。未装解码器时明确失败，不静默降为头部检查。省略该选项只验文件/原ID/头部关系，不算完整解码。四组件各三条已在CPU完整执行此入口。当前范围获取使用本任务的私有传输控制，公共有界媒体获取/准备入口还未完整迁入；不能把“已有媒体可校验”说成使用者已能一键准备全部输入。原模型、NPU执行、适用评分仍未完成，不勾选整个来源。

媒体准备入口：`scripts/prepare_wavcaps_original_media.py`默认只打印固定原前三条的有界范围计划，不联网。显式`--download`才请求固定发布版本的压缩成员范围；服务器必须返回精确206和Content-Range，否则失败，不退回整包下载。输出必须匹配原成员CRC和已核验SHA256；已有不兼容文件不覆盖。

```bash
# 仅计划；四组件名称同上表
python scripts/prepare_wavcaps_original_media.py --component AudioSet_SL
# 服务器显式下载；不要将数据放进仓库
python scripts/prepare_wavcaps_original_media.py --component AudioSet_SL \
  --download --metadata-root /srv/datasets/wavcaps-metadata \
  --media-root /srv/datasets/wavcaps-media/AudioSet_SL --decode
```

已有原压缩范围可用`--cache-root /srv/cache/wavcaps/AudioSet_SL`替换`--download`。缓存命名为`<part>-range-<start>-<end>.private.bin`，以离线计划为准；原SoundBible缓存若无`zip-`前缀，可在服务器另目录用硬链接建立该名称，不修改原文件。四组件共12条通过公共缓存准备CLI重新解压并完整PCM校验，回执`/root/wavcaps-public-media-preparation-20261031-001/qualification-public.json`。**网络下载模式尚未真实核验**，不可将缓存实跑或HTTP夹具测试宣称为网络获取已跑通。模型和评分仍未完成。

元数据复用现有获取器。默认只显示计划；服务器上可显式获取某份原始元数据，已自行取得的文件使用只读校验：

```bash
python scripts/prepare_reused_research_assets.py WAVCAPS
python scripts/prepare_reused_research_assets.py WAVCAPS --mode download \
  --data-root /private/wavcaps-original-metadata --max-bytes 1000000 \
  --files data/json_files/SoundBible/sb_final.json
python scripts/prepare_reused_research_assets.py WAVCAPS --mode verify \
  --data-root /private/wavcaps-original-metadata \
  --files data/json_files/SoundBible/sb_final.json
```

`download`仅对新外部目录或显式匹配的恢复目录执行，不覆盖旧数据；`verify`已在服务器对四份原文件实际执行通过。下载仍受网络、来源条款和显式容量预算约束，失败不会换题或改文件。FreeSound已用同一获取器从原Git
LFS URL取得并通过发布方SHA256，不能将该URL与GitHub普通blob传输混用。

`workload-data catalog download-plan WAVCAPS`提供作者Git固定版本的四份原始JSON及身份，不下载音频或模型；这不是将Git元数据冒称为另一个Hub
revision。发布方限制学术用途，音频还须保留原来源条件，不根据Hub许可标签推断可商用或再分发。当前CPU取得SoundBible、BBC、AudioSet、FreeSound四份，合计403050条、174782992字节；使用者本机默认读取四份时缺失仍报错，部分使用必须显式选文件。

安装本数据包后，可以对自行合法取得并通过固定哈希核验的原JSON使用现有接口：

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

records = iter_workload(
    "WAVCAPS",
    data_root=Path("/private/wavcaps-original-metadata"),
    files=["data/json_files/SoundBible/sb_final.json"],
)
for record in records:
    audio_metadata = record.task["input"]  # 保留原字段及duration；wav_path占位符不能直接使用
    weak_reference = record.outcome_record()  # caption及存在的原描述/类别，不送入模型
```

原`num_captions_per_audio=1`、原文件名和作者/下载定位信息保留在annotations，各组件身份由源文件路径区分；caption、原描述、类别及FreeSound标签只作参考，没有新造prompt、任务顺序或五条参考。CPU对四份全部403050条完成原字段还原及输入/参考分离核验。实际CI安装包的74份软件/资源身份核验通过，12条原始固定样本与研究工作树中未修改的消费端读取一致，见[元数据CI交接回执](../docs/workload-data/validation/wavcaps-original-metadata-ci-handoff-20261031.json)。这些只证明原始元数据链路，不表示音频已取得、原生模型已运行或质量已评分。作者音频模型、完整原媒体与适用原评价协议仍待接通，不能把caption当输入答案或用Qwen文本生成替代音频推理。

## BixBench 原科学任务准备与读取（原生执行结束，评分待核验）

本接口用于官方科学Agent任务，不用零样本问答替代Notebook工作流。原问题JSONL实际为205条扁平记录、17字段，不能按旧数据卡的嵌套`questions`结构加载。文件版本和64份ZIP身份见[固定原资产清单](data/catalogs/bixbench_original_files_v1.json)；205条问题引用其中59份。当前只取得固定原前三题所需的两份原ZIP，其余57份尚未取得。

使用包含本适配器的本批组件版本；较早的固定安装锁不自动包含后续新增来源。目录由使用者显式提供，原`BixBench.jsonl`和ZIP文件名不变：

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.bixbench import original_capsule_path, restore_original_task

root = Path("/private/bixbench-original")
for record in iter_workload("BIXBENCH", data_root=root, limit=3):
    original_input = record.task["input"]  # 原question及data_folder，不含参考答案
    verified_zip = original_capsule_path(record, data_root=root)
    original_row = restore_original_task(record)  # 含参考侧，仅供可信原生环境/评测器
```

读取保留原问题顺序、参考答案/假说/结果、元数据和未知扩展字段，不生成新prompt。参考侧不得整体作为模型输入；`request_body()`会明确拒绝把该环境任务当成现成API请求。`original_capsule_path`验证官方ZIP字节身份，不解压、不执行代码，也不代表Notebook环境可用。运行副本仍须经过作者预处理，移除参考Notebook，而不是把原ZIP内的全部内容暴露给Agent。

完整元数据索引可以在部分ZIP尚缺时建立；这只证明问题引用可读取，不能据此声明所有任务可执行。现有pipeline配置中的来源项可写为`{"source_id":"BIXBENCH","data_root":"/private/bixbench-original"}`，输出目录必须与来源目录分离；研究消费继续使用原`resolve_input`接口。记录的`runtime.assets`列出对应原ZIP身份，适配阶段不会把缺失ZIP自动下载或替换成别的语料。

服务器已安装[实际CI工件](../docs/workload-data/validation/bixbench-actual-ci-handoff-20261030.json)，71个运行载荷与Git及安装文件逐字节一致；全部205条原字段还原、复用未改变的205条索引及固定前三条公共/未修改研究消费回查、原ZIP摘要校验均通过。这不是科学Agent执行或评分；[早期候选交接证据](../docs/workload-data/validation/bixbench-candidate-handoff-20261030.json)保留原阶段范围。Python3.12以上的官方科学运行环境、NPU模型绑定及原评分仍待完成。

官方软件准备入口固定发布方27份原文件（627,314字节），包含许可证、原Agent/评分源码和`uv.lock`，不包含任务数据或模型。以下命令分别预览、获取与核验；获取需要显式软件字节预算，核验不会导入或执行作者代码：

```bash
python scripts/prepare_jsonschemabench_author_source.py --source bixbench --author-root /external/bixbench-author
python scripts/prepare_jsonschemabench_author_source.py --source bixbench --author-root /external/bixbench-author --action prepare --max-bytes 627314
python -I scripts/prepare_jsonschemabench_author_source.py --source bixbench --author-root /external/bixbench-author --action verify
```

运行环境应独立于模型服务和源码目录：使用Python3.12以上和作者原`uv.lock`，将`UV_PROJECT_ENVIRONMENT`、`UV_CACHE_DIR`及`TMPDIR`显式设到外部目录，再执行`uv sync --frozen --no-dev --no-python-downloads --python <Python3.12路径> --project /external/bixbench-author`。专用CPU的Python3.12.13环境已完成这一步及依赖检查、安装后原源码核验。作者包安装后，在外部工作目录直接导入`bixbench`会报模块未找到；显式设置`PYTHONPATH=/external/bixbench-author`后，使用独立环境Python执行`-m bixbench.generate_trajectories --help`通过。这只是客户端入口加载验证；官方ARM64镜像及原作者Notebook
API的隔离可信夹具也已通过。原生Agent曾实际发出四次API请求，均因消息模板不兼容返回400，零有效模型完成、零任务完成、未评分；不能将请求尝试当成推理通过。不得通过修改锁文件、重排原消息或切成零样本问答来宣称原流程完成。

数据准备公共入口使用已取得的原JSONL与ZIP，不自动下载。先用上述独立客户端Python运行以下命令查看计划，确认两个全新外部输出目录及空间后再加`--execute`：

```bash
/external/bixbench-client/bin/python scripts/prepare_bixbench_original_tasks.py \
  --author-root /external/bixbench-author --data-root /private/bixbench-original \
  --output /private/bixbench-prepared --workspace /private/bixbench-workspaces --limit 3
```

执行时固定核验发布方软件及数据摘要，按原JSONL顺序选择前N题，调用作者预处理和科学环境构造；仅修改新副本中的运行路径，原ZIP及JSONL不变。作者预处理删除的是运行副本中的参考Notebook及复制ZIP，不应把来源目录作为输出。此公共入口已在服务器复用同一批资产验证原前三题，数据文件数为4、4、6；见[原任务准备证据](../docs/workload-data/validation/bixbench-selected-task-preparation-20261030.json)。它不会启动Notebook、Agent或模型，也不评分。

后续隔离执行还要求workspace在Docker
daemon与客户端两侧有相同路径，并由操作者仅对运行副本设置非root容器所需的组权限；不得递归修改原始数据或共享目录。已验证的可信夹具使用UID/GID65534、运行目录setgid2770及控制器进程局部umask0007，避免root创建的0600文件阻止非root读取。任务容器保持离线、只读根、去除能力且只挂载本任务workspace；原执行器必须显式启用Docker，不能退回宿主机执行生成代码。[镜像与夹具证据](../docs/workload-data/validation/bixbench-notebook-preparation-20261030.json)不代表原生科学任务已跑通。

原生Agent公共入口（默认仅输出计划，不调用模型）：

```bash
/external/bixbench-client/bin/python scripts/run_bixbench_original_agent.py \
  --author-root /external/bixbench-author --data-root /private/bixbench-original \
  --prepared /private/bixbench-prepared --output /private/bixbench-native-output \
  --workspace /private/bixbench-native-workspaces --limit 3 \
  --model openai/QUALIFIED_LOCAL_MODEL --api-base http://127.0.0.1:19466/v1 \
  --docker-host unix:///run/owned-docker/docker.sock
```

将模型名、端口及Docker套接字替换为操作者已核验的专用环境；输出与workspace须是两个全新外部目录，不能覆盖准备目录。使用含`prepared_files`摘要的新准备回执，入口会再次核对原JSONL、ZIP、发布方软件和预处理文件身份。模型窗口空闲、容器资源及模型原消息模板兼容性就绪后才加`--execute`。远端模型可使用已授权的回环转发，或显式`--model-queue`配合前述可信worker；入口不自动建立网络连接或启动模型。

对于已确认响应可能超过客户端默认60秒的本地模型，可显式加`--request-timeout 600`。这是传输等待时间，不是生成token预算或Agent步数；默认不覆盖作者客户端配置。须在回执披露该差异，不能通过重复请求或截断响应来掩盖超时。范围为60至1800秒，仍须核验模型绑定、客户端实际解析及原生任务终态。

固定客户端实际核验表明，顶层`llm_model.timeout`会被忽略，必须使用`llm_model.router_kwargs.timeout`。公共入口已修正这一映射，服务器真实构造验证600秒、温度1.0、4096-token预算和原重试参数均保留；这仍不等于原生任务或评分完成。[原客户端核验证据](../docs/workload-data/validation/bixbench-official-client-20261030.json)

执行沿用原40步科学Agent、双阶段工具交互、原生成预算及原轨迹保存，不是静态问答。当前9B绑定拒绝原角色序列`system,user,user,system`，报`System message must be at the beginning`；保留失败容器和日志，不能合并或删除末尾system来凑通过。需另行验证兼容模型；资源替代模型不得标为官方推荐。此入口不包含评分，作者默认开放答案judge为GPT-4o；未授权调用付费judge，不以轨迹生成、计划输出或Notebook夹具代替官方评分与来源总勾。

开放答案评分入口（默认仅计划，服务器原生评分尚待验证）：

执行后端须区分：本公共顺序验证入口调用发布方原`TrajectoryGenerator.vanilla_rollout`，其内部保留原40步、Agent状态转移、工具执行及完整轨迹；不是原配置默认的Aviary批处理后端复现，也不报告其并发或性能。回执分别记录原配置`rollout_type`与实际调用API，不能用配置值替代实际后端说明。需要默认批处理后端复现时应使用原发布方配置入口及其`batch_rollout`路径。

工作区中断后的恢复：先确认旧Agent进程确已终止，保留旧输出，在新的私有输出和工作目录运行同一固定前缀，并传入`--reuse-completed-from /private/previous-native-output`。入口核验旧模型、原Agent/生成/步数配置、已完成题目的连续前缀、全部原始任务字段及参考答案，同时要求结果JSON和完整轨迹JSONL均存在；通过后逐字节复制已完成轨迹，并记录哈希。未给出答案的已完成任务同样保留，不因质量失败重跑。未完整保存的中断任务从原生流程起点重新执行，不宣称恢复其内存状态；传输超时可修正，但不能改变题目、生成预算或原生步数。该恢复能力已用服务器保存的第一题文件核验，尚不代表固定三题已完成或官方评分已通过。

先将完整保存的原生轨迹与同一原始数据版本逐题对齐，准备官方评分输入：

```bash
/external/bixbench-client/bin/python scripts/prepare_bixbench_original_grade_input.py \
  --author-root /external/bixbench-author --data-root /private/bixbench-original \
  --native-output /private/bixbench-native-output \
  --output /private/bixbench-new-grade-input --limit 3
```

默认仅计划；确认固定前N题的原生运行全部保存后加`--execute`。入口复用官方`load_answer`与单答案提取规则，不生成新答案；核验题目ID与原参考答案，将原JSONL的`eval_mode`原值映射到评分CSV的`evaluation_mode`，不允许缺失、冲突或未知模式静默退化为LLM默认判定。缺题、重复或不一致直接报错，空答案不丢弃。已使用两份真实保存的原轨迹验证逐行映射及原空答案解析；固定三题全部保存后的完整CSV写出仍待验证。下面的`--input-file`指向其输出`original-grade-input.private.csv`。

```bash
/external/bixbench-client/bin/python scripts/run_bixbench_original_grader.py \
  --author-root /external/bixbench-author \
  --input-file /private/bixbench-original-grading-input.csv \
  --output /private/bixbench-new-grading-output \
  --model openai/QUALIFIED_LOCAL_JUDGE --api-base http://127.0.0.1:19466/v1
```

输入遵循发布方`grade_outputs.py`的CSV约定：`question`为原题、`target`为原参考答案、`predicted`为已保存的原生Agent答案；已有`evaluation_mode`原值须保留，不得改题或用模型重新生成答案。该入口不自动将轨迹改写成CSV，调用方须先核对逐题对应关系。加`--execute`后直接调用固定发布方的`grade_answers`，保留开放答案评分规则、温度1.0与作者默认生成配置；仅显式绑定已核验的本地judge和回环API。发布方会原地更新CSV，因此入口只传新私有副本，不覆盖输入。评分失败保留输出并报错。当前仅软件边界测试通过，尚无真实评分成功声明；替代judge的结果也不等于GPT-4o结果复现，必须披露模型差异，不能据此单独打来源总勾。

跨节点评分可追加`--model-queue /private/bixbench-grading-queue`，复用生成阶段的`model_proxy`传输；API须为`127.0.0.1`且端口匹配，队列、输出、原输入及作者目录相互分离，均不得写入软件仓库。默认计划不创建代理或目录；执行时仅转发官方评分函数生成的原请求，不改提示、答案或计分。当前已在专用CPU真实客户端通过队列模式的计划检查，未执行评分。需要复现GPT-4o成绩时，应使用固定发布方原`grade_outputs.py`及操作者自行授权的官方认证/费用配置，不能将本地替代judge验证结果冒充该成绩。

### DailyTalk：官方对话语音合成准备入口（原生验证未完成）

最新第三次原参数尝试已进入声学模型，但当前NPU软件栈的`DynamicGRUV2Hidden`拒绝原FP32权重（报错仅列出FP16/FRACTAL_Z支持）；不能据此宣称所有精度均不支持GRU。声码器尚未进入，合成未完成。前两次环境错误不能再作为当前唯一阻塞。保留原检查点及失败证据，不擅自降低精度、不删除GRU、不改网络结构，也不回退CPU冒充NPU通过。

[DailyTalk 官方发布](https://github.com/keonlee9420/DailyTalk)提供带历史上下文的对话语音合成基线及预训练模型，不是把录音转成普通文本问答的
benchmark。官方上下文模型使用批量推理；抽样须保留原验证划分及所需完整对话历史，不能将单个话轮拆成无上下文题目。源码、转写/说话人/统计元数据与词典、原始音频、声学模型、HiFi-GAN声码器须分别取得和核验。

以下入口只下载或核验原发布文件，不运行模型；默认`plan`无下载。软件固定52份文件，元数据与词典固定11份文件，均核验Git
blob、字节数和SHA256。不包含音频或模型权重，不能据此声称完整语音任务可用。

```bash
python scripts/prepare_jsonschemabench_author_source.py --source dailytalk --author-root /private/dailytalk/author --action prepare --max-bytes 326159
python scripts/prepare_dailytalk_original_metadata.py --data-root /private/dailytalk/original-assets --action prepare --max-bytes 12631639
# 已取得文件只核验，不重新下载：
python scripts/prepare_jsonschemabench_author_source.py --source dailytalk --author-root /private/dailytalk/author --action verify
python scripts/prepare_dailytalk_original_metadata.py --data-root /private/dailytalk/original-assets --action verify
```

固定原音频准备入口（需要该执行节点能访问Google Drive，不能忽略网络失败改用其他录音）：

```bash
python scripts/prepare_dailytalk_original_audio.py --data-root /private/dailytalk/audio --metadata-root /private/dailytalk/original-assets --action prepare --max-network-bytes 67108864 --max-extracted-bytes 33554432
python scripts/prepare_dailytalk_original_audio.py --data-root /private/dailytalk/audio --metadata-root /private/dailytalk/original-assets --action verify
```

该入口按未改动的官方验证清单确定前3个完整对话，保留全部29个话轮的原WAV、转写及原始语料元数据，共59份原文件。只使用固定ZIP范围，不保存重打包数据或整份压缩包；校验归档版本、原成员CRC及固定SHA256。`verify`与完整文件的重复`prepare`不访问网络；已有文件不一致、遗留`.partial`、缺失话轮、服务器不支持Range或预算不足时保留现场并报错，不更换样本。它不是全部128个验证对话的原生合成验证；全量原语料仍可从发布方原目录取得。

原模型工件固定在`configs/workload-native/dailytalk-original-model-assets.json`：`history_guo`声学检查点900000步、原配置指定的universal
HiFi-GAN声码器、原预处理指定的`distiluse-base-multilingual-cased-v1`512维文本编码器。模型来源与字节身份分别记录，不把无历史模型、LJSpeech声码器或其他维度编码器混用。取得原发布文件后可统一核验；声码器只解压发布方外层ZIP，保留其中的`generator_universal.pth.tar`检查点文件。

```bash
python scripts/verify_dailytalk_original_models.py --acoustic-checkpoint /private/dailytalk/models/900000.pth.tar --vocoder-checkpoint /private/dailytalk/models/generator_universal.pth.tar --encoder-root /private/dailytalk/distiluse --action verify
```

该核验入口不下载、反序列化或运行模型。原`TextDataset`的Guo批量推理路径读取原音素、说话人/情感映射及本轮/历史文本嵌入；保留完整原对话关系，并保持模型原有的最多10轮历史窗口，不改用无历史分支，也不将完整训练期重预处理当作该推理路径的必需步骤。

取得原元数据和所选音频后，可准备原Guo批量输入。该入口保留原始行字节与顺序，复制原说话人、情绪、统计映射，并逐轮登记原有最近10轮历史和待生成的512维特征文件名；缺少历史话轮或原文件校验不符时直接报错。

```bash
python scripts/prepare_dailytalk_original_guo_inputs.py --metadata-root /private/dailytalk/original-assets --audio-root /private/dailytalk/audio --output /private/dailytalk/guo-inputs --action prepare
python scripts/prepare_dailytalk_original_guo_inputs.py --metadata-root /private/dailytalk/original-assets --audio-root /private/dailytalk/audio --output /private/dailytalk/guo-inputs --action verify
```

输出包含原行子集`val_phone.fixed3.private.txt`、原映射和`guo-input-contract.private.json`。这不是嵌入或合成结果；仍需用原文本编码器执行发布方的`encode([raw_text])[0]`并运行原Guo声学模型和声码器。两个命令已在服务器上对全部29个固定原话轮实际通过，不加载模型、不访问网络，也不将输入准备计为来源完成。

评价边界：[原论文第4节](https://arxiv.org/html/2207.01063)使用人工MOS及比较式CMOS评价自然度和对话上下文，不能用WAV格式检查或文本LLM打分替代。当前29话轮是固定原生流程验证样本，并非论文质量实验的260话轮样本；人工听评未开展，不声明复现论文质量分数。原生合成输出、文件完整性检查与人工质量评价分别记录。

当前服务器已通过软件和元数据公开`verify`入口，42份原Python文件语法检查通过。已取得原`val_phone.txt`按首次出现顺序的前3个完整对话（23、30、59）全部29段原始WAV及转写，保留原元数据；所选ZIP成员的原CRC和SHA256已核验，未下载或验证整份5.34GB压缩包。上述公开音频入口的`verify`与复用式`prepare`也已通过服务器实测，均无新下载。该CPU节点不能直连Google
Drive，首次音频取得使用已披露的纯内存HTTPS中转；当前公开下载函数已做真实HTTP范围探测，完整首次下载路径只有离线夹具覆盖，不能称为该节点直连下载通过。官方`history_guo`的900000步检查点也已取得并完成全文件SHA256与ZIP成员CRC校验，取得阶段未反序列化或运行。随后两个生成器检查点已用`weights_only=True`加载为`meta`张量核验结构：声学检查点仅明确放行其NumPy标量及对应dtype类型，声码器不使用该白名单；没有关闭受限加载、初始化NPU或执行推理。音频和检查点只在服务器保存。

universal声码器及原512维文本编码器文件也已取得并校验。29话轮已按原编码入口完成真实NPU编码，保留512维特征，不重复生成。声学合成仍未通过：第一次尝试缺少CANN
Python路径，第二次在编译器子进程遇到`AF_UNIX path too long`；均保留失败记录，尚无已核验的合成WAV。短临时目录的独立多进程探测已通过，后续重试不改变原模型、历史窗口、批大小和生成控制。独立软件环境不等于复现原Torch1.7/TensorFlow2.5.1环境；保留原文件的兼容副本仅调整可选依赖按需导入和显式NPU设备选择。声学及声码器完整NPU前向、原始评测与研究侧接入仍待完成；不得静默回退CPU。不能用现有Qwen文本服务替代声学模型，也不能把编码成功或检查点字节校验算作该来源完成。原公开资源与代码许可按发布方及组件各自LICENSE执行，本仓库不打包音频和权重。
