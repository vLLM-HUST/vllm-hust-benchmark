# BBH 原始任务执行证据与覆盖边界

Boolean原任务读取和执行入口已包含在草稿PR链#428→#432中，尚未合并。固定研究侧已有该来源的消费回执；CI/software
checks验证了入口。当前准备器、读取适配和通用原生runner覆盖固定版本的27个原始任务文件（对应发布方论文的23类任务）；除Boolean外的26项尚无模型/NPU执行证据，故来源仍未完成，不能将代码入口表述为运行验证。

## 原始来源与可用资产

发布方为[suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)，固定提交`9ee07bd481feebf959a6b59d61ea57bdcf30964d`。
发布论文描述23项任务；固定提交中有27个任务JSON文件，共6511条实例，部分类别含不同配置的独立任务文件。文件数不等于workload类别数。
固定清单登记了27个原始任务JSON文件及其27个配套原始CoT提示文件，逐个固定路径、归档成员、行数、字节数和SHA-256。原任务字段不改写；`target`只保留在参考侧，不进入模型输入或来源索引。原归档和成员校验和固定；保留发布方MIT仓库声明和上游归属，本仓不重新分发原始题目、提示示例、答案或私有响应。

## 使用链路

安装本批匹配的根CLI和数据组件，按发布方条款自行取得固定原归档：

```sh
# 在测试服务器的可写目录执行；不要下载到开发者本地电脑
export BBH_ROOT=/path/on/server/bbh
mkdir -p "$BBH_ROOT"
curl --fail --location --output "$BBH_ROOT/publisher-legacy.zip" https://codeload.github.com/suzgunmirac/BIG-Bench-Hard/legacy.zip/9ee07bd481feebf959a6b59d61ea57bdcf30964d
echo "8406fd092f7050638d60e7560952a62f78eeea38ac5a2e5755dff1e12efe44fc  $BBH_ROOT/publisher-legacy.zip" | sha256sum --check
workload-data prepare-bbh-original --data-archive "$BBH_ROOT/publisher-legacy.zip" --output "$BBH_ROOT/original-all"
# 旧Boolean准备入口仍可使用，须选择另一输出目录
workload-data prepare-bbh-boolean --data-archive "$BBH_ROOT/publisher-legacy.zip" --output "$BBH_ROOT/original-boolean"
```

归档校验对应上面的固定`legacy.zip`入口；其他GitHub
ZIP入口的归档封装和摘要可能不同，不能据此跳过校验。准备后任务目录可命名为示例中的`/data/bbh-original-all`或任何服务器可写路径。

输出目录必须不存在。准备接口验证固定原归档、所有选中成员的唯一性、SHA/字节数、任务行数与字段、原始提示分隔符和发布方示例数；原样复制任务JSON与提示文件，0模型调用、0评分，不解压无关文件。旧Boolean命令保留兼容；全量准备命令复制54个原始文件。

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

for record in iter_workload("BBH", data_root=Path("/data/bbh-original-all"), limit=3):
    original_input = record.task["input"]  # 只有原始input
    scorer_reference = record.outcome_record()  # target，不传给模型
```

`iter_workload("BBH", ...)`可逐文件读取27个原始任务；支持通用bind、pipeline和resolve_input。原任务没有独立ID，使用文件校验和与原行序号组成身份，不编造到达时间；完整记录比较时应使用同一绑定和相同limit等选择契约，不能删除出处或选择字段来制造相等。

## 分层验证记录

- 2026-10-24在测试服务器的独立Python
  3.12.13环境构建并安装当前未提交候选wheel，SHA-256为`21d12bc002757a1b54955c0e6b18bdd03a09f5fa7637cf5a377cd37051d8cddb`。公开CLI从固定归档准备全部54个文件；27个任务共6511条逐行核对原始`input`与`target`，每个任务原前三条共81条完成bind→pipeline→resolve，输入、参考和来源身份一致；27份提示的字节摘要和原三示例结构全部核验。数据、wheel和索引仅在服务器，本地仅保存[脱敏回执](validation/bbh-all-original-assets-server-20261024.json)。这是已安装候选的真实数据准备/读取证据，0模型调用、0原生任务执行、0评分；不是CI最终工件或研究消费端的新联调证据。
- CPU夹具覆盖原字段、非空文本检查、参考隔离、错误归档/成员拒绝、原字节复制、已有目录拒绝覆盖，以及显式绑定的索引回查。这些人工夹具不是BBH任务证据。
- 服务器已独立安装本地候选数据wheel，SHA256
  `55c77277f73c1e3f757bfc66ac08be84966cbfb904f03ada4aa2f4a1b99b3ce8`。全部250行逐条对照原始input/target，前三条bind→pipeline→resolve回查通过，0模型调用。候选不是CI工件，未与研究侧联调。
- 另一次模型探测在已有lm-eval
  0.4.11环境中，使用发布方原始三示例CoT提示与原任务索引0、1、2，已有Qwen3.5-27B、双910B2、BF16、TP2、graph、非思考配置。模型原生chat传输不是论文原模型设置。
- 模型探测三条均为完整输入，实际/预检均532 tokens；每条输出48
  tokens。三条均`finish_reason=stop`，`stop_reason`符合固定generation配置；固定regex/exact_match评分完成，结果0/3。模型未输出可提取的最终答案，因此得到低分；这是一条有效模型结果，不是runner、stop协议或评分器故障。
- 私有执行回执SHA256
  `284e07c8f6398be081b84598e14a6cccf73cc17c76c6b3d44fc58393bd4c72a9`；原提示和任务身份不变。核验脚本首次使用不存在的record_id属性、第二次混比全文件读取与limit=3索引记录，两次失败脚本保留；统一绑定和limit后通过，没有放宽原字段检查。

lm-eval提供固定参考评测实现，不是发布方原始评分源码；其BBH说明也未确认论文方法等价性。因此不称本次为发布方完整评测、论文复现、全量模型验证或性能提升。保留0/3，不删题、不改停止词或提取规则来获取通过。0/3不构成运行失败；若后续验证另一种停止/传输设置，必须单独登记为不同运行协议，不能替换本次结果。

## 可复用原生执行入口

通用runner支持全部27项，包括Boolean；使用者可选择任意原始任务并完整运行。本轮验证复用Boolean已有回执，选择其余26项补证据，不重复运行同一原题、模型及协议。runner使用固定`lm-eval==0.4.11`环境，验证任务/提示、harness配方和评分模块校验和，不修改停止词或提取规则。默认每个显式选择任务按原顺序取前三条；`--full`才跑所选任务全部原始行。输入超上下文或执行失败都原样记录，不换题、不截断、不重试凑通过：

```sh
python scripts/run_bbh_original_tasks.py --data-root /data/bbh-original-all --output /data/bbh-run-private --endpoint http://127.0.0.1:19466/v1 --model MODEL_ALIAS --max-model-len 131072 --task-ids causal_judgement date_understanding
# 显式全量：只将所选任务的前三条扩为该任务全部原始行
python scripts/run_bbh_original_tasks.py --data-root /data/bbh-original-all --output /data/bbh-run-full-private --endpoint http://127.0.0.1:19466/v1 --model MODEL_ALIAS --max-model-len 131072 --task-ids causal_judgement --full
```

runner只允许无凭据的本机HTTP接口，输出目录必须不存在；原始请求、响应及参考答案仅写入该私有目录。所有评测输入保留原文与原行序；参考答案不进入请求。运行前需自行准备已启动的兼容模型服务及固定harness环境；脚本不启动模型、不下载数据、不执行任务代码。`--max-model-len`必须等于当前部署配置；runner先做完整输入token预检，超限即失败，不截断。执行协议使用发布方原始三示例CoT提示、固定harness的generation/regex/exact-match配置和模型原生chat传输；这不等于发布方论文设置或发布方评分器。未迁入观测钩子或研究特征分析。

根候选wheel与数据候选wheel安装后的`prepare-bbh-boolean`已在服务器通过；公开脚本按同一配置复验同三题并得到0/3，输入token一致，重复运行不增加唯一题数。见[脱敏记录](validation/bbh-boolean-original-npu-20261023.json)。这些原题评分可复用，不重跑同配置、不改stop。脚本正确性夹具加入CPU
smoke，不在CI跑真实模型。

2026-10-24，余下26个任务变体各按预先固定的原索引0、1、2在独立双NPU窗口完成生成与固定harness评分，共78条，全部正常停止，未自动重试或换题。随后在兼容依赖环境中对保存的原响应离线复核，78条评分全部一致，没有再次请求模型。见[脱敏原生执行回执](validation/bbh-all-task-variants-original-npu-20261024.json)。加上复用的Boolean三题，已覆盖全部27个任务变体的81个唯一原题；不是6511题全量评测，也不合并不同运行的模型准确率。78条中23条提取到答案，其余依固定评分规则保留低分，不改停止规则。

评测客户端使用独立Python
3.12环境：`lm-eval==0.4.11`、`datasets==4.3.0`、`pyarrow==22.0.0`、`numpy==1.26.4`、`huggingface-hub==1.33.0`。生成时环境的可选Arrow依赖存在兼容问题，但本运行的原JSON读取与固定评分器不依赖Arrow；任务结束后修正客户端依赖并离线复核原响应，未修改共享推理环境。数据工具独立环境依赖检查通过，8项迁移工具测试通过；这些软件测试与原题执行分别记账。

固定提交`3f397e69`的CPU smoke、lint和主CI
tests全部通过。其CI数据wheel在服务器独立环境安装后，全部27变体的81条原记录回查通过；原生执行、固定评分、来源身份、环境及命令文档均已收口，按本轮固定抽样交付范围打勾，见[交付回执](validation/bbh-native-delivery-20261024.json)。这不是6511题全量或论文成绩复现；研究侧的新wheel锁定没有在本次验证中宣称完成。

## 本轮已验证的双NPU模型服务配置

BBH与MATH500本轮复用同一现有模型服务。模型为`Qwen/Qwen3.5-27B`，revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`；11份权重及9份运行文件与[固定工件摘要](validation/qwen35-27b-deployable-artifact-20261022.json)再次逐一匹配。硬件是两张Ascend
910B2，BF16、TP2；不是原论文模型，也不称官方容器复现。

实测服务环境为Python 3.12.13、CANN
9.1.0、`torch==2.10.0+cpu`、`torch-npu==2.10.0.post4`、`vllm==0.23.0+empty`、`vllm-ascend==0.23.0`。两份已安装的可编辑源码树均干净，提交分别是`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`和`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。这里记录的是复用现有环境的真实验证，不声称已完成全新机器的编译安装验证；使用者需要匹配Ascend驱动、CANN及这两份源码工件。

在已分配两张NPU的环境内，确认原服务端口空闲、没有占用他人的卡后运行下列配置；`MODEL_DIR`指向与上述摘要匹配的外部模型目录，`RUNTIME_DIR`是外部缓存/日志目录：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
: "${MODEL_DIR:?set an external verified model directory}"
: "${RUNTIME_DIR:?set an external runtime cache and log directory}"
export VLLM_PLUGINS=ascend
export TMPDIR="$RUNTIME_DIR/tmp"
export XDG_CACHE_HOME="$RUNTIME_DIR/cache"
export HF_HOME="$RUNTIME_DIR/huggingface"
export VLLM_CACHE_ROOT="$RUNTIME_DIR/vllm-cache"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_DIR/torchinductor"
export TRITON_CACHE_DIR="$RUNTIME_DIR/triton"
export ASCEND_PROCESS_LOG_PATH="$RUNTIME_DIR/ascend-log"
export ASCEND_WORK_PATH="$RUNTIME_DIR/ascend-work"
export HCCL_NPU_SOCKET_PORT_RANGE=20600-20700
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$ASCEND_PROCESS_LOG_PATH" "$ASCEND_WORK_PATH"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --host 127.0.0.1 --port 19466 \
  --served-model-name workload-qwen35-27b-128k-nonthinking \
  --dtype bfloat16 --tensor-parallel-size 2 --gpu-memory-utilization 0.75 \
  --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 2048 \
  --language-model-only --trust-remote-code --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --additional-config '{"enable_cpu_binding":false}'
```

服务就绪后检查`/health`及`/v1/models`。没有强制eager，日志中实际出现graph
replay；不从本轮功能探测推导性能结论。本轮容器已经正确映射两张设备，不再手工覆盖`ASCEND_RT_VISIBLE_DEVICES`。客户端环境必须与服务端隔离，不用评测依赖修复去修改共享模型服务。
