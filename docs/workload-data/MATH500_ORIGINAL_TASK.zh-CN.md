# MATH500 原生评测入口与数据准备适配

## 来源身份与边界

本入口将 `MATH500` 归入 `standalone-reasoning`。来源身份固定为
[`openai/prm800k` 的指定提交](https://github.com/openai/prm800k/tree/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/math_splits)，
文件为 `prm800k/math_splits/test.jsonl`。读取镜像固定在
[`HuggingFaceH4/MATH-500` 指定提交](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be)，
其 LFS SHA-256 与上游文件身份一致： `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132`；文件含 500
个唯一原题。 许可依据是 PRM800K 与来源 MATH 仓库的 MIT 声明。代码只登记来源与哈希，不把任务文件、答案或解答随软件包发布。

每个原始 JSONL 行保留为一个任务。`problem` 是唯一模型输入；`answer` 和 `solution`
分别保留为评分目标与参考材料，不进入模型输入；`unique_id`、`subject` 和 `level` 作为来源标注。适配器不改写题目，不生成替代题，也不以合成数据回退。

## 权威原生评测入口

本轮模型工件、双Ascend服务环境及完整启动命令复用[BBH文档的实测服务配置](BBH_ORIGINAL_INPUTS.zh-CN.md#%E6%9C%AC%E8%BD%AE%E5%B7%B2%E9%AA%8C%E8%AF%81%E7%9A%84%E5%8F%8Cnpu%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E9%85%8D%E7%BD%AE)；客户端独立安装，不能用下面的评测依赖覆盖模型服务环境。

交付评测以 HuggingFaceH4 在上述固定提交发布的
[`eval.yaml`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/blob/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/eval.yaml)
为准，由 Inspect AI 按[官方 Hugging Face task 入口](https://inspect.aisi.org.uk/tasks.html)直接加载该任务定义，并用
`@revision` 固定版本。 此配置固定 `test` split、4 个 epochs、`pass_at_1` 归约，以及 `model_graded_fact`
评分器。依[Inspect官方模型评分规则](https://inspect.aisi.org.uk/model-graded.html)，任务未显式指定 grader model
时会回退到本次被测模型，因此本入口使用同一个本地服务模型完成生成和官方裁判，不切换到外部模型或自建评分器；服务接入按[Inspect官方 vLLM provider](https://inspect.aisi.org.uk/providers.html)连接既有
`/v1` endpoint。 运行器固定调用 `hf/HuggingFaceH4/MATH-500@6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`。

评测运行环境固定 [inspect-ai==0.3.283](https://pypi.org/project/inspect-ai/0.3.283/)；运行器会拒绝其他版本。
在服务器已有模型服务可用、研究任务已释放 NPU 后，先把 Hub 缓存和 Inspect 日志放在仓库外，再执行：

```sh
export HF_HOME=/data/huggingface-cache/math500
python -m pip install 'inspect-ai==0.3.283'
python scripts/run_math500_native.py \
  --model-id "$MODEL_ID_FROM_SERVER" \
  --base-url http://127.0.0.1:19466/v1 \
  --log-dir /data/private-runs/math500-inspect
```

2026-10-24服务器验证客户端还固定安装`datasets==4.3.0`、`huggingface-hub==1.33.0`、`openai==3.37.0`、`s3fs==2025.9.0`、`boto3==1.41.5`，解决其共享`fsspec`/`botocore`版本约束，独立环境`pip check`通过。服务器直连Hub超时时使用`HF_ENDPOINT=https://hf-mirror.com`作为下载通道；固定revision不变，原`test.jsonl`再次核验上述发布文件SHA及500条数量，原`eval.yaml`的4
epochs、pass_at_1及model_graded_fact不变。镜像不是另一套任务，也不改变任务内容。

入口显式设置`--max-retries 0`，不自动重试失败的API请求；不启用`--retry-on-error`，不覆盖上游生成预算、epoch数或评分器。

不带 `--full` 时，入口固定运行原始 test 顺序的前三题；数据读取回执已将其与固定来源文件的前三个原题 ID 逐项核对。每题仍执行上游完整的 4 轮生成和
`model_graded_fact` 评分，不缩减任务步骤、预算或评分过程。 Inspect 原始日志可能包含题目、答案和模型输出，必须保留在仓库外。`HF_HOME`
也必须位于仓库外，避免下载的原始数据进入 Git。

若以后依据预估确认完整运行与评分不超过 30 分钟，可显式追加 `--full` 运行 500 题；否则沿用预先固定的前三题样本。 按官方配置每题包含 4 轮生成及对应模型评分，完整执行涉及约
4,000 次模型调用，当前采用固定样本；尚未在NPU上完整运行500题。本轮原前三题四轮流程及声明范围的软件交付核验已完成，详见下方回执，不将抽样称为全量。

此入口会直接通过固定 revision 从 Hugging Face 加载官方评测任务及数据，适合原生评测；它不替代下文的数据包读取/准备验证。

## 数据包读取、准备和诊断执行

先从发布方取得固定版本文件，并将它放在显式数据根目录下：

```text
/data/math500/prm800k/math_splits/test.jsonl
```

在独立诊断环境安装固定版本上游 `grading/grader.py` 和 `grading/math_normalize.py`，并安装其 `sympy`、`pylatexenc`
依赖；运行器会逐文件校验上游 SHA。 模型 tokenizer 必须从本地模型目录读取，禁止运行时网络拉取或改变引擎服务配置。

```sh
python scripts/run_math500_original_samples.py \
  --data-root /data/math500 \
  --grader-root /opt/openai-prm800k/prm800k \
  --tokenizer /workspace/models/Qwen3.5-27B \
  --endpoint http://127.0.0.1:19466/v1/completions \
  --model workload-qwen35-27b-128k-nonthinking \
  --output /workspace/private-runs/math500-001
```

执行器先通过 workload-data 适配器读取并准备前三条原题，再经 consumer 逐条解析，核对输入与来源适配结果一致，随后向现有文本补全服务各发送一次请求。请求使用温度 0、固定
seed、最多 2048 个生成 token；超过显式模型上下文上限的输入在请求前失败，不截断、不重试。输出目录包含可回查的私有准备数据库，必须留在仓库之外，不得上传到公开 PR。

## 评分协议与不可混淆项

固定版本的
[`eval.yaml`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/blob/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/eval.yaml)
声明 `problem`/`answer` 字段、四个 epoch、`pass_at_1`，并使用 Inspect AI `model_graded_fact`。它的提示要求逐步推理且最后给出
`\\boxed{...}`。

下面的 `run_math500_original_samples.py` 是数据包与确定性 grader 的诊断入口，不是交付用的原生评测入口：它沿用原题和提示模板，但不复现上述四轮 Inspect
配方。为避免额外裁判模型，该诊断脚本只生成一次，并使用固定上游提交中的
[`OpenAI PRM800K 确定性答案评分器`](https://github.com/openai/prm800k/blob/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/grading/grader.py)
评分抽取出的 boxed 最终答案。回执会分别记录终止原因、是否抽取到答案和评分结果；模型答错仍是一次成功执行，不会替换样本或改写题目。该诊断结果不称为 HuggingFaceH4 的原生
MATH-500 评测分数，也不能用于本来源交付勾选。

仓库历史上已有一个 `MATH-500` 负载性能结果：200 个计划请求、200 完成、0 失败。但当时来源到运行的绑定评级为 D，且该结果没有证明与本入口固定的 500 个原题
ID、适配器输入投影和确定性评分器一致。因此它保留为历史性能材料，不被重复跑作性能测试，也不替代这里所需的原始任务读取/官方评分接入核验。两种证据不能混为一个准确率或验收结论。

## 已验证范围与当前缺口

2026-10-24在服务器独立环境完成原前三题完整4轮Inspect原生执行及`model_graded_fact`评分，12个题目×轮次均正常stop、无执行错误；逐条核对原输入、参考、每题四个epoch及评分存在性通过。见[脱敏原生回执](validation/math500-upstream-inspect-native-20261024.json)。裁判遵循上游默认使用被测模型本身，不把自评分解释成外部独立质量认证。固定提交3f397e69三项CI通过，其实际CI
wheel独立安装后全500条原输入/参考字段核对及前三条prepare/resolve通过，0新增模型调用；见[固定工件回查](validation/math500-ci-reader-3f397e69-20261024.json)和[抽样交付回执](validation/math500-native-delivery-20261024.json)。据此按声明范围打勾，不称全量或独立质量认证。先前20项CPU专项证据继续保留，不能替代原生运行。

真实来源取得、500条适配器读取与数据哈希已有回执；CPU
fixture覆盖投影、答案隔离、准备/consumer往返及运行命令。原前三题四轮任务和评分另有上述真实NPU回执，不能混用两类证据。固定工件和复现说明已核验；仍未做500题全量，也不宣称研究侧新依赖锁定或完整新机器重建。
