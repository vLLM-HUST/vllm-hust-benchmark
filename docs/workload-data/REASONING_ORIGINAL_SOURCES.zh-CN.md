# 独立推理：原始来源与增量验证

类别：`standalone-reasoning`。本批在既有 GSM8K、ARC-Challenge 基础上增加 MMLU-Pro biology/chemistry各自原始前三题及 GSM8K
连续预选样本。此类是独立知识/推理任务，不等同科学实验工作流或Agent。

本节保留历史验证截面；MMLU-Pro属于本轮五个仅分类来源，不据此开展新增接入或模型验证。当前交付状态以统一机器台账和清单为准。

## 已验证范围

| 来源     | 执行与评分                                                                                          | 未覆盖                                |
| -------- | --------------------------------------------------------------------------------------------------- | ------------------------------------- |
| GSM8K    | 固定revision下去重覆盖test:0–200共201题，另有尾部test:1316–1318三题；累计204/1319唯一题号；错误保留 | 其余1115题、全量质量及性能结论        |
| AI2-ARC  | 两子集各原前三题及末三题，均为原版似然执行与acc/acc_norm评分；累计12/3548唯一原题                   | 两子集其余3536题、全量质量及性能结论  |
| MMLU-Pro | biology前三题3/3正确；chemistry前三题3/3正确，对应原test行3405–3407、question_id3526–3528           | 其余12个学科、各学科余题、全量12032题 |

GSM8K逐批[历史快照](validation/gsm8k-incremental-progress-20261022.json)和[恢复去重审计](validation/gsm8k-resumed-coverage-20261022.json)覆盖test:0–200；[累计回执](validation/gsm8k-cumulative-coverage-20261023.json)再纳入尾部1316–1318，合计204个唯一ID，不是全量。初始三题和后续批次使用的运行配置不完全相同，不把跨配置覆盖数当成同规格准确率或性能比较。显式sample
ID时lm-eval 0.4.11的汇总effective字段会显示全split，脚本据实际logged doc_id校验唯一任务数，不能把该字段当实际运行1319题。

MMLU-Pro
biology原前三条完成NPU评分及数据组件准备/回查；chemistry则按官方学科任务配置直接使用同一原始test文件，执行原行3405–3407。每个学科实际模型任务均只有三条。biology回查确认问题/选项原文一致，task
input未混入答案字段。见[biology脱敏回执](validation/mmlu-pro-biology-npu-20261022.json)与[chemistry脱敏回执](validation/mmlu-pro-chemistry-npu-20261023.json)。

## 固定数据及官方执行器

- GSM8K：`openai/gsm8k`，revision `740312add88f781978c0658806c59bc2815b9866`。
- MMLU-Pro：`TIGER-Lab/MMLU-Pro`，revision
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`。取该版本完整test/validation
  parquet，脚本核对完整SHA；不改题或重新生成fewshot。
- `lm-eval[api]==0.4.11`、`datasets==3.6.0`、`pyarrow==19.0.1`、`huggingface-hub==1.31.0`。MMLU-Pro保留对应学科原生过滤、5条validation示例、提示、2048生成预算、stop及答案正则评分。
- 现有双卡910B2、Qwen3.5-27B、vLLM-Ascend graph、BF16、本地NPU。chat传输与非思考设置明确披露，不等同上游所有参考设置。

## 复现

```sh
python scripts/run_gsm8k_incremental_samples.py \
  --data-root /authorized/gsm8k/main --endpoint http://127.0.0.1:19466/v1/chat/completions \
  --model workload-qwen35-27b-128k-nonthinking --start 3 --count 6 \
  --output /private/fresh-gsm-run
python scripts/run_mmlu_pro_original_samples.py \
  --data-root /authorized/mmlu-pro/data --endpoint http://127.0.0.1:19466/v1/chat/completions \
  --model workload-qwen35-27b-128k-nonthinking --subject biology --output /private/fresh-mmlu-run
```

使用新建私有输出目录并设置umask
077；private-results包含原题、参考答案与响应，不提交Git/CI。测试集由使用者按上游条款取得。CI只做CPU参数/安全/软件检查，不安装模型或评分任务环境，也不访问真实数据。

小数据集按实际推理成本评估是否全量；本批三套的全量任务仍多于抽样范围，不能因为数据文件仅数MB就称全量验证成本小。增量队列在没有原生长任务和服务请求时继续新样本，不空转，不为模型答错而换样本。

## 固定工件及完整可用范围（2026-10-24）

GSM8K原train/test文件与ARC两子集六文件在服务器均通过完整SHA、全行原输入/参考字段比对，以及逐文件原前三条prepare/resolve回查；固定数据组件来自3f397e69的实际CI
wheel。GSM8K尾部三题及ARC各三题的保存原生结果逐题与当前原文件一致，原metrics保留，没有新模型调用或重算分数。见[联合复核回执](validation/reasoning-ci-assets-20261024.json)。GSM8K的204/1319累计去重范围继续由原累计回执证明，不把三条尾部回查冒充重新验证全部204次模型执行。

GSM8K入口的`--start/--count`可选择原test连续范围；`--start 0 --count 1319`执行全test，但本轮未跑全量。保持上游5条train
fewshot、提示、stop及严格/宽松数值提取与exact_match；2048生成预算是声明的模型调用配置，固定上游gsm8k.yaml未规定另一条必须保留的max_gen_toks，不以缩短预算凑通过。`--data-root`须直接包含train/test
Parquet，即组件规范目录`EXTERNAL_GSM8K_ROOT/main`，而组件自身数据根目录是其父目录。

模型/服务完整配置见[实测双NPU环境](BBH_ORIGINAL_INPUTS.zh-CN.md#%E6%9C%AC%E8%BD%AE%E5%B7%B2%E9%AA%8C%E8%AF%81%E7%9A%84%E5%8F%8Cnpu%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E9%85%8D%E7%BD%AE)；运行客户端使用上文独立固定harness环境，不能覆盖共享服务环境。原始数据可按固定Hub
revision由`huggingface_hub.snapshot_download`取得到外部目录，GSM8K选择`main/train-00000-of-00001.parquet`和`main/test-00000-of-00001.parquet`；ARC选择其两子集train/validation/test六文件。遇到下载通道变化，只允许验证为同一固定原资产的传输，不替换题库。新结果、参考及完整索引只留私有目录。
