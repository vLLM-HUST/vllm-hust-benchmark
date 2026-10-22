# 独立推理：原始来源与增量验证

类别：`standalone-reasoning`。本批在既有 GSM8K、ARC-Challenge 基础上增加 MMLU-Pro biology 原始前三题及 GSM8K
连续预选样本。此类是独立知识/推理任务，不等同科学实验工作流或Agent。

## 已验证范围

| 来源          | 执行与评分                                                                    | 未覆盖                                  |
| ------------- | ----------------------------------------------------------------------------- | --------------------------------------- |
| GSM8K         | 原test:0–2既有评分；增量按固定连续原ID分批执行，去重快照165/1319；错误保留    | 剩余题、全量质量及性能结论              |
| ARC-Challenge | 既有原test:0–2，官方似然评分，2对1错                                          | ARC-Easy及Challenge其余题               |
| MMLU-Pro      | biology筛选后原始前三题，对应原test行2688–2690、question_id2804–2806；3/3正确 | 其他学科、biology剩余714题、全量12032题 |

GSM8K增量[快照](validation/gsm8k-incremental-progress-20261022.json)逐批记录ID及回执SHA，去除重复运行；不是全量完成。初始三题使用既有配置，增量使用独立128K/非思考配置，不把跨配置合计覆盖数当成同规格性能或总体准确率比较。显式sample
ID时lm-eval 0.4.11的汇总effective字段会显示全split，脚本据实际logged doc_id校验唯一任务数，不能把该字段当实际运行1319题。

MMLU-Pro模型执行与数据组件准备/回查分别验证：准备2691条引用仅为了覆盖原行2688–2690，实际模型任务只有三条。回查确认问题/选项原文一致，task
input未混入answer、answer_index、cot_content。见[脱敏回执](validation/mmlu-pro-biology-npu-20261022.json)。

## 固定数据及官方执行器

- GSM8K：`openai/gsm8k`，revision `740312add88f781978c0658806c59bc2815b9866`。
- MMLU-Pro：`TIGER-Lab/MMLU-Pro`，revision
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`。取该版本完整test/validation
  parquet，脚本核对完整SHA；不改题或重新生成fewshot。
- `lm-eval[api]==0.4.11`、`datasets==3.6.0`、`pyarrow==19.0.1`、`huggingface-hub==1.31.0`。MMLU-Pro保留该版本biology原生过滤、5条validation示例、提示、2048生成预算、stop及答案正则评分。
- 现有双卡910B2、Qwen3.5-27B、vLLM-Ascend graph、BF16、本地NPU。chat传输与非思考设置明确披露，不等同上游所有参考设置。

## 复现

```sh
python scripts/run_gsm8k_incremental_samples.py \
  --data-root /authorized/gsm8k --endpoint http://127.0.0.1:19466/v1/chat/completions \
  --model workload-qwen35-27b-128k-nonthinking --start 3 --count 6 \
  --output /private/fresh-gsm-run
python scripts/run_mmlu_pro_original_samples.py \
  --data-root /authorized/mmlu-pro/data --endpoint http://127.0.0.1:19466/v1/chat/completions \
  --model workload-qwen35-27b-128k-nonthinking --output /private/fresh-mmlu-run
```

使用新建私有输出目录并设置umask
077；private-results包含原题、参考答案与响应，不提交Git/CI。测试集由使用者按上游条款取得。CI只做CPU参数/安全/软件检查，不安装模型或评分任务环境，也不访问真实数据。

小数据集按实际推理成本评估是否全量；本批三套的全量任务仍多于抽样范围，不能因为数据文件仅数MB就称全量验证成本小。增量队列在没有原生长任务和服务请求时继续新样本，不空转，不为模型答错而换样本。
