# AI2-ARC 原版任务抽样验证

本入口覆盖知识/科学问答类独立请求，不是交互式科学 Agent 或完整 AI4Science 实验链。使用固定revision的
[AI2 ARC原始数据](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453)，包括ARC-Challenge与ARC-Easy两个子集；每个子集的原始
Parquet文件SHA均在脚本中核验，不重新造题，不拼入参考答案。

采用
[lm-eval 0.4.11 ARC配置](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.11/lm_eval/tasks/arc)及相应子集的原生任务定义：保留原始问题、候选答案、零示例、多选似然评分和
acc/acc_norm 指标。通过 completions 获取候选答案似然，**不改成聊天生成再抽取答案**。

## 数据与运行

使用者按数据条款自行取得原始文件，目录保持：

```text
/data/ai2-arc/ARC-Challenge/train-00000-of-00001.parquet
/data/ai2-arc/ARC-Challenge/validation-00000-of-00001.parquet
/data/ai2-arc/ARC-Challenge/test-00000-of-00001.parquet
```

在独立环境安装数据包与固定评测依赖，使用已有 NPU 模型服务；客户端仅分词、组织请求及评分，不加载模型权重。

```sh
python -m pip install './workload-preparation[data]'
python -m pip install 'lm-eval[api]==0.4.11' 'datasets==3.6.0' 'pyarrow==19.0.1' 'huggingface-hub==1.31.0'
python scripts/run_arc_original_samples.py --data-root /data/ai2-arc --suite ARC-Challenge --endpoint http://127.0.0.1:19466/v1/completions --model workload-qwen35-27b --tokenizer /workspace/models/Qwen3.5-27B --max-length 16384 --output /private/arc-challenge-run
python scripts/run_arc_original_samples.py --data-root /data/ai2-arc --suite ARC-Easy --endpoint http://127.0.0.1:19466/v1/completions --model workload-qwen35-27b-128k-nonthinking --tokenizer /workspace/models/Qwen3.5-27B --max-length 16384 --output /private/arc-easy-run
# 任意原题区间：显式加 --start ORIGINAL_START --count ORIGINAL_COUNT。
# 全部原始test：显式加 --full；不改变多选似然或上游评分。
```

tokenizer 必须匹配服务模型，max-length 与实际服务配置核对。脚本关闭静默截断及自动重试。输出目录必须不存在，包含原题、答案和响应，仅保留在私有存储；公开脱敏元数据，不公开运行载荷。

## 本次实测与边界

执行前分别固定两个子集的test:0–2，每个子集3题、每题产生4个候选似然请求，候选请求不算独立题目。双卡910B2和已有Qwen3.5-27B完成原文回查、执行及上游评分：Challenge
2/3、Easy 1/3，错误结果均保留。

2026-10-24另按执行前冻结的末尾题号验证新增区间入口：Challenge原题1169–1171、Easy原题2373–2375，各3题完成原生候选似然及上游评分。逐题输入、候选、参考答案与原Parquet完全一致，实际保存的doc_id集合、24个候选似然结果和运行脚本摘要已复核；见[新增原题区间审计](validation/arc-tail-native-audit-20261024.json)。本轮客户端使用lm-eval[api]
0.4.11、datasets 4.3.0、pyarrow 22.0.0、numpy 1.26.4、huggingface-hub 1.33.0及tenacity
9.2.1；服务使用当前128K模型别名。安装API可选依赖前的一次尝试在模型调用前失败，修复隔离客户端依赖后保持原定六题重试，没有换题。累计两个子集各6题、共12/3548个唯一原题；不同历史服务配置的分数不合并。

[Challenge验证记录](validation/arc-challenge-npu-20261022.json)与[Easy验证记录](validation/arc-easy-npu-20261023.json)包含历史前三题的固定版本、评分、实际范围和脱敏证据。历史范围为6/3548，新增末尾六题后累计12/3548，分批报告分数，不合并成统一准确率。来源级完成状态仍以统一交付清单为准，不表示任一子集全量、性能提升或统计代表性。旧Easy运行的引擎服务上下文上限为128K，评测tokenizer最大长度为16K；抽测输入未截断。Challenge首次bring-up与公开脚本复跑使用相同三题，不重复计数。

## 固定工件复核及范围入口补齐（2026-10-24）

固定3f397e69
CI数据wheel在服务器独立环境读取两个子集的六个原Parquet文件：文件SHA、全部行的原问题/候选与参考隔离一致，每文件原前三条prepare/resolve通过。旧两子集六题的保存结果逐题与原数据核对，未重跑模型；见[联合原资产复核](validation/reasoning-ci-assets-20261024.json)。

旧执行入口固定只跑前三题；现在保留默认0–2，增加`--start/--count`原始连续题号及`--full`全split选择，支持Easy 2376题、Challenge
1172题。所有选择都在执行前固定，不改题、不重选失败项；完整文件仍走同一官方多选似然/acc及acc_norm，不改为聊天生成。实际日志的doc_id集合用于核对执行范围，不把harness汇总effective字段误当实际题数。旧回执的helper摘要描述历史实跑文件，不冒称新增范围选择代码已经做了全量NPU执行。新增入口正做软件和固定提交CI收口，来源级暂不提前打勾。

模型工件与当前服务启动说明见[实测双NPU环境](BBH_ORIGINAL_INPUTS.zh-CN.md#%E6%9C%AC%E8%BD%AE%E5%B7%B2%E9%AA%8C%E8%AF%81%E7%9A%84%E5%8F%8Cnpu%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E9%85%8D%E7%BD%AE)。复用历史任务结果不意味着可比较不同服务配置的性能；上述旧Challenge模型别名属于历史16K服务，使用当前128K服务时应明确绑定其实际别名，并按实际部署核对`--max-length`和tokenizer，不默默切换模型。
