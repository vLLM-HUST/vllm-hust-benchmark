# ARC-Challenge 原版任务抽样验证

本入口覆盖知识/科学问答类独立请求，不是交互式科学 Agent 或完整 AI4Science 实验链。使用
[AI2 ARC 原始数据](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453)，版本固定；三个原始
Parquet 文件的 SHA 在脚本中核验，不重新造题，不拼入参考答案。

采用
[lm-eval 0.4.11 ARC-Challenge 配置](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/arc/arc_challenge.yaml)及其继承的任务定义：保留原始问题、候选答案、零示例、多选似然评分和
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
python scripts/run_arc_original_samples.py --data-root /data/ai2-arc --endpoint http://127.0.0.1:19466/v1/completions --model workload-qwen35-27b --tokenizer /workspace/models/Qwen3.5-27B --max-length 16384 --output /private/arc-new-run
```

tokenizer 必须匹配服务模型，max-length 与实际服务配置核对。脚本关闭静默截断及自动重试。输出目录必须不存在，包含原题、答案和响应，仅保留在私有存储；公开脱敏元数据，不公开运行载荷。

## 本次实测与边界

执行前固定 test:0–2，3 题产生 12 个候选似然请求，不能算成 12 题。双卡 910B2、已有 Qwen3.5-27B、vLLM-Ascend graph
模式完成原文回查、执行及上游评分，2 题正确、1 题错误，均保留。

[验证记录](validation/arc-challenge-npu-20261022.json)包含配置、分词器及脚本指纹、实际范围和聚合结果。清单勾选表示三条完成接入闭环，不表示模型全答对、ARC-Easy
已验证、1172 题全量已跑、性能提升或统计代表性。首次 bring-up 与公开脚本复跑使用相同三题，不重复计数。
