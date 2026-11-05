# Qwen3.5-35B-A3B：INT8 KV 准确率配对

2026-11-05，在同一宿主、模型和双卡环境下，切换 BF16 / INT8 KV cache，完成 GSM8K 固定前 200 题配对。本轮子集未观察到准确率下降；不代表总体提升、统计非劣或长上下文无损。

## 结果

| 指标 | BF16 KV (`auto`) | INT8 KV (`int8`) |
| --- | ---: | ---: |
| 正确题数 | 194 / 200 | 195 / 200 |
| 准确率 | 97.0% | 97.5% |
| HTTP 成功请求 | 200 / 200 | 200 / 200 |
| 输出达到长度上限 | 1 | 2 |

INT8 相对 BF16 为 **+0.5 个百分点**。严格匹配与宽松数字提取结果一致；`test:89` 从错变对，没有从对变错，其余 199 题正确性一致。截断题保留原评分，没有排除或事后改判。

两组正式请求的规范化 SHA256 多重集合一致，题目 ID、文档摘要、提示摘要逐题一致。实际输入为 **761–1654 token**，不覆盖 16K/32K 长上下文。

## 模型、服务与评分配置

- 模型：`/data/shared_models/Qwen3.5-35B-A3B`，权重 BF16，TP=2；逻辑设备 `ASCEND_RT_VISIBLE_DEVICES=0,1`，本轮物理 NPU 2/7，均为 910B2。
- 两臂采用同一 `vllm-hust-dev` 环境及插件工作区；唯一主动切换的服务参数为 `--kv-cache-dtype auto/int8`。
- `max-model-len=16384`、`max-num-seqs=8`、`max-num-batched-tokens=2048`、`gpu-memory-utilization=0.80`；开启 chunked prefill，关闭 prefix caching，无 speculative decoding，使用 `FULL_DECODE_ONLY` graph。
- 复用 benchmark 仓库 `scripts/run_gsm8k_incremental_samples.py`，原始 GSM8K 5-shot 提示和 exact-match 评分保持不变。客户端并发改为 8，显式设置 `temperature=0`、`seed=0`，关闭 thinking，输出上限 2048 token，保留原 stop 条件。
- `lm-eval=0.4.11`、`datasets=3.6.0`、`pyarrow=19.0.1`、`huggingface-hub=2.2.0`、`transformers=5.19.0`。后两项不同于旧示例环境，但两臂完全相同。
- 正式请求前，各臂使用相同训练集预热：前五题带答案、第六题不带答案，最多生成 256 token。插件首次有效 K/V 校准 scale 后复用；预热不计入正式评分。

## 版本与结论边界

| 仓库 | 运行时 HEAD |
| --- | --- |
| vllm-hust | `4f61604bea01ce56afc320e0f0566166fd79a562` |
| vllm-ascend-hust | `e5870e657a2add3182551c70bbad5919fc96d899` |
| 本插件 | `2c2866b9991d056c869191f0c1536c3421055ea2` |
| vllm-hust-benchmark | `2cc0fae34b57a639c819da38ea33af2235f1973f` |

运行包含未提交修改，**不能只用 HEAD 复现**。原目录保存工作区 diff、status 和插件 Python 源码快照；configuration 中记录源码 SHA256，配对期间插件源码摘要一致。模型 JSON 有摘要；权重分片仅记录 size/mtime，未做完整字节哈希。

这是固定子集、固定校准与并发下的一轮比较，不是 GSM8K 全量成绩，也不是官方榜单数据。summary 中的 bootstrap 区间仅对固定子集重采样，不构成总体非劣证据。INT4 未在此轮评估。

`lm-eval` 的 `n-samples.effective` 显示 1319 是该版本显式选择题目时的计数限制；实际日志验证为 200 个独立 ID。每臂 400 条 harness 样本对应两种 filter，各 200 题，不是 400 个独立问题。

## 保存的证据与复算

仓库内保存：

- [逐题标量与配对汇总](./summary.json)。
- [环境、版本和源码摘要](./configuration.json)。
- [文件校验与完整原件定位](./manifest.json)；同目录保存两臂启动命令、评分 receipt 和硬件记录。

完整题文、参考答案、模型输出、HTTP 响应、源码快照及评分环境保留在本机 `/root/kv-int8-accuracy-20261105`。manifest 的 `external_evidence` 记录路径和 SHA256；这些外部原件没有随插件复制，单独克隆插件不能恢复完整原件。

在保留原件的机器上复算：

```bash
/root/kv-int8-accuracy-20261105/client/bin/python \
  /root/kv-int8-accuracy-20261105/summarize.py
```

首次准备尝试的下载校验失败和等待超时保存在 `/root/kv-int8-accuracy-20261104`，未产生正式评分，不混入本结果。成功配对结束后服务已停止，NPU 已释放。
