# BigCodeBench 原始任务生成阶段验证

## 已完成范围

2026-10-23 在现有双卡 Ascend 910B2、Qwen3.5-27B、TP=2、BF16 服务上， 预选原始 v0.1.4 文件前三条 BigCodeBench/0、/1、/2，使用
Instruct 视图生成。 与先前读取检查是同三条任务，不累计为六个独立任务。

| 原始任务       | prompt tokens | completion tokens | 终止 | 全输入预检匹配 |
| -------------- | ------------: | ----------------: | ---- | -------------- |
| BigCodeBench/0 |           150 |               287 | stop | 是             |
| BigCodeBench/1 |           150 |               226 | stop | 是             |
| BigCodeBench/2 |           130 |               348 | stop | 是             |

原数据固定于
[b74c0d0b 版本](https://huggingface.co/datasets/bigcode/bigcodebench/tree/b74c0d0bf70d2c0bc459be537895cca163007f1a)，
文件为 data/v0.1.4-00000-of-00001.parquet，1140条；按来源目录核验摘要。 未缩短或改写原任务，参考实现与测试代码未送入模型。

## 原版协议与差异

固定发布方
[09dd993f 代码](https://github.com/bigcode-project/bigcodebench/tree/09dd993f46c3fbf3a799465bb96d524edcb0b199)，
直接加载未修改的 provider/utility.py 中 make_raw_chat_prompt， 以及 gen/util/openai_request.py 中
make_request；执行前核验原文件摘要。 采用发布方 run_codegen 默认 instruction/response prefix 与1280输出预算，
greedy为temperature=0、n=1，请求函数保留其top_p=0.95。 首尾strip是发布方函数行为，磁盘原始prompt不变。

本次不是完整官方CLI/参考环境复现：模型替换为已有本机NPU服务，服务默认关闭thinking。
未使用发布方无限重试循环，而调用单次请求函数，客户端timeout=300、max_retries=0。 未调用sanitize、代码执行或评分函数，也没有观测/采集钩子。
请求和输出仅保留于服务器私有目录，公开记录不含生成代码或答案。

## 检查清单

- [x] 三个原始任务读取、prepare、resolve输入/参考独立往返。
- [x] 原版提示/请求函数驱动NPU生成；三条完整输入预检匹配、正常stop。
- [ ] 可复用公开生成入口、CPU协议测试及交接复跑。
- [ ] 原版功能测试与适用pass@1评分。
- [ ] 完整1140条验证、本批最终工件及研究端交接。

当前未找到Docker/Podman或Docker socket，已有HumanEval隔离探测也未通过。 这不证明每种沙箱都不可行，但不能绕过隔离裸执行生成代码。
必须记为生成阶段完成、功能评分未运行，不能据HTTP成功或stop宣称代码正确、 官方pass@1、完整任务验证或性能提升。
