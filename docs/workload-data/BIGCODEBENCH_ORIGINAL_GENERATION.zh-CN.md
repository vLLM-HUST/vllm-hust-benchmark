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

## 独立生成入口

`scripts/run_bigcodebench_original_generation.py` 只执行生成阶段，原始数据与发布方代码 由使用者在私有目录取得。`--official-code`
应包含上述固定版本的 bigcodebench/provider/utility.py 和 bigcodebench/gen/util/openai_request.py；
入口校验其字节摘要，不修改或分发第三方代码。

本次独立客户端环境为 Python3.12、openai3.1.0、transformers5.5.4、pyarrow19.0.1。
这些是本次探测版本，不是发布方完整官方环境；没有加载CPU模型或CUDA推理。

```bash
python -m pip install openai==3.1.0 transformers==5.5.4 pyarrow==19.0.1
python scripts/run_bigcodebench_original_generation.py \
  --data-root /private/bigcodebench \
  --official-code /private/bigcodebench-code \
  --output /private/bigcodebench-generation-new \
  --model workload-qwen35-27b-128k-nonthinking \
  --origin http://127.0.0.1:19466 --context-capacity 131072
```

原始数据根目录下须有 data/v0.1.4-00000-of-00001.parquet。输出目录必须不存在， 且不能与原始数据或代码目录互相包含。只允许无凭据的本机HTTP服务，不自动下载、
换模型、缩短题目或执行生成代码。完整输入先经服务tokenize预检，随后核对实际usage。 length单独标为截断；空输出、请求失败或预检不符保留失败类型。
只在三条生成均正常stop时返回0，但返回0仍不代表功能正确、评分或全套完成。 原始请求和响应留在权限受限的私有目录，不在终端打印代码或答案。

## 检查清单

当前服务器已重做[隔离预检](validation/code-execution-isolation-preflight-20261024.json)：网络namespace可创建，但mount/PID隔离和私有bubblewrap无害探测失败，且无Docker/Podman/socket。不能据此安全执行生成代码；原功能评分仍未运行。保存的生成结果继续复用，待可验证的隔离环境就绪后接原sanitize/功能测试及评分，不裸执行、不重生成刷分。

- [x] 三个原始任务读取、prepare、resolve输入/参考独立往返。
- [x] 原版提示/请求函数驱动NPU生成；三条完整输入预检匹配、正常stop。
- [x] 可复用公开生成入口、CPU协议测试及服务器同三题交接复跑；结果一致，不累计独立任务数。
- [ ] 原版功能测试与适用pass@1评分。
- [ ] 完整1140条验证、本批最终工件及研究端交接。

当前未找到Docker/Podman或Docker socket，已有HumanEval隔离探测也未通过。 这不证明每种沙箱都不可行，但不能绕过隔离裸执行生成代码。
必须记为生成阶段完成、功能评分未运行，不能据HTTP成功或stop宣称代码正确、 官方pass@1、完整任务验证或性能提升。
