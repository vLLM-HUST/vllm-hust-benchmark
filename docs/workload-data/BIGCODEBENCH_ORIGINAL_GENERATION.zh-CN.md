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

2026-10-25入口扩展：默认仍选原始0、1、2；可添加 `--indices 5 8 11`、
`--start 3 --count 3` 或 `--full`（1140题）选择模式。显式ID须有序、唯一、至少三条，
不得与范围/全量模式混用；私有结果文件名、回执和准备后的答案均保留原始位置及任务ID。
准备入口自动复用生成回执中的选择，并保留完整1140条原始数据供官方
`BIGCODEBENCH_OVERRIDE_PATH` 使用。所有模式仍采用原版Instruct提示、1280预算、
greedy设置，不因失败换题或重写输出。退出0要求全部选定题正常stop，而非仅默认三题。

这些通用选择模式现有CPU夹具测试，尚无新服务器运行证据；上文2026-10-23的回执仍指向
当时固定前三题及旧脚本摘要，未改写为新版本已验证。后续功能评分仍须执行原版sanitize、
原始测试和官方评分，生成及准备成功均不足以勾选完成。

## 检查清单

### 2026-10-25 后续评分准备（尚未完成）

已在服务器获取上述固定 `09dd993f` 版本的完整官方源码归档，并解包至独立私有目录；
未重新生成已有三题，也未执行其候选代码。官方 `Docker/Evaluate.Dockerfile` 使用
Python 3.10，并安装独立的 `Requirements/requirements-eval.txt`。其构建步骤还引用动态
main 分支的依赖文件，复现时应使用本次固定源码中的文件，记录最终实际依赖，不能声称
仅下载源码即已复现官方环境。

当前待办：核验官方 sanitize、原始测试和本地评分入口，准备隔离环境及其依赖，复用
已保存三题输出执行原生功能评分，再核验公共入口及研究侧消费。此前 HumanEval、
DS1000 的隔离验证不能替代 BigCodeBench 的依赖和运行验证；本项仍不打勾。

可先运行静态预检（只读取原始数据和源码，绝不导入官方模块或执行候选代码）：

```bash
python scripts/audit_bigcodebench_scoring_assets.py \
  --official-code /private/bigcodebench-code \
  --data-root /private/bigcodebench
```

该入口复用生成阶段的 PyArrow 数据读取依赖，核对固定源码摘要、原始 Parquet 摘要和
完整1140条原始ID顺序，列出固定前三题参考实现及测试的静态导入。动态导入、运行时资源、
sanitize、参考实现检查和候选代码评分均不在其证明范围内。当前只通过软件夹具测试；
上传执行时 SSH 32048 返回连接拒绝，尚无服务器执行本入口的回执，不能计为原生验证。

2026-10-24历史服务器的[隔离预检](validation/code-execution-isolation-preflight-20261024.json)：网络namespace可创建，但mount/PID隔离和私有bubblewrap无害探测失败，且无Docker/Podman/socket。该历史结论不代表2026-10-25已验证的其他来源隔离环境；BigCodeBench自身功能评分仍未运行。保存的生成结果继续复用，待其隔离环境与依赖验证后接原sanitize/功能测试及评分，不裸执行、不重生成刷分。

- [x] 三个原始任务读取、prepare、resolve输入/参考独立往返。
- [x] 原版提示/请求函数驱动NPU生成；三条完整输入预检匹配、正常stop。
- [x] 可复用公开生成入口、CPU协议测试及服务器同三题交接复跑；结果一致，不累计独立任务数。
- [ ] 原版功能测试与适用pass@1评分。
- [ ] 完整1140条验证、本批最终工件及研究端交接。

当前未找到Docker/Podman或Docker socket，已有HumanEval隔离探测也未通过。 这不证明每种沙箱都不可行，但不能绕过隔离裸执行生成代码。
必须记为生成阶段完成、功能评分未运行，不能据HTTP成功或stop宣称代码正确、 官方pass@1、完整任务验证或性能提升。
