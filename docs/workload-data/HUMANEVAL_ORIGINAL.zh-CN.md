# HumanEval 原始代码生成任务

所属workload：`code-generation-with-tests`。来源为OpenAI原始164道Python函数补全任务，
不是QBI改写任务或Agent交互环境。官方仓库现注明已停止维护，不将其宣传为最新代码评测。

## 验证状态

- [x] 官方提交、原压缩文件SHA及164条记录结构核查。
- [x] 原prompt及空白保持不变，参考实现、测试和entry_point只放在核验侧。
- [x] 安装本次数据包，公开生成入口在NPU完成HumanEval/0、/1、/2，均stop，生成45/115/97 token。
- [ ] 官方功能评分：未执行生成代码，无pass@1或正确率。
- [ ] 164题全量、统计代表性、性能评价及研究端新增来源消费联调。

见[脱敏回执](validation/humaneval-original-generation-npu-20261023.json)。私有预备入口跑的是同三题，
不计新增题数。来源级仍不打通过勾：生成不能替代原始测试执行和功能评分。

## 数据准备与使用

固定提交`b9453e8f1865df0a0057a1383767caa400acfbcf`，
[官方HumanEval.jsonl.gz](https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz)
44877字节，SHA256为`b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`。
官方项目采用MIT许可证，原始数据不随包分发。

安装本批精确提交的数据组件，不能只按0.1.0版本号判断能力；将原文件以原名放在显式授权目录：

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

records = list(iter_workload("HUMANEVAL", data_root=Path("/authorized/humaneval"), limit=3))
for record in records:
    prompt = record.task["input"]["prompt"]
    scorer_only = record.outcome_record()
    # 不把scorer_only传给模型；此处不执行代码。
```

```sh
python scripts/run_humaneval_original_generation.py \
  --data-root /authorized/humaneval --output /private/fresh-generation \
  --endpoint http://127.0.0.1:19466/v1 --model workload-qwen35-27b-128k-nonthinking
```

入口预选原前三题，raw completions发送完整原prompt，不套chat模板、不附答案、不截断、不拼测试。
temperature=0、max_tokens=2048、每题一次生成是明确部署配置，不声称复现原论文采样协议。 模型使用现有双910B2
NPU服务，不在CPU加载模型。输出含模型补全，只限私有保存。

## 官方评分隔离要求

2026-10-24在当前服务器重新完成[隔离能力预检](validation/code-execution-isolation-preflight-20261024.json)：user/network
namespace无害探测成功，但mount/PID隔离及原私有bubblewrap的`/bin/true`探测均因挂载传播权限失败；未发现Docker/Podman或socket。因此功能评分仍阻塞，未执行生成代码或参考代码，未改主机安全配置。此结论基于当前实例，不沿用旧服务器的占卡描述。

[官方README](https://github.com/openai/human-eval/blob/b9453e8f1865df0a0057a1383767caa400acfbcf/README.md)
要求隔离运行模型生成代码；reliability_guard不是安全沙箱。 本次从官方openEuler软件源取得bubblewrap0.8.0并验证RPM签名和载荷摘要，
仅解包到任务私有目录，无系统包安装或模型服务环境修改。 实际无害探测在创建隔离挂载时返回`Failed to make / slave: Permission denied`。
namespace探测成功不代表安全沙箱可用，因此未裸跑生成代码。

后续须提供经验证的独立代码执行环境，再按固定官方测试、终止及评分协议核验同三条补全。 不以字符串检查、自编判题、执行参考答案或改题代替；答错保留为真实结果。 CPU
CI仅做软件夹具、字段隔离和接口检查，不下载原题、不执行生成代码、不跑模型性能。
