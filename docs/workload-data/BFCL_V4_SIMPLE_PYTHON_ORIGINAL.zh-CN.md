# BFCL v4 simple_python 原始函数调用任务

登记类别现已完成[400题原生交付](validation/bfcl-native-delivery-20261024.json)。[完整审计](validation/bfcl-native-complete-audit-20261024.json)覆盖原0–399，复用首尾六题，仅新增中间394题，题号无重复/遗漏；全部原prompt、输入token、响应摘要及官方AST
valid判定一致，固定CI包400条索引/prepare→resolve通过。以下保留历史阶段记录，较早“中间待验”不代表当前状态。原始模型服务配置见[双NPU环境](BBH_ORIGINAL_INPUTS.zh-CN.md#%E6%9C%AC%E8%BD%AE%E5%B7%B2%E9%AA%8C%E8%AF%81%E7%9A%84%E5%8F%8Cnpu%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E9%85%8D%E7%BD%AE)。不将simple_python完整覆盖称为全部BFCL或真实工具执行。

所属workload为`tool-call-generation`。本批只覆盖原始单轮函数选择与参数生成， 官方AST checker匹配函数名及参数，不执行工具或模型生成代码。
不能据此声称应用闭环、检索、记忆、多轮、多函数、并行调用或全部BFCL已完成。

## 验证状态

2026-10-24已完成[首尾六题及固定CI工件审计](validation/bfcl-native-edges-audit-20261024.json)：原0–2复用，不重跑；新增397–399由新入口实际生成及官方评分。原handler重建prompt、完整输入token数、保存响应摘要及原AST
valid判定均一致。固定3f397e69
CI包全400题原字段及索引一致，6条prepare/resolve一致；全部110个原运行源码文件与归档字节匹配。0新增审计生成调用，静态评分未执行工具或生成代码。中间3–396共394题仍待终态/审计，未完成前来源级不打勾。

2026-10-24补齐`--start/--count`原始连续题号及`--full`完整400题入口；默认仍为原0–2。选择在模型调用前冻结并与原任务逐字段核对，不重复或替换失败题，其他BFCL类别仍不在此入口范围。新入口改为直接调用固定官方`eval_runner._evaluate_single_ast_entry`，保留官方AST解析失败/输出格式错误的invalid判定，而非把模型答错当程序崩溃；原预算内length若已取得响应并完成原评分，按实际终态记录，不仅因非stop就判链路未完成。

4项CPU测试及代码检查通过；在原BFCL独立环境用人为无效输出夹具验证官方函数返回`ast_decoder:decoder_failed`且valid=false，0模型调用、未执行工具或代码。这不是原题模型实测。历史三题回执保持旧脚本身份；新增范围与评分入口仍待新的原题NPU验证，不提前打勾。

- [x] 官方固定源码/任务/独立评分参考字节及400条ID对齐核验。
- [x] 安装根包和数据包后，通过公开准备命令保持原文件名和内容。
- [x] BFCL原前三题simple_python_0/1/2的公开入口NPU复验，均stop，官方valid均为true。
- [x] 安装的110个上游Python文件与固定官方归档逐字节一致；独立环境pip check通过。
- [x] 登记simple_python完整400题原生生成、官方AST静态评分与安装工件回查。
- [ ] 其他BFCL类别、统计代表性、真实工具执行或性能评价。
- [ ] 本批新数据API的研究端精确工件消费联调。

[脱敏回执](validation/bfcl-v4-simple-python-public-handoff-npu-20261023.json)。私有预备和公开复验是同三题，
不重复计数。部分官方结果同时包含valid=true及error_type=unclear，按原valid字段记录静态判定，原字段一并保留。
这不是实际工具执行正确率，也不解决HumanEval安全沙箱受阻的功能评分。

## 固定原始来源

官方ShishirPatil/gorilla提交`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`，
[源码归档](https://api.github.com/repos/ShishirPatil/gorilla/zipball/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8)
45045507字节，SHA256为`23db73c7b67850696bf4316a92fe97ba54a5dfb46e822bc294e5158b16efffb0`。
所用项目位于berkeley-function-call-leaderboard/，其pyproject声明Apache2.0许可证；原数据和第三方运行包不随本项目wheel重新分发。

- 原任务BFCL_v4_simple_python.json：283274字节、400条，SHA256为`82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991`。
- 独立参考possible_answer/BFCL_v4_simple_python.json：63627字节、400条，SHA256为`90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc`。

原.json文件实际采用逐行JSON。读取器按显式row_format=jsonl解析，不改后缀、不猜测或重写数据； 默认.json数组行为不变。格式声明贯通绑定、直接读取、索引生成及原行回查。
任务输入只含原question/function，评分参考保留为单独资产，不进入模型输入或研究索引。

## 安装与准备

先安装本批精确提交的根包与数据组件，不能只按0.1.0版本号判断能力。

```sh
workload-data prepare-bfcl-simple-python \
  --official-archive /authorized/bfcl-source.zip --output /private/fresh-bfcl-data
```

API为`vllm_hust_workload_data.provider.prepare_bfcl_simple_python`。仅提取固定两个成员到新私有目录，
检查原SHA及行数，不下载、不调用模型或评分。准备成功不代表任务完成。

完整原生运行环境与引擎环境分离，使用Python3.12；Ascend主机对这个准备/评分进程清除继承的
PYTHONPATH、PYTHONHOME、LD_PRELOAD，不修改模型服务环境。原源码按固定归档解包，不打代码补丁。

```sh
python3.12 -m venv /private/bfcl-env
/private/bfcl-env/bin/python -m pip install --no-deps torch==2.8.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
/private/bfcl-env/bin/python -m pip install \
  -r configs/workload-native/bfcl-simple-python-py312-dependencies.txt
/private/bfcl-env/bin/python -m pip install --no-deps \
  /authorized/gorilla-source/berkeley-function-call-leaderboard
/private/bfcl-env/bin/python -m pip check
```

CPU版Torch只满足官方SDK/Transformers导入依赖，不加载或运行CPU模型。 不安装vLLM/sglang/CUDA extras；模型推理始终由已有双910B2 NPU服务完成。
该源码归档采用官方SCM fallback版本0.0.0.dev0，实际身份以源码提交、归档SHA和110文件字节核验为准， 不将fallback字符串冒充作者正式发布版本。
曾发现qwen-agent默认依赖缺少soundfile，首次导入失败且0模型调用；固定快照已补soundfile0.13.1，未修改上游Python。

## 原生执行入口

```sh
env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD /private/bfcl-env/bin/python \
  scripts/run_bfcl_v4_simple_python_original.py \
  --official-archive /authorized/bfcl-source.zip --data-root /private/fresh-bfcl-data \
  --output /private/fresh-bfcl-run --tokenizer /workspace/models/Qwen3.5-27B \
  --endpoint http://127.0.0.1:19466/v1 --model workload-qwen35-27b-128k-nonthinking
```

原QwenFCHandler负责原任务预处理、提示格式、查询和解析，原ast_checker评分；不调用decode_execute或真实工具。
保留官方OSSHandler的min(4096,剩余上下文-2)预算，temperature=0，完整输入token预检与实际233/170/257一致。
tokenizer仅从本地读取，禁止下载远程代码；不启动或重启推理服务。

官方该版本没有Qwen3.5注册项。本批仅新增运行时模型路由配置，沿用原本地Qwen/Qwen3-32B-FC的
QwenFCHandler与underscore_to_dot=false，模型名/URL指向实际Qwen3.5部署。
这不是切换到Qwen3-32B权重，也不是作者原模型/论文协议等价声明，不改题目、参数规则或评分器源码。

输出含原提示、模型内容和静态评分明细，只限私有保存。公开只有来源、ID、SHA、配置、终态和静态判定。 CPU CI只跑软件夹具、JSONL/索引回查、字段隔离及接口检查，不下载原题、不跑模型/性能。
没有迁入科研观测、采集钩子、机制干预、特征挖掘或研究分析。
