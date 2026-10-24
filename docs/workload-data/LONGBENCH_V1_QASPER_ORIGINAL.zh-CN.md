# LongBench v1 Qasper 原始文档问答

本批只接入LongBench v1发布的Qasper科学文档问答子集，对应`document-grounded-analysis`。
不是单独原版Qasper全库、LongBench-E、LongBench v2或其他LongBench任务已验证。

## 验证范围

- [x] 固定官方数据/源码归档、SHA和Qasper原始200条记录。
- [x] 安装根包与数据包后，通过公开命令提取原始文件，不改字节。
- [x] 原前三题完整文档、官方提示、原128-token输出预算，NPU生成、正常stop和官方QA-F1评分。
- [x] 公开执行入口同三题复验，官方F1百分制为19.05、36、12.5，汇总22.52。
- [x] 当前固定CI数据包逐文件匹配，原200条文档/问题/ID及参考字段逐条相等，全部准备索引，首尾6条完成prepare→resolve回查；复用旧三题保存请求，原提示、128-token预算及官方评分重算均一致，模型/Tokenizer
  API调用0。[CPU审计回执](validation/longbench-qasper-cpu-audit-20261024.json)。
- [x] 公开入口补齐`--start/--count`原始行范围及`--full`全200题选项；越界或全量/范围冲突拒绝，不按模型结果筛题。新原生运行使用原行3..199，不重跑已有前三题。
- [x] 原200题均已实际执行并完成官方QA-F1评分：复用3题、新跑197题；全量原输入/参考答案、请求token计数和私有响应SHA复核，逐题及全部200题官方评分均重新计算一致，总体QA-F1百分制50.70。预算仍为128，不改题、不截断。见[全200题原生回执](validation/longbench-qasper-all200-native-20261024.json)。
- [ ] 其他LongBench子集/类别；不由本Qasper结果推定完成。
- [x] 固定CI数据包32份文件及研究消费包40份Python文件与固定源码逐字节一致；服务器独立干净环境完成原前三题的公共索引→研究读取及研究直接prepare→resolve两条路径，完整输入、参考和来源身份一致，pip
  check通过。[研究交接回执](validation/longbench-qasper-research-handoff-20261024.json)。不代表200条全部经过研究端，也未由此重跑模型。

[公开入口复验回执](validation/longbench-v1-qasper-public-handoff-npu-20261023.json)
只证明具名三题完整接入，F1不是答题正确率，也不是通过质量阈值。 私有预备运行和公开复验使用同三条ID，不重复增加任务数。

该三题回执是历史证据；后续197题与前三题组成原200题全覆盖，来自两个执行批次，不伪称单次连续运行。全量复核仍不等于模型质量阈值、场景代表性或引擎性能评价，也不是原版Qasper全库、检索链路或科学工具执行已验证。具名范围交付见[完成回执](validation/longbench-qasper-native-delivery-20261024.json)。

## 固定来源及数据准备

- 数据：THUDM/LongBench，提交`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`。
- [官方data.zip](https://huggingface.co/datasets/THUDM/LongBench/resolve/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip)：
  113932529字节，SHA256为`cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`，与发布方LFS摘要一致。
- 原成员`data/qasper.jsonl`：4829368字节、200条，SHA256为`29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854`。
- [官方源码归档](https://api.github.com/repos/THUDM/LongBench/zipball/2e00731f8d0bff23dc4325161044d0ed8af94c1e)：
  SHA256为`2f92f9953e3dd2fd3915a132a0a4cb53acb879af6ef1e98962129a6f3fe262bb`。
  使用其中LongBench/目录下的原eval.py、metrics.py、dataset2prompt和dataset2maxlen。

安装本批精确提交的根包和独立数据包；仅凭0.1.0版本号不足以判断功能。原始数据、题目、答案与模型输出不随wheel分发。

```sh
workload-data prepare-longbench-qasper \
  --data-archive /authorized/data.zip --output /private/fresh-qasper-data
```

只提取固定原始成员到新目录，检查归档SHA、成员SHA及行数。API为
`vllm_hust_workload_data.provider.prepare_longbench_qasper`，不下载、不调用模型或评分。
准备成功不代表任务执行。来源记录只将context/input传入任务输入，answers/all_classes留在核验侧； 原all_classes为null时保留null，不伪造标签。

## 独立执行环境与运行

不改模型服务环境，在独立Python>=3.10环境安装本批根包/数据包及以下实测评分依赖：

```sh
python -m pip install numpy==2.2.6 jieba==0.42.1 rouge==1.0.1 fuzzywuzzy==0.18.0 six==1.17.0
python -m pip check
python scripts/run_longbench_v1_qasper_original.py \
  --data-root /private/fresh-qasper-data --official-archive /authorized/longbench-source.zip \
  --output /private/fresh-qasper-run --endpoint http://127.0.0.1:19466/v1 \
  --model workload-qwen35-27b-128k-nonthinking --context-limit 131072
```

默认仍选择原始前三题。添加`--start 197 --count 3`选择原文件最后三题，添加`--full`选择完整200题；不能将`--full`与非默认范围混用。沿用旧三题的后续验证可用`--start 3 --count 197`执行尚未验证的部分，不重跑0–2；是否完整执行仍按预估30分钟规则确定。原文件SHA、200条顺序和既有前三题ID在执行前核对，私有输出中的行号/文件名保留原始位置。任何范围仍使用完整文档、相同官方提示与原128-token预算，超上下文停止，不截断或换题。

Ascend主机应对该独立进程清除继承的PYTHONPATH、PYTHONHOME、LD_PRELOAD，避免CANN包污染准备/评分环境，
不更改模型服务的环境。评分在CPU执行普通文本指标，模型推理仍由现有双910B2 NPU服务完成。 context-limit是明确部署配置，不是测得的性能；输入不适配时停止，不截断、不换题。

## 模型传输与评分边界

官方旧build_chat没有Qwen3.5适配。本批使用其原始Qasper提示正文作为单条user消息，
再由现有模型的原生聊天模板编码；temperature=0、enable_thinking=false明确披露。
这是模型部署适配，不宣称作者原模型/容器/论文采样协议等价。三条输入token预检与实际使用量一致：4103、3440、4461。
保持原128-token输出预算、完整文档和提示；原eval.py的scorer负责QA-F1及多个原参考答案取最大值，不重写评分器。

此前raw completion诊断三条均耗尽128-token预算，官方F1为2.05，见
[历史诊断](validation/longbench-v1-qasper-raw-diagnostic-npu-20261023.json)。其结果保留，不冒充通过、不删除失败证据。
两种传输不能当作引擎性能提升的因果基线。未读取答案来改题或扩大预算。

旧执行器曾要求finish_reason为stop才具备来源勾选资格，该历史回执保持原样。当前入口将合法length终止与服务/流程错误区分：保持128-token原预算及原输出，获得完整官方评分的任务可计作执行证据，不代表模型答对。修改后入口已实际执行其余197题；本批全200题均为stop，不声称本批额外实测了length分支。

输出含原提示、回答及评分参考，只能私有保存；公开回执只保留ID、SHA、配置、终态及分数。 CPU CI仅运行软件夹具、字段隔离、接口与身份校验，不下载原数据、不跑模型/性能、不包含科研观测或分析。
