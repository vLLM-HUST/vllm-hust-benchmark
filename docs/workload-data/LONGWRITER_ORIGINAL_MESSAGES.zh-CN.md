# 长输出写作：LongWriter-6k 原始消息接入

任务类别：`general-single-turn-assistant`；本批单独验证跨任务的“长输出”运行维度，不额外虚构第37个任务类别。来源是作者发布的LongWriter-6k原始长输出训练语料，不是LongBench-Write或LongWrite-Ruler评测套件，也不是企业生产流量。作者公开构造的语料与本仓库临时自行生成的数据必须区分。

## 原数据与部署

固定[zai-org/LongWriter-6k](https://huggingface.co/datasets/zai-org/LongWriter-6k) revision
`0db15c0624f19d63e2efe1021595af933cc5b6cc`，`long.jsonl`共6000行，完整SHA与来源目录匹配后才读取。原文件按发布方Apache-2.0条款单独取得，不随软件分发。

原始`messages`中最后一个assistant消息是参考回答，数据组件仅把此前消息作为输入，并保留原reference_message供评分侧使用。输入必须以原user请求结束；不会用参考回答、重新构造的问题、压缩消息或新增任务指令来代替原输入。运行前回查原文件，输入和参考两侧必须分别与原行相等。

首批预选原train行0、1、2。使用现有Qwen3.5-27B双卡910B2、graph/BF16、本地chat服务，不额外安装作者LongWriter权重；这属于显式替代模型，不是作者论文结果的复现。采用temperature=0.5、max_tokens=32768作为本次声明的生成配置。该语料不是API请求数据，没有可声称“原样保留”的请求预算；不把本配置伪装成原始请求参数或官方评估协议。

每题在生成前使用服务端tokenizer检查完整原消息与声明的输出预算是否适配当前128K窗口。超限拒绝，不截断。响应原文、原任务和参考回答保留在私有输出目录；公开只提交来源信息与脱敏回执。

## 验证边界

- [x] 原train行0–2三条消息回查、NPU生成、完整输入token一致、三条stop及响应完整性。[脱敏回执](validation/longwriter-original-npu-20261022.json)。实际输出2168、3064、5497
  token；未评估原题要求篇幅是否达标，不能把32768预算称实际生成长度。
- [ ] 适用语义质量/长度达标评估、其他任务和全部6000条。
- [x] 固定CI数据包逐文件核对、原6000条输入与独立参考字段逐条核查、全量索引和首尾6条prepare→resolve回查完成；旧三题保存输入引用、响应完整性和token记录已复核，模型与Tokenizer
  API调用均0。[CPU审计](validation/longwriter-cpu-audit-20261024.json)。旧运行未保存HTTP请求报文，不能将响应/原始输入复核说成有逐请求wire
  capture。
- [x] 原生入口补齐原始行区间与全量选项，保留默认前三题、原始行号、完整消息和声明的生成配置；新入口已实际执行末三条5997–5999并完成完整性复核，输出2398、2368、1557
  token，全部stop。与复用前三题合计6条唯一原始记录，不声称全部6000条或长度质量达标。[范围入口原生审计](validation/longwriter-tail-native-audit-20261024.json)。
- [x] 固定数据组件与研究消费包实际安装后，原前三条通过公共索引→研究读取和研究直接prepare→resolve两条路径，完整输入、独立参考与来源身份一致。没有新模型调用，也没有迁入研究观测/分析。[研究交接](validation/longwriter-research-handoff-20261024.json)。

该来源提供原始长写作请求，但没有为此接入配置部署的官方评分器。正常stop不等于满足原题要求的字数或质量；length终止不冒充正常生成，回答太短也不因为协议完整就声称长输出能力通过。这里不是原文生成长度和参考长度的研究分析，更不是正式性能验收。

## 使用

按[数据组件说明](../../workload-preparation/README.md)安装根包和独立数据包，然后运行：

```sh
python scripts/run_longwriter_original_samples.py \
  --data-root /authorized/LongWriter-6k-original \
  --output /private/fresh-longwriter-run \
  --endpoint http://127.0.0.1:19466/v1 \
  --model workload-qwen35-27b-128k-nonthinking
```

输出目录必须不存在。执行器使用标准库HTTP，不新增生产依赖。CPU smoke只测试软件夹具与边界，不加载模型、下载原始语料或运行性能测试。无需迁移科研观测、采集钩子或特征分析代码。

添加`--start 5997 --count 3`选择原文件最后三行，或`--full`选择完整6000行；全量与非默认区间不能混用，越界拒绝。准备索引包含所选原始位置所需前缀，模型只执行指定行，私有响应文件仍用原始行号命名。默认生成配置始终为temperature=0.5、max_tokens=32768，不因换区间而收缩输出预算或压缩输入；全部6000条是可执行入口能力，不是本次已跑完的声明。

运行环境需要Python>=3.10、精确数据组件及根包的响应完整性检查器；本入口HTTP与选择逻辑使用标准库，不要求另装官方语义评分器、CPU模型或科研依赖。末三条实跑使用Python3.12.13独立客户端，pip
check通过，安装的数据组件32份文件与固定CI工件逐字节一致，根包响应检查器与当前源码SHA一致。模型及完整启动参数复用[已核验双NPU部署](BBH_ORIGINAL_INPUTS.zh-CN.md)。原6000条内容仍只在授权服务器取得，不放入仓库/CI。具名语料生成接入完成见[交付回执](validation/longwriter-native-delivery-20261024.json)；语义质量、字数达标及全6000条仍未验证，不与接入完成混淆。
