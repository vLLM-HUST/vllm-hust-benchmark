# JSONSchemaBench 原始约束资产接入与验证边界

## 输入与读取

JSONSchemaBench 是发布方用于评估结构化生成的 schema 集合，不是线上请求记录。 `JSONSCHEMABENCH` 适配器将每个原始 JSON 文件作为一个约束资产，保留全部
schema 关键字，不生成 prompt、不补 `required`、不修改 `additionalProperties`，也不访问 `$ref`。
文件字节摘要与行索引共同定位输入；一文件计一条，不把 schema 的属性拆成请求。 对象和布尔 schema 均可读取；读取成功不代表后端支持全部关键字。

当前生产目录尚未固定下载清单。使用方必须通过现有 bind 入口显式绑定自己取得的 原始文件及来源版本；数据不会随 wheel 分发，读取入口不隐式下载。

## 已执行的原始样本

数据版本为 `guidance-ai/jsonschemabench` 的 `9a94995b9279ae3af3aed4b2629172790b968d14`；发布方代码版本为
`epfl-dlab/jsonschemabench` 的 `8003e8405c4d8d8b327b1eb472c9856297d75493`。 在执行前固定选择 `Github_easy`
目录按路径字典序排列的前三个文件。 未改 schema，未按生成结果重新挑选样本。

| 原始文件    | 生成终止 | JSON 可解析 | 发布方 schema 校验 | 失败关键字                    |
| ----------- | -------- | ----------- | ------------------ | ----------------------------- |
| o10008.json | stop     | 是          | 通过               | 无                            |
| o10009.json | stop     | 是          | 不通过             | properties.clientId.maxLength |
| o10010.json | stop     | 是          | 不通过             | properties.clientId.maxLength |

三个输入分别为 788、828、845 prompt tokens；预检与实际 usage 相符。 三个输出分别为 21、88、190 tokens。独立使用 jsonschema 标准
FormatChecker 复核已有输出，得到相同的两处 maxLength 错误；独立诊断不替换发布方评分。 这些结果证明三次生成已结束，但原约束符合率仅 1/3，不能记为全部通过。
目前不能仅凭结果判定是约束后端忽略了关键字还是其他后端限制。

## 执行环境与协议差异

使用现有双卡 Ascend 910B2、Qwen3.5-27B、TP=2、BF16、131072 上下文服务。 模型不是发布方参考模型；本次不产生性能比较或论文复现结论。 调用固定发布方
`core/messages.py` 的原始 few-shot formatter（两组演示）和 `core/evaluator.py` 的原始校验函数；模型输入使用原 schema 的
OpenAI-compatible `response_format.json_schema`，非流式、temperature=0、
max_tokens=1024、enable_thinking=false。

发布方 OpenAIEngine 的 adapt_schema 会改变约束，包括 required 和 additionalProperties。本次没有使用这种改写，不应冒称该参考引擎复现。 原始纯
prompt/validator 模块通过独立 namespace 加载，绕过自动导入可选模型引擎的 core 初始化器；原文件未修改，但这不是完整参考环境。 原始输出、演示和 schema
仅保留在服务器私有目录，不随本说明发布。

## 待完成检查

- [x] 原始三个约束的 NPU 生成与发布方校验。
- [x] 对失败输出做独立标准校验，保留原始失败结论。
- [x] whole-schema 读取适配与合成夹具 CPU 单测。
- [ ] 最终 wheel 安装后原始三文件读取、prepare、消费索引往返。
- [ ] 可复用执行入口及固定环境安装文档。
- [ ] 与研究消费端完成最终固定工件联调。
- [ ] 完整子集及其他 schema 子集验证。

上述未完成项不能由本次三样本探测或 CPU 测试代替。该来源与长文档理解类别 一起进入下一批 PR，不为单个数据集创建独立 PR。
