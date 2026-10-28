# JSONSchemaBench 原始约束资产接入与验证边界

## 输入与读取

JSONSchemaBench 是发布方用于评估结构化生成的 schema 集合，不是线上请求记录。 `JSONSCHEMABENCH` 适配器将每个原始 JSON 文件作为一个约束资产，保留全部
schema 关键字，不生成 prompt、不补 `required`、不修改 `additionalProperties`，也不访问 `$ref`。
文件字节摘要与行索引共同定位输入；一文件计一条，不把 schema 的属性拆成请求。 对象和布尔 schema 均可读取；读取成功不代表后端支持全部关键字。

来源目录固定了本次三个原始文件的版本、摘要、大小和一文件一条的计数。使用方可按目录核验后读取，或通过现有 bind 入口显式绑定自己取得的其他原始文件及来源版本；数据不会随 wheel
分发，读取入口不隐式下载。

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

## 安装与显式执行

在独立 Python 3.12 环境安装纯 prompt/validator 所需依赖，不安装或启动发布方 可选 CUDA 引擎，不改变现有 NPU 服务环境：

```bash
python3.12 -m venv env-jsonschema
env-jsonschema/bin/python -m pip install jsonschema==4.23.0 numpy==2.2.3 dacite==1.9.2 omegaconf==2.3.0 matplotlib==3.10.1 prettytable==3.15.1
env-jsonschema/bin/python scripts/run_jsonschemabench_original_schema.py \
  --schema-root /private/original-data \
  --official-code /private/original-code \
  --output /private/run-new-001 \
  --model workload-qwen35-27b-128k-nonthinking \
  --origin http://127.0.0.1:19466 --context-capacity 131072
```

`schema-root` 下必须有 `Github_easy/o10008.json`、`o10009.json`、`o10010.json`； `official-code` 下必须有固定版本的
`core/`。入口核验原始文件、prompt 和 validator 摘要；来源版本见前文。首次执行前自行取得原始代码和数据，并确认适用许可。 入口仅允许本机 HTTP
服务，不接收远端凭据，不自动下载或回退模型。 输出目录必须不存在且与原始数据、代码目录分离；请求和输出保存在私有文件。 完整 prompt 预检、实际 usage、终止原因、JSON 可解析与
schema 合法性分别记录。 `GENERATED` 不是正确性通过；HTTP 拒绝和执行异常另记为失败。 本入口明确限制到三个已固定的无 `$ref` 原始资产，不代表支持全库引用解析。

## 待完成检查

- [x] 原始三个约束的 NPU 生成与发布方校验。
- [x] 对失败输出做独立标准校验，保留原始失败结论。
- [x] whole-schema 读取适配与合成夹具 CPU 单测。
- [x] 候选 wheel 在服务器独立安装，原始三文件逐一 bind、prepare、resolve，输入对象完全相等；此项未调用模型。
- [x] 同一候选 wheel 回归 QuALITY 2086 问题读取/建索引及三个原始问题往返。
- [x] 固定CI `3f397e69`
  数据wheel已安装并逐文件核对；原始三文件读取、bind、prepare、resolve逐对象相等，保存的发布方two-shot提示、原schema请求和官方validator重算均一致。[CPU复核](validation/jsonschemabench-cpu-audit-20261024.json)。模型及Tokenizer
  API调用0，保留两条maxLength失败，不新增独立任务数。
- [x] 可复用执行入口及固定环境安装文档；已在现有 NPU 服务执行相同三个任务。
- [x] 原三份schema资产与研究消费端完成固定工件双路径联调：公共索引→研究读取、研究直接prepare→resolve各3条，完整原schema和来源身份一致，不增删约束、不调用模型。[研究接通回执](validation/jsonschemabench-research-handoff-20261024.json)。
- [ ] 完整官方`core.bench.bench`流程及其他原分区。已核查固定官方`core.dataset.Dataset`读取`epfl-dlab/JSONSchemaBench`的`test`及`json_schema`字段、10个具名分区；官方允许实现`core.engine.Engine`自定义后端，后续应复用这一协议而非仅拼接formatter/validator。服务器取得该公开仓库元数据时返回`httpx.ConnectError: Errno101 Network is unreachable`，因此尚未取得并核实其固定revision，不填写预期摘要冒充下载成功。不改网络、不从本地缓存原数据，保留原三份样本和失败结果。[官方流程准备及阻塞](validation/jsonschemabench-official-workflow-readiness-20261024.json)。来源整项仍不打勾。
- [ ] 完整子集及其他 schema 子集验证。

上述未完成项不能由本次三样本探测或 CPU 测试代替。该来源与长文档理解类别 一起进入下一批 PR，不为单个数据集创建独立 PR。

### 安装验证记录

候选 wheel SHA256： `a3e11c1b64f3edef3f94f5482fc18b253610da06388a40f3cb608928a727b251`。 服务器公开入口复跑目录为
`jsonschemabench-public-runner-002`，三个任务结果与首次探测一致。 这是三个独立任务的重复执行，不累计为六个或九个独立任务。 最终 CI
工件及消费端联调仍待完成，不能用此本地构建候选替代。 首次私有安装验证助手误将“无 outcome”假定为空对象；按公共类型契约应为 None。
该次助手断言失败已保留，修正助手后重新验证通过，未修改生产契约或放宽输入比较。

2026-10-24补充：在独立原评分环境（jsonschema4.23.0、numpy2.2.3）安装并核对指定成功CI数据工件后，上述三资产的准备及旧响应复核已完成；研究端最终工件联调仍未完成。先前候选wheel/复跑记录保留为历史，不与当前固定CI来源混用，也不把三文件验证外推到全部schema集合。等待研究窗口期间不调用模型服务、不启动新的生成。
