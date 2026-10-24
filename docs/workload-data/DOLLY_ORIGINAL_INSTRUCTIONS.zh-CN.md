# 通用单轮指令：Dolly 原始字段生成接入

类别：`general-single-turn-assistant`。Dolly 是 Databricks
发布的指令语料，不是带原生执行环境和官方评分器的独立benchmark。这里只交付原始字段读取、准备、输入隔离和生成完整性验证，不将其包装成官方质量测评。

## 固定来源与输入契约

原数据来自
[databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)，固定
revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`，完整JSONL共15011行，使用仓库来源目录中的完整SHA校验。数据按发布方CC BY-SA
3.0条款单独取得；本PR不重新分发题目或参考回答。

数据组件返回原始`instruction`、`context`，`response`仅进入outcome，不进入模型输入。预先选定原train行0、1、2，不按结果筛选，对应closed_qa、classification、open_qa，不代表所有类别均验证。

提示使用发布方[固定源码training/consts.py](https://github.com/databrickslabs/dolly/blob/e2d664ddf06e0723d818dc56d6c522ead4bd881d/training/consts.py)中有/无context的训练模板，在`{response}`槽之前取前缀，分别代入未经修改的instruction/context。源码文件完整SHA验证后才能加载。此处是明确的推理接入选择，并非作者的官方评估协议；特别是有context时不假装其与上游无context的生成示例完全相同。原参考response和训练结束标记不拼入输入。

部署使用已有Qwen3.5-27B双卡910B2服务，而非额外部署发布方Dolly权重；模型替换、chat传输、temperature=0、max_tokens=2048均显式记录。每题先核对完整模板化输入容量，实际usage须与预检一致，不截断原始字段。超预算、服务错误或完整性失败不改记成功。

## 评分边界

- [x] 完整原文件SHA、前三条原字段回查、发布方固定模板身份验证。
- [x] 三条NPU生成、完整输入token一致、三条stop及响应完整性。[脱敏回执](validation/dolly-original-npu-20261022.json)。回执/退出状态/格式加固过程中的重复验证不增加唯一任务数，合计仍为三题。
- [ ] 原参考回答的适用语义评价、其他类别和完整15011条。

原数据含参考回答，但没有为这一生成接入提供的官方语义评分器。不以任意字符串完全相等、合法文本或非空输出冒充回答正确。生成核验只检查完整输入、响应协议、内容与终止完整性；参考回答保留在原文件/评分侧，下游可以另行选择并声明适用的质量评测方法。

15011条不属于几分钟即可默认全跑的小集，首批先验证原始前三条；不会以三条样本推断类别覆盖、代表性、准确率或性能。CPU CI只运行软件夹具，不拉取真实数据、不启动模型。

## 使用

```sh
python scripts/run_dolly_original_samples.py \
  --data-root /authorized/dolly-original \
  --publisher-constants /authorized/dolly-fixed/training/consts.py \
  --output /private/fresh-dolly-run \
  --endpoint http://127.0.0.1:19466/v1 \
  --model workload-qwen35-27b-128k-nonthinking
```

先按[数据组件安装说明](../../workload-preparation/README.md)安装根包及独立数据包。执行器使用标准库HTTP，不另加生产依赖；真实文件、固定发布方源码、原始输入/输出和参考回答均留在授权目录。输出目录必须新建。不是将科研观测、特征分析或机制实验迁入benchmark。

## 八类原请求的固定覆盖（2026-10-24）

以上三题回执保留为历史证据。本轮按每个原始`category`在原文件中首次出现的三条选择，八类共24个唯一原请求；其中原索引0、1、2复用上述回执，新增21条在双NPU服务完成生成和响应完整性核验，均正常stop。没有换题、截断或参考答案注入；数据集整体仍为15011条，不把24条称为全量。见[类别覆盖回执](validation/dolly-original-categories-20261024.json)。

新入口复用旧入口的发布方模板拼接和HTTP代码，不另造提示或评分器。服务器以该入口对保存的21条实际请求/响应离线复核，逐条请求与原字段和同一模板一致、响应完整，新增模型调用为0；CPU类别选择/协议复用/路径、端点及回执检查与旧入口合计12项通过。固定提交8c4c62e0的CPU
smoke、lint及主CI tests均通过，再使用固定3f397e69
CI数据wheel运行相同离线核对，通过21条新请求及3条旧证据复用，0新增模型调用。见[请求语料抽样交付回执](validation/dolly-request-corpus-delivery-20261024.json)，按此范围打勾，非全量或官方质量测评。

客户端仅需匹配根软件包和独立数据wheel，使用Python
3.12标准库；模型工件及完整双NPU启动命令见[已验证服务环境](BBH_ORIGINAL_INPUTS.zh-CN.md#%E6%9C%AC%E8%BD%AE%E5%B7%B2%E9%AA%8C%E8%AF%81%E7%9A%84%E5%8F%8Cnpu%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E9%85%8D%E7%BD%AE)。调用者须核实服务实际模型工件与配置；相同模型别名本身不证明权重身份相同。

```sh
# 全新验证：按原类别各取前三条，少于三条时取全部。
python scripts/run_dolly_original_categories.py \
  --data-root /authorized/dolly-original \
  --publisher-constants /authorized/dolly-fixed/training/consts.py \
  --output /private/fresh-dolly-categories \
  --max-model-len 131072
# 若有同来源/模板/模型/预算且完整性已验证的旧回执，可额外给出：
# --reuse-receipt /private/prior-dolly-receipt.json
# 所有15011条：在上述命令后显式追加 --full。
```

`--verify-existing-run EXTERNAL_RUN`替代`--output`，检查已有私有运行的固定选择、终态、原请求和保存响应，不发模型请求。该模式适用于本入口格式及本轮具名私有运行；不把任意第三方日志解释成已验证结果。不支持任意改选索引、缩短预算、自动失败重试或换题；参考回答仍保留在原文件与outcome，不进入生成请求。没有独立语义评分器，因此输出完整性不等于回答正确，亦不报告虚构的准确率。
