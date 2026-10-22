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
