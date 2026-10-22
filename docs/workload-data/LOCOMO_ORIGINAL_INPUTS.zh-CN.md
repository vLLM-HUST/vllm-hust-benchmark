# LoCoMo 原始长会话问答输入

来源为[snap-research/locomo](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)，固定原始文件为`data/locomo10.json`。
该发布版有10段会话及1986道问答，是发布方构造并标注的长会话评测数据，不能称为企业生产请求或真实用户流量。
发布方声明CC-BY-NC-4.0；使用者须遵守原始条款，不能把本仓入口理解为无限制商业授权。本仓不分发原始会话、答案、发布方代码或图像。

## 输入与身份

每条读取记录对应原始一题，默认输入只含完整`conversation`和该题原始`question`。
保留所有原始会话轮次、日期字符串、说话人、`dia_id`、图像URL及发布方图像描述；不缩短、重排或补造会话。 日期字符串不是推理请求到达时间，也不隐含时区或实测KV复用。

答案、证据和对抗题字段在参考侧；类别、原始`sample_id`及父会话/题目位置在出处注释侧。 题目身份采用原文件摘要与展平序号，另保留父会话行号和该会话内题号，不用重复的会话ID冒充唯一问题ID。
发布方生成的`observation`、`session_summary`及人工`event_summary`不作为默认输入，不执行生成、检索、科研观测或分析。

第5类对抗题中有444条只有`adversarial_answer`而没有普通`answer`。读取器保留原字段，不改名、不填入自造答案、不丢弃该类题。
固定发布方`gpt_utils.py`的第5类生成路径存在直接访问`answer`的代码，不能仅凭读取成功声称这一原生路径可用；该兼容问题和随机选项构造须单独验证。
原发布版没有提供图像字节，URL和图像描述字段的存在不等于视觉推理已验证。

## 使用

按发布方条款单独取得固定文件，保存为`/data/locomo/data/locomo10.json`。下载后须与来源目录中的字节数及SHA256一致。
可通过固定Git版本的原始地址或GitHub官方Contents API取得，原始地址连接失败不应换用未核验镜像。

```python
from pathlib import Path
from vllm_hust_workload_data import iter_workload

for record in iter_workload("LOCOMO", data_root=Path("/data/locomo"), limit=3):
    task_input = record.task["input"]
    reference = record.outcome_record()  # 不传给模型
```

通用bind、pipeline和resolve_input共用同一题目展开视图。配置示例：

```json
{"sources":[{"source_id":"LOCOMO","data_root":"/data/locomo","limit":3}]}
```

准备输出须与原始数据目录分离；默认选原始展平0/1/2题，均来自首段完整会话，类别为2/2/3。
这不代表抽取了三段会话，更不覆盖全部类别。模型输入的提示格式、类别指令、上下文预算及评分实现须由原生执行协议另行明确。

## 当前证据

- [x] 官方固定文件取得，本地Git blob与发布方固定元数据一致，服务器SHA256再次一致。
- [x] 候选wheel独立安装，1986题输入和参考逐项对原始JSON检查，完整会话及原父/题位置一致；前三条准备及回查通过。
- [x] CPU夹具覆盖字段隔离、日期/轮次契约、对抗题、重复ID、字段冲突和输入回查；夹具不是原题执行证据。
- [ ] 固定最终CI工件及指定研究端消费联调。
- [ ] 原始前三题模型执行、终止、预算和适用的发布方评分：首次原0题请求HTTP500，编译阶段异常后服务退出，未返回答案，评分未运行；见[失败记录](validation/locomo-original-qa-failure-20261023.json)。未改题、裁剪上下文或提高预算。
- [ ] 第5类官方生成路径兼容、其他问题与图像任务。

详见[候选读取回执](validation/locomo-original-reader-20261023.json)。零模型调用、零任务执行、零评分，不因此勾选来源完成。
本能力与相关长会话/历史来源合批交付，不创建LoCoMo单独PR；原始载荷仅由使用者私有保存。
