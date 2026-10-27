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

### 固定原版第5类失败复现（2026-10-24）

原版共446条第5类题，其中444条缺少`answer`。在完整原文件与两份发布方源码SHA不变的条件下，固定对话0最先出现的三道第5类题，直接调用原`get_gpt_answers`与`eval_question_answering`函数：生成在模型请求前、评分在进入第5类评分分支前，均触发`KeyError: answer`。评分探测只用空的诊断预测触发路径，不将其当模型结果；模型API调用尝试为0，不据此声称统计了所有依赖模块的网络行为。见[原版预检回执](validation/locomo-category5-original-preflight-20261024.json)。

实时核对发布方main仍为固定3eb6f2c提交；官方仓库中的[PR #11](https://github.com/snap-research/locomo/pull/11)与[PR #41](https://github.com/snap-research/locomo/pull/41)提出同类字段读取修复，但截至本次核查均OPEN、未合并。不能把它们称为官方已修复版本，也不能把题目复制一份并填入`answer`来假装原版通过。后续兼容实现必须明确补丁来源、确切源码差异、原数据未改、类别1–4行为未改及第5类原始MC选项/评分规则保留；验证完成前此来源继续未勾。

## 显式最小兼容接入（2026-10-25）

提供公共入口`scripts/run_locomo_original_qa.py`，调用固定发布方`get_gpt_answers/get_input_context/get_cat_5_answer`与`eval_question_answering`，而不是重新手写第5类题目或换用QBI派生协议。原始数据和两份源码文件均按前述SHA校验且不改写；代码从固定模块选取所需定义，避免导入不用的外部付费模型、RAG和BERTScore依赖。

兼容开关`--allow-cat5-missing-answer-compat`默认关闭，启用时仅替换生成函数4处、评分函数2处`answer`读取：已有`answer`原值优先；只在第5类缺键时回退该题原`adversarial_answer`。其他类别缺键仍报错。第5类有两题同时存在两种字段且值不同，不能直接全局替换；本兼容实现保留两题原本工作的读取行为。原MC选项顺序由固定随机种子产生、`Not mentioned`选项、选项到预测的原映射和原评分分支均保留，不向原数据补`answer`。

这是本仓显式兼容实现，不是上游已合并修复；问题参考上游[PR11](https://github.com/snap-research/locomo/pull/11)（head
`82306173f26cb1474d79ffdc6010b48e2ff9ab97`）及[PR41](https://github.com/snap-research/locomo/pull/41)（head
`e671eb2731b648f9a83d94494705d221f8ec4082`），不能声称逐字应用这两个提案。与直接替换的区别是保留所有原本存在的`answer`，而不重新解释两个冲突样本。

[服务器CPU兼容回执](validation/locomo-missing-key-compatibility-20261025.json)核查全部1986项：1542次原值读取保留，444次缺键回退；1540个类别1–4的诊断评分路径一致，旧9个真实输出的原评分复算完全一致。空诊断预测只用于CPU分支测试，不是模型输出或新题通过。独立客户端安装及pip
check已通过，见[完整依赖快照](../../configs/workload-native/locomo-text-qa-py312-dependencies.txt)。

本轮原生选择在新结果产生前固定：第3类共96题，按每题约8秒及全会话前缀预留600秒估计全类约1368秒，因此全跑；其余较大类别各固定原顺序前三题。直接复用原9题，仅新增99题。估计不是实测性能结论，实际变慢也未缩小已固定范围。现新增99题全部完成原生生成及原评分，CPU审计逐题重放原提示、检查保存响应摘要及原选项映射，官方评分复算一致；审计模型调用0。合计108个唯一原题覆盖第3类全96题及其他四类各原前三题。见[原生审计](validation/locomo-original-native-audit-20261025.json)及[交付回执](validation/locomo-native-delivery-20261025.json)。支持按原入口继续全量，不宣称全部1986题或图像/RAG已执行。

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

### 原生生成及评分入口

原始文件由使用者按原条款在服务器取得；固定源码为同一提交的`task_eval/gpt_utils.py`及`task_eval/evaluation.py`。文件名可自选，内容必须匹配前述固定SHA。安装根源码以取得脚本，不需要科研观测包、RAG模型或BERTScore；另建Python3.12客户端：

```sh
python3.12 -m venv /private/locomo-client
/private/locomo-client/bin/python -m pip install -r configs/workload-native/locomo-text-qa-py312-dependencies.txt
/private/locomo-client/bin/python -m pip check
env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 \
  TIKTOKEN_CACHE_DIR=/private/locomo-client/tokenizer-cache \
  /private/locomo-client/bin/python scripts/run_locomo_original_qa.py \
  --data-file /private/locomo/data/locomo10.json \
  --gpt-source /private/locomo/task_eval/gpt_utils.py \
  --evaluation-source /private/locomo/task_eval/evaluation.py \
  --output /private/fresh-locomo-category5-run \
  --category 5 --allow-cat5-missing-answer-compat
```

默认原前三题；`--category 5`选择该类原顺序前三题，`--category 3 --full`选择全部96道第3类原题，`--full`选择全部1986题；重复`--flat-index N`显式绑定唯一、递增的原展平位置，不能与category/full混用。没有按结果重新取样、失败自动换题或自动重试。私有输出目录必须尚不存在，保存请求、响应、完整原QA及经原选项映射后的评分输入；不要公开这些文件。

真实执行保留原32-token预算、temperature0、batch size1、原函数内5秒等待、原日期提示及完整上下文。发布方tiktoken/cl100k预算计算与实际Qwen
tokenizer的131072部署窗口分别核查；任何原上下文截断或实际预算溢出均停止，不裁剪、不换题。模型和服务启动配置复用[现有双NPU部署](BBH_ORIGINAL_INPUTS.zh-CN.md)，不是GPT-4-Turbo论文结果复现。

`--preflight-only`仅准备原提示，跳过墙钟等待，用空的诊断返回值推动原生成分支，绝不发起模型请求或执行任务评分，不计原生完成。实际执行不使用这些占位值或跳过等待。CI只跑合成软件夹具，不下载数据/模型、不运行性能测试。

## 当前证据

- [x] 官方固定文件取得，本地Git blob与发布方固定元数据一致，服务器SHA256再次一致。
- [x] 候选wheel独立安装，1986题输入和参考逐项对原始JSON检查，完整会话及原父/题位置一致；前三条准备及回查通过。
- [x] CPU夹具覆盖字段隔离、日期/轮次契约、对抗题、重复ID、字段冲突和输入回查；夹具不是原题执行证据。
- [x] 固定04113e35
  CI工件及指定研究端b3e1f9f消费联调：31个软件payload核对、服务器1986题字段与前三条回查、双模式前三条及旧九条回归通过；两端对应CI均成功。见[固定工件消费记录](validation/locomo-fixed-input-consumption-20261023.json)。
- [x] 原始前三题模型执行、终止、预算和适用的发布方评分：恢复服务后原索引0–2均完成，三次请求均正常终止，发布方F1为0.6667/0/0.2857；使用Qwen3.5-27B替代发布方GPT-4-Turbo。见[首批重试记录](validation/locomo-original-qa-recovery-20261023.json)。
- [x] 追加原索引3–5：三次请求均正常终止，发布方F1为1/0.5/0.75；六题汇总均值0.5337。见[追加记录](validation/locomo-original-qa-recovery-batch2-20261023.json)及[六题汇总](validation/locomo-original-qa-sampled-20261023.json)。
- [x] 公共生成/评分入口及显式缺键兼容的CPU回归；旧9题保存结果的评分不变，详见上方兼容说明，不冒充新增模型运行。
- [x] 固定3f397e69数据组件实际安装后，原首三题与三道原第5类题共6条完成公共prepare→resolve及研究直接prepare→resolve；公共索引→研究分析入口另复核首三条。完整会话、问题及参考字段不变，第5类仍保留只有adversarial_answer而没有answer的原始结构。[研究端交接](validation/locomo-research-handoff-20261025.json)。这不是模型执行或研究观测验收，兼容开关没有改写数据层。
- [x] 新99题原生批次及保存结果审计完成，复用原9题合计108个唯一原题；公共执行脚本与服务器实际执行文件SHA一致，原始输入、32-token预算、原等待、选项映射和评分均核查。
- [ ] 其余1878题的原生执行、图像字节任务、RAG及生成摘要协议未验证；抽样交付不代表这些范围完成。

首次原题执行的[HTTP500失败回执](validation/locomo-original-qa-failure-20261023.json)仍保留。重试只把编译临时目录切到实例本地`/tmp`并使用新缓存目录；该改动没有改变模型和请求参数，也没有证明NFS故障根因。读取及跨仓固定工件回查均为零模型调用、零任务执行、零评分，相关回执继续独立保存。
本能力与相关长会话/历史来源合批交付，不创建LoCoMo单独PR；原始载荷仅由使用者私有保存。

## 第4类原题补验（2026-10-24）

原对话0中第4类最先出现的三题（原问题索引82、83、84）在结果产生前固定选择，沿用固定发布方上下文拼接、提示和F1函数；完整原会话未截断、不注入生成记忆，保留原32-token预算。现有双NPU
Qwen3.5-27B作为公开GPT-4配置的替代模型，不称论文复现。

三次生成均正常stop，服务输入token与预检一致（18098、18098、18096），发布方F1为1、0.8571、0.4762。见[脱敏回执](validation/locomo-category4-native-20261024.json)。当时累计9个唯一原题覆盖第1–4类，旧回执的第5类未验证状态不改写；现交付状态由上面的新增审计与交付回执补足。不合并跨运行成绩、不称全量，也没有把QBI派生题或FrozenEvidence引入默认路径。
