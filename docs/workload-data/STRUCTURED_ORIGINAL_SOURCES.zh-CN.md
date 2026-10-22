# 结构化输出：原始 Text→JSON 任务与企业请求

类别：`structured-generation`。采用发布方原始任务，不用自行重写的prompt替代。企业PAIO-JSON是请求数据，不称独立权威benchmark；其既有三条回放/格式核验没有独立语义gold，继续单列。

## 已验证与未验证

- [x] StructEval原始Text→JSON前三条：task_id
  000500/000501/000502，原文件行1085–1087。官方语法/路径评分1、1、0.68，错误保留。[回执](validation/structeval-json-original-npu-20261022.json)。
- [x] 修正后的数据包已在服务器安装，对全部50条Text→JSON原题进行准备/回查，原query一致，VQA与raw_output_metric未进入输入/注释。[组件回查](validation/structeval-provider-roundtrip-20261022.json)。此步骤模型调用0，不是额外50条推理。
- [x] 全部50条原始Text→JSON已完成NPU执行及官方评分，平均final_eval_score为0.9616，其中3条未满分。[完整子集回执](validation/structeval-text-json-full-npu-20261022.json)。它是语法/路径分数，不是语义准确率；与前三条重叠，合计唯一任务仍为50。
- [ ] 其他输入格式转JSON、其他输出格式、可视渲染/VQA均未验证。完整StructEval共2035条，不能把Text→JSON子集叫完整套件。
- [ ] PAIO-JSON的语义准确率：未提供独立gold；保持不评估，不因合法JSON或schema通过而打勾。

## 原始数据和官方实现

固定HF数据 `TIGER-Lab/StructEval` revision `936be10548688c6d417cd435dd906158f19ba14b`，原train
parquet完整SHA校验。该文件在数据卡中对应test split，不依据文件名把它当训练集重划。

官方仓库作者推荐的LiteLLM分支固定提交
`7781339a5b033965ac664133543ebd10773643dd`，完整源码归档SHA与安装Python源码一致。官方`build_prompt(query)`加原有代码标记，官方workflow执行生成、语法解析和路径评分；不是自行追加新提示/答案或自制评分器。

使用现有Qwen3.5-27B、双卡910B2、BF16/graph、本地NPU服务，text-only/nonthinking配置。原官方workflow的temperature=0、max_tokens=None，concurrency=1、render_concurrency=1，关闭重试并设置请求timeout300秒。只对非渲染JSON调用原生语法/路径检查，不调用VLM或外部judge；这一分数不等同通用语义正确率。

## 输入与评分隔离修正

原目录把`VQA`参考问答置于annotation_fields，而输入envelope包含replay.annotations，边界不严谨。本批移到outcome_fields，与raw_output_metric一起仅供评分侧。原始文件不修改，task
query不改；当前官方模型运行只读取query，因此既有Text→JSON得分不是因这次修正而改变。回归测试检查整个input_record，而非只检查task.input，避免注释旁路带入答案。

## 运行

```sh
python scripts/run_structeval_original_json.py \
  --data-file /authorized/StructEval/data/train-00000-of-00001.parquet \
  --official-archive /authorized/structeval-official.zip \
  --output /private/fresh-three \
  --endpoint http://127.0.0.1:19466/v1 --model workload-qwen35-27b-128k-nonthinking
```

添加`--all-json`执行全部50条原始Text→JSON，不包括格式转换到JSON。程序强制新建私有输出目录，校验原始数据/源码，模型传输仅许可批准的本地端点。原始选择及推理输出包含题目、评分参考与模型回答，不发布Git/CI。

任务环境独立安装官方源码及render extra（上游模块导入需要，JSON子集不会启动浏览器），固定LiteLLM1.65.0、OpenAI
SDK2.54.0。库修复后的数据wheel与原始官方执行器是两个组件；研究侧只消费数据组件，不迁入观测、性能剖析或研究分析。CI只验证临时软件夹具和CLI边界，不取真实数据或模型。
