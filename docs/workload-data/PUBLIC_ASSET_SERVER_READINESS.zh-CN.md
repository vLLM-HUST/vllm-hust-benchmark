# 公开原始资产：服务器准备与任务验证缺口

## 已验证的数据链路

2026-10-23 将此前仅本地验证的七个来源的选定原始文件传到当前授权服务器。 传输前对照固定来源目录核验原始摘要；服务器读取时再次核验文件摘要。 使用 #431 固定提交
[84b901f6](https://github.com/vLLM-HUST/vllm-hust-benchmark/commit/84b901f673c30ba8dce2c73038dc52e1c5cc9932)
的成功 CPU CI 39862575744 工件，实际安装于独立 site-packages 环境。
每源预选原始行0、1、2，完成原始读取、prepare、resolve，并分别比较输入和参考字段。 本记录不是模型执行或官方评分结果；原始文件、输入、答案、图像及完整数据库不随文档发布。

| 来源              | 选定原始文件                                      | 数据链路 | 运行类型            | 原生任务与评分                                                      |
| ----------------- | ------------------------------------------------- | -------- | ------------------- | ------------------------------------------------------------------- |
| BIGCODEBENCH      | data/v0.1.4-00000-of-00001.parquet                | 三条通过 | task_dataset        | 原三题NPU生成3stop；功能执行与评分尚需隔离环境                      |
| DS1000            | test.jsonl                                        | 三条通过 | task_dataset        | 原三题NPU生成2stop/1length；原版功能判定未运行                      |
| SWE_VERIFIED      | data/test-00000-of-00001.parquet                  | 三条通过 | task_dataset        | 未运行；需原仓库版本、修复 Agent、测试容器                          |
| SCIENCEAGENTBENCH | data/verified-00000-of-00001.parquet              | 三条通过 | task_dataset        | 未运行；需原任务资产、执行环境和判定                                |
| TOOLACE           | data.json                                         | 三条通过 | historical_dialogue | 原始发布方合成历史工具对话；不是可执行 Agent 环境，不生成假 episode |
| CHARTQA           | data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet | 三条通过 | task_dataset        | 未运行；当前服务为 language-model-only，不能据此宣称图像推理        |
| TRACELAB          | syfi_coding_trace.jsonl.gz                        | 三条通过 | invocation_metadata | 元数据读取不是原始请求重放；未调用模型/工具或运行采集分析           |

所有选定记录均通过，合计21条；每个来源只取表中一个原始文件。 ChartQA 的训练/验证分片未传到本次服务器，不将这三条结论扩展到其他文件。 来源版本、原路径和摘要由该固定工件中的
source_manifests_v1.json 定义， 使用者不能更换同名文件后沿用结论。三个代码/修复任务也没有因能读入就执行生成代码。

## 执行与证据边界

- [x] 七个选定原始文件按固定目录摘要核验并私有传入服务器。
- [x] 固定 CI wheel 安装，21条输入及参考的独立往返通过。
- [ ] 适用的模型生成、原生完整任务终止和官方评分。
- [ ] 各原生执行环境、模态能力和安全隔离逐项核验。
- [x] 固定数据版本653ad351的最终CI工件已在服务器重新安装并完成七源21条往返；研究端原PR
  #37亦完成双模式21条及旧来源9条回归，四项CI成功。见[固定版本消费记录](DATA_CONSUMPTION_VALIDATION.zh-CN.md)。

上述七来源数据链路检查的模型调用0、原生任务执行0、评分调用0。
后续同三条BigCodeBench原题已另完成NPU生成阶段，见[独立记录](BIGCODEBENCH_ORIGINAL_GENERATION.zh-CN.md)；未运行功能评分，不改变数据链路回执范围。
DS-1000同三条Pandas原题也完成[单独生成探测和入口复跑](DS1000_ORIGINAL_GENERATION.zh-CN.md)，其中一条原预算length；未运行功能评分，不证明科学闭环。
这里不改写题目、不注入解题证据，不将历史对话或元数据伪装成在线任务。 已有HumanEval隔离探测失败不能被绕过后裸执行生成代码；其他套件仍需逐项检查。
本批随[Draft PR #432](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/432)合并交付多个来源；不是逐数据集开PR。最终增量CI工件与消费端核验状态见PR，不将生成阶段或读取结果混作完整任务通过。

## 独立QASPER原发布版接入准备（2026-10-25）

这是登记来源`QASPER`的原v0.3任务包，不是已经交付的`LONGBENCH`派生Qasper200题。发布方固定[HF加载器](https://huggingface.co/datasets/allenai/qasper/blob/7bde7e4a9f805bed6c6e79e664c37057263dbc1d/qasper.py)给出原S3归档地址；服务器直接取得两个原归档，字节数和SHA与Hub记录一致，未使用镜像或改题。参见[获取与原题清点](validation/qasper-original-acquisition-20261025.json)。

原train/dev/test分别为888/281/416篇论文、2593/1005/1451个问题，合计1585篇论文、5049个问题。按原paper对象顺序和原qas顺序展开，一题一记录；输入保留完整title、abstract、full_text和figures_and_tables原caption/file字段，以及该题原question。所有答案标注、证据、yes/no、不可回答标签及高亮证据完整留在参考侧；不把兄弟题、参考答案或证据注入输入。保留原paper
ID、question ID、父/题位置和问卷元数据，不把论文数当问题数。caption/file不是图像字节，也不证明视觉推理可用。

安装包含此增量的匹配数据组件后，在服务器私有目录取得原归档：

```sh
python -I scripts/prepare_qasper_original_assets.py \
  --train-dev-archive /private/qasper-train-dev-v0.3.tgz \
  --test-archive /private/qasper-test-and-evaluator-v0.3.tgz \
  --output /private/fresh-qasper-v03
```

入口只核验并提取三个原JSON及归档自带`publisher-assets/qasper_evaluator.py`，保留原字节，拒绝错摘要、歧义成员及覆盖输出目录；不执行归档代码、不下载数据或调用模型。`workload-data catalog download-plan QASPER`列出归档URL和成员名；URL指向归档，不能把归档直接另存为同名JSON。再使用`iter_workload("QASPER", data_root=Path("/private/fresh-qasper-v03"), files=["qasper-dev-v0.3.json"])`读取指定split，或不指定files读取三个split；不指定limit就是全量。所有记录均拒绝伪造现成API请求体。

统一索引配置为`{"sources":[{"source_id":"QASPER","data_root":"/private/fresh-qasper-v03"}]}`，沿用pipeline
plan/run及resolve_input入口。服务器独立安装候选wheel后，33个包内payload逐字节一致；全部5049条输入、参考及父题关系与原JSON逐项一致，全部5049条索引完成，各split首尾各3条共18条回查通过，见[安装候选包CPU审计](validation/qasper-original-provider-cpu-audit-20261025.json)。这是本地构建的软件候选工件，不冒充成功CI工件或研究端已接通；没有在本地下载原数据。

原任务同时评价答案与证据，归档中的发布方评分器已保留，但本轮尚未进行模型生成、官方评分或研究消费端新版本联调。后续须采用适配模型的完整原任务执行与原评分；不能用LongBench派生提示及QA-F1替代原QASPER证据评价，也不能用诊断预测或读取通过充当原生执行。本来源保持未勾，企业数据及科研观测/分析均不在本增量中。
